# file: hypergraph_extractor/out_sem_2/upload_to_atomspace.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import json, re, sys, logging, csv, itertools, collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from opencog.atomspace import AtomSpace, types
from opencog.type_constructors import TruthValue
import hyperon
from hyperon import SymbolAtom, ExpressionAtom, GroundedAtom 
import itertools
import re
try:
    # Hyperon ≥0.4:  MeTTa in root package
    from hyperon import *
except ImportError:
    # Hyperon 0.3.x:  MeTTa lives in runner
    from hyperon.runner import MeTTa
    
import importlib.resources

stdlib_path = importlib.resources.files("hyperon").joinpath("stdlib.metta")
print(stdlib_path.as_posix())
logging.basicConfig(
    level=logging.DEBUG if "-v" in sys.argv else logging.INFO,
    format="%(levelname)s | %(message)s")
log = logging.getLogger("loader")

try:
    from hyperon import  GroundingSpace, S, E, SpaceRef
except ImportError as e:
    log.error("No hyperon {e}")
    Space = S = E = None

def _sym(txt: str):
    """Return a Symbol atom (stripped of surrounding quotes)."""
    return S(str(txt).strip('"'))

def json_to_hyperon_space(hg_json: dict) -> "SpaceRef":
    """
    Convert an AtomSpace-style hypergraph JSON into a Hyperon Space.
    – Nodes   →  Symbol atoms (S)
    – Links   →  Expression atoms (E) whose head is a Symbol of link_type
    – Truth-values: not yet in python-hyperon → ignored (can be re-added
      later with grounded atoms when the spec is stable)  :contentReference[oaicite:0]{index=0}
    – Attributes on EvaluationLinks become
      (E (S 'has-attr') <link-expr> (S key) (S val))
    """
    space = SpaceRef(GroundingSpace())           # empty in-memory space
    id2atom: dict[str, "Atom"] = {}

    # 1.  nodes ──────────────────────────────
    for n in hg_json.get("nodes", []):
        id2atom[n["id"]] = _sym(n["name"])

    # 2.  links ──────────────────────────────
    def _expr_for_link(link: dict):
        # build child atoms (recursively for nested links)
        child_atoms = []
        for tok in link.get("arguments", []):
            if tok in id2atom:          # node or previously built link
                child_atoms.append(id2atom[tok])
            else:                       # forward ref → first build link
                child_atoms.append(_expr_for_link(link_defs[tok]))
        # predicate-only (EvaluationLink) uses separate key
        if link["link_type"] == "EvaluationLink":
            pred = id2atom[link["predicate"]]
            e = E(_sym("EvaluationLink"), pred, *child_atoms)
        elif link["link_type"] == "ContextLink":
            e = E(_sym("ContextLink"), *child_atoms)
        elif link["link_type"] == "ImplicationLink":
            e = E(_sym("ImplicationLink"), *child_atoms)
        else:                                   # ListLink or other
            e = E(_sym(link["link_type"]), *child_atoms)
        id2atom[link["id"]] = e
        space.add_atom(e)

        # optional JSON attributes  → has-attr
        if attrs := link.get("attributes"):
            for k, v in attrs.items():
                space.add_atom(E(_sym("has-attr"), e, _sym(k), _sym(v)))
        return e

    link_defs = {l["id"]: l for l in hg_json.get("links", [])}
    for lid in link_defs:
        if lid not in id2atom:          # build each link once
            _expr_for_link(link_defs[lid])

    # 3.  (optional) metadata → separate branch
    meta = hg_json.get("metadata") or {}
    if ss := meta.get("source_sentences"):
        pred = _sym("raw_text")
        for sent in ss:
            sent_sym = _sym(sent["id"])
            txt_sym  = _sym(sent["text"])
            # print(txt_sym)
            space.add_atom(E(pred, sent_sym, txt_sym))

    return space



def _children(expr):
    """
    Return iterable of children on any Hyperon version:
    • 0.4.x:        expr.get_children()
    • older ← C++:  expr.size() / expr.get_child(i)
    """
    if hasattr(expr, "get_children"):            # 0.4, dev branch
        return expr.get_children()
    if hasattr(expr, "size"):                    # 0.3 / legacy
        return [expr.get_child(i) for i in range(expr.size())]
    raise TypeError("Cannot iterate children of ExpressionAtom")

def to_metta(atom) -> str:
    """Force explicit (S …) / (E …) wrappers, version-agnostic."""
    # — symbol --------------------------------------------------------
    if isinstance(atom, SymbolAtom):
        name = str(atom)                # str() works on all builds
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", name):
            return f"(S {name})"
        return f'(S "{name}")'

    # — expression ----------------------------------------------------
    if isinstance(atom, ExpressionAtom):
        parts = " ".join(to_metta(c) for c in _children(atom))
        return f"(E {parts})"

    # variables, grounded atoms, etc.
    return str(atom)

def export_to_metta(hg_json: dict, out_path: Path, space_var: str = "&hg"):
    nodes      = {n["id"]: S(n["name"])     for n in hg_json["nodes"]}
    link_defs  = {l["id"]: l                for l in hg_json["links"]}
    id2atom    = dict(nodes)

    # ❶  guarantees the same handle EVERYWHERE
    #header = f"!(bind! {space_var} (new-space))"
    #lines: list[str] = [header]
    lines = [
    f'!(import! &self neurospace)',
    f'!(bind! {space_var} (new-space))', 
    ]
    print(lines)

    # ───────── internal builder ─────────
    def build(uid: str):
        if uid in id2atom:
            return id2atom[uid]

        link = link_defs[uid]
        lt   = link["link_type"]
        A    = lambda x: build(x)           # resolve id recursively

        if lt == "EvaluationLink":
            if link.get("wrap_listlink", True):
                lst = E(S("ListLink"), *(A(a) for a in link["arguments"]))
                lines.append(f"!(add-atom {space_var} {to_metta(lst)})")
            head = A(link["predicate"])
            expr = E(head, *(A(a) for a in link["arguments"]))

        elif lt == "ContextLink":
            *ctx, thing = link["arguments"]
            expr = E(S("context"), *(A(c) for c in ctx), A(thing))

        elif lt == "ImplicationLink":
            prem  = [A(link["arguments"][0])]
            concl = [A(link["arguments"][1])]
            expr  = E(S("=>"),
                       E(*prem)  if len(prem)  > 1 else prem[0],
                       E(*concl) if len(concl) > 1 else concl[0])

        elif lt in {"TypeLink", "MemberLink"}:
            expr = E(S(lt), *(A(a) for a in link["arguments"]))

        else:                               # fallback
            expr = E(S(lt), *(A(a) for a in link.get("arguments", [])))

        id2atom[uid] = expr
        return expr

    # build every link exactly once
    for lid in link_defs:
        build(lid)

    # ❷  write *nodes first*, then links (order is cosmetic but clearer)
    for sym in nodes.values():
        lines.append(f"!(add-atom {space_var} {to_metta(sym)})")

    for lid, expr in id2atom.items():
        tv = link_defs.get(lid, {}).get("truth_value")
        tv_s = f"; tv {tv[0]} {tv[1]}" if tv else ""
        lines.append(f"!(add-atom {space_var} {to_metta(expr)}){tv_s}")

    # optional metadata
    meta = hg_json.get("metadata") or {}
    if ss := meta.get("source_sentences"):
        pred = S("raw_text")                # ❸  no hidden helper
        for sent in ss:
            sent_sym = S(sent["id"])
            txt_sym  = S(sent["text"])
            expr     = E(pred, sent_sym, txt_sym)
            lines.append(f"!(add-atom {space_var} {to_metta(expr)})")

    out_path.write_text("\n".join(lines) + "\n")
    log.info("✅  wrote %s  (%d atoms)", out_path, len(link_defs))

    
    
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?") # search numbers in sheme output of atomspace
def _num(txt: str) -> str:
    m = _NUM_RE.search(str(txt))
    return m.group() if m else "0"
    
# What if I can get the turthvalue as value at one point but not a string sheme output
def _tv(js) -> TruthValue | None:
    if isinstance(js, (list, tuple)) and len(js) == 2:
        try:
            return TruthValue(float(js[0]), float(js[1]))
        except Exception:
            pass
    return None

_ANON_SEQ  = itertools.count(1)          # bypassing opencog atomspace's no id no handle rule however we cannot map anotmore original json file 
_ANON_MAP: dict = {}      # bypassing opencog atomspace's no id no handle rule however we cannot map anotmore original json file 
def _anon_id(atom) -> str:
    """Return a stable synthetic ID for an anonymous atom."""
    if atom not in _ANON_MAP:
        _ANON_MAP[atom] = f"ANON_{next(_ANON_SEQ)}"
    return _ANON_MAP[atom]

# Nodes I can Extract from gazillion Pages of OpenCog Wikis
_NODE_TYPES = {
    "ConceptNode":   types.ConceptNode,
    "PredicateNode": types.PredicateNode,
    "SentenceNode":  getattr(types, "SentenceNode", types.ConceptNode),
    "TimestampNode": getattr(types, "TimesNode", types.ConceptNode),
    "NumberNode":    types.NumberNode,
}

def _node(asp: AtomSpace, atype: str, name: str):
    node_type = _NODE_TYPES.get(atype, types.ConceptNode)
    if node_type is types.NumberNode:
        name = _num(name)
    h = asp.add_node(node_type, str(name))
    #log.debug("NODE %-15s %s", atype, name)
    return h

# Links I can Extract from gazillion Pages of OpenCog Wikis
_LINK = {
    "EvaluationLink":  types.EvaluationLink,
    "ContextLink":     types.ContextLink,
    "ImplicationLink": types.ImplicationLink,
    "ListLink":        types.ListLink,
}

def _list_link(asp: AtomSpace, children: List["AtomHandle"]):
    return asp.add_link(types.ListLink, children)

def _context_link(
        asp: AtomSpace,
        thing: "AtomHandle",
        ctx_atoms: List["AtomHandle"],
        tv: TruthValue | None = None):
    """
    Builds a schema-correct 2-child ContextLink.  Apparently all atomspace is a normal Knowledge Graphs nested.
    """
    if len(ctx_atoms) == 1:
        return asp.add_link(types.ContextLink, [thing, ctx_atoms[0]], tv=tv)
    ctx_ll = _list_link(asp, ctx_atoms)
    return asp.add_link(types.ContextLink, [thing, ctx_ll], tv=tv)

def _wrap_if_many(asp: AtomSpace, atoms: List["AtomHandle"]) -> "AtomHandle":
    """Return the single atom if len==1 else a ListLink of them."""
    return atoms[0] if len(atoms) == 1 else _list_link(asp, atoms)

# ───────── main loader ─────────
def load_json_into_atomspace(
        jpath: Path,
        asp: AtomSpace,
        with_metadata: bool = True) -> Dict[str, "AtomHandle"]:
    data = json.loads(jpath.read_text())
    json_id_to_atom: Dict[str, "AtomHandle"] = {}
    link_defs = {l["id"]: l for l in data.get("links", [])}
    log.info(link_defs)
    # 1. plain nodes
    for n in data.get("nodes", []):
        json_id_to_atom[n["id"]] = _node(
            asp, n.get("atom_type", "ConceptNode"), n["name"])

    # 2. recursive link builder
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def get_or_build(aid: str):
        if aid in json_id_to_atom:
            return json_id_to_atom[aid]
        if aid not in link_defs:
            raise KeyError(f"ID '{aid}' not declared as node or link")
        l = link_defs[aid]
        ltype, tv = l["link_type"], _tv(l.get("truth_value"))
        ctor = _LINK.get(ltype)
        if not ctor:
            raise ValueError(f"Unsupported link_type '{ltype}'")

        #EvaluationLink
        if ltype == "EvaluationLink":
            pred = get_or_build(l["predicate"])
            args = [_list_link(asp, [get_or_build(a) for a in l["arguments"]])]

            el_handle = asp.add_link(types.EvaluationLink, [pred, args[0]], tv=tv)
            json_id_to_atom[aid] = el_handle

            # convert JSON attributes to has-attr links
            for k, v in (l.get("attributes") or {}).items():
                key_n = _node(asp, "ConceptNode", str(k))
                val_n = _node(
                    asp,
                    "NumberNode" if str(v).replace(".", "", 1).isdigit() else "ConceptNode",
                    str(v),
                )
                asp.add_link(
                    types.EvaluationLink,
                    [ _node(asp, "PredicateNode", "has-attr"),
                      _list_link(asp, [el_handle, key_n, val_n]) ])
            return el_handle

        #  ImplicationLink 
        if ltype == "ImplicationLink":
            log.info(l)
            log.info(l["arguments"][0])
            premises = _wrap_if_many(asp, [get_or_build(l["arguments"][0])])
            conclusions = _wrap_if_many(asp, [get_or_build(l["arguments"][1])])
            h = asp.add_link(types.ImplicationLink, [premises, conclusions], tv=tv)
            json_id_to_atom[aid] = h
            return h

        #  ContextLink 
        if ltype == "ContextLink":
            if "arguments" in l:  # non-canonical form: list of >=2 items
                *ctx_ids, thing_id = l["arguments"]
                thing = get_or_build(thing_id)
                ctx_atoms = [get_or_build(x) for x in ctx_ids]
                h = _context_link(asp, thing, ctx_atoms, tv)
            else:
                thing = get_or_build(l["thing"])
                ctx_atom = get_or_build(l["context"])
                h = _context_link(asp, thing, [ctx_atom], tv)
            json_id_to_atom[aid] = h
            return h

        #  ListLink or other simple 
        children = [get_or_build(c) for c in l.get("arguments", [])]
        h = asp.add_link(ctor, children, tv=tv)
        json_id_to_atom[aid] = h
        return h

    # build all links
    for lid in link_defs:
        get_or_build(lid)

    # 3.  metadata ingestion
    if with_metadata and (meta := data.get("metadata")):
        doc_node = _node(asp, "ConceptNode", f"Document:{meta.get('document_id', jpath.name)}")
        sent_pred = _node(asp, "PredicateNode", "raw_text")

        def _sentence_node(sent: dict):
            """Return cached SentenceNode or create & cache it."""
            sid = sent["id"]
            if sid not in json_id_to_atom:
                json_id_to_atom[sid] = _node(asp, "SentenceNode", sid)
            return json_id_to_atom[sid]

        for s in (meta.get("source_sentences") or []):
            s_node  = _sentence_node(s)
            txt_node = _node(asp, "ConceptNode", s["text"])
            txt_ll   = _list_link(asp, [s_node, txt_node])
            asp.add_link(types.EvaluationLink, [sent_pred, txt_ll])
            asp.add_link(types.MemberLink, [s_node, doc_node])

    log.info("✅  Imported %d atoms.", len(asp))
    return json_id_to_atom

#  CSV exporter 
def save_atomspace_to_csv(
        json_id_to_atom: Dict[str, "AtomHandle"],
        asp: AtomSpace,
        nodes_path: str,
        links_path: str,
        include_anon: bool = True):

    rev_map = {a: jid for jid, a in json_id_to_atom.items()}

    nodes_rows, links_rows = [], []

    for atom in asp:
        tv = atom.tv
        tv_s, tv_c = (None, None)
        #print(atom)
        if tv:
            #print(tv)
            #print(type(tv))
            m = re.search(r"\(stv\s+([0-9eE\.\-]+)\s+([0-9eE\.\-]+)\)", str(tv))
            tv_s, tv_c = (float(m.group(1)), float(m.group(2))) if m else (None, None)

        if atom.is_node():
            if not include_anon and atom not in rev_map:
                continue
            nodes_rows.append({
                "id": rev_map.get(atom, _anon_id(atom)),
                "type": atom.type_name,
                "name": atom.name,
                "tv_strength": tv_s,
                "tv_confidence": tv_c,
            })
        else:
            child_ids = [rev_map.get(c, _anon_id(c)) for c in atom]
            links_rows.append({
                "id": rev_map.get(atom, _anon_id(atom)),
                "type": atom.type_name,
                "child_ids": "|".join(child_ids),
                "tv_strength": tv_s,
                "tv_confidence": tv_c,
            })

    # write nodes
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=nodes_rows[0].keys())
        w.writeheader(); w.writerows(nodes_rows)
    # write links
    with open(links_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=links_rows[0].keys())
        w.writeheader(); w.writerows(links_rows)
    log.info("✅  CSV saved (%d nodes, %d links).", len(nodes_rows), len(links_rows))

def diagnose_atom_types(engine, space_name: str, num_to_check: int = 10):
    """
    Inspects and prints the types of the first few atoms in a given space.
    """
    print(f"\n--- DIAGNOSING ATOM TYPES IN '{space_name}' ---")
    try:
        # Get the list of atom objects from the space
        atoms = engine.run(f"! (get-atoms {space_name})")[0]
        print(f"Found {len(atoms)} total atoms. Checking the first {num_to_check}...")
        if not atoms:
            print("Space is empty or could not be read.")
            print("--- END DIAGNOSIS ---\n")
            return

        # Loop through a sample of atoms
        for i, atom in enumerate(atoms[:num_to_check]):
            py_type = type(atom).__name__
            # Check against the specific Hyperon types
            is_expr = isinstance(atom, ExpressionAtom)
            is_sym = isinstance(atom, SymbolAtom)
            is_grounded = isinstance(atom, GroundedAtom)
            
            print(
                f"Atom #{i+1}: {str(atom)}\n"
                f"  - Python Type: {py_type}\n"
                f"  - Is ExpressionAtom? {is_expr}\n"
                f"  - Is SymbolAtom?     {is_sym}\n"
                f"  - Is GroundedAtom?   {is_grounded}\n"
            )
    except Exception as e:
        print(f"An error occurred during diagnosis: {e}")
    print(f"--- END DIAGNOSIS ---\n")


if __name__ == "__main__":
    if len(sys.argv) < 1:
        sys.exit("Use: python upload_to_atomspace.py <hypergraph.json> [-v] [--no-meta] [--hyperon-out]")
    base = Path.cwd()
    hg_json   = base/ "HG_merged/final_merged_hg.json"    
    jfile = Path(hg_json).expanduser()
    with_meta = "--no-meta" not in sys.argv

    asp = AtomSpace()
    id_map = load_json_into_atomspace(jfile, asp, with_metadata=with_meta)

    nodes_csv = jfile.with_suffix(".nodes.csv").name
    links_csv = jfile.with_suffix(".links.csv").name
    save_atomspace_to_csv(id_map, asp, nodes_csv, links_csv)

    nodes_csv = Path(nodes_csv)
    links_csv = Path(links_csv)

    ids_in_nodes = set()
    with nodes_csv.open() as f:
        r = csv.DictReader(f)
        ids_in_nodes |= {row["id"] for row in r}

    ids_in_links = set()
    child_ids_seen = collections.Counter()

    with links_csv.open() as f:
        for row in csv.DictReader(f):
            ids_in_links.add(row["id"])
            for cid in (row["child_ids"] or "").split("|"):
                child_ids_seen[cid] += 1

    orphaned = [cid for cid in child_ids_seen
                if cid not in ids_in_nodes and cid not in ids_in_links]

    print(f"child IDs total      : {sum(child_ids_seen.values())}")
    print(f"unique child IDs     : {len(child_ids_seen)}")
    print(f"orphaned child IDs   : {len(orphaned)}")
    if orphaned:
        print("Sample:", orphaned[:10])

    # --- Hyperon ---
    import os
    import neurospace
    log.info("Converting AtomSpace to Hyperon Space ...")
    out_path = jfile.with_suffix(".metta")
    hg_data  = json.loads(jfile.read_text())
    hspace   = json_to_hyperon_space(hg_data)
    export_to_metta(hg_data, out_path, space_var="&hg")
    log.info("✅  MeTTa file written → %s", out_path)


    engine = MeTTa()
    engine.run('! (register-module! "neurospace.py")')
    engine.run('!(import! &self module:"neurospace.py")')
    
    try:
        atoms_to_register = neurospace.my_operations(engine)
        for name, atom in atoms_to_register.items():
            engine.register_atom(name, atom)
        log.info("✅ Custom operations 'ask-qa' and 'ask-clas' registered.")
    except Exception as e:
        log.error(f"❌ Failed to register custom operations: {e}")

    engine.run('!(bind! &hg (new-space))')
    for ln in out_path.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("!(bind!") or ln.startswith("!(import!"):
            continue
        if ln.startswith("!(add-atom"):
            ln = re.sub(r"&self", "&hg", ln)
            engine.run(ln)
    raw   = len(engine.run('! (get-atoms &hg)')[0])
    print("RAW_hg:", raw)    
    
    engine.run('!(bind! &qa-space (new-space))')
    for ln in out_path.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("!(bind!") or ln.startswith("!(import!"):
            continue
        if ln.startswith("!(add-atom"):
            ln = re.sub(r"&hg", "&qa-space", ln)
            ln = re.sub(r"&self", "&qa-space", ln)
            engine.run(ln)
    raw   = len(engine.run('! (get-atoms &qa-space)')[0])
    print("RAW:", raw)    
    
    engine.run('!(bind! &clas-space (new-space))')
    for ln in out_path.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("!(bind!") or ln.startswith("!(import!"):
            continue
        if ln.startswith("!(add-atom"):
            ln = re.sub(r"&hg", "&clas-space", ln)
            ln = re.sub(r"&self", "&clas-space", ln)
            engine.run(ln)
    raw   = len(engine.run('! (get-atoms &clas-space)')[0])
    print("RAW_clas:", raw)        

    diagnose_atom_types(engine, "&qa-space")
    diagnose_atom_types(engine, "&clas-space")
    # Query for QA using 'ask-qa'
    # We now pass the space name AS A STRING LITERAL
    
    
    # EITHER A RAG OR A DATABASE RESEARCH TRANSLATOR 
    
    #q = ''' 
    #! (match &hg
     #      (E (S $a)
     #         (S Choledocholithiasis))
     #      $a)
    #'''
    #q ="! (match &hg ((S $x)) (match &hg ($y Choledocholithiasis) $y ))"
    
    #print("▶ relates Choledocholithiasis :", engine.run(q))
    
    q_qa = """!(ask-qa &qa-space
        (What $x associated with Choledocholithiasis ?))"""
    print("▶ Query Result [QA]:", engine.run(q_qa))

    # Query for CLAS using 'ask-clas'
    # We now pass the space name AS A STRING LITERAL
    q_clas = """!(ask-clas &clas-space
        ("$x Choledocholithiasis"))"""
    print("▶ Query Result [CLAS]:", engine.run(q_clas))