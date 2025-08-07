# file: hypergraph_extractor/out_sem_2/json2FB.py
#  1. Read final_merged_hg.json  &  final_merged_hrkg.json
#  2. Inline nested links / facts  →  *_inline.json where every link contains all information necessary except metadata (TRIAL)
#  3. Build lookup TSVs            →  hg_entities_facts_literals_mapping.tsv
#                                     hrkg_entities_facts_literals_mapping.tsv
from __future__ import annotations


import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Set
import pprint
pp = pprint.PrettyPrinter(indent=2).pformat

# ── metadata predicates ─────────────────────────────────────────
P_HAS_TIMESTAMP   = "has_timestamp"
P_HAS_SENTENCE    = "has_sentence"
P_HAS_TRUTH_VALUE = "has_truth_value"

# ── helper: short slug for literals ─────────────────────────────

def _slugify(value: Any) -> str:
    if value is None:
        value = "null"
    if not isinstance(value, str):
        value = str(value)
    value = value.lower().strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_%.><=-]", "", value)[:32]
    return value or hashlib.md5(value.encode()).hexdigest()[:8]


# ───────────────────────────────────────────────────────────────
#  1. Inline JSON utilities
# ───────────────────────────────────────────────────────────────

def inline_hg_links(data: Dict[str, Any]) -> Dict[str, Any]:
    links_by_id = {l["id"]: l for l in data["links"]}

    def _inline(link, seen=None):
        seen = seen or set()
        if link["id"] in seen:
            return link
        seen.add(link["id"])
        new = deepcopy(link)
        new_args = []
        for arg in new.get("arguments", []):
            if isinstance(arg, str) and arg.startswith("L") and arg in links_by_id:
                new_args.append(_inline(links_by_id[arg], seen))
            else:
                new_args.append(arg)
        new["arguments"] = new_args
        return new

    data["links"] = [_inline(l) for l in data["links"]]
    return data


def inline_hrkg_facts(data: Dict[str, Any]) -> Dict[str, Any]:
    facts_by_id = {f["id"]: f for f in data["facts"]}

    def _inline(fact, seen=None):
        seen = seen or set()
        if fact["id"] in seen:
            return fact
        seen.add(fact["id"])
        new = deepcopy(fact)
        if isinstance(new.get("arguments"), list):
            new_args: List[Any] = []
            for a in new["arguments"]:
                if isinstance(a, str) and a in facts_by_id:
                    new_args.append(_inline(facts_by_id[a], seen))
                else:
                    new_args.append(a)
            new["arguments"] = new_args
        return new

    data["facts"] = [_inline(f) for f in data["facts"]]
    return data


# ───────────────────────────────────────────────────────────────
#  2. Mapping TSV writers  →  {id → label}
# ───────────────────────────────────────────────────────────────

def _safe_label(val: str | None, fallback: str) -> str:
    val = (val or "").strip()
    return val if val else fallback


def write_hg_mapping(data, out_path: Path) -> Dict[str, str]:
    id2lbl = {n["id"]: _safe_label(n.get("name"), n["id"]) for n in data["nodes"]}
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write("label\tkind\tid\tname\tatom_type\n")
        for n in data["nodes"]:
            fp.write(
                f"{id2lbl[n['id']]}\tnode\t{n['id']}\t{n.get('name','')}\t{n.get('atom_type','')}\n"
            )
    return id2lbl


def write_hrkg_mapping(data, out_path: Path):
    ent_lbl = {e["id"]: _safe_label(e.get("name"), e["id"]) for e in data["entities"]}
    fact_lbl = {f["id"]: _safe_label(f.get("predicate"), f["id"]) for f in data["facts"]}
    lit_lbl: Dict[str, str] = {}

    def _ensure_lit(val: Any):
        if val not in lit_lbl:
            lit_lbl[val] = str(val)
        return lit_lbl[val]

    for f in data["facts"]:
        for t in (f.get("tuple") or []):
            if t.get("entity") is None and t.get("literal") is not None:
                _ensure_lit(t["literal"])

    with out_path.open("w", encoding="utf-8") as fp:
        fp.write("label\tkind\tid_or_literal\tname_or_value\trole_or_type\n")
        for e in data["entities"]:
            fp.write(
                f"{ent_lbl[e['id']]}\tentity\t{e['id']}\t{e.get('name','')}\t{e.get('type','')}\n"
            )
        for f in data["facts"]:
            fp.write(
                f"{fact_lbl[f['id']]}\tfact\t{f['id']}\t{f.get('predicate','')}\t.\n"
            )
        for lit, lbl in lit_lbl.items():
            fp.write(f"{lbl}\tliteral\t{lit}\t{lit}\t.\n")

    return ent_lbl, fact_lbl, lit_lbl


# ───────────────────────────────────────────────────────────────
#  Shared helpers
# ───────────────────────────────────────────────────────────────

def _truth_value_scalar(tv):
    if tv is None:
        return None
    if isinstance(tv, list):
        return tv[1] if len(tv) > 1 else tv[0]
    return tv


def _attr_suffix(attrs: dict | None):
    if not attrs:
        return ""
    vals = [str(v) for k, v in sorted(attrs.items())]
    return "/" + "/".join(vals)

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
SENT_RE = re.compile(r"SENT_\d+", re.I)


# ───────────────────────────────────────────────────────────────
#  3a. Hypergraph → triples + metadata triples
# ───────────────────────────────────────────────────────────────

def build_hg_triples(data, id2lbl) -> Counter:
    triples: Counter[tuple[str, str, str]] = Counter()
    id_to_node = {n["id"]: n for n in data["nodes"]}

    _id   = lambda x: x["id"] if isinstance(x, dict) else x
    _pred = lambda pid: id_to_node.get(pid, {}).get("name") or f"predicate_{pid}"

    def _context_info(nid_a: str, nid_b: str):
        ts, sid, s_txt = None, None, None
        for nid in (nid_a, nid_b):
            node = id_to_node.get(nid, {})
            typ  = node.get("atom_type", "")
            name = node.get("name", "")
            if typ in {"TimeStampNode", "TimeNode", "DateNode"} or DATE_RE.fullmatch(name):
                ts = name
            if typ == "SentenceNode" or SENT_RE.fullmatch(name):
                sid = name
        if sid and "metadata" in data and "sentences" in data["metadata"]:
            for sent in data["metadata"]["sentences"]:
                if sent.get("id") == sid:
                    s_txt = sent.get("text")
                    break
        return ts, sid, s_txt

    def _emit(eval_link, ts=None, sid=None, tv=None):
        base_relation = _pred(eval_link["predicate"]) + _attr_suffix(eval_link.get("attributes"))
        args = [_id(a) for a in eval_link.get("arguments", [])]
        if len(args) < 2:
            return

        # helper to record one primary triple
        def _record(subj, obj):
            if subj in id2lbl and obj in id2lbl:
                triples[(id2lbl[subj], base_relation, id2lbl[obj])] += 1

        # -- CARTESIAN PRODUCT: every ordered pair (subj, obj) --
        for subj in args:
            for obj in args:
                if subj == obj:
                    continue
                _record(subj, obj)

        # metadata triples (once per link)
        if ts:
            triples[(base_relation, P_HAS_TIMESTAMP, ts)] += 1
        if sid:
            triples[(base_relation, P_HAS_SENTENCE, sid)] += 1
        if tv is not None:
            triples[(base_relation, P_HAS_TRUTH_VALUE, str(tv))] += 1
            triples[(base_relation, P_HAS_TRUTH_VALUE, str(tv))] += 1

    def _walk_link(L):
        """Recursive descent that handles EvaluationLink and ContextLink.
        ContextLinks can be nested arbitrarily; for each one we:
        1. Locate the *first* EvaluationLink anywhere in its argument list.
        2. Emit that EvaluationLink as usual, using the two context nodes (if any)
           as timestamp/sentence detectors **and** as explicit has_context triples.
        3. Continue recursion so that deeper ContextLinks are reached too.
        """
        if L.get("link_type") == "EvaluationLink":
            _emit(
                L,
                tv=_truth_value_scalar(L.get("truth_value"))
            )

        elif L.get("link_type") == "ContextLink":
            args = L.get("arguments", [])
            if len(args) < 3:
                # malformed ContextLink → still recurse into any link args
                for a in args:
                    if isinstance(a, dict) and a.get("link_type"):
                        _walk_link(a)
                return

            ctx_nodes = [a for a in args[:2] if isinstance(a, (str, dict))]
            # find the first EvaluationLink object anywhere in remaining args
            def _find_eval(obj):
                if isinstance(obj, dict):
                    if obj.get("link_type") == "EvaluationLink":
                        return obj
                    for sub in obj.get("arguments", []):
                        res = _find_eval(sub)
                        if res is not None:
                            return res
                return None
            inner_eval = None
            for a in args[2:]:
                inner_eval = _find_eval(a)
                if inner_eval is not None:
                    break

            # Use context nodes to derive ts/sid + emit has_context triples
            ts, sid, _ = (None, None, None)
            if len(ctx_nodes) >= 2:
                ts, sid, _ = _context_info(_id(ctx_nodes[0]), _id(ctx_nodes[1]))

            if inner_eval is not None:
                base_pred = _pred(inner_eval["predicate"]) + _attr_suffix(inner_eval.get("attributes"))
                # 1) Emit the evaluation itself
                _emit(
                    inner_eval,
                    ts=ts,
                    sid=sid,
                    tv=_truth_value_scalar(inner_eval.get("truth_value"))
                )
                # 2) has_context triples
                for ctx in ctx_nodes:
                    ctx_id = _id(ctx)
                    if ctx_id in id2lbl:
                        triples[(base_pred, "has_context", id2lbl[ctx_id])] += 1

            # recurse further to catch nested ContextLinks / EvaluationLinks
            for a in args:
                if isinstance(a, dict) and a.get("link_type"):
                    _walk_link(a)

        else:
            # Other link types (if any) – just recurse
            for a in L.get("arguments", []):
                if isinstance(a, dict) and a.get("link_type"):
                    _walk_link(a)

    # kick off traversal
    for root in data["links"]:
        _walk_link(root)

    return triples




# ───────────────────────────────────────────────────────────────
#  3b. HRKG → triples + metadata triples
# ───────────────────────────────────────────────────────────────

def build_hrkg_triples(data, ent_lbl, fact_lbl, lit_lbl) -> Counter:
    triples: Counter[tuple[str, str, str]] = Counter()

    def _lbl_for_literal(val, dtype=None):
        if dtype:
            val = f"{val}/{dtype}"
        if val not in lit_lbl:
            lit_lbl[val] = str(val)
        return lit_lbl[val]

    seen_facts: Set[str] = set()

    def _collect_role2lbls(fact: Dict[str, Any]):
        fid = fact["id"]
        if fid in seen_facts:
            return {}
        seen_facts.add(fid)

        pred = fact.get("predicate", "unknown_predicate")
        role2: Dict[str, List[str]] = defaultdict(list)

        # tuples ------------------------------------------------------
        for tup in fact.get("tuple") or []:
            role = tup["role"]
            if tup.get("entity"):
                lbl = ent_lbl.get(tup["entity"])
            else:
                lbl = _lbl_for_literal(tup["literal"], tup.get("datatype"))
            if lbl:
                role2[role].append(lbl)

        # nested facts as arguments ----------------------------------
        for arg in fact.get("arguments") or []:
            if isinstance(arg, dict):
                lbl = fact_lbl.get(arg["id"])
                if lbl:
                    role2["fact_arg"].append(lbl)
                sub = _collect_role2lbls(arg)
                for r, vals in sub.items():
                    role2[f"fact_arg->{r}"].extend(vals)
            elif isinstance(arg, str):
                lbl = fact_lbl.get(arg)
                if lbl:
                    role2["fact_arg"].append(lbl)

        # --- emit triples for THIS fact instance --------------------
        ts  = fact.get("timestamp")
        sid = fact.get("sentence_id")
        tv  = _truth_value_scalar(fact.get("truth_value"))

        for r1, l1s in role2.items():
            for r2, l2s in role2.items():
                if r1 == r2:
                    continue
                base_relation = f"[{r1}]:{pred}:[{r2}]"
                for l1 in l1s:
                    for l2 in l2s:
                        triples[(l1, base_relation, l2)] += 1
                # metadata triples (once per relation instance) -------
                if ts:
                    triples[(base_relation, P_HAS_TIMESTAMP, ts)] += 1
                if sid:
                    triples[(base_relation, P_HAS_SENTENCE, sid)] += 1
                if tv is not None:
                    triples[(base_relation, P_HAS_TRUTH_VALUE, str(tv))] += 1
        return role2

    for f in data["facts"]:
        _collect_role2lbls(f)

    return triples


# ───────────────────────────────────────────────────────────────
#  4. Main pipeline
# ───────────────────────────────────────────────────────────────

def main():
    base = Path.cwd()
    hg_json   = base/ "HG_merged/final_merged_hg.json"
    hrkg_json = base / "HRKG_merged/final_merged_hrkg.json"

    # -------- Hypergraph branch ------------------------------------
    hg_data = inline_hg_links(json.loads(hg_json.read_text()))
    (base / "HG_merged/final_merged_hg_inline.json").write_text(json.dumps(hg_data, indent=4))
    id2lbl = write_hg_mapping(hg_data, base / "HG_merged/hg_entities_facts_literals_mapping.tsv")
    hg_triples = build_hg_triples(hg_data, id2lbl)
    with (base / "HG_merged/hg_triples.txt").open("w", encoding="utf-8") as fp:
        for s, p, o in hg_triples:
            fp.write(f"{s}\t{p}\t{o}\n")

    # -------- HRKG branch ------------------------------------------
    hrkg_data = inline_hrkg_facts(json.loads(hrkg_json.read_text()))
    (base / "HRKG_merged/final_merged_hrkg_inline.json").write_text(json.dumps(hrkg_data, indent=4))
    ent_lbl, fact_lbl, lit_lbl = write_hrkg_mapping(
        hrkg_data, base / "HRKG_merged/hrkg_entities_facts_literals_mapping.tsv")
    hrkg_triples = build_hrkg_triples(hrkg_data, ent_lbl, fact_lbl, lit_lbl)
    with (base / "HRKG_merged/hrkg_triples.txt").open("w", encoding="utf-8") as fp:
        for s, p, o in hrkg_triples:
            fp.write(f"{s}\t{p}\t{o}\n")

    print("✓ Pipeline finished – flattened triples + metadata")


if __name__ == "__main__":
    main()
