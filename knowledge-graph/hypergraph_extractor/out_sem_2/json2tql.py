# json_to_tql.py  ▸ v3.2 – 2025‑07‑10
"""
Convert a Lab‑Interpretation KG JSON file (e.g. *kg_trial_1.json*) into
  • **schema_<stem>.tql** – TypeDB 3.x schema (matching your *schema_1.tql*)
  • **data_<stem>.tql**   – data inserts.

Changes in 3.2
==============
* **Restored `plays` lines** – each entity now explicitly plays its roles.
* Still *no* separate role declarations; roles live inline in `relation … relates …`.
* Previous bugfixes (f‑string, attribute types, flattening, provenance) remain.
"""
import argparse, json, pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

#############################
# Helpers                   #
#############################

TYPEDB_VALUE_MAP = {bool: "boolean", int: "long", float: "double"}
ISO_FMT_LEN = 10  # yyyy‑mm‑dd

def detect_attr_type(v: Any) -> str:
    if isinstance(v, bool):
        return TYPEDB_VALUE_MAP[bool]
    if isinstance(v, int):
        return TYPEDB_VALUE_MAP[int]
    if isinstance(v, float):
        return TYPEDB_VALUE_MAP[float]
    if isinstance(v, str) and len(v) >= ISO_FMT_LEN and v[4] == "-" and v[7] == "-":
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return "datetime"
        except ValueError:
            pass
    return "string"

def merge_attr(attr: str, tp: str, store: Dict[str, str]):
    if store.get(attr, "string") == "string":
        store[attr] = tp

#############################
# Flatten nested dicts      #
#############################

def flatten(prefix: str, value: Union[dict, Any]):
    if not isinstance(value, dict):
        yield prefix, value
        return
    for k, v in value.items():
        new_p = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(new_p, v)
        else:
            yield new_p, v

#############################
# Schema discovery          #
#############################

def discover_schema(data):
    nodes       = data.get("nodes", [])
    edges       = data.get("edges", [])
    hypers      = data.get("hyperedges", [])
    guidelines  = data.get("guideline_thresholds", [])

    id2type = {n["id"]: n.get("type", "Unknown") for n in nodes}
    id2type.update({g["id"]: "GuidelineThreshold" for g in guidelines})

    ent_attrs: Dict[str, Dict[str, str]] = defaultdict(dict)
    rel_attrs: Dict[str, Dict[str, str]] = defaultdict(dict)
    rel_roles: Dict[str, Dict[str, str]] = defaultdict(dict)  # rel → role → entity‑type

    # entities
    for obj in nodes + guidelines:
        etype = obj.get("type", "GuidelineThreshold")
        for k, v in obj.items():
            if k in {"id", "type"}: continue
            for fk, fv in flatten(k, v):
                merge_attr(fk, detect_attr_type(fv), ent_attrs[etype])

    def add_role(rel, role, node_id):
        rel_roles[rel][role] = id2type.get(node_id, "Unknown")

    # edges
    for e in edges:
        rel = e["predicate"]
        add_role(rel, "subject", e["source"])
        add_role(rel, "object",  e["target"])
        for k, v in e.items():
            if k in {"predicate", "source", "target"}: continue
            if isinstance(v, str) and v in id2type:
                add_role(rel, k, v)
            else:
                for fk, fv in flatten(k, v):
                    merge_attr(fk, detect_attr_type(fv), rel_attrs[rel])

    # hyperedges
    for h in hypers:
        rel = h["type"]
        for k, v in h.items():
            if k in {"id", "type"}: continue
            if isinstance(v, str) and v in id2type:
                add_role(rel, k, v)
            else:
                for fk, fv in flatten(k, v):
                    merge_attr(fk, detect_attr_type(fv), rel_attrs[rel])

    return ent_attrs, rel_attrs, rel_roles

#############################
# Emit schema               #
#############################

def emit_schema(ent_attrs, rel_attrs, rel_roles):
    lines = ["define"]
    # attributes
    all_attrs = {a: t for d in [*ent_attrs.values(), *rel_attrs.values()] for a, t in d.items()}
    for a, t in sorted(all_attrs.items()):
        lines.append(f"attribute {a}, value {t};")
    lines.append("")
    # entities
    for etype, attrs in sorted(ent_attrs.items()):
        owns = ", ".join(sorted(attrs))
        owns_clause = f", owns {owns}" if owns else ""
        lines.append(f"entity {etype}{owns_clause};")
    lines.append("")
    # relations
    for rel, roles in sorted(rel_roles.items()):
        role_block = ",\n        relates ".join(sorted(roles))
        attr_block = rel_attrs.get(rel, {})
        owns_block = ",\n        owns " + ", ".join(sorted(attr_block)) if attr_block else ""
        lines.append(f"relation {rel}\n        relates {role_block}{owns_block};\n")
    # plays
    for ent, attrs in sorted(ent_attrs.items()):
        pass  # placeholder to keep ordering nice
    for rel, rolemap in sorted(rel_roles.items()):
        for role, etype in sorted(rolemap.items()):
            lines.append(f"{etype} plays {rel}:{role};")
    return "\n".join(lines)

#############################
# Emit data                 #
#############################

def q(v):
    return json.dumps(v)

def emit_data(data):
    nodes       = data.get("nodes", [])
    edges       = data.get("edges", [])
    hypers      = data.get("hyperedges", [])
    guidelines  = data.get("guideline_thresholds", [])

    var = {}
    out = ["insert"]

    # entities
    for obj in nodes + guidelines:
        etype = obj.get("type", "GuidelineThreshold")
        vn = f"${obj['id']}"
        var[obj['id']] = vn
        attrs = []
        for k, v in obj.items():
            if k in {"id", "type"}: continue
            for fk, fv in flatten(k, v):
                attrs.append(f"has {fk} {q(fv)}")
        out.append(f"  {vn} isa {etype} {' '.join(attrs)};")

    def lbl(r):
        return r

    # edges
    for e in edges:
        rel = e['predicate']
        pairs = [f"subject: {var[e['source']]}", f"object: {var[e['target']]}" ]
        attrs = []
        for k, v in e.items():
            if k in {'predicate', 'source', 'target'}: continue
            if isinstance(v, str) and v in var:
                pairs.append(f"{lbl(k)}: {var[v]}")
            else:
                for fk, fv in flatten(k, v):
                    attrs.append(f"has {fk} {q(fv)}")
        out.append(f"  $rel_{len(out)} ({', '.join(pairs)}) isa {rel} {' '.join(attrs)};")

    # hyperedges
    for h in hypers:
        rel = h['type']
        pairs, attrs = [], []
        for k, v in h.items():
            if k in {'id', 'type'}: continue
            if isinstance(v, str) and v in var:
                pairs.append(f"{lbl(k)}: {var[v]}")
            else:
                for fk, fv in flatten(k, v):
                    attrs.append(f"has {fk} {q(fv)}")
        out.append(f"  $rel_{len(out)} ({', '.join(pairs)}) isa {rel} {' '.join(attrs)};")

    return "\n".join(out)

#############################
# CLI                       #
#############################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("."))
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    ent_attrs, rel_attrs, rel_roles = discover_schema(data)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.json.stem
    (args.out / f"schema_{stem}.tql").write_text(emit_schema(ent_attrs, rel_attrs, rel_roles))
    (args.out / f"data_{stem}.tql").write_text(emit_data(data))
    print("✓ generated", f"schema_{stem}.tql", "and", f"data_{stem}.tql", "in", args.out)

if __name__ == "__main__":
    main()
