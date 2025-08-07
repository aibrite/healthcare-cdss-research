
import argparse, json, pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
TYPEDB_VALUE_MAP = {bool: "boolean", int: "long", float: "double"}
ISO_FMT_LEN = 10  # yyyy-mm-dd

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

def flatten(prefix: str, value: Union[dict, Any]):
    """Flatten nested dicts into (key, value) pairs with underscore path."""
    if not isinstance(value, dict):
        yield prefix, value
        return
    for k, v in value.items():
        new_p = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(new_p, v)
        else:
            yield new_p, v

# ---------------------------------------------------------------------
# Discovery from JSON
# ---------------------------------------------------------------------
def discover_schema(data: Dict[str, Any]):
    nodes       = data.get("nodes", [])
    edges       = data.get("edges", [])
    hypers      = data.get("hyperedges", [])
    guidelines  = data.get("guideline_thresholds", [])

    id2type = {n["id"]: n.get("type", "Unknown") for n in nodes}
    id2type.update({g["id"]: "GuidelineThreshold" for g in guidelines})

    entity_attrs: Dict[str, Dict[str, str]] = defaultdict(dict)              # etype -> {attr:type}
    relation_attrs: Dict[str, Dict[str, str]] = defaultdict(dict)            # rel -> {attr:type}
    relation_roles: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))  # rel -> role -> set(etype)

    # ---- entities & their attributes
    for obj in nodes + guidelines:
        etype = obj.get("type", "GuidelineThreshold")
        for k, v in obj.items():
            if k in {"id", "type"}:
                continue
            for fk, fv in flatten(k, v):
                t = detect_attr_type(fv)
                if entity_attrs[etype].get(fk, t) == "string":
                    entity_attrs[etype][fk] = t
                else:
                    entity_attrs[etype][fk] = entity_attrs[etype].get(fk, t)

    # helper to add role mapping
    def add_role(rel, role, node_id):
        relation_roles[rel][role].add(id2type.get(node_id, "Unknown"))

    # ---- edges (binary relations)
    for e in edges:
        rel = e["predicate"]
        add_role(rel, "subject", e["source"])
        add_role(rel, "object",  e["target"])
        for k, v in e.items():
            if k in {"predicate", "source", "target"}:
                continue
            if isinstance(v, str) and v in id2type:
                add_role(rel, k, v)  # provenance link etc.
            else:
                for fk, fv in flatten(k, v):
                    t = detect_attr_type(fv)
                    if relation_attrs[rel].get(fk, t) == "string":
                        relation_attrs[rel][fk] = t
                    else:
                        relation_attrs[rel][fk] = relation_attrs[rel].get(fk, t)

    # ---- hyperedges
    for h in hypers:
        rel = h["type"]
        for k, v in h.items():
            if k in {"id", "type"}:
                continue
            if isinstance(v, str) and v in id2type:
                add_role(rel, k, v)
            else:
                for fk, fv in flatten(k, v):
                    t = detect_attr_type(fv)
                    if relation_attrs[rel].get(fk, t) == "string":
                        relation_attrs[rel][fk] = t
                    else:
                        relation_attrs[rel][fk] = relation_attrs[rel].get(fk, t)

    return entity_attrs, relation_attrs, relation_roles

# ---------------------------------------------------------------------
# Emit schema
# ---------------------------------------------------------------------
def emit_schema(entity_attrs, relation_attrs, relation_roles):
    lines: List[str] = ["define"]

    # attribute declarations
    seen_attrs: Dict[str, str] = {}
    for adict in list(entity_attrs.values()) + list(relation_attrs.values()):
        for a, t in adict.items():
            seen_attrs[a] = t
    for a, t in sorted(seen_attrs.items()):
        lines.append(f"attribute {a}, value {t};")
    lines.append("")

    # entities
    for etype, attrs in sorted(entity_attrs.items()):
        owns = ", owns " + ", ".join(sorted(attrs)) if attrs else ""
        lines.append(f"entity {etype}{owns};")
    lines.append("")

    # relations
    for rel, roles in sorted(relation_roles.items()):
        rrole_block = ",\n        relates ".join(sorted(roles))
    relates ".join(sorted(roles))
        owns_dict  = relation_attrs.get(rel, {})
        owns_block = ",
    owns " + ", ".join(sorted(owns_dict)) if owns_dict else ""
        lines.append(f"relation {rel}
    relates {role_block}{owns_block};
")

    # plays lines
    for rel, rolemap in sorted(relation_roles.items()):
        for role, etypes in sorted(rolemap.items()):
            for etype in sorted(etypes):
                lines.append(f"{etype} plays {rel}:{role};")

    return "
".join(lines)

# ---------------------------------------------------------------------
# Emit data
# ---------------------------------------------------------------------
def js(v): return json.dumps(v)

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
        var[obj["id"]] = vn
        attrs = []
        for k, v in obj.items():
            if k in {"id", "type"}:
                continue
            for fk, fv in flatten(k, v):
                attrs.append(f"has {fk} {js(fv)}")
        out.append(f"  {vn} isa {etype} {' '.join(attrs)};")

    def lbl(x): return x

    # edges
    for e in edges:
        rel = e["predicate"]
        pairs = [f"subject: {var[e['source']]}", f"object: {var[e['target']]}"]
        attrs = []
        for k, v in e.items():
            if k in {"predicate", "source", "target"}:
                continue
            if isinstance(v, str) and v in var:
                pairs.append(f"{lbl(k)}: {var[v]}")
            else:
                for fk, fv in flatten(k, v):
                    attrs.append(f"has {fk} {js(fv)}")
        out.append(f"  $rel_{len(out)} ({', '.join(pairs)}) isa {rel} {' '.join(attrs)};")

    # hyperedges
    for h in hypers:
        rel = h["type"]
        pairs, attrs = [], []
        for k, v in h.items():
            if k in {"id", "type"}:
                continue
            if isinstance(v, str) and v in var:
                pairs.append(f"{lbl(k)}: {var[v]}")
            else:
                for fk, fv in flatten(k, v):
                    attrs.append(f"has {fk} {js(fv)}")
        out.append(f"  $rel_{len(out)} ({', '.join(pairs)}) isa {rel} {' '.join(attrs)};")

    return "\n".join(out)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", type=pathlib.Path, help="KG JSON file")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("."), help="Output directory")
    args = p.parse_args()

    data = json.loads(args.json.read_text())

    ent_attrs, rel_attrs, rel_roles = discover_schema(data)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.json.stem
    (args.out / f"schema_{stem}.tql").write_text(emit_schema(ent_attrs, rel_attrs, rel_roles))
    (args.out / f"data_{stem}.tql").write_text(emit_data(data))
    print("✓ generated", f"schema_{stem}.tql and data_{stem}.tql in", args.out)

if __name__ == "__main__":
    main()
