#!/usr/bin/env python3
"""
Generates a robust schema for TypeDB by defining each relation independently.
This version fixes the Python NameError.
  - schema_auto.tql   (load first)
  - data_auto.tql     (load second)
"""
import json
import pathlib
import re
import sys
from collections import defaultdict, OrderedDict

# ───────────── LABEL & VARIABLE HELPERS ────────────────────────────────────

def l(label: str) -> str:
    """Creates a kebab-case, valid TypeDB identifier."""
    s = re.sub(r"[^A-Za-z0-9\-]", "-", label.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s if s and s[0].isalpha() else "n" + s

def topo(facts):
    """Topologically sorts facts based on their dependencies."""
    done, todo = set(), list(facts)
    sorted_facts = []
    while todo:
        found_cycle = True
        for f in todo[:]:
            deps = {t["entity"] for t in f["tuple"] if t.get("entity", "").startswith("F")}
            if deps.issubset(done):
                sorted_facts.append(f)
                done.add(f["id"])
                todo.remove(f)
                found_cycle = False
        if found_cycle:
            sorted_facts.extend(todo)
            break
    return sorted_facts

# ───────────── SCHEMA BUILDER (Final Explicit Version) ───────────────

def build_schema(js):
    """Builds the schema by defining each relation and its rules independently."""
    # --- Collect all schema information first ---
    attrs = OrderedDict([
        ("name", "string"), ("fact-id", "string"), ("timestamp", "datetime-tz"), ("truth-value", "double"),
    ])
    for f in js["facts"]:
        for t in f["tuple"]:
            if "literal" in t: attrs[l(t["datatype"])] = "double"
    for k in js.get("metadata", {}): attrs[l(k)] = "string"

    base_entity_types = {l(e["type"]) for e in js["entities"]}
    fact_entity_types = {l(f["predicate"]) for f in js["facts"]}
    fact_attrs = defaultdict(set)
    for f in js["facts"]:
        fact_type = l(f["predicate"])
        for t in f["tuple"]:
            if "literal" in t: fact_attrs[fact_type].add(l(t["datatype"]))
    roles = {l(t["role"]) for f in js["facts"] for t in f["tuple"] if "entity" in t}

    # --- Generate Schema ---
    lines = ["define"]
    for name, vtype in attrs.items():
        lines.append(f"attribute {name} value {vtype};")

    # Use '@abstract' annotation after type definition for TypeDB 3.x
    lines.append("entity data-entity @abstract, owns name;")
    for etype in sorted(base_entity_types):
        # Generate dynamic extra_attrs for study entities by checking metadata keys
        if etype == "study":
            extra_attrs = []
            # Check for specific metadata keys and convert them using l() helper
            for meta_key in ["source-pmid", "publication-year"]:
                # Convert to actual metadata key format (underscore)
                actual_key = meta_key.replace('-', '_')
                if actual_key in js.get("metadata", {}):
                    extra_attrs.append(l(actual_key))
            
            # Build owns clause for study entities
            if extra_attrs:
                owns_clause = ", ".join(f"owns {attr}" for attr in extra_attrs)
                lines.append(f"entity {etype} sub data-entity, {owns_clause};")
            else:
                lines.append(f"entity {etype} sub data-entity;")
        else:
            # For non-study entities, no extra attributes
            lines.append(f"entity {etype} sub data-entity;")

    lines.append("\nentity fact @abstract, owns fact-id, owns timestamp, owns truth-value;")
    for ftype in sorted(fact_entity_types):
        owns_clause = ", ".join(f"owns {attr}" for attr in sorted(fact_attrs[ftype]))
        lines.append(f"entity {ftype} sub fact{(', ' + owns_clause) if owns_clause else ''};")

    # For each role, create a unique relation with specific role names
    for role in sorted(roles):
        relation_name = f"has-{role}"
        subject_role = f"has-{role}-subject"
        object_role = f"has-{role}-object"
        
        # Define the relation with its unique roles
        lines.append(f"relation {relation_name}")
        lines.append(f"  relates {subject_role},")
        lines.append(f"  relates {object_role};")
        
        # Define the plays declarations in correct order: type plays relation:role
        lines.append(f"fact plays {relation_name}:{subject_role};")
        lines.append(f"fact plays {relation_name}:{object_role};")
        lines.append(f"data-entity plays {relation_name}:{object_role};")
        lines.append("") # Add a blank line for readability
    
    return "\n".join(lines)


# ───────────── DATA BUILDER (Unchanged) ─────────────────────────────────────

def build_data(js):
    """Builds the data with reified fact entities and linking relations."""
    ents = {e["id"]: e for e in js["entities"]}
    facts_map = {f["id"]: f for f in js["facts"]}
    out = ["insert"]
    for e in js["entities"]:
        attrs = [f'has name "{e["name"]}"']
        if e["type"] == "Study":
            # Use the same logic as in schema generation
            for meta_key in ["source-pmid", "publication-year"]:
                # Convert to actual metadata key format (underscore)
                actual_key = meta_key.replace('-', '_')
                if actual_key in js.get("metadata", {}):
                    # Use l() helper to convert the metadata key to TypeDB format
                    attr_name = l(actual_key)
                    attrs.append(f'has {attr_name} "{js["metadata"][actual_key]}"')
        out.append(f'  ${e["id"]} isa {l(e["type"])}, {", ".join(attrs)};')
    out.append("")
    for f in topo(js["facts"]):
        fid, pred = f["id"], l(f["predicate"])
        ts = f["timestamp"] + "T00:00:00Z"
        tv = f["truth_value"][0] if isinstance(f["truth_value"], list) else f["truth_value"]
        fact_attrs = [f'has fact-id "{fid}"', f'has timestamp {ts}', f'has truth-value {tv:.2f}']
        for t in f["tuple"]:
            if "literal" in t: fact_attrs.append(f'has {l(t["datatype"])} {t["literal"]}')
        out.append(f'insert ${fid} isa {pred}, {", ".join(fact_attrs)};\n')
        for t in f["tuple"]:
            if "entity" in t:
                role = l(t["role"])
                player_id = t["entity"]
                # Use new role names for TypeDB 3.x compatibility
                subject_role = f"has-{role}-subject"
                object_role = f"has-{role}-object"
                
                match_lines = [f'match', f'  $sub isa fact, has fact-id "{fid}";']
                if player_id.startswith("F"):
                    player_fact = facts_map[player_id]
                    obj_var = f'obj_{player_id}'  # Create a unique var name for each object
                    match_lines.append(f'  ${obj_var} isa {l(player_fact["predicate"])}, has fact-id "{player_id}";')
                else:
                    player_ent = ents[player_id]
                    obj_var = f'obj_{player_id}'  # Create a unique var name for each object
                    match_lines.append(f'  ${obj_var} isa {l(player_ent["type"])}, has name "{player_ent["name"]}";')
                insert_line = f'insert ({subject_role}: $sub, {object_role}: ${obj_var}) isa has-{role};' # Use the unique obj_var
                out.append("\n".join(match_lines))
                out.append(insert_line)
                out.append("")
    return "\n".join(out)

# ───────────── MAIN ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 typedb_trial.py trial_1.json")
    input_path = pathlib.Path(sys.argv[1])
    if not input_path.exists():
        sys.exit(f"Error: Input file not found at {input_path}")
    js = json.loads(input_path.read_text())
    schema_content = build_schema(js)
    pathlib.Path("schema_auto.tql").write_text(schema_content)
    data_content = build_data(js)
    pathlib.Path("data_auto.tql").write_text(data_content)
    print("✓ Python script fixed. Schema generated with independent relations.")