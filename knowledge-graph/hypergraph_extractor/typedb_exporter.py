# file: hypergraph_extractor/typedb_exporter.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import json
import logging
from collections import defaultdict, Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from models import NTKG, Entity, Fact  # type: ignore
from models import *

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# helper utils
# ─────────────────────────────────────────────────────────────────────────────
def _safe(lbl: str) -> str:
    return lbl.strip().lower().replace(" ", "_").replace("-", "_")


def l(label: str) -> str:
    """Creates a kebab-case, valid TypeDB identifier."""
    import re
    s = re.sub(r"[^A-Za-z0-9\-]", "-", label.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s if s and s[0].isalpha() else "n" + s


def _quote(val) -> str:
    return json.dumps(val)


def _unique(base: str, used: Set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    used.add(f"{base}_{i}")
    return f"{base}_{i}"


# reserved keywords + built-ins we already use
_RESERVED = {
    "attribute",
    "relation",
    "entity",
    "match",
    "define",
    "insert",
    "delete",
    "get",
    "count",
    "rule",
    "plays",
    "owns",
    "relates",
    "alias",
    "value",
    "name",
    "timestamp",
    "truth_value",
}

# ---------------------------------------------------------------------------


def _analyse(
    ntkg: NTKG,
) -> tuple[
    Set[str],                                # entity types
    Dict[str, Set[str]],                     # predicate → safe roles
    Dict[str, Set[Tuple[str, str]]],         # entityType → {(predicate, safeRole)}
    Dict[str, Dict[str, bool]],              # needs_many[pred][safeRole] = True
    Dict[str, str],                          # role_map   rawRole → safeRole
    Dict[str, str],                          # attr_map   rawRole → safeAttr
    Dict[str, str],                          # metadata_attr_map   rawKey → safeAttr
]:
    ent_types: Set[str] = {e.type for e in ntkg.entities}
    ent_by_id = {e.id: e for e in ntkg.entities}

    # raw collections
    rel_roles_raw: Dict[str, Set[str]] = defaultdict(set)
    plays_raw: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    literal_roles: Set[str] = set()
    # multiplicity counter
    max_mult: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for fact in ntkg.facts:
        if not fact.tuple:
            continue
        pred = _safe(fact.predicate)
        # count multiplicities inside this fact
        counts = Counter(r.role for r in fact.tuple)
        for raw_role, multiplicity in counts.items():
            if multiplicity > max_mult[pred][raw_role]:
                max_mult[pred][raw_role] = multiplicity

        for r in fact.tuple:
            if r.entity:
                rel_roles_raw[pred].add(r.role)
                plays_raw[ent_by_id[r.entity].type].add((pred, r.role))
            elif r.literal is not None:
                literal_roles.add(r.role)

    # build collision-free label maps
    used: Set[str] = {_safe(t) for t in ent_types} | set(rel_roles_raw) | _RESERVED
    builtin_attrs = {"name", "timestamp", "truth_value"}
    used |= builtin_attrs

    role_map: Dict[str, str] = {}
    all_roles = {r for s in rel_roles_raw.values() for r in s}
    for raw in sorted(all_roles):
        safe = _safe(raw)
        if safe in used:
            safe = _unique(f"{safe}_role", used)
        else:
            used.add(safe)
        role_map[raw] = safe

    attr_map: Dict[str, str] = {}
    for raw in sorted(literal_roles):
        safe = _safe(raw)
        if safe in used:
            safe = _unique(f"a_{safe}", used)
        else:
            used.add(safe)
        attr_map[raw] = safe
    
    # handle metadata attributes
    metadata_attr_map: Dict[str, str] = {}
    if ntkg.metadata:
        for raw_key in sorted(ntkg.metadata.keys()):
            safe = _safe(raw_key)
            if safe in used:
                safe = _unique(f"m_{safe}", used)
            else:
                used.add(safe)
            metadata_attr_map[raw_key] = safe

    # translate sets with safe labels
    rel_roles: Dict[str, Set[str]] = defaultdict(set)
    for p, roles in rel_roles_raw.items():
        rel_roles[p] = {role_map[r] for r in roles}

    plays: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for et, pairs in plays_raw.items():
        plays[et] = {(p, role_map[r]) for p, r in pairs}

    # multiplicity → needs_many
    needs_many: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for pred, m in max_mult.items():
        for raw_role, count in m.items():
            if raw_role not in role_map:          # ← NEW: skip literal-only roles
                continue
            needs_many[pred][role_map[raw_role]] = count > 1

    return (
        ent_types,
        rel_roles,
        plays,
        needs_many,
        role_map,
        attr_map,
        metadata_attr_map,
    )


# ════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ════════════════════════════════════════════════════════════════════════════
def export_typeql_schema(ntkg: NTKG, out: Path) -> None:
    ents, rel_roles, plays, many, _, attr_map, metadata_attr_map = _analyse(ntkg)

    lines = ["define"]

    # attributes
    lines += [
        "attribute name value string;",
        "attribute timestamp value string;",
        "attribute truth_value value double;",
    ]
    for a in sorted(attr_map.values()):
        lines.append(f"attribute {a} value string;")
    for a in sorted(metadata_attr_map.values()):
        lines.append(f"attribute {a} value string;")

    # base data-entity
    lines.append("entity data-entity @abstract;")
    lines.append("")
    
    # entity subtypes
    for et in sorted(ents):
        safe = _safe(et)
        etype = safe
        
        # Collect entity-specific attributes (excluding "name")
        if etype == "study":
            # Dynamically generate extra_attrs for study entities based on metadata keys
            extra_attrs = []
            if ntkg.metadata:
                # Check for specific metadata keys and convert them using l() helper
                for raw_key in ["source-pmid", "publication-year"]:
                    # Convert to the actual metadata key format that would be in ntkg.metadata
                    original_key = raw_key.replace('-', '_')
                    if original_key in ntkg.metadata and original_key in metadata_attr_map:
                        extra_attrs.append(metadata_attr_map[original_key])
        else:
            extra_attrs = []
        
        # Build the statement according to the task requirements
        if extra_attrs:
            # If extra_attrs is not empty: entity {etype} sub data-entity, owns {attr1}, owns {attr2}, ...
            owns_clause = ", " + ", ".join(f"owns {attr}" for attr in extra_attrs)
        else:
            # If extra_attrs is empty: entity {etype} sub data-entity;
            owns_clause = ""
        
        # Create the entity statement
        stmt = f"entity {etype} sub data-entity{owns_clause}"
        
        # Add plays clauses
        for pred, role in sorted(plays.get(et, set())):
            stmt += f", plays {pred}:{role}"
        
        lines.append(stmt + ";")

    # relations
    attr_clause = (
        ", owns " + ", owns ".join(sorted(attr_map.values()))
        if attr_map else ""
    )
    for pred, roles in sorted(rel_roles.items()):
        stmt = f"relation {pred}, owns timestamp, owns truth_value{attr_clause}"
        for role in sorted(roles):
            if many[pred].get(role):
                stmt += f", relates {role} @card(0..)"
            else:
                stmt += f", relates {role}"
        lines.append(stmt + ";")

    out.write_text("\n".join(lines) + "\n", "utf-8")
    logger.info("📝  Wrote schema → %s", out)


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════
def export_ntkg_to_typeql(ntkg: NTKG, out: Path) -> None:
    _, _, _, _, role_map, attr_map, metadata_attr_map = _analyse(ntkg)

    var: Dict[str, str] = {}
    lines: List[str] = []

    # entities
    for i, e in enumerate(ntkg.entities, 1):
        v = f"$e{i}"
        var[e.id] = v
        
        # Build attributes list starting with name
        attrs = [f'has name {_quote(e.name)}']
        
        # Add metadata attributes for specific entity types
        if e.type.lower() == "study" and ntkg.metadata:
            for raw_key in ["source-pmid", "publication-year"]:
                # Convert to the actual metadata key format that would be in ntkg.metadata
                original_key = raw_key.replace('-', '_')
                if original_key in metadata_attr_map and original_key in ntkg.metadata:
                    safe_attr = metadata_attr_map[original_key]
                    value = ntkg.metadata[original_key]
                    attrs.append(f'has {safe_attr} {_quote(value)}')
        
        lines.append(f'insert {v} isa {_safe(e.type)}, {", ".join(attrs)};')

    # relations
    rid = 0
    for f in ntkg.facts:
        if not f.tuple or f.arguments:
            continue
        rid += 1
        rv = f"$r{rid}"
        rtype = _safe(f.predicate)

        role_parts: List[str] = []
        attr_parts: List[str] = [
            f'has timestamp {_quote(f.timestamp.isoformat() if isinstance(f.timestamp, (date, datetime)) else str(f.timestamp))}'
        ]
        if f.truth_value is not None:
            tv = f.truth_value[0] if isinstance(f.truth_value, list) else f.truth_value
            attr_parts.append(f"has truth_value {tv}")

        for r in f.tuple:
            if r.entity:
                role_parts.append(f"{role_map[r.role]}: {var[r.entity]}")
            elif r.literal is not None:
                attr_parts.append(f"has {attr_map[r.role]} {_quote(r.literal)}")

        lines.append(
            f"insert {rv} isa {rtype} ({', '.join(role_parts)}), {', '.join(attr_parts)};"
        )

    out.write_text("\n".join(lines) + "\n", "utf-8")
    logger.info("📦  Wrote data   → %s", out)
