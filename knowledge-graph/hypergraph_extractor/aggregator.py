# file: hypergraph_extractor/aggregator.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, TypeVar, Union
import copy

import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


from models import Entity, Fact, Hypergraph, NTKG, Sentence
from models import *
logger = logging.getLogger(__name__)

T = TypeVar("T", NTKG, Hypergraph)


def _merge_metadata(graphs: Sequence[T]) -> Dict[str, Any]:
    """Merges metadata from a list of graphs, collecting unique PMIDs, years, and all source sentences."""
    merged_meta = {
        "source_pmids": set(),
        "publication_years": set(),
        "source_sentences": [],
    }
    for g in graphs:
        if not g or not g.metadata:
            continue
        if pmid := g.metadata.get("source_pmid"):
            merged_meta["source_pmids"].add(pmid)
        if year := g.metadata.get("publication_year"):
            merged_meta["publication_years"].add(year)
        if sentence := g.metadata.get("source_sentence"):
            merged_meta["source_sentences"].append(
                {"id": g.metadata.get("sentence_id", "N/A"), "text": sentence}
            )

    merged_meta["source_pmids"] = sorted(list(merged_meta["source_pmids"]))
    merged_meta["publication_years"] = sorted(list(merged_meta["publication_years"]))
    return merged_meta

def _merge_meta(target: dict | None, source: dict | None) -> dict | None:
    """
    Very simple metadata merge strategy:
    • If both are None → None
    • Otherwise shallow-merge source into target (source wins on conflict)
    """
    if target is None and source is None:
        return None
    merged = dict(target or {})
    merged.update(source or {})
    return merged

def merge_ntkg(graphs: List[NTKG], document_id: str) -> NTKG:
    """
    Merges multiple per-sentence NTKG fragments into a single, de-duplicated graph.
    This function correctly handles:
    - De-duplication of entities by name.
    - Assignment of new, globally unique IDs to all entities and facts.
    - Rewriting of entity IDs within fact tuples.
    - Rewriting of fact IDs within the `arguments` of higher-order facts.
    - Comprehensive metadata aggregation.

    Parameters:
    ----------
    graphs : List[NTKG]
        A list of NTKG objects, typically one per sentence.
    document_id : str
        An identifier for the source document.

    Returns:
    -------
    NTKG
        A single, consolidated NTKG.
    """
    if not any(graphs):
        return NTKG(
            graph_type="N-tuple Hyper-Relational Temporal Knowledge Graph",
            entities=[],
            facts=[],
            metadata={"document_id": document_id, "fragments_merged": 0},
        )

    valid_graphs = [g for g in graphs if g]

    # --- 1. De-duplicate entities and create final entity list ---
    ent_by_name: "OrderedDict[str, Entity]" = OrderedDict()
    for g in valid_graphs:
        for e in g.entities:
            ent_by_name.setdefault(e.name, e)

    name_to_new_id: Dict[str, str] = {}
    merged_entities: List[Entity] = []
    for idx, (name, ent) in enumerate(ent_by_name.items(), start=1):
        new_id = f"E{idx}"
        name_to_new_id[name] = new_id
        merged_entities.append(Entity(id=new_id, name=name, type=ent.type))

    # --- 2. Process all facts, assign new IDs, and rewrite entity references ---
    rewritten_facts = []
    old_fact_id_to_new_id: Dict[str, str] = {}
    fact_counter = 0
    logger.info(document_id)
    filename = str(document_id).split("/")[-1]
    clean_doc_id = filename.replace(".", "_")
    #clean_doc_id = document_id.replace("/", "_").replace(".", "_")

    for g in valid_graphs:
        local_id_to_name = {e.id: e.name for e in g.entities}
        for f in g.facts:
            fact_counter += 1
            new_fact = f.model_copy(deep=True)

            # Assign a new, globally unique fact ID and map the old one to it.
            # The agent MUST provide a temporary fact ID for this to work.
            new_id = f"F{fact_counter}_{new_fact.sentence_id}_{clean_doc_id}"
            if f.id:
                old_fact_id_to_new_id[f.id] = new_id
            new_fact.id = new_id

            # Rewrite entity IDs within the fact's tuple
            if new_fact.tuple:
                for role in new_fact.tuple:
                    if role.entity and role.entity in local_id_to_name:
                        entity_name = local_id_to_name[role.entity]
                        role.entity = name_to_new_id.get(entity_name)

            rewritten_facts.append(new_fact)

    # --- 3. Second pass: rewrite fact IDs in higher-order facts ---
    for fact in rewritten_facts:
        if fact.arguments:
            fact.arguments = [
                old_fact_id_to_new_id.get(arg_id, arg_id)
                for arg_id in fact.arguments
            ]

    # --- 4. Consolidate metadata from all fragments ---
    source_sentences = []
    processed_sent_ids = set()
    for g in valid_graphs:
        if g.metadata:
            sent_id = g.metadata.get("sentence_id")
            if sent_id and sent_id not in processed_sent_ids:
                source_sentences.append({
                    "id": sent_id,
                    "text": g.metadata.get("source_sentence", "")
                })
                processed_sent_ids.add(sent_id)

    # Sort sentences by their numeric ID part for correct ordering
    source_sentences.sort(key=lambda x: int(x['id'].split('_')[1]))

    final_meta = {
        "document_id": document_id,
        "fragments_merged": len(valid_graphs),
        "source_sentences": source_sentences,
    }

    logger.info(
        "Merged %d NTKG fragments → %d entities, %d facts",
        len(valid_graphs),
        len(merged_entities),
        len(rewritten_facts),
    )

    return NTKG(
        graph_type="N-tuple Hyper-Relational Temporal Knowledge Graph",
        entities=merged_entities,
        facts=rewritten_facts,
        metadata=final_meta,
    )


def merge_hg(graphs: List[Hypergraph], document_id: str, sentences: List[Sentence]) -> Hypergraph:
    """Merges sentence-level Hypergraphs, de-duplicates nodes and links, and rewrites IDs."""
    valid_graphs = [g for g in graphs if g]
    if not valid_graphs:
        logger.warning("`merge_hg` received no valid graphs. Returning empty Hypergraph.")
        return Hypergraph(graph_type="AtomSpace Hypergraph", nodes=[], links=[], metadata={"document_id": document_id, "fragments_merged": 0})

    # --- 1. Merge Nodes ---
    node_by_name: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    old_node_id_to_new_id: Dict[str, str] = {}

    for g in valid_graphs:
        for n in g.nodes:
            key = n.get("name")
            if not key:
                logger.warning("Node missing 'name', using ID '%s' as key. This can cause merge errors.", n['id'])
                key = n['id']
            if key not in node_by_name:
                node_by_name[key] = copy.deepcopy(n)
    
    merged_nodes = []
    for i, node in enumerate(node_by_name.values()):
        new_id = f"N{i+1}"
        # Rewrite the ID in the object itself
        node['id'] = new_id
        merged_nodes.append(node)
    
    # Build the full mapping from any old ID to the new canonical ID
    for g in valid_graphs:
        for n in g.nodes:
            key = n.get("name", n['id'])
            if key in node_by_name:
                old_node_id_to_new_id[n["id"]] = node_by_name[key]["id"]

    # --- 2. Merge Links ---
    merged_links_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    old_link_id_to_new_id: Dict[str, str] = {}
    
    def _rewrite_ids(obj: Any, id_map: Dict[str, str]) -> Any:
        if isinstance(obj, str): return id_map.get(obj, obj)
        if isinstance(obj, list): return [_rewrite_ids(v, id_map) for v in obj]
        if isinstance(obj, dict): return {k: _rewrite_ids(v, id_map) for k, v in obj.items()}
        return obj
    
    # First pass: de-duplicate links after rewriting node IDs
    for g in valid_graphs:
        for link in g.links:
            # Create a copy where all node IDs are already rewritten to their final form
            rewritten_link = _rewrite_ids(copy.deepcopy(link), old_node_id_to_new_id)
            # Create a key based on content to find duplicate links
            link_content_key = json.dumps({k: v for k, v in rewritten_link.items() if k != 'id'}, sort_keys=True)
            
            if link_content_key not in merged_links_map:
                new_link_id = f"L{len(merged_links_map) + 1}"
                if link.get('id'):
                    old_link_id_to_new_id[link['id']] = new_link_id
                rewritten_link['id'] = new_link_id
                merged_links_map[link_content_key] = rewritten_link
            elif link.get('id'):
                # Map the old ID of this duplicate link to the ID of the canonical one
                old_link_id_to_new_id[link['id']] = merged_links_map[link_content_key]['id']

    merged_links = list(merged_links_map.values())
    
    # --- 3. Second Pass: Rewrite link IDs within link arguments ---
    for link in merged_links:
        if isinstance(link.get("arguments"), list):
            # Rewrite arguments using the complete ID map (nodes and links)
            rewritten_args = _rewrite_ids(link["arguments"], {**old_node_id_to_new_id, **old_link_id_to_new_id})
            
            # Filter out any non-string arguments that may have been produced by the LLM
            link['arguments'] = [arg for arg in rewritten_args if isinstance(arg, str)]


    # --- 4. Final Metadata ---
    final_meta = {
        "document_id": document_id,
        "fragments_merged": len(valid_graphs),
        "source_sentences": [
            {"id": s.id, "text": s.text} for s in sentences
        ]
    }

    merged = Hypergraph(
        graph_type="AtomSpace Hypergraph",
        nodes=merged_nodes,
        links=merged_links,
        metadata=final_meta,
    )
    logger.info(
        "Merged %d HG fragments → %d nodes, %d links",
        len(valid_graphs),
        len(merged_nodes),
        len(merged_links),
    )
    return merged