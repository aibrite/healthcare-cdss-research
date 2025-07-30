# file: hypergraph_extractor/pipeline.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import asyncio
import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel

import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from aggregator import merge_hg, merge_ntkg
from agents import HGExtractionAgent, HRKGExtractionAgent, SentenceSplitterAgent
from config import load_settings
from models import Hypergraph, NTKG, Sentence
from typedb_exporter import (
    export_ntkg_to_typeql,
    export_typeql_schema,
)



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

def _write_json(path: Path, model: BaseModel):
    """Serializes a Pydantic model to a JSON file with indentation."""
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


async def _run_async_pipeline(text: str, out_dir: Path, document_id: str):
    """The asynchronous part of the pipeline."""
    settings = load_settings()

    # ── 1 · sentence splitting ────────────────────────────────────────────────
    splitter = SentenceSplitterAgent()
    sentences: List[Sentence] = splitter.run(text)
    if not sentences:
        logger.warning("No sentences were found in the input text. Exiting.")
        return

    # ── 2 · concurrent extraction ─────────────────────────────────────────────
    hg_agent = HGExtractionAgent(settings)
    hrkg_agent = HRKGExtractionAgent(settings)
    logger.info("Starting concurrent extraction for %d sentences...", len(sentences))
    hg_results, hrkg_results = await asyncio.gather(
        hg_agent.run(sentences),
        hrkg_agent.run(sentences),
    )

    hg_graphs: List[Hypergraph] = [g for g in hg_results if g]
    ntkgs: List[NTKG] = [g for g in hrkg_results if g]
    logger.info(
        "Extraction complete. Found %d valid Hypergraphs and %d valid NTKGs.",
        len(hg_graphs),
        len(ntkgs),
    )

    # ── 3 · per-sentence raw JSON (debug) ─────────────────────────────────────
    debug_dir = out_dir / "per_sentence_outputs"
    debug_dir.mkdir(exist_ok=True, parents=True)
    for sent, hg, kg in zip(sentences, hg_results, hrkg_results):
        if hg:
            _write_json(debug_dir / f"{sent.id}_hg.json", hg)
        if kg:
            _write_json(debug_dir / f"{sent.id}_hrkg.json", kg)

    # ── 4 · merge NTKGs ───────────────────────────────────────────────────────
    if ntkgs:
        hrkg_dir = out_dir / "HRKG_merged"
        hrkg_dir.mkdir(exist_ok=True, parents=True)
        merged_ntkg = merge_ntkg(ntkgs, document_id=document_id)
        _write_json(hrkg_dir / "final_merged_hrkg.json", merged_ntkg)
        logger.info("✅ HRKG merge finished → %s", hrkg_dir)
        
        # I CAN ADD json2FB HERE FIRST AND TRY inline json for TYPEDB also
        
        
        # NEW ▼  Generate TypeDB artefacts ------------------------------------
        typedb_dir = out_dir / "TypeDB"
        typedb_dir.mkdir(exist_ok=True, parents=True)
        export_typeql_schema(merged_ntkg, typedb_dir / "schema.tql")
        export_ntkg_to_typeql(merged_ntkg, typedb_dir / "data.tql")
        logger.info("✅ TypeDB files ready     → %s", typedb_dir)
    else:
        logger.warning("No valid NTKG graphs to merge.")

    # ── 5 · merge HGs ─────────────────────────────────────────────────────────
    if hg_graphs:
        hg_dir = out_dir / "HG_merged"
        hg_dir.mkdir(exist_ok=True, parents=True)
        merged_hg = merge_hg(
            [g for g in hg_graphs if g],
            document_id=document_id,
            sentences=sentences,
        )
        _write_json(hg_dir / "final_merged_hg.json", merged_hg)
        logger.info("✅ HG merge finished → %s", hg_dir)
        
        # I CAN ADD upload_to_atomspace HERE AND TRY to automatize for AtomSpace and MeTTa also
        
        
    else:
        logger.warning("No valid Hypergraphs to merge.")
    
        # IF I ADD json2FB I can add graph_processor here to automatize the graph production pipeline also

def run_pipeline(text: str, out_dir: Path, document_id: str):
    """
    Main pipeline entry point. Orchestrates sentence splitting,
    concurrent extraction, and graph aggregation.
    """
    if os.path.isdir(str(out_dir)):
        print("OK")
        asyncio.run(_run_async_pipeline(text, out_dir, document_id))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(_run_async_pipeline(text, out_dir, document_id))
    
