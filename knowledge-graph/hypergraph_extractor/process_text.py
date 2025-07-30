# file: hypergraph_extractor/process_text.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import sys
from pathlib import Path

from pipeline import run_pipeline

if __name__ == "__main__":
    
    text_path, out_dir = Path(__file__).parents[1] / "input.txt", Path('out_sem_2')
    text = (
        sys.stdin.read()
        if text_path == "-"
        else Path(text_path).read_text(encoding="utf-8")
    )

    run_pipeline(text, out_dir, document_id=text_path)
