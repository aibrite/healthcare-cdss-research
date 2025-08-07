# file: hypergraph_extractor/agents.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import asyncio
import copy
import json
import json5
import logging
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Type
import semantic_chunker

import spacy
from fastcoref import spacy_component

import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


from config import Settings
from models import Hypergraph, NTKG, Sentence, default_timestamp, year_from_text
from openai_client import chat_completion

logger = logging.getLogger(__name__)

try:
    _nlp = spacy.load("en_core_sci_scibert", exclude=["parser", "lemmatizer", "ner", "textcat"])
    #if "sentencizer" not in _nlp.pipe_names:
        #_nlp.add_pipe("sentencizer", first=True)
    _nlp.add_pipe("fastcoref")
except OSError:
    logger.error("spaCy model 'en_core_sci_scibert' not found.")
    logger.error("Please run: python -m spacy download en_core_sci_scibert")
    _nlp = None

_DEBUG_DIR = Path("out/debug")
_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
_PLACEHOLDER_LITERAL = "<<MISSING_LITERAL>>"
MAX_RETRIES = 3

# --- Default truth values based on predicate type, as a fallback ---
PREDICATE_TRUTH_VALUES = {
    "implies": [0.85, 0.70],
    "causes": [0.85, 0.70],
    "therapy_effect": 0.90,
    "adverse_effect": 0.80,
    "requires_monitoring": 0.60,
    "requires_investigation": 0.50,
}
DEFAULT_TRUTH_VALUE = 0.5 # A neutral default if predicate is unknown

def _strip_trailing_commas(raw: str) -> str:
    """
    json5.loads → json.dumps round-trip removes *all* trailing commas,
    even nested ones. We lose comments but preserve formatting.
    """
    try:
        obj = json5.loads(raw)
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        return raw

_trailing_comma_re = re.compile(r",\s*([}\]])")


def _extract_json_from_text(text: str) -> str:
    """
    Pull out the *first* JSON object from an LLM reply and
    strip any trailing commas that would break `json.loads`.
    """
    # prefer fenced ```json ``` blocks
    for pat in (r"```json\s*(\{.*?})\s*```", r"```\s*(\{.*?})\s*```"):
        m = re.search(pat, text, re.S)
        if m:
            candidate = _strip_trailing_commas(m.group(1))
            return candidate

    # fallback: first {...} in the text
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = _trailing_comma_re.sub(r"\1", text[first : last + 1])
        return candidate

    raise ValueError("No JSON object found")


def _sanitize_ntkg_dict(d: dict, sent: "Sentence | None" = None) -> dict:
    """
    Performs robust cleaning on raw LLM output for NTKG graphs before Pydantic validation.
    - Ensures metadata is present.
    - Normalizes common alternative keys (e.g., 'value' -> 'literal').
    - **Removes malformed facts (e.g., empty tuples).**
    - **Injects a default `truth_value` if it's missing, based on the predicate.**
    """
    if not d.get("graph_type", "").startswith("N-tuple"):
        return d

    if "metadata" not in d or d["metadata"] is None:
        d["metadata"] = {}
    if sent is not None:
        d["metadata"].setdefault("sentence_id", sent.id)
        d["metadata"].setdefault("source_sentence", sent.text)

    new_facts = []
    if "facts" in d and isinstance(d["facts"], list):
        for fact in d["facts"]:
            if not isinstance(fact, dict):
                continue # Skip malformed fact entries

            # --- Structural validation: Ensure fact is not empty ---
            has_tuple = isinstance(fact.get("tuple"), list) and len(fact.get("tuple", [])) > 0
            has_args = isinstance(fact.get("arguments"), list) and len(fact.get("arguments", [])) > 0
            if not has_tuple and not has_args:
                logger.warning("Sanitizer removing malformed fact with no content: %s", fact.get('id', 'N/A'))
                continue # Discard this fact, it's invalid

            # --- Add sentence_id to fact if missing ---
            if "sentence_id" not in fact and sent is not None:
                fact["sentence_id"] = sent.id
            
            # --- Ensure truth_value exists (CRITICAL FIX) ---
            if "truth_value" not in fact or fact["truth_value"] is None:
                predicate = fact.get("predicate", "")
                default_tv = PREDICATE_TRUTH_VALUES.get(predicate, DEFAULT_TRUTH_VALUE)
                fact["truth_value"] = default_tv
                logger.warning("Sanitizer injected missing 'truth_value': %s for predicate '%s'", default_tv, predicate)

            # --- Clean up roles within the tuple ---
            if has_tuple:
                new_tuple = []
                for role_obj in fact.get("tuple", []):
                    if not isinstance(role_obj, dict):
                        continue # Skip malformed roles
                    r = copy.deepcopy(role_obj)

                    # Normalize literal and datatype keys
                    if "literal" not in r:
                        for alias in ("value", "val", "amount"):
                            if alias in r:
                                r["literal"] = r.pop(alias)
                                break
                    if "datatype" not in r and "data_type" in r:
                        r["datatype"] = r.pop("data_type")

                    # Handle case where entity/literal is missing
                    if r.get("entity") is None and r.get("literal") in (None, "", []):
                        r["literal"] = _PLACEHOLDER_LITERAL
                    
                    new_tuple.append(r)
                fact["tuple"] = new_tuple
            
            new_facts.append(fact)

    d["facts"] = new_facts
    return d


# 3. Agents
class SentenceSplitterAgent:
    """
    A robust sentence splitter for scientific text. It uses a multi-stage
    process to handle complex punctuation and structures.
    1. Pre-processes text by converting semicolons and list markers to periods.
    2. Uses spaCy's powerful sentencizer for the core splitting logic.
    3. Filters out empty results and provides a clean list of sentences.
    """

    def run(self, text: str) -> List[Sentence]:
        # Stage 1: Pre-process the text to normalize clause-separating punctuation
        # Turn semicolons into periods.
        #processed_text = re.sub(r';', '.', text)
        # Turn bullet points or list items into sentences.
        processed_text = re.sub(r'\n\s*[-*•]\s*', '. ', text)
        # Collapse multiple whitespace characters into a single space
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        processed_text = _nlp(      # for multiple texts use nlp.pipe
           processed_text, 
           component_cfg={"fastcoref": {'resolve_text': True}}
        )
        processed_text = processed_text._.resolved_text
        Path("input_coref.txt").write_text(processed_text)
         # ID WILL ADD THE PROMPTER BASED BASIC NER HERE 
         
         
         # raw_sents = []
         # try:
            # Stage 2: Use spaCy's sentencizer on the pre-processed text
             # if not _nlp:
               #   raise RuntimeError("spaCy model not loaded.")
             # doc = _nlp(processed_text)
            # Stage 3: Filter and clean the output
             # raw_sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 1]

         # except Exception as exc:
            #  logger.error(
               #   "spaCy sentenciser failed (%s) – falling back to basic regex", exc
             # )
            # Basic fallback for catastrophic failures
             # raw_sents = re.split(r'(?<=[.!?])\s+', processed_text)

        # Final creation of Sentence objects, ensuring no empty strings get through
        try:
            chunker = semantic_chunker.get_chunker(
                "gpt-4.1",
                chunking_type="text", 
                max_tokens=150, 
                trim=False,  
                overlap=0, 
            )
        except Exception as e:
            print(e)

        chunks = chunker.chunks(processed_text)  # list[str]
        sentences = [
            Sentence(id=f"SENT_{i+1}", text=t) 
            for i, t in enumerate(chunks) if t
        ]
        logger.info("SentenceSplitterAgent produced %d sentences", len(sentences))
        return sentences


class LLMExtractionAgent:
    """
    Base-class.
    """

    prompt_template: str
    example_json: str
    out_model = NTKG

    def __init__(self, settings: Settings):
        self.settings = settings

    async def _extract_one(self, sent: Sentence):
        msgs = [
            {"role": "system", "content": self.settings.system_prompt},
            {
                "role": "assistant",
                "content": self.settings.example_prompt.replace(
                    "{EXAMPLE_JSON}", self.example_json
                ),
            },
            {
                "role": "user",
                "content": self.prompt_template.replace(
                    "{SENTENCE}", sent.text
                ).replace(
                    "{SENTENCE_ID}", sent.id
                ),
            },
        ]

        for attempt in range(1, MAX_RETRIES + 2): 
            content = await chat_completion(
                model=self.settings.agent.model_name,
                messages=msgs,
                max_completion_tokens=self.settings.agent.max_completion_tokens,
                temperature=self.settings.agent.temperature,
            )

            (_DEBUG_DIR / f"debug_{self.__class__.__name__}_{sent.id}_try{attempt}.txt"
            ).write_text(content, encoding="utf-8")
        
            try:
                json_block = _extract_json_from_text(content)
                if self.out_model is NTKG:
                    payload = _sanitize_ntkg_dict(json.loads(json_block), sent)
                    graph = self.out_model.model_validate(payload)
                else: 
                    graph = self.out_model.model_validate_json(json_block)
            except Exception as exc:
                logger.warning("%s parse failed (try %d): %s", sent.id, attempt, exc)
                graph = None

            def _is_empty(g):
                if g is None:
                    return True
                if isinstance(g, NTKG):
                    return len(g.entities) == 0 and len(g.facts) == 0
                if isinstance(g, Hypergraph):
                    return len(g.nodes) == 0
                return True  # shouldn’t happen

            if not _is_empty(graph):
                return graph

            logger.info(
                "%s produced empty graph on try %d/%d",
                sent.id,
                attempt,
                MAX_RETRIES + 1,
            )
            if attempt > MAX_RETRIES:
                break
        return None
	
    async def run(self, sentences: Iterable[Sentence]) -> Sequence[NTKG | Hypergraph]:
        sem = asyncio.Semaphore(int(self.settings.agent.rate_limit_rps))

        async def guard(coro):
            async with sem:
                return await coro

        tasks = [guard(self._extract_one(s)) for s in sentences]
        return await asyncio.gather(*tasks)


class HGExtractionAgent(LLMExtractionAgent):
    out_model = Hypergraph

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.prompt_template = settings.prompt_hg_template
        self.example_json = settings.example_json_hg


class HRKGExtractionAgent(LLMExtractionAgent):
    out_model = NTKG

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.prompt_template = settings.prompt_hrkg_template
        self.example_json = settings.example_json_hrkg