import json
import os
import re
import threading
from ftplib import all_errors
from functools import partial
from json import load
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

import dspy
from dotenv import load_dotenv
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Import the optimizer
from dspy.teleprompt import BootstrapFewShot, BootstrapFinetune, MIPROv2
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from prefect import flow, get_run_logger, task
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from sklearn.model_selection import train_test_split
import spacy


load_dotenv()


llms = {
    "default": "openai/gpt-4.1-mini-2025-04-14",
    "ner_task": "openai/gpt-4.1-mini-2025-04-14",
    "semantic_chunker": "openai/gpt-4.1-nano-2025-04-14",
    "ner_task_judge": "openai/gpt-4.1-mini-2025-04-14",
}


ENTITY_LABEL_PATTERN = re.compile(r"^(.*?)(\[\d+\])?$")


class Span(BaseModel):
    start: int = Field(description="index of the first token in the span (inclusive)")
    end: int = Field(description="index just after the last token (exclusive)")
    label: str = Field(
        description="the appropriate UMLS semantic type (e.g., FINDING_T033)"
    )
    text: str = Field(description="the exact phrase covered by the token")

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.label))

    def __eq__(self, other):
        if not isinstance(other, Span):
            return NotImplemented
        return (self.start, self.end, self.label) == (
            other.start,
            other.end,
            other.label,
        )


class SpanImproved(Span):
    """
    The improved output of the NER task.
    """

    umls_pref_label: Optional[str] = Field(
        default=None,
    )

    comments: Optional[str] = Field(
        default=None,
        description="Comments on the NER output, explaining improvements made to the NER output. Return null if no improvements were made.",
    )


class SpanValidated(SpanImproved):
    pass


class SpanList(BaseModel):
    spans: List[Span] = Field(description="List of annotated spans.")


class ImprovedSpanList(BaseModel):
    spans: List[SpanImproved] = Field(description="List of annotated spans.")


class SemanticParagraph(BaseModel):
    # Optional: The LLM can generate a topic for the paragraph
    heading: str = dspy.OutputField(
        desc="A concise heading for the semantic group (e.g., 'Discussion of Liver Enzymes', 'Comparison with PBC')."
    )
    # The actual sentences, verbatim
    sentences: List[str] = dspy.OutputField(
        desc="A list of original, unchanged sentences that belong to this group."
    )


class GroupedText(BaseModel):
    paragraphs: List[SemanticParagraph] = dspy.OutputField(
        desc="The full text, reorganized into a list of semantically coherent paragraphs."
    )


class SentenceWithTokens(BaseModel):
    sentence: str = dspy.InputField()
    tokens: List[str] = dspy.InputField()


class SpansOutput(BaseModel):
    sentence: str = dspy.OutputField(desc="original sentence")
    span_list: SpanList = dspy.OutputField(desc="spans for the sentence")


class ExtractSignature(dspy.Signature):
    """
    You are an expert medical annotator specializing in UMLS semantic typing and clinical information extraction. Your task is critical for building a trustworthy Clinical Decision Support System (CDSS) that can improve patient care. Strive for absolute precision and rigor.

    UMLS Semantic Types (Annotate ONLY these five types):
    * **Sign or Symptom (T184):** An observable manifestation of a disease or a subjective experience reported by a patient. This is typically reserved for general, subjective, or descriptive experiences (e.g., "pain," "nausea," "fatigue," "tenderness") that are not formally classified as diseases themselves.
    * **Diagnostic Procedure (T060):** The act of performing a procedure. *Examples: "liver biopsy," "endoscopy".*
    * **Finding (T033):** An evaluative or comparative observation about clinical data.
        * **Litmus Test:** A span is a `Finding` **if and only if** it contains an explicit evaluation (e.g., "high," "elevated," "ratio > 1"), a comparison ("greater than," "rather than"), or an interpretation ("deficiency," "suppression").
        * A finding can combine multiple measurements if they are part of a single, unified evaluation.
    * **Laboratory Procedure (T059):** A raw laboratory test name without any evaluation or result. These are always atomic (e.g., "ALT," "AST," "GGT"). "AST:ALT ratio" is a lab procedure; "AST:ALT ratio > 1.5" is a finding.
    * **Disease or Syndrome (T047):** A pathological condition that alters a normal process (e.g., "diabetes," "cirrhosis," "Primary Sclerosing Cholangitis"). This refers to the state itself, not the observation of its markers.

    ⸻

    For each sentence in input:

    **CRITICAL ANNOTATION RULES**

    **Part A: Technical Span Integrity (Check First!)**
    1.  **Perfect Token Reconstruction:** The `text` of your annotated span MUST be an exact match of the tokens joined by spaces.
        * Verify: `span.text == " ".join(tokens[span.start:span.end])`
        * Example: If `tokens` are `['AST', ':', 'ALT', 'ratio', 'of', '<', '1.5']`, the span for "AST : ALT ratio of < 1.5" must be `start=0, end=7`. The text will be `"AST : ALT ratio of < 1.5"`. Pay close attention to spaces around punctuation and symbols.
    2.  **Valid Indices:** Span indices must be valid.
        * Verify: `0 ≤ span.start < span.end ≤ len(tokens)`

    **Part B: Semantic Annotation Rules**
    3. Prioritize Cause-Effect Separation for CDSS: In sentences where a finding indicates, suggests, or is a risk factor for a disease, you MUST separate the cause and the effect into distinct spans.
        The 'Cause' Span: This is typically the Finding (e.g., "AST:ALT ratio > 1.5").
        The 'Effect' Span: This is the Disease or Syndrome that is being indicated. The span for the disease should exclude the relational/linking words (e.g., "indicates," "suggests," "is more likely").
    4.  **Longest Contiguous Span:** Always annotate the longest possible contiguous span that represents a single, coherent semantic type.
        * *Example:* For "raised GGT with a normal ALP," the correct span is the full phrase, not just "raised GGT."
    5.  **No Leading/Trailing Waste:** Trim the span to begin and end with a "content" token. Never include leading determiners (the, a, an) or trailing punctuation unless it's part of the final token itself (e.g., '1000I/U.').
    6.  **Handle Parentheses Correctly:**
        * *Synonyms & Abbreviations*: If parentheses contain a direct synonym or abbreviation, annotate the main term and the term in the parentheses as two separate spans. The parenthesis tokens ( and ) themselves are never included in any span.
        * *Integral Clarifications*: If parentheses contain a clarification that is an essential part of the description of the main entity, you MUST INCLUDE the parentheses and their content within the main entity's span. This often occurs within a broad Finding or Disease description
        * *Interrupting Explanations*: If parentheses provide an external comparison, contrast, or explanation that is not an essential part of the preceding entity, you MUST EXCLUDE the parentheses and their content from the preceding span. The preceding span must end before the opening parenthesis. Any distinct entities inside the parenthesis should be annotated separately.
    7.  **Resolve Overlaps:** If two potential spans overlap, you must choose only one. Annotate the one that represents the longest, most complete semantic concept.
    8.  **T047 Precedence for Named Conditions**: For named medical conditions that could be considered both a sign and a disease (e.g., 'macrocytosis', 'anemia', 'jaundice', 'cholestasis'), you MUST use the DISEASE_OR_SYNDROME_T047 label. The SIGN_OR_SYMPTOM_T184 label should only be used if the term does not represent a formally named pathological state.
    ⸻

    **Chain of Thought Process**

    1.  **Understand the Sentence:** Read the sentence to grasp the full clinical context.

    2.  **Identify Potential Spans:** Identify all words and phrases that could potentially be one of the five UMLS types. Be inclusive at this stage.

    3.  **Filter and Refine Spans based on Rules:** For **each potential span**, rigorously apply the rules:
        * **Apply the `Finding` Litmus Test:** Does this span contain a clear evaluative or comparative term? If yes, it's a `Finding`. If no, it CANNOT be a `Finding`. Is it a raw test name instead? Then it's a `Laboratory Procedure`.
        * **Check for Longest Span:** Is this the longest possible span for this concept? Did I miss adjacent words that complete the idea (e.g., "...tend to fluctuate")?
        * **Check Parentheses:** Did I handle any parentheses according to Rule #5?
        * **Check for Chronic vs. Acute State:** Is this a chronic disease state (`Disease or Syndrome`) or an immediate observation/deficiency (`Finding`)? "Dietary deficiency of folate" is a `Finding`.
        * **Check for Overlaps:** Have I created overlapping spans? If so, resolve them now using Rule #6.

    4.  **Final Quality and Technical Check:** Before concluding, perform this final review:
        * ✅ For every span, have I mentally confirmed that `span.text` will exactly match `" ".join(tokens[span.start:span.end])`?
        * ✅ For every `Finding`, can I point to the specific word that makes it a finding?
        * ✅ For every `Laboratory Procedure`, is it truly atomic and free of evaluation?
        * ✅ Did I apply the longest span principle correctly, especially for complex findings?
        * ✅ Is length of input equal to length of output ?
        * ❤️‍🔥 As an expert annotator, does this output reflect the highest standard of accuracy, knowing it will be used to support clinical decisions?
        ⸻

        Correct Example Labels:

        - Sentence: "ALT(SGPT) is an important test."
          Labels:
          "ALT" ->  LABORATORY_PROCEDURE_T059
          "SGPT" -> LABORATORY_PROCEDURE_T059
        - Sentence: "Raised LDL with slightly high AST, GGT and often ALP:AST > 1 is a risk factor for CHD."
          Labels:
          "Raised LDL with slightly high AST, GGT and often ALP:AST > 1" -> FINDING_T033
          "CHD" -> DISEASE_OR_SYNDROME_T047
        - Sentence: "Cholesterol measured high or borderline in comparision to LDL indicates liver disease."
          Labels:
          "Cholesterol measured high or borderline in comparision to LDL" -> FINDING_T033
          "liver disease" -> DISEASE_OR_SYNDROME_T047
        - Sentence: "Constantly high ALT rather than AST in young population with fast-food culture is a risk"
          Labels:
          "Constantly high ALT rather than AST" -> FINDING_T033
        - Sentence: "Alcohol causes levels of ALP and GGT (without AST) tend to fluctuate in comparison to increase in AST:ALT"
          Labels:
          "levels of ALP and GGT (without AST) tend to fluctuate in comparison to increase in AST:ALT" -> FINDING_T033

        Wrong Labels:
        - Sentence: "AST and GGT are elevated (contrary to ALP) in patients"
          Labels:
          "AST and GGT are elevated" -> FINDING_T033 (longest span is "AST and GGT are elevated (contrary to ALP)")
        - Sentence: "The ALP and AST test is useful to understand the liver function."
          Labels:
          "ALP" -> FINDING_T033 (ALP here is not a finding but a laboratory procedure)
          "ALP and AST"->LABORATORY_PROCEDURE_T059 (not atomic)
    """

    previous_run: str = dspy.InputField(
        default=None,
        description="The previous run of the NER task, if any. This is used to provide a correction for the current run.",
    )
    input: List[SentenceWithTokens,] = dspy.InputField(
        description="The input sentences to be annotated.", default_factory=list
    )
    output: List[SpansOutput] = dspy.OutputField(
        desc="Output list based on input order.", default_factory=list
    )


class NerJudgeSignature(dspy.Signature):
    """
    You are a Senior Medical Annotator responsible for quality assurance.
    Your task is to meticulously review a set of proposed NER annotations, correct any errors, and provide a final, high-quality output.
    You will be given the original sentence, the prompt/rules used for the initial annotation, and the proposed annotations.

    Your goal is to ensure the final output is 100% compliant with all rules outlined in the provided prompt.

    Cognitive Process for Review:
    1.  **Initial Setup**:
        - Carefully read the original `sentence`.
        - Carefully read the `prompt_used` to fully understand all annotation rules, definitions, and constraints.
        - Review the `ner_proposed` output (tokens and spans).

    2.  **Systematic Span-by-Span Verification**:
        - For each `span` in `ner_proposed.spans`, verify its correctness against the rules from the `prompt_used`. Ask yourself:
            - **Rule Adherence**: Does this span violate any of the "Annotation Rules"?
                - Is it the *longest possible* contiguous span for this concept? (Rule 1, 4, 5)
                - Are compound expressions correctly kept together? (Rule 2)
                - Is the span properly trimmed (no leading/trailing determiners or punctuation)? (Rule 3)
                - Are synonyms in parentheses handled correctly as separate spans? (Rule 6)
                - If the label is `FINDING_T033`, does the text contain an explicit evaluation (e.g., 'high', 'elevated', 'ratio > 1')? (Rule 7)
            - **Label Accuracy**: Is the `label` (e.g., `FINDING_T033`, `DISEASE_OR_SYNDROME_T047`) the most accurate one based on the definitions in the prompt? Is a `LABORATORY_PROCEDURE_T059` being confused with a `FINDING_T033`?
            - **Boundary and Text Accuracy**: Are the `start` and `end` token indices correct? Does the `text` field perfectly match the concatenation of `tokens[start:end]`?

    3.  **Identification of Omissions**:
        - After reviewing all proposed spans, read the `sentence` one more time.
        - Have any valid medical concepts that conform to the annotation rules been missed entirely?

    4.  **Correction and Synthesis**:
        - Based on your verification and omission checks, create a definitive list of final spans. This may involve:
            - **Keeping** correct spans from the proposal.
            - **Deleting** incorrect spans from the proposal.
            - **Modifying** spans from the proposal (e.g., adjusting boundaries, changing the label).
            - **Adding** new spans that were missed.

    5.  **Normalize to UMLS:** After finalizing a span, determine its `umls_pref_label` based on its semantic type (e.g TXXX).
        * **Example:** If the `text` is "CHD", the `umls_pref_label` is "Coronary Heart Disease".
        * **Handle Non-Applicable Cases:** For descriptive spans (e.g., "Raised LDL and GGT") or terms without a direct UMLS concept, this field **must be null**. Do not alter the span to force a match.

    6.  **Generate Final Output and Comments**:
        - Construct the `ner_final` object with the corrected list of spans.
        - In the `comments` field, provide a clear and concise explanation for every change you made (deletions, modifications, additions).
        - For each change, reference the specific rule(s) from the prompt that motivated the correction.
        - Fill `umls_pref_label` field if applicable.
        - If the proposed output was already perfect and no changes were needed, set `comments` to null.
        - Yo don't need to generate a comment about setting umls_pref_label.
    """

    prompt_used: str = dspy.InputField(
        desc="The full prompt, including rules, used to generate the proposed output."
    )
    sentence: str = dspy.InputField(desc="The original input sentence.")
    tokens: List[str] = dspy.InputField(desc="The tokens of the input sentence.")
    ner_proposed: SpanList = dspy.InputField(
        desc="The proposed NER annotations to be reviewed."
    )
    ner_final: ImprovedSpanList = dspy.OutputField(
        desc="The final, corrected NER annotations with explanatory comments."
    )


class SemanticChunkerSignature(dspy.Signature):
    """You are an expert technical writer and medical informatician. Your task is to analyze a list of clinical sentences and group them into semantically coherent paragraphs. Each paragraph should focus on a single topic or a closely related set of ideas.

    **Critical Rules:**
    1.  You MUST NOT change, rephrase, edit, or alter the original sentences in any way.
    2.  You MUST preserve the original order of sentences within each group.
    3.  Every sentence from the input list must be included in exactly one group. Do not omit any sentences.

    Group the sentences and provide a descriptive heading for each group.
    """

    text: str = dspy.InputField(desc="text to be grouped.")
    grouped_document: GroupedText = dspy.OutputField(
        desc="The document, restructured into semantic paragraphs."
    )


T = TypeVar("T", bound=SpanList)


class NerResult(GenericModel, Generic[T]):
    sentence: str
    tokens: list[str]
    output: T


class LLMModule(dspy.Module):
    """
    Base module with built-in singleton logic and cost calculation.
    """

    _instances = {}
    _lock = threading.Lock()

    def __init__(self, lm, cache):
        self.lm = lm
        self.cache = False  # cache

    @classmethod
    def get_instance(cls, *args, **kwargs):
        """
        This class method returns the single instance of a given subclass.
        It ensures that each subclass (e.g., NERModule, JudgeModule)
        has its own distinct singleton instance.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    # Create a new instance of the specific subclass (cls)
                    # and pass any arguments to its __init__
                    cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]

    def calculate_cost(self):
        # This implementation remains the same
        cost = sum([x["cost"] for x in self.lm.history if x["cost"] is not None])
        return cost


class ExtractModule(LLMModule):
    def __init__(self, lm=None, cache=False):
        lm = lm or dspy.LM(
            llms["ner_task"], temperature=1.0, max_tokens=32000, cache=cache
        )
        super().__init__(lm=lm, cache=cache)
        self.ner_predictor = dspy.ChainOfThought(ExtractSignature)

    def forward(self, input, previous_run: Optional[str] = None) -> dspy.Prediction:

        with dspy.context(lm=self.lm):
            res = self.ner_predictor(input=input, previous_run=previous_run)

        return res


class JudgeModule(LLMModule):
    def __init__(self, lm=None, cache=False):
        lm = lm or dspy.LM(
            llms["ner_task_judge"], temperature=0.0, max_tokens=32000, cache=cache
        )
        super().__init__(lm=lm, cache=cache)
        self.ner_predictor = dspy.ChainOfThought(NerJudgeSignature)

    def forward(
        self,
        sentence: str,
        tokens: list[str],
        prompt_used,
        ner_proposed: Optional[str] = None,
    ) -> dspy.Prediction:
        with dspy.context(lm=self.lm):

            # with mlflow.start_run(run_name="judge"):

            res = self.ner_predictor(
                sentence=sentence,
                tokens=tokens,
                prompt_used=prompt_used,
                ner_proposed=ner_proposed,
            )

        return res


class SemanticChunkerModule(LLMModule):
    def __init__(self, lm=None, cache=True):
        lm = lm or dspy.LM(
            llms["semantic_chunker"], temperature=0.0, max_tokens=32000, cache=cache
        )
        super().__init__(lm=lm, cache=cache)
        self.ner_predictor = dspy.ChainOfThought(SemanticChunkerSignature)

    def forward(self, text: str) -> dspy.Prediction:
        with dspy.context(lm=self.lm):

            # with mlflow.start_run(run_name="judge"):

            res = self.ner_predictor(
                text=text,
            )

        return res


def _calculate_prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Helper function to calculate precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_ner_metrics_with_output(
    gold_outputs: List[SpanList],
    predicted_outputs: List[SpanList],
) -> Dict[str, Any]:
    per_label_tp: Dict[str, int] = defaultdict(int)
    per_label_fp: Dict[str, int] = defaultdict(int)
    per_label_fn: Dict[str, int] = defaultdict(int)
    all_labels: Set[str] = set()
    per_example_metrics = []

    # --- Loop through each example ---
    for gold_output, predicted_output in zip(gold_outputs, predicted_outputs):
        gold_spans_set = set(gold_output.spans)
        predicted_spans_set = set(predicted_output.spans)

        # --- NEW: Calculate metrics for this single example for Macro-Average ---
        example_tp = len(gold_spans_set.intersection(predicted_spans_set))
        example_fp = len(predicted_spans_set - gold_spans_set)
        example_fn = len(gold_spans_set - predicted_spans_set)
        per_example_metrics.append(_calculate_prf1(example_tp, example_fp, example_fn))

        # --- Accumulate per-label stats (existing logic) ---
        tp_spans = gold_spans_set.intersection(predicted_spans_set)
        fp_spans = predicted_spans_set - gold_spans_set
        fn_spans = gold_spans_set - predicted_spans_set

        for span in gold_spans_set.union(predicted_spans_set):
            all_labels.add(span.label)
        for span in tp_spans:
            per_label_tp[span.label] += 1
        for span in fp_spans:
            per_label_fp[span.label] += 1
        for span in fn_spans:
            per_label_fn[span.label] += 1

    # --- Final Calculations ---
    results: Dict[str, Any] = {}

    # --- MODIFIED: Calculate the overall score using MACRO-averaging ---
    num_examples = len(gold_outputs)
    if num_examples > 0:
        macro_precision = (
            sum(ex["precision"] for ex in per_example_metrics) / num_examples
        )
        macro_recall = sum(ex["recall"] for ex in per_example_metrics) / num_examples
        macro_f1 = sum(ex["f1"] for ex in per_example_metrics) / num_examples
        # The total TP/FP/FN are still useful to return for context
        total_tp = sum(ex["tp"] for ex in per_example_metrics)
        total_fp = sum(ex["fp"] for ex in per_example_metrics)
        total_fn = sum(ex["fn"] for ex in per_example_metrics)
    else:
        macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
        total_tp, total_fp, total_fn = 0, 0, 0

    results["overall"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    # --- Calculate per-label scores (this logic remains the same) ---
    results["per_label"] = {}
    for label in all_labels:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        results["per_label"][label] = _calculate_prf1(tp, fp, fn)

    return results


def calculate_ner_metrics(
    gold_outputs: List[NerResult[SpanList]],
    predicted_outputs: List[NerResult[SpanList]],
) -> Dict[str, Any]:

    gold = [gold_output.output for gold_output in gold_outputs]
    pred = [pred_output.output for pred_output in predicted_outputs]

    return calculate_ner_metrics_with_output(gold, pred)


import re
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Optional, Tuple

PIPE_SPLIT = re.compile(r"\s*\|\s*")
ID_RX = re.compile(r"\[(\d+)]$")  # …T047[12]  → 12


def _parse_ann(raw: str) -> Optional[Tuple[str, Optional[int]]]:
    """
    Returns (label, id) or None for '_'.
    Handles `LABEL[3]`  or  just `LABEL`.
    """
    raw = raw.strip()
    if raw == "_" or not raw:
        return None
    raw = PIPE_SPLIT.split(raw)[0]  # keep only 1st layer
    m = ID_RX.search(raw)
    if m:
        return raw[: m.start()], int(m.group(1))
    return raw, None  # idless span


nlp = spacy.load("en_core_sci_sm")  # or en_core_sci_lg


def tokenize(text: str) -> List[str]:
    doc = nlp(text)
    with doc.retokenize() as retokenizer:
        return [token.text for token in doc]


# ── robust loader ─────────────────────────────────────────────────────────────
def new_webanno_tsv_to_ner_span_outputs(tsv: str) -> List[NerResult[SpanList]]:
    """
    Parse a WebAnno TSV-3.3 string into NerResult objects.
    Tolerant to: trailing tabs, mixed spaces/tabs, no span-id, absolute char ranges.
    """
    results: List[NerResult[SpanList]] = []
    lines = tsv.splitlines()

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        # skip headers / multiple blank lines
        if not ln or ln.startswith("#FORMAT") or ln.startswith("#T_"):
            i += 1
            continue

        if not ln.startswith("#Text="):
            raise ValueError(f"Expected #Text= at line {i+1}")
        sentence_text = ln[6:]
        i += 1

        tokens: List[str] = []
        ann_per_tok: List[Optional[Tuple[str, Optional[int]]]] = []

        while i < len(lines) and lines[i].strip():  # until blank line
            cols = lines[i].split("\t")
            while cols and cols[-1] == "":  # drop trailing empties
                cols.pop()
            if len(cols) < 4:
                raise ValueError(f"Bad token line @ {i+1}: {lines[i]!r}")

            _, _char_rng, tok, raw_ann = cols[:4]
            tokens.append(tok)
            ann_per_tok.append(_parse_ann(raw_ann))
            i += 1
        i += 1  # skip the blank line after this sentence

        # ── build spans by grouping contiguous identical annotations ─────────
        spans: List[Span] = []
        span_start = None
        current_ann = None

        for idx, ann in enumerate(ann_per_tok):
            if ann != current_ann:
                if current_ann is not None:  # close previous
                    spans.append(
                        Span(
                            start=span_start,
                            end=idx,
                            label=current_ann[0],
                            text=" ".join(tokens[span_start:idx]),
                        )
                    )
                current_ann = ann
                span_start = idx if ann is not None else None
            # else: continue current span
        if current_ann is not None:  # close final span
            spans.append(
                Span(
                    start=span_start,
                    end=len(tokens),
                    label=current_ann[0],
                    text=" ".join(tokens[span_start:]),
                )
            )

        results.append(
            NerResult[SpanList](
                sentence=sentence_text,
                tokens=tokens,
                output=SpanList(spans=spans),
            )
        )

    return results


def load_from_tsv(tsv_file_path: str) -> List[NerResult[SpanList]]:
    with open(tsv_file_path, "r", encoding="utf-8") as f:
        return new_webanno_tsv_to_ner_span_outputs(f.read())


class ValidatedSpanList(SpanList):
    """
    The validated output of the NER task.
    """

    # cuis: Optional[list[str]] = Field(description="List of CUI codes corresponding to the tokens")
    # semantic_types: Optional[list[str]] = Field(
    #     description="List of semantic types corresponding to the tokens"
    # )
    # status: Optional[Literal["valid", "partial", "invalid"]] = Field(
    #     default="valid",
    #     description="Status of the NER output, indicating whether it is valid or invalid.",
    # )
    # comments: Optional[str] = Field(
    #     default=None,
    #     description="Comments on the NER output, explaining why it is valid or invalid.",
    # )


class NerState(BaseModel):
    source_text: Optional[str] = Field(default=None)
    sentences: Optional[list[str]] = Field(default=None)
    grouped_document: GroupedText = Field(default=None)
    tokens: Optional[list[list[str]]] = Field(default=None)
    ner_proposed: Optional[list[NerResult[SpanList]]] = Field(default=None)
    ner_improved: Optional[list[NerResult[ImprovedSpanList]]] = Field(default=None)
    ner_validated: Optional[list[NerResult[ValidatedSpanList]]] = Field(default=None)


@task(retries=2, retry_delay_seconds=5)
def process_one(paragraph: SemanticParagraph) -> List[NerResult[SpanList]]:
    logger = get_run_logger()
    # logger.info(f"Processing: {sentence!r}")

    model = ExtractModule.get_instance()
    input = [
        SentenceWithTokens(sentence=sentence, tokens=tokenize(sentence))
        for sentence in paragraph.sentences
    ]

    res: List[SpansOutput] = model(input=input, previous_run=None).output
    if len(res) != len(input):
        print(
            f"Expected {len(paragraph.sentences        )} results, got {len(res.output)}."
        )
    result = []
    for i, si in enumerate(input):
        r = NerResult[SpanList](
            sentence=si.sentence, tokens=si.tokens, output=res[i].span_list
        )
        result.append(r)
    return result


@flow(name="extract_flow")
def ner_task(state: NerState):
    outputs: List[NerResult[SpanList]] = process_one.map(
        state.grouped_document.paragraphs
    )

    flat_results: list[NerResult[SpanList]] = []

    for future in outputs:
        results_from_one_paragraph = future.result()

        # Add the results from this paragraph to your flat list.
        if results_from_one_paragraph:
            flat_results.extend(results_from_one_paragraph)

    # 3. The 'flat_results' list now contains all the NerResult objects from all paragraphs.
    return {"ner_proposed": flat_results}


@task(retries=2, retry_delay_seconds=5)
def process_one_judge(
    proposed: NerResult[SpanList],
) -> NerResult[ImprovedSpanList]:
    logger = get_run_logger()
    logger.info(f"Processing improvement: {proposed.sentence!r}")

    model = JudgeModule.get_instance()
    improved = model(
        sentence=proposed.sentence,
        tokens=proposed.tokens,
        ner_proposed=proposed.output,
        prompt_used=ExtractSignature.__doc__,
    )
    return NerResult[ImprovedSpanList](
        sentence=proposed.sentence, tokens=proposed.tokens, output=improved.ner_final
    )


@flow(name="judge_flow")
def judge(state: NerState) -> None:
    outputs: list[ImprovedSpanList] = process_one_judge.map(state.ner_proposed)
    return {"ner_improved": outputs}


def validate_spans(tokens: list[str], item: SpanList) -> list[str]:
    errors = []
    n_tokens = len(tokens)

    for i, span in enumerate(item.spans):
        if not (0 <= span.start < span.end <= n_tokens):
            errors.append(
                f"Span[{i}] index error: span.end should have been < {n_tokens}. This time pay attention indices."
            )
            errors.append(f"")
            continue

        gold = " ".join(tokens[span.start : span.end]).strip()
        if span.text.strip() != gold:
            errors.append(
                f"Span[{i}] text mismatch: «{span.text}» should have been: «{gold}» This time pay attention text = tokens[span.start:span.end]"
            )

    return errors


@task(name="validate_ner")
def validate_ner(state: NerState) -> None:

    source = state.ner_improved if state.ner_improved else state.ner_proposed
    logger = get_run_logger()
    validated_list: list[NerResult[ValidatedSpanList]] = []
    model = ExtractModule.get_instance()
    print("Ner cost", model.calculate_cost())
    for item in source:
        previous_run = None
        all_errors = []
        validating_item = item.output

        for attempt in range(1, 5):  # 1, 2, 3
            errors = validate_spans(item.tokens, validating_item)

            if len(errors) == 0:
                validated_list.append(
                    NerResult[ValidatedSpanList](
                        sentence=item.sentence,
                        tokens=item.tokens,
                        output=ValidatedSpanList(spans=validating_item.spans),
                    )
                )
                break

            msg = (
                f"Your previous attempt #{attempt} to annotate this sentence had errors:\n"
                + "\n".join(f"{i}. {e}" for i, e in enumerate(errors, 1))
                # + "\n.Output was:\n"
                # + item.model_dump_json(indent=0)
                + "\nBefore submitting, ensure you have fixed the errors above."
            )

            all_errors.append(f"{msg}\n\n")
            previous_run = "".join(all_errors)

            logger.warning(all_errors)

            res = model(
                input=[SentenceWithTokens(sentence=item.sentence, tokens=item.tokens)],
                previous_run=previous_run,
            )
            validating_item = res.output[0].span_list
        else:
            logger.error(f"Failed after 3 attempts: {item.sentence!r}")
            validated_list.append(
                NerResult[ValidatedSpanList](
                    sentence=item.sentence,
                    tokens=item.tokens,
                    output=ValidatedSpanList(spans=validating_item.spans),
                )
            )
    logger.info(f"Validated:Proposed {len(validated_list)}/{len(source)} sentences")
    print("Total cost", model.calculate_cost())
    return {"ner_validated": validated_list}


@task(name="needs_judgement")
def needs_judgement(state: NerState) -> str:
    judge = True
    if judge:
        return "judge"
    else:
        return "validate_ner"


@task(name="split_sentences")
def split_sentences(state: NerState) -> None:
    sentences = state.source_text.split("\n")
    tokens = []
    for sentence in sentences:

        doc = nlp(sentence)

        stokens = []
        with doc.retokenize() as retokenizer:
            for token in doc:
                stokens.append(token.text)
        tokens.append(stokens)

    return {"sentences": sentences, "tokens": tokens}


# ---------------------------------------------------------------------
#  parser
# ---------------------------------------------------------------------
_RE_TOKEN_LINE = re.compile(
    r"(?P<sid>\d+)-(?P<tid>\d+)\t"
    r"(?P<start>\d+)-(?P<end>\d+)\t"
    r"(?P<token>.+?)\t"
    r"(?P<tag>.+)"
)

_RE_TAG = re.compile(r"^(?P<label>[A-Z0-9_]+)\[(?P<idx>\d+)]$")


def read_webanno_tsv(source: Union[str, Path]) -> list[SpanList]:
    """
    Read a WebAnno TSV 3.x file (or a text string containing it) and
    return a list of NerSpanOutput objects.
    """
    txt = (
        Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    )
    lines = txt.splitlines()

    outputs: list[SpanList] = []
    tokens: list[str] = []
    span_map: dict[str, tuple[str, int, int]] = {}  # key=id  -> (label, start, end)

    for ln in lines:
        if ln.startswith("#Text="):
            # -- starting a new sentence ------------------------------
            if tokens:  # flush previous sentence
                outputs.append(_flush_sentence(tokens, span_map))
                tokens, span_map = [], {}

            continue  # no further processing needed for #Text

        if not ln or ln[0] == "#":
            continue  # skip blank lines and other comments

        m = _RE_TOKEN_LINE.match(ln)
        if not m:  # malformed line – ignore
            continue

        tok_idx = len(tokens)
        token = m.group("token")
        tag = m.group("tag")

        tokens.append(token)

        if tag != "_":
            tag_m = _RE_TAG.match(tag)
            if tag_m:
                key = tag_m.group("idx")  # unique id in brackets
                label = tag_m.group("label")

                if key not in span_map:
                    span_map[key] = [label, tok_idx, tok_idx + 1]  # new span
                else:
                    span_map[key][2] = tok_idx + 1  # extend span

    # flush last sentence if file doesn't end with blank line
    if tokens:
        outputs.append(_flush_sentence(tokens, span_map))

    return outputs


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _flush_sentence(
    tokens: list[str], span_map: dict[str, tuple[str, int, int]]
) -> SpanList:
    spans = [
        Span(start=st, end=en, label=label, text=" ".join(tokens[st:en]))
        for label, st, en in span_map.values()
    ]
    spans.sort(key=lambda s: (s.start, s.end))
    return SpanList(tokens=tokens.copy(), spans=spans)


# ---------------------------------------------------------------------
#  helpers for “smart” spacing
# ---------------------------------------------------------------------
_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", "%", ")", "]", "}"}
_NO_SPACE_AFTER = {"(", "[", "{", "<"}

WS_RE = re.compile(r"\s+")


def _rebuild_sentence(tokens: list[str]):
    """Return sentence string plus per-token local offsets (start, end)."""
    parts, offsets, pos = [], [], 0
    for i, tok in enumerate(tokens):
        if (
            i > 0
            and tok not in _NO_SPACE_BEFORE
            and tokens[i - 1] not in _NO_SPACE_AFTER
        ):
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(tok)
        pos += len(tok)
        offsets.append((start, pos))
    return "".join(parts), offsets


# ---------------------------------------------------------------------
#  main conversion routine
# ---------------------------------------------------------------------
# def ner_span_outputs_to_webanno_tsv(ner_outputs: list[NerResult[NerSpanOutput]]) -> str:
#     """
#     Convert NerSpanOutput list -> WebAnno TSV 3.x string
#     (document-level offsets, unique span IDs).
#     """
#     lines = [
#         "#FORMAT=WebAnno TSV 3.3",
#         "#T_SP=webanno.custom.NamedEntityexactmatch|value",
#         "",
#         "",  # blank after header
#     ]

#     doc_pos = 0  # running char position across document
#     global_id = 1  # **unique** index for every span

#     for sent_id, res in enumerate(ner_outputs, start=1):

#         item = res.output
#         # 1) sentence text + local offsets
#         sent_text, local_offsets = _rebuild_sentence(item.tokens)
#         lines.append(f"#Text={sent_text}")

#         # 2) fill annotation array
#         ann = ["_"] * len(item.tokens)
#         for span in item.spans:
#             if not (0 <= span.start < span.end <= len(item.tokens)):
#                 # log and skip
#                 print(
#                     f"Skipping span {span.label} in sentence {sent_id}: "
#                     f"indices [{span.start}, {span.end}) out of range (len={len(item.tokens)})"
#                 )
#                 continue
#             for tok_i in range(span.start, span.end):
#                 ann[tok_i] = f"{span.label}[{global_id}]"
#             global_id += 1

#         # 3) emit token rows with DOC-level offsets
#         for tok_idx, ((loc_start, loc_end), tok, tag) in enumerate(
#             zip(local_offsets, item.tokens, ann), start=1
#         ):
#             lines.append(
#                 f"{sent_id}-{tok_idx}\t{doc_pos + loc_start}-{doc_pos + loc_end}\t{tok}\t{tag}"
#             )

#         lines.append("")  # blank line between sentences
#         doc_pos += len(sent_text) + 1  # +1 for the newline between sents

#     return "\n".join(lines)


from itertools import count
from typing import Dict, List


def new_ner_span_outputs_to_webanno_tsv(
    ner_outputs: list[NerResult[SpanList]],
) -> str:
    """
    Converts a list of NerResult objects to a valid WebAnno TSV 3.3 string.

    This function correctly adheres to the "one line per token" rule and maintains
    continuous character offsets and span IDs across all sentences.

    Args:
        ner_outputs: A list of NerResult objects. The tokenization within these
                     objects is expected to be correct and final.

    Returns:
        A string formatted according to the WebAnno TSV 3.3 specification.
    """
    tsv_parts = []
    header = [
        "#FORMAT=WebAnno TSV 3.3",
        "#T_SP=webanno.custom.NamedEntityexactmatch|value\n\n",
    ]
    tsv_parts.extend(header)

    global_char_offset = 0
    global_span_id_counter = 1

    for sent_idx, ner_result in enumerate(ner_outputs, 1):
        if sent_idx > 1:
            tsv_parts.append("")

        tsv_parts.append(f"#Text={ner_result.sentence}")

        token_annotations = ["_"] * len(ner_result.tokens)

        # Assign unique, continuous IDs to each span
        sorted_spans = sorted(ner_result.output.spans, key=lambda s: s.start)
        for span in sorted_spans:
            annotation = f"{span.label}[{global_span_id_counter}]"
            for i in range(span.start, span.end):
                if i < len(token_annotations):
                    token_annotations[i] = annotation
            global_span_id_counter += 1

        local_char_offset = 0
        # --- FIX: Reverted to a stable, one-line-per-token iteration ---
        for token_idx, token in enumerate(ner_result.tokens):
            # Find the token in the original sentence string
            start_char_local = ner_result.sentence.find(token, local_char_offset)
            if start_char_local == -1:
                start_char_local = local_char_offset  # Fallback

            start_char_global = global_char_offset + start_char_local
            end_char_global = start_char_global + len(token)

            # The TSV token index is 1-based
            tsv_token_num = token_idx + 1

            tsv_line = (
                f"{sent_idx}-{tsv_token_num}\t"
                f"{start_char_global}-{end_char_global}\t"
                f"{token}\t"
                f"{token_annotations[token_idx]}"
            )
            tsv_parts.append(tsv_line)

            # Update local offset for the next token search
            local_char_offset = start_char_local + len(token)

        # Update global offset for the next sentence
        global_char_offset += len(ner_result.sentence) + 1

    return "\n".join(tsv_parts)


@task()
def save_output(state: NerState) -> None:
    with open("ner_output.json", "w", encoding="utf-8") as f:
        f.write(state.model_dump_json(indent=2))

    output = new_ner_span_outputs_to_webanno_tsv(state.ner_validated)
    with open("ner_annotations.tsv", "w", encoding="utf-8") as f:
        f.write(output)

    # cas = ner_span_outputs_to_cas(state.ner_validated)
    # cas.to_xmi("export.xmi")


def semantic_chunk(state: NerState):
    model = SemanticChunkerModule.get_instance()
    res = model(text=state.source_text)

    group: GroupedText = res.grouped_document

    for p in group.paragraphs:
        print(p.heading, "\n")
        for s in p.sentences:
            print(s, "\n")

    return {"grouped_document": res.grouped_document}


def build_graph() -> CompiledStateGraph:
    gb = StateGraph(NerState)

    gb.add_node("ner", ner_task)
    gb.add_node("semantic_chunk", semantic_chunk)
    gb.add_node("split_sentences", split_sentences)
    gb.add_node("judge", judge)
    gb.add_node("save_output", save_output)
    gb.add_node("validate_ner", validate_ner)
    gb.add_node("needs_judgement", needs_judgement)

    gb.add_edge(START, "split_sentences")
    gb.add_edge("split_sentences", "semantic_chunk")
    gb.add_edge("semantic_chunk", "ner")
    gb.add_conditional_edges(
        "ner", needs_judgement, {k: k for k in ("judge", "validate_ner")}
    )
    gb.add_edge("judge", "validate_ner")
    gb.add_edge("validate_ner", "save_output")
    gb.add_edge("save_output", END)

    return gb.compile()


def dspy_ner_metric_implementation_corrected(
    gold_example: dspy.Example,  # Receives a single dspy.Example for gold
    prediction: dspy.Prediction,  # Receives a single dspy.Prediction from the program
    trace=None,  # Accepts the trace argument from DSPy
    *,  # Makes subsequent arguments strictly keyword-only
    metric_to_optimize: str = "f1",
) -> float:

    gold_ner_output_obj = gold_example.output
    # Access the predicted NerSpanOutput from the 'output' field of the prediction
    # This 'output' field name also comes from the NerSpanSignature
    if not hasattr(prediction, "output"):
        print(
            f"ERROR: prediction is missing the 'output' attribute. Object: {prediction}"
        )
        return 0.0  # Or raise an error
    predicted_ner_output_obj = prediction.output

    # Ensure they are the correct Pydantic type (especially if they were loaded from demos)
    if not isinstance(gold_ner_output_obj, SpanList):
        print(
            f"ERROR: gold_example.output is not a NerSpanOutput. Type: {type(gold_ner_output_obj)}"
        )
        # Potentially try to re-parse if it's a dict, though it should be correct from dspy.Example
        if isinstance(gold_ner_output_obj, dict):
            try:
                gold_ner_output_obj = SpanList(**gold_ner_output_obj)
            except Exception as e_parse_gold:
                print(f"Failed to parse gold_ner_output_obj from dict: {e_parse_gold}")
                return 0.0
        else:
            return 0.0

    if not isinstance(predicted_ner_output_obj, SpanList):
        print(
            f"ERROR: prediction.output is not a NerSpanOutput. Type: {type(predicted_ner_output_obj)}"
        )
        if isinstance(predicted_ner_output_obj, dict):
            try:
                predicted_ner_output_obj = SpanList(**predicted_ner_output_obj)
            except Exception as e_parse_pred:
                print(
                    f"Failed to parse predicted_ner_output_obj from dict: {e_parse_pred}"
                )
                return 0.0
        else:
            return 0.0

    # calculate_ner_metrics expects List[NerSpanOutput]
    # Since the metric is called per example, wrap them in lists.
    stats_dict = calculate_ner_metrics_with_output(
        [gold_ner_output_obj], [predicted_ner_output_obj]
    )

    print(
        f"\n--- Optimization Step Detailed Stats (optimizing for '{metric_to_optimize}') ---"
    )
    print(gold_example.sentence)
    # print(json.dumps(stats_dict, indent=2))

    if "overall" not in stats_dict or metric_to_optimize not in stats_dict["overall"]:
        print(
            f"Warning: Metric '{metric_to_optimize}' not found in overall stats. Stats: {stats_dict}"
        )
        print(
            f"Available keys in 'overall': {stats_dict.get('overall', {}).keys()}. Returning 0.0"
        )
        return 0.0

    score_to_return = float(stats_dict["overall"][metric_to_optimize])

    # has_finding = (
    #     "per_label" in stats_dict and "FINDING_T033" in stats_dict["per_label"]
    # )
    # has_lab = (
    #     "per_label" in stats_dict
    #     and "LABORATORY_PROCEDURE_T059" in stats_dict["per_label"]
    # )

    # if has_finding and has_lab:
    #     score_to_return = (
    #         stats_dict["per_label"]["FINDING_T033"][metric_to_optimize]
    #         + stats_dict["per_label"]["LABORATORY_PROCEDURE_T059"][metric_to_optimize]
    #     ) / 2.0
    # elif has_finding:
    #     score_to_return = stats_dict["per_label"]["FINDING_T033"][metric_to_optimize]
    # elif has_lab:
    #     score_to_return = stats_dict["per_label"]["LABORATORY_PROCEDURE_T059"][
    #         metric_to_optimize
    #     ]

    print(
        f"Current score for optimization ('{metric_to_optimize}'): {score_to_return:.4f}\n"
    )

    return score_to_return


import json

import dspy


def save_optimized_program_demos(compiled_program: ExtractModule, save_path: str):
    """
    Saves the demos from the optimized NER predictor to a JSON file.
    """

    compiled_program.save(path=save_path, save_program=False)

    if hasattr(compiled_program, "ner_predictor") and hasattr(
        compiled_program.ner_predictor, "demos"
    ):
        demos = compiled_program.ner_predictor.demos

        # dspy.Example has a toDict() method for serialization
        demos_to_save = [demo.toDict() for demo in demos]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(demos_to_save, f, indent=2)
        print(f"Successfully saved {len(demos_to_save)} demos to {save_path}")
    else:
        print("No demos found in the compiled program's ner_predictor to save.")


# Example usage after compilation:
# save_optimized_program_demos(compiled_ner_program, "optimized_ner_demos.json")


def eval(examples, load=True):

    evaluate_correctness = dspy.Evaluate(
        devset=examples,
        metric=dspy_ner_metric_implementation_corrected,
        num_threads=24,
        display_progress=True,
        display_table=True,
    )

    model = ExtractModule.get_instance()
    if load:
        model.load(path="optimized_ner_demos.json")
    res = evaluate_correctness(model)

    print(f"SCORE IS: {res}")


def generate_examples(results: list[NerResult[SpanList]]):
    examples = []

    for result in results:
        example = dspy.Example(
            sentence=result.sentence,
            tokens=result.tokens,
            previous_run=None,
            output=result.output,
        ).with_inputs("sentence", "tokens", "previous_run")
        examples.append(example)
    return examples


def evaluate(gold: list[NerResult[SpanList]]) -> dict[str, float]:
    examples = generate_examples(gold)

    train_examples, test_examples = train_test_split(
        examples, train_size=0.2, random_state=123
    )

    eval(examples, load=False)

    teleprompter = MIPROv2(
        metric=dspy_ner_metric_implementation_corrected,
        auto="heavy",  # Can choose between light, medium, and heavy optimization runs
    )

    # Optimize program
    print(f"Optimizing program with MIPROv2...")

    uncompiled_ner_program = ExtractModule()

    # dspy.configure(lm=get_ner_llm())

    optimized_program = teleprompter.compile(
        student=uncompiled_ner_program,
        trainset=train_examples,
        valset=test_examples,
        max_bootstrapped_demos=4,
        requires_permission_to_run=False,
        minibatch=False,
    )

    print("MIPROv2 optimization complete.")

    save_optimized_program_demos(optimized_program, "optimized_ner_demos.json")

    eval(examples, load=True)


def evaluate2(gold: list[NerResult[SpanList]]) -> dict[str, float]:
    train_examples = generate_examples(gold)

    optimizer = BootstrapFewShot(
        metric=dspy_ner_metric_implementation_corrected,  # Your metric function
        max_bootstrapped_demos=4,  # Number of few-shot examples to generate for the student
        max_labeled_demos=16,  # Number of few-shot examples to provide to the teacher (if different)
        # teacher_settings = {}    # Optional: if your teacher LM needs different settings
    )

    uncompiled_ner_program = ExtractModule()

    teacher = ExtractModule(
        dspy.LM(llms["teacher"], temperature=0.1, max_tokens=32000, cache=False)
    )
    eval(train_examples, False)

    print(f"Starting DSPy optimization with {len(train_examples)} examples...")
    compiled_ner_program = optimizer.compile(
        student=uncompiled_ner_program,
        trainset=train_examples,
        teacher=teacher,
    )
    print("DSPy optimization complete.")

    save_optimized_program_demos(compiled_ner_program, "optimized_ner_demos.json")

    eval(train_examples)


def evaluate3(gold: list[NerResult[SpanList]]) -> dict[str, float]:
    examples = generate_examples(gold)

    train_examples, test_examples = train_test_split(
        examples, train_size=0.2, random_state=123
    )
    train_examples = examples

    # lm = get_ner_llm()
    # dspy.configure(lm=lm)

    # evaluate_correctness = dspy.Evaluate(
    #     devset=examples,
    #     metric=dspy_ner_metric_implementation_corrected,
    #     num_threads=24,
    #     display_progress=True,
    #     display_table=True,
    # )

    model = ExtractModule()
    # model.load(path="optimized_ner_demos.json")
    # res = evaluate_correctness(model)

    # print(res)

    optimizer = BootstrapFinetune(
        metric=dspy_ner_metric_implementation_corrected,  # Your metric function
        num_threads=16,
    )

    p = model.predictors()
    p[0].lm = model.lm

    print(f"Starting DSPy optimization with {len(train_examples)} examples...")
    compiled_ner_program = optimizer.compile(
        student=model,
        trainset=train_examples,
    )
    print("DSPy optimization complete.")

    save_optimized_program_demos(compiled_ner_program, "optimized_ner_demos.json")


graph = build_graph()
# mlflow.dspy.autolog()
# mlflow.set_experiment("CDSS Experiment")


def diff(gold: list[NerResult[SpanList]], pred: list[NerResult[SpanList]]):

    for i, (g, p) in enumerate(zip(gold, pred)):
        for gs, ps in zip(g.output.spans, p.output.spans):
            if gs != ps:
                print(f"{i+1}\nPred: {ps}\nGold: {gs}\n")


@flow(name="main_ner_flow")
def invoke_graph(initial_state: NerState):
    result = graph.invoke(initial_state)
    # get_run_logger().info(f"NER Result: {result}")
    return result


if __name__ == "__main__":
    with open("./data/sample.txt", "r") as f:
        data = f.read()
    initial_state: NerState = NerState(source_text=data)
    result = invoke_graph(initial_state)

    print("-" * 10)
    print(f"Cost of extraction flow: {ExtractModule.get_instance().calculate_cost()} ")
    print(f"Cost of judge flow: {JudgeModule.get_instance().calculate_cost()} ")
    print("-" * 10)

    gold = load_from_tsv("./gold/ner_annotations_corrected.tsv")
    pred = load_from_tsv("./output/ner_annotations.tsv")

    # # print(result)
    # examples = generate_examples(gold)
    # eval(examples, load=False)

    stats = calculate_ner_metrics(gold, pred)
    print(stats)

    diff(gold, pred)

    print("-" * 10)
    print(
        f"Final Cost of extraction flow: {ExtractModule.get_instance().calculate_cost()} "
    )
    print(f"Final Cost of judge flow: {JudgeModule.get_instance().calculate_cost()} ")
    print("-" * 10)
