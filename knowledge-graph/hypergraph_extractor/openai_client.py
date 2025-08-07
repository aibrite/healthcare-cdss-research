# file: hypergraph_extractor/openai_client.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

import backoff
import httpx
import tiktoken
import pydantic
from pydantic import BaseModel, ConfigDict
from packaging.version import parse as parse_version
_PD_V2 = parse_version(pydantic.__version__).major >= 2
if _PD_V2:                    # real v2
    from pydantic import ConfigDict
logger = logging.getLogger(__name__)

_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("Environment variable OPENAI_API_KEY is not set")

# ---- helper ---------------------------------------------------------------
# Rough but fast – good enough for sizing the request
_token_cache: dict[str, tiktoken.Encoding] = {}

if not hasattr(BaseModel, "model_dump"):        # v1 → add polyfill
    # ------------------------- helpers -----------------------------------
    def _choose_mode(kwargs):
        """
        Return 'python' or 'json'.

        Priority   1) explicit mode     (kwargs['mode'])
                    2) file_type alias   (kwargs['file_type'])
                    3) default 'python'
        """
        mode = kwargs.pop("mode", None)
        if mode is not None:
            return mode
        file_type = kwargs.pop("file_type", None)
        if file_type is not None:
            return "json" if str(file_type).lower() == "json" else "python"
        return "python"

    # ------------------------- BaseModel.model_dump ----------------------
    def _model_dump(self, **kwargs):
        """
        v2-style .model_dump for Pydantic v1.

        Accepts *exactly* the same kwargs as v2, plus the alias
        `file_type='json' | 'dict'` for backward-compat convenience.
        """
        mode = _choose_mode(kwargs)

        if mode == "json":
            return self.json(**kwargs)            # v1's `.json()`
        elif mode == "python":
            return self.dict(**kwargs)            # v1's `.dict()`
        else:
            raise ValueError("mode must be 'python' or 'json'")

    # ------------------------- BaseModel.model_dump_json -----------------
    def _model_dump_json(self, **kwargs):
        """v2 alias for .json() on Pydantic v1."""
        return self.json(**kwargs)

    # ------------------------- monkey-patch ------------------------------
    BaseModel.model_dump = _model_dump
    BaseModel.model_dump_json = _model_dump_json


def _count_tokens(model: str, messages: List[Dict[str, str]]) -> int:
    enc = _token_cache.setdefault(model, tiktoken.encoding_for_model(model))
    try:
        enc = _token_cache.setdefault(model, tiktoken.encoding_for_model(model))
    except KeyError:
        enc = _token_cache.setdefault(model, tiktoken.get_encoding("cl100k_base"))
    num = 0
    for m in messages:
        # <|start|>{role/name}\n{content}<|end|>\n  → ~4 overhead per message
        num += 4 + len(enc.encode(m["content"]))
    return num + 2  # priming + assistant reply prefix


def _available_completion_tokens(model: str, prompt_tokens: int) -> int:
    # hard-coded for the models we use
    CONTEXT_WINDOWS = {
        "gpt-4.1": 1_000_000,
        "o3": 100_000,
        "gpt-4.1-mini": 128_000,
        "o3-mini": 100_000,
    }
    limit = CONTEXT_WINDOWS.get(model, 16300)
    return max(1, limit - prompt_tokens)
    # never return 0 – the API would reject it


# ---- Pydantic request model ----------------------------------------------
class _ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float | None = None
    max_completion_tokens: int | None = None
    stream: bool = False

    # ---------------- configuration --------------------------------------
    if _PD_V2:                      # Pydantic 2.x  → use ConfigDict sentinel
        model_config = ConfigDict(
            extra="forbid",
            populate_by_name=True,
            strict=True,
        )
    else:                           # Pydantic 1.x  → classic inner Config
        class Config:
            extra = "forbid"
            allow_population_by_field_name = True
    #model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


# ---- HTTP orchestration ---------------------------------------------------
@backoff.on_exception(
    backoff.expo,
    (httpx.HTTPError, httpx.ReadTimeout),
    max_tries=10,
    jitter=backoff.full_jitter,
)
async def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            _OPENAI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if r.status_code >= 400:
            try:
                err = r.json()
            except ValueError:
                err = r.text
            logger.error("OpenAI returned %s → %s", r.status_code, err)
            r.raise_for_status()

        return r.json()


async def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    # auto-size if caller didn’t decide
    if max_completion_tokens is None:
        prompt_len = _count_tokens(model, messages)
        max_completion_tokens = _available_completion_tokens(model, prompt_len)

    req = _ChatRequest(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    ).model_dump(exclude_none=True)
    if "max_completion_tokens" in req:
        req["max_tokens"] = req.pop("max_completion_tokens")
    #req["response_format"] = {"type": "json_object"}
    resp = await _post(req)
    return resp["choices"][0]["message"]["content"]

