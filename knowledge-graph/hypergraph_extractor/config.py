# file: hypergraph_extractor/config.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata  # noqa: F401

from pathlib import Path
from typing import Final

import tomli
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,   # v2 field-level validator
)


# ──────────────────────────────────────────────────────────────────────────────
# Agent-level settings
# ──────────────────────────────────────────────────────────────────────────────
class _AgentSettings(BaseModel):
    model_name: str = "gpt-4.1"
    max_completion_tokens: int | None = 16_300
    temperature: float | None = 0.1
    rate_limit_rps: float = 3.0
    max_retries: int = 3

    # Apply special rules *after* the object is built
    @model_validator(mode="after")
    def _o3_rules(self):
        if self.model_name == "o3":
            # safest: return a patched copy (keeps model immutable if so configured)
            return self.model_copy(
                update={
                    "temperature": 1.0,
                    "max_completion_tokens": None,
                }
            )
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Global settings loaded from `pyproject.toml`
# ──────────────────────────────────────────────────────────────────────────────
class Settings(BaseModel):
    """Immutable runtime configuration (loaded from `[tool.hypergraph-extractor]`)."""

    system_prompt: str
    example_prompt: str
    example_json_hg: str
    example_json_hrkg: str
    prompt_hg_template: str
    prompt_hrkg_template: str
    agent: _AgentSettings = Field(default_factory=_AgentSettings)

    # expand any field that points to a file → replace with file contents
    @field_validator(
        "system_prompt",
        "example_prompt",
        "example_json_hg",
        "example_json_hrkg",
        "prompt_hg_template",
        "prompt_hrkg_template",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, v):
        if isinstance(v, str):
            p = Path(v)
            if p.is_file():
                return p.read_text(encoding="utf-8")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Loader helper
# ──────────────────────────────────────────────────────────────────────────────
def load_settings(pyproject: Path | None = None) -> Settings:
    """
    Read `[tool.hypergraph-extractor]` section from *pyproject.toml*
    and return a fully validated `Settings` object.
    """
    pyproject = pyproject or Path(__file__).parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        raw = tomli.load(f)

    section: Final = raw["tool"]["hypergraph-extractor"]
    return Settings(**section)
