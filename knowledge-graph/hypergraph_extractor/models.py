# file: hypergraph_extractor/models.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata  # noqa: F401

import datetime as dt
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    BeforeValidator,
    PlainSerializer,
    model_validator,          # v2-style validators
)
from typing_extensions import Annotated

# ──────────────────────────────────────────────────────────────────────────────
# Type-aliases & custom types
# ──────────────────────────────────────────────────────────────────────────────

class IsoDate(dt.date):
    """
    A `datetime.date` that can be parsed from an ISO-formatted string
    and serialised back to the same string form.
    """

    # ---- parsing ------------------------------------------------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):  # noqa: D401  (simple validation helper)
        if isinstance(v, str):
            try:
                return dt.date.fromisoformat(v)
            except ValueError as exc:
                raise ValueError(
                    "Timestamp must be in ISO format 'YYYY-MM-DD'"
                ) from exc
        if isinstance(v, dt.date):
            return v
        raise TypeError("Timestamp must be str or datetime.date")

    # ---- JSON-schema hint (optional) ----------------------------------------
    @classmethod
    def __modify_schema__(cls, field_schema):  # noqa: D401
        field_schema.update(type="string", format="date")


Timestamp = Annotated[
    dt.date,
    BeforeValidator(
        lambda v: dt.date.fromisoformat(v) if isinstance(v, str) else v
    ),
    PlainSerializer(lambda x: x.isoformat(), return_type=str),
]

# ──────────────────────────────────────────────────────────────────────────────
# Core models
# ──────────────────────────────────────────────────────────────────────────────


class Sentence(BaseModel):
    id: str
    text: str


class Entity(BaseModel):
    id: str
    name: str
    type: str


class FactRole(BaseModel):
    role: str
    entity: Optional[str] = None
    literal: Optional[Union[int, float, str]] = Field(
        default=None, alias="value"
    )
    datatype: Optional[str] = Field(default=None, alias="data_type")

    model_config = ConfigDict(populate_by_name=True)

    # ----- cross-field validation -------------------------------------------
    @model_validator(mode="after")
    def _check_role_content(self):
        has_entity = self.entity is not None
        has_literal = self.literal not in (None, "", " ")

        if not has_entity and not has_literal:
            raise ValueError(
                "FactRole must provide at least `entity` or `literal`."
            )

        if self.datatype is not None and not has_literal:
            raise ValueError(
                "`datatype` can only be provided when `literal` is present."
            )
        return self


class Fact(BaseModel):
    id: Optional[str] = None
    predicate: str
    timestamp: Timestamp
    truth_value: Optional[Union[float, List[float]]] = None
    sentence_id: str
    tuple: Optional[List[FactRole]] = None
    arguments: Optional[List[str]] = None

    # ----- ensure either tuple *or* arguments, not both ----------------------
    @model_validator(mode="after")
    def _check_tuple_or_arguments(self):
        has_tuple = bool(self.tuple)
        has_args = bool(self.arguments)

        if has_tuple and has_args:
            raise ValueError(
                "A Fact cannot have both `tuple` and `arguments`."
            )
        if not has_tuple and not has_args:
            raise ValueError(
                "A Fact must have either `tuple` or `arguments`."
            )
        return self


class Hypergraph(BaseModel):
    graph_type: Literal["AtomSpace HyperGraph", "AtomSpace Hypergraph"]
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class NTKG(BaseModel):
    graph_type: Literal[
        "N-tuple Hyper-Relational Temporal Knowledge Graph"
    ]
    entities: List[Entity]
    facts: List[Fact]
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────


def default_timestamp(year: int | None) -> dt.date:
    """Return *1 Jan <year>* as a `datetime.date` (or 1900 if `None`)."""
    return dt.date(year or 1900, 1, 1)


def year_from_text(txt: str) -> int | None:
    """Extract the first 4-digit year (1900–2099) from text, if present."""
    if m := re.search(r"\b(19|20)\d{2}\b", txt):
        return int(m.group())
    return None
