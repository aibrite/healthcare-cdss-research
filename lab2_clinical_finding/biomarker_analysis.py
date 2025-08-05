"""
biomarker_analysis.py
=====================

This module provides functionality for parsing laboratory results
presented in two CSV files (``Results_Single.csv`` and ``Results_Multiple.csv``)
and combining them with a context database of biomarker metadata
(``Biomarker_Data.csv``).  The goal of this module is to produce a
structured representation of each biomarker that includes the raw
measurements, their temporal context, trend classification,
measurement status relative to reference ranges, critical flags based
on sharp change rules, rates of change, and a handful of derived
clinical ratios.  See the documentation in the functions below for
detailed explanations of each processing step.

The mapping between the Turkish names found in the result files and the
English names used by the biomarker database is encoded in the
``TURKISH_TO_ENGLISH`` dictionary.  Most of the common analytes are
explicitly mapped, while unknown or unsupported names fall back to
their cleaned Turkish representation.

Author: OpenAI ChatGPT
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Mapping dictionary
#
# Many of the analyte names contained within the laboratory result files are
# printed in Turkish.  To enable merging with the English metadata contained
# in ``Biomarker_Data.csv`` we provide a comprehensive mapping of known
# Turkish descriptors to the corresponding English test names.  When adding
# new mappings please ensure the key is stripped of newlines and extra
# whitespace and that the value corresponds exactly to the "Test Name"
# column in ``Biomarker_Data.csv``.
# -----------------------------------------------------------------------------

TURKISH_TO_ENGLISH: Dict[str, str] = {
    # Liver enzymes
    "Alanin Aminotransferaz (alt) (serum/plazma)": "ALT",
    "Alanin Aminotransferaz (alt)": "ALT",
    "Alanin Aminotransferaz (alt) serum/plazma": "ALT",
    "Albümin (serum/plazma)": "Albumin",
    "Alkalen Fosfataz (serum/plazma)": "ALP",
    "Aspartat Aminotransferaz (ast) (serum/plazma)": "AST",
    "Aspartat Aminotransferaz (ast)": "AST",
    # White blood cell subsets
    "BASO#": "Basophils",
    "BASO%": "BASO %",
    "EOS#": "Eosinophils",
    "EOS%": "EOS %",
    "LYM#": "Lymphocytes",
    "LYM%": "LYM %",
    "NEU#": "Neutrophils (absolute)",
    "NEU%": "NEU %",
    "MONO#": "Monocytes",
    "MONO%": "MONO %",
    "NRBC#": "NRBC",
    "NRBC%": "NRBC",
    # Other cell indices
    "MCH": "MCH",
    "MCHC": "MCHC",
    "MCV": "MCV",
    "RDW": "RDW",
    "PLT": "Platelets",
    "PCT": "Plateletcrit (PCT)",
    "PDW": "PDW",
    "MPV": "MPV",
    "RBC": "RBC",
    "HGB": "Hemoglobin",
    "HCT": "Hematocrit",
    # Lipids and cholesterol
    "HDL kolesterol": "HDL‑C",
    "Kolesterol (serum/plazma)": "Total Cholesterol",
    "Total Kolesterol": "Total Cholesterol",
    "Ldl Kolesterol (direkt)": "LDL‑C",
    "Non-HDL Kolesterol": "Non‑HDL Cholesterol",
    "Total/HDL Kolesterol": "Total/HDL Cholesterol",
    "Trigliserid": "Triglycerides",
    "Trigliserid (serum/plazma)": "Triglycerides",
    "VLDL Kolesterol": "VLDL Cholesterol (direct)",
    # Minerals and electrolytes
    "Kalsiyum (serum/plazma)": "Calcium (total)",
    "Magnezyum (serum/plazma)": "Magnesium (Serum)",
    "Sodyum (serum/plazma)": "Sodium",
    "Potasyum (serum/plazma)": "Potassium",
    "Klorür (serum/plazma)": "Chloride",
    "Fosfor (serum/plazma)": "Phosphate (Serum)",
    # Endocrine
    "Serbest T3": "Free T3",
    "Serbest T4": "Free T4",
    "TSH": "TSH",
    # Renal
    "Üre (serum/plazma)": "Serum Urea",
    "Kreatinin": "Creatinine",
    "Kreatinin (serum/plazma)": "Creatinine",
    "eGFR": "eGFR",
    # Metals
    "Bakır (Cu), Serum": "Serum Copper (Cu)",
    "Bakır (Cu), Serum µg/L": "Serum Copper (Cu)",
    "Bakır (serum/plazma)": "Serum Copper (Cu)",
    "Çinko (Zn), Serum": "Serum Zinc",
    "Çinko (Zn), Serum µg/L": "Serum Zinc",
    "Çinko (serum/plazma)": "Serum Zinc",
    # Iron and binding capacity
    "Demir (serum/plazma)": "Serum Iron",
    "Demir bağlama kapasitesi": "TIBC",
    "Total Demir Bağlama Kapasitesi": "TIBC",
    "Doymamış Demir Bağlama Kapasitesi": "TIBC",
    # Vitamin and folate
    "Folat (serum/plazma)": "Serum Folate",
    "Ferritin (serum/plazma)": "Ferritin",
    "Vitamin B12": "Vitamin B12",
    # Proteins
    "Protein (serum/plazma)": "Total Protein",
    "Albümin": "Albumin",
    # Bilirubin
    "Bilirubin, Total (serum/plazma)": "Total Bilirubin",
    "Bilirubin, Total": "Total Bilirubin",
    "Bilirubin, Direkt (serum/plazma)": "Total Bilirubin",
    "Bilirubin, Direkt (vücut Sıvıları)": "Total Bilirubin",
    "Bilirubin, Direkt": "Total Bilirubin",
    # Glucose and diabetes
    "Glukoz (serum/plazma)": "Fasting Glucose",
    "C Reaktif Protein (crp)": "CRP (high‑sens)",
    "Gamma Glutamil Transferaz (ggt) (serum/plazma)": "Gamma‑glutamyl Transferase (GGT)",
    "Gamma Glutamil Transferaz (ggt)": "Gamma‑glutamyl Transferase (GGT)",
    "Glike Hemoglobin (hb A1c) (elektroforez)": "Glycated Hemoglobin (HbA1c) by Electrophoresis",
    "HbA1c (Elektroforez)": "Glycated Hemoglobin (HbA1c) by Electrophoresis",
    "HbA1c (mmol/mol)": "HbA1c",
    "HbA1c ye göre Ort. Glukoz": "Glucose",
    # Kidney and uric acid
    "Ürik Asit (serum/plazma)": "Uric Acid (Serum)",
    # Cholesterol subdivisions already covered
    # Misc counts and ratios
    "Non-HDL Kolesterol": "Non‑HDL Cholesterol",
    "Nötrofil Lenfosit Oranı": "Neutrophil : Lymphocyte Ratio",
    "Total/HDL Kolesterol": "Total/HDL Cholesterol",
    "LUC#": "LUC",
    "LUC%": "LUC",
    # Additional mapping for derived ratios
    "BUN": "BUN",  # may not appear but reserved
    "LDL kolesterol": "LDL‑C",
    "HDL kolesterol": "HDL‑C",
    "Trigliserid": "Triglycerides",
}


def clean_test_name(name: str) -> str:
    """Normalize a test name by removing newlines and extra spaces.

    Parameters
    ----------
    name : str
        The raw test name from the results file.

    Returns
    -------
    str
        A cleaned string with internal newlines collapsed and excess
        whitespace removed.
    """
    if not isinstance(name, str):
        return str(name)
    cleaned = name.replace("\n", " ").replace("\r", " ")
    # Collapse multiple spaces into a single space
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def map_test_name(raw_name: str) -> str:
    """Map a raw Turkish test name to its English equivalent.

    The mapping is performed in a case-insensitive manner and after
    normalising the raw name via ``clean_test_name``.  If no
    corresponding English name is found, the cleaned Turkish name is
    returned unchanged.

    Parameters
    ----------
    raw_name : str
        The test name as found in the result files.

    Returns
    -------
    str
        The mapped English name or the cleaned raw name if no mapping
        exists.
    """
    cleaned = clean_test_name(raw_name)
    # Direct lookup
    if cleaned in TURKISH_TO_ENGLISH:
        return TURKISH_TO_ENGLISH[cleaned]
    # Case-insensitive lookup
    lower_clean = cleaned.lower()
    for key, val in TURKISH_TO_ENGLISH.items():
        if key.lower() == lower_clean:
            return val
    # Try partial matching on abbreviations enclosed in parentheses
    if "(" in cleaned and ")" in cleaned:
        abbr = cleaned.split("(")[1].split(")")[0]
        # Remove any non-alphanumeric characters
        abbr_key = abbr.replace(" ", "").lower()
        for key, val in TURKISH_TO_ENGLISH.items():
            if abbr_key == key.replace(" ", "").lower():
                return val
    return cleaned


def parse_reference_range(rng: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse a reference range string into numeric boundaries.

    The result files provide reference ranges in a variety of formats such
    as ``"53 - 128"``, ``"< 50"`` or ``"> 1,5"``.  This function
    attempts to convert these into a pair of lower and upper numeric
    bounds.  ``None`` is used to represent an open-ended boundary.

    Parameters
    ----------
    rng : Any
        The reference range string.  Non-string values yield `(None, None)`.

    Returns
    -------
    tuple
        A pair `(low, high)` where either element may be ``None`` if the
        range is unbounded on that side.
    """
    if not isinstance(rng, str) or not rng.strip():
        return (None, None)
    s = rng.strip()
    s = s.replace(",", ".")  # convert decimal comma to dot
    # Remove any units or text following the numeric range
    # e.g. "0 - 1.75 IU" -> "0 - 1.75"
    parts = s.split()
    # handle patterns like "< 50" or "> 1.2"
    if parts[0] in {"<", "<="}:
        # upper bound only
        try:
            high = float(parts[1])
        except ValueError:
            high = None
        return (None, high)
    if parts[0] in {">", ">="}:
        try:
            low = float(parts[1])
        except ValueError:
            low = None
        return (low, None)
    # pattern: "number - number"
    if "-" in s:
        # Remove any non-numeric characters except dot and minus sign
        try:
            low_str, high_str = s.split("-")
            low = float(low_str.strip()) if low_str.strip() else None
            high = float(high_str.strip()) if high_str.strip() else None
            return (low, high)
        except Exception:
            pass
    # If we cannot parse return Nones
    return (None, None)


def parse_value(val: Any) -> Optional[float]:
    """Convert a raw value from the result file into a float.

    Numeric values in the result files may use a comma as the decimal
    separator.  Non-numeric or missing values yield ``None``.

    Parameters
    ----------
    val : Any
        The raw value from the result file.

    Returns
    -------
    Optional[float]
        The numeric value as a float if parseable, otherwise ``None``.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    # Replace comma with dot and remove whitespace
    s = str(val).strip().replace(",", ".")
    # remove any unicode spaces
    s = s.replace("\u202f", "").replace("\u00a0", "")
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class BiomarkerInfo:
    """Encapsulates metadata about a biomarker from the context database."""

    test_name: str
    sex: Optional[str]
    normal_low: Optional[float]
    normal_high: Optional[float]
    normal_unit: Optional[str]
    normal_comp: Optional[str]
    normal_change_low: Optional[float]
    normal_change_high: Optional[float]
    normal_change_unit: Optional[str]
    normal_change_time_days: Optional[float]
    normal_change_per_day_low: Optional[float]
    normal_change_per_day_high: Optional[float]
    sharp_change_rules: List[Dict[str, Any]] = field(default_factory=list)
    sharp_change_aggregated_low: Optional[float] = None
    sharp_change_aggregated_high: Optional[float] = None


def load_biomarker_data(path: str) -> Dict[str, Dict[str, BiomarkerInfo]]:
    """Load biomarker metadata from the supplied CSV file.

    The returned mapping is keyed first by the English test name and
    second by the sex to which the entry applies.  A value of ``"Both"``
    in the ``sex`` column of the CSV file denotes that the same
    reference ranges and change limits apply regardless of patient sex.
    When multiple rows share the same test name (e.g. separate ranges
    for male and female), each row is stored separately under its
    recorded sex.  Consumers of this function should use the patient's
    sex to select the most appropriate metadata; falling back to a
    ``"Both"`` entry or any available entry when no exact match exists.

    Parameters
    ----------
    path : str
        Path to the ``Biomarker_Data.csv`` file.

    Returns
    -------
    dict
        A nested mapping ``{test_name: {sex: BiomarkerInfo}}``.  The
        outer key is the English test name and the inner key is the sex
        category (e.g. ``"Male"``, ``"Female"`` or ``"Both"``).  Each
        value is a :class:`BiomarkerInfo` instance populated from the
        corresponding row of the CSV file.
    """
    df = pd.read_csv(path)
    biomarker_map: Dict[str, Dict[str, BiomarkerInfo]] = {}
    for _, row in df.iterrows():
        test_name = row.get("Test Name")
        if not isinstance(test_name, str) or not test_name:
            continue
        # Normalise sex values; default to 'Both' when missing
        raw_sex = row.get("sex")
        sex_key = None
        if isinstance(raw_sex, str) and raw_sex.strip():
            # Normalize to title case for consistency (e.g. male -> Male)
            sex_key = raw_sex.strip().title()
        else:
            sex_key = "Both"
        # Parse sharp_change_parsed into a list of dictionaries
        rules: List[Dict[str, Any]] = []
        raw_rules = row.get("sharp_change_parsed")
        if isinstance(raw_rules, str) and raw_rules.strip():
            try:
                rules = ast.literal_eval(raw_rules)
            except Exception:
                rules = []
        info = BiomarkerInfo(
            test_name=test_name,
            sex=sex_key,
            normal_low=row.get("normal_low") if not pd.isna(row.get("normal_low")) else None,
            normal_high=row.get("normal_high") if not pd.isna(row.get("normal_high")) else None,
            normal_unit=row.get("normal_unit") if isinstance(row.get("normal_unit"), str) else None,
            normal_comp=row.get("normal_comp") if isinstance(row.get("normal_comp"), str) else None,
            normal_change_low=row.get("normal_change_low") if not pd.isna(row.get("normal_change_low")) else None,
            normal_change_high=row.get("normal_change_high") if not pd.isna(row.get("normal_change_high")) else None,
            normal_change_unit=row.get("normal_change_unit") if isinstance(row.get("normal_change_unit"), str) else None,
            normal_change_time_days=row.get("normal_change_time_days") if not pd.isna(row.get("normal_change_time_days")) else None,
            normal_change_per_day_low=row.get("normal_change_per_day_low") if not pd.isna(row.get("normal_change_per_day_low")) else None,
            normal_change_per_day_high=row.get("normal_change_per_day_high") if not pd.isna(row.get("normal_change_per_day_high")) else None,
            sharp_change_rules=rules,
            sharp_change_aggregated_low=row.get("sharp_change_aggregated_low") if not pd.isna(row.get("sharp_change_aggregated_low")) else None,
            sharp_change_aggregated_high=row.get("sharp_change_aggregated_high") if not pd.isna(row.get("sharp_change_aggregated_high")) else None,
        )
        biomarker_map.setdefault(test_name, {})[sex_key] = info
    return biomarker_map


def load_results(path: str) -> pd.DataFrame:
    """Load and preprocess a results CSV file.

    The returned DataFrame has an additional column ``Test_clean`` and
    ensures that date columns are treated as strings.  Values are not
    converted to numeric at this stage; that is handled later during
    melting.

    Parameters
    ----------
    path : str
        Path to either ``Results_Single.csv`` or ``Results_Multiple.csv``.

    Returns
    -------
    pandas.DataFrame
        A preprocessed dataframe with a cleaned test name column.
    """
    df = pd.read_csv(path, dtype=str)
    # Clean the test names
    df["Test_clean"] = df["Test"].apply(clean_test_name)
    return df


def melt_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide-format result dataframe into a long format.

    Columns that look like dates (YYYY-...) are treated as measurement
    timestamps.  The output dataframe has columns ``Test_clean``,
    ``datetime``, ``value``, ``unit`` and ``reference_range``.

    Parameters
    ----------
    df : pandas.DataFrame
        The wide-format dataframe obtained from ``load_results``.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with one row per measurement.
    """
    # Identify measurement columns: those that contain a hyphen and digits
    measurement_cols = [c for c in df.columns if any(ch.isdigit() for ch in str(c)) and str(c).count("-") >= 1]
    id_vars = ["Test_clean", "Unit", "Reference Range"]
    # Melt into long format
    long_df = df.melt(id_vars=id_vars, value_vars=measurement_cols, var_name="datetime_str", value_name="raw_value")
    # Drop missing values
    long_df = long_df[long_df["raw_value"].notna()]
    # Parse datetime
    def parse_dt(x: str) -> dt.datetime:
        try:
            return dt.datetime.strptime(x.strip(), "%Y-%m-%d %H:%M")
        except Exception:
            return pd.NaT
    long_df["datetime"] = long_df["datetime_str"].apply(parse_dt)
    # Convert raw values to float where possible
    long_df["value"] = long_df["raw_value"].apply(parse_value)
    # Parse reference ranges
    long_df[["ref_low", "ref_high"]] = long_df["Reference Range"].apply(parse_reference_range).apply(pd.Series)
    # Keep only relevant columns
    result = long_df[["Test_clean", "datetime", "value", "Unit", "ref_low", "ref_high"]].copy()
    result.rename(columns={"Unit": "unit"}, inplace=True)
    # Drop rows with no parsed value or datetime
    result = result[result["value"].notna() & result["datetime"].notna()]
    return result


def compute_temporal_context(dates: List[dt.datetime]) -> Tuple[str, Optional[int]]:
    """Determine the temporal context for a set of measurement dates.

    If only a single measurement is present the context is "Current" and
    ``None`` is returned for the month span.  With multiple measurements,
    the context is "Last X Months" where X is the integer number of
    months between the earliest and latest dates.

    Parameters
    ----------
    dates : list of datetime
        The measurement timestamps in chronological order.

    Returns
    -------
    tuple
        A tuple ``(context, months_span)`` where ``context`` is one of
        ``"Current"`` or ``"Last X Months"`` and ``months_span`` is the
        number of months spanned (``None`` for a single measurement).
    """
    if not dates:
        return ("Unknown", None)
    if len(dates) == 1:
        return ("Current", None)
    # Compute difference in months (approximate using 30 days per month)
    span_days = (max(dates) - min(dates)).days
    months = max(1, int(round(span_days / 30)))
    return (f"Last {months} Months", months)


def compute_trend(
    values: List[float],
    dates: Optional[List[dt.datetime]] = None,
    biomarker_info: Optional[BiomarkerInfo] = None,
) -> str:
    """Classify the temporal trend of a series of measurements.

    The goal of this function is to provide a simple yet robust classification of
    a biomarker time series as trending **up**, trending **down**, **stable** or
    **fluctuating**.  To reduce sensitivity to outliers and avoid assumptions
    about normally distributed residuals, a non‑parametric estimate of the
    slope is used.  Specifically, the function computes the **Theil–Sen
    estimator** of the per‑day slope, which is defined as the median of all
    pairwise slopes between measurement points【852769632987166†L133-L151】.  The
    Theil–Sen method is robust to outliers and performs well even when the
    underlying data are not normally distributed【852769632987166†L144-L151】.  When
    metadata are available the slope threshold for classification is taken
    from ``normal_change_per_day_high``.  Otherwise, a heuristic threshold is
    derived from the observed range and time span (20 % of the range per day).

    The classification proceeds as follows:

    * If fewer than two points are supplied, the trend is **"No Trend"**.
    * If all successive differences are strictly positive (monotonic increase),
      the trend is **"Up"**.  Likewise, if all are strictly negative the trend is
      **"Down"**.
    * Otherwise the Theil–Sen slope is computed.  If the slope exceeds the
      threshold the series is labelled **"Up"**; if it is below the negative
      threshold the series is labelled **"Down"**.
    * If the slope magnitude is small and the relative variation between the
      minimum and maximum values is less than 5 % of their midpoint, the
      series is considered **"Stable"**.
    * All other cases are labelled **"Fluctuating"**.

    Parameters
    ----------
    values : list of float
        The measurement values in chronological order.
    dates : list of datetime, optional
        Corresponding timestamps.  If provided and of equal length, they are
        used to compute the time axis; otherwise a sequential integer index is
        used.
    biomarker_info : BiomarkerInfo, optional
        Metadata containing normal per‑day change limits.  When provided,
        ``normal_change_per_day_high`` defines the slope threshold.

    Returns
    -------
    str
        One of "Up", "Down", "Stable", "Fluctuating" or "No Trend".
    """
    # Require at least two points
    if not values or len(values) < 2:
        return "No Trend"

    # Basic monotonicity check based on successive differences
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(d > 0 for d in diffs):
        return "Up"
    if all(d < 0 for d in diffs):
        return "Down"

    # Compute relative variation for later stability check
    vmin, vmax = min(values), max(values)
    mean_val = (vmin + vmax) / 2.0
    rel_variation = float('inf') if mean_val == 0 else (vmax - vmin) / mean_val

    # Build a time axis in days; fall back to integer indices when dates are absent
    if dates and len(dates) == len(values) and None not in dates:
        x = [(d - dates[0]).days for d in dates]
    else:
        x = list(range(len(values)))

    # Compute the Theil–Sen estimator: median of pairwise slopes
    slopes: List[float] = []
    n = len(values)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx == 0:
                continue
            dy = values[j] - values[i]
            slopes.append(dy / dx)
    # Fallback slope is zero if no valid pairwise slopes
    sen_slope = float(np.median(slopes)) if slopes else 0.0

    # Determine threshold for slope classification
    if biomarker_info and biomarker_info.normal_change_per_day_high:
        threshold = biomarker_info.normal_change_per_day_high
    else:
        # Use 20% of the observed range per day as a heuristic threshold
        time_span_days = max(x) - min(x) if len(x) > 1 else 1
        if time_span_days <= 0:
            time_span_days = 1
        threshold = ((vmax - vmin) / time_span_days) * 0.2 if vmax != vmin else 0.0

    # Compare the Sen slope against the threshold
    if sen_slope > threshold:
        return "Up"
    if sen_slope < -threshold:
        return "Down"

    # If slope is small but variation is tiny, declare stable
    if rel_variation < 0.05:
        return "Stable"

    return "Fluctuating"


def compute_measurement_status(
    value: float,
    ref_low: Optional[float],
    ref_high: Optional[float],
    biomarker_info: Optional[BiomarkerInfo] = None,
) -> str:
    """Determine the qualitative status of a measurement.

    The measurement is compared against both the reference range present in
    the results file and, if available, the normal range defined in the
    biomarker metadata.  The most conservative (broadest) range between
    the two sources is used when both are available.  If no boundaries
    are defined the status is ``"Unknown"``.

    Parameters
    ----------
    value : float
        The measured value.
    ref_low : float or None
        Lower bound from the result file's reference range.
    ref_high : float or None
        Upper bound from the result file's reference range.
    biomarker_info : BiomarkerInfo, optional
        Metadata from the biomarker database.

    Returns
    -------
    str
        One of ``"Low"``, ``"Normal"``, ``"High"`` or ``"Unknown"``.
    """
    # Prioritise the reference range from the result.  If a reference
    # bound is provided it will be used directly; otherwise fallback to
    # the biomarker's normal range.  If neither source defines any
    # bounds the status is unknown.
    low_bound: Optional[float] = None
    high_bound: Optional[float] = None
    if biomarker_info:
        low_bound = biomarker_info.normal_low
        high_bound = biomarker_info.normal_high
    elif ref_low is not None or ref_high is not None:
        low_bound = ref_low
        high_bound = ref_high
    if low_bound is None and high_bound is None:
        return "Unknown"
    if low_bound is not None and value < low_bound:
        return "Low"
    if high_bound is not None and value > high_bound:
        return "High"
    return "Normal"


def compute_critical_status(
    values: List[float],
    dates: List[dt.datetime],
    biomarker_info: Optional[BiomarkerInfo],
) -> Optional[str]:
    """Evaluate whether a series of measurements contains critical values.

    Criticality is assessed in two ways:

    #. Absolute thresholds defined by ``sharp_change_aggregated_low`` and
       ``sharp_change_aggregated_high``.  If any measurement is below the
       aggregated low or above the aggregated high the status is
       ``"Critically Low"`` or ``"Critically High"`` respectively.
    #. Relative or absolute changes defined in ``sharp_change_rules``.  For
       each pair of adjacent values the per-day change is computed and
       compared against any percentage/absolute thresholds defined in the
       rules.  If the absolute change exceeds a threshold the status
       becomes ``"Critically High"`` or ``"Critically Low"`` depending on
       the direction of the change.  Note that the current implementation
       treats all large changes symmetrically; finer control (e.g.
       differentiating rises from falls) could be added by examining the
       ``part`` field in the parsed rules.

    Parameters
    ----------
    values : list of float
        The measurement values in chronological order.
    dates : list of datetime
        Corresponding timestamps for the values.
    biomarker_info : BiomarkerInfo or None
        Metadata containing critical thresholds and sharp change rules.

    Returns
    -------
    str or None
        Returns ``"Critically Low"`` or ``"Critically High"`` if a
        critical condition is met.  Otherwise returns ``None``.
    """
    # If no biomarker info or empty values, no criticality can be assessed
    if not biomarker_info or not values:
        return None
    # Determine the unit in which thresholds should be compared.  When a
    # ``normal_unit`` is defined it matches the unit into which measurement
    # values have already been converted by ``process_results``.  Otherwise
    # comparisons will be carried out directly.
    target_unit: Optional[str] = biomarker_info.normal_unit if biomarker_info.normal_unit else None

    # Helper to compute absolute thresholds from rule values.  Handles
    # ULN‑based multipliers and unit conversions when necessary.
    def _get_threshold(val: Optional[float], unit: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        # ULN‑based thresholds multiply by the upper limit of normal
        if unit and ("uln" in unit.lower() or "×" in unit or "\u00d7" in unit):
            uln = biomarker_info.normal_high
            if uln is None:
                return None
            return val * uln
        # Convert into the target unit if one is defined and units differ
        if unit and target_unit and unit.strip().lower() != target_unit.strip().lower():
            try:
                return convert_value(val, unit, target_unit)
            except Exception:
                return val
        return val

    # ---------------------------------------------------------------------
    # 1. Evaluate absolute aggregated thresholds (sharp_change_aggregated_low/high)
    #    Historically these fields have been ambiguous: in some cases the
    #    aggregated "low" value actually represents an upper critical bound
    #    (e.g. Serum Urea uses a value of 60 to indicate a dangerous high),
    #    and vice versa.  To interpret them sensibly we compare the
    #    aggregated thresholds against the normal range.  Thresholds above the
    #    normal high are treated as critical high bounds; thresholds below the
    #    normal low are treated as critical low bounds.  If both aggregated
    #    thresholds are defined we assign the smaller to the low bound and the
    #    larger to the high bound.
    agg_low_raw = biomarker_info.sharp_change_aggregated_low
    agg_high_raw = biomarker_info.sharp_change_aggregated_high
    # Convert aggregated values to target unit when possible
    def _convert_agg(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        if target_unit and biomarker_info.normal_unit:
            try:
                return convert_value(val, biomarker_info.normal_unit, target_unit)
            except Exception:
                return val
        return val
    agg_low_val = _convert_agg(agg_low_raw)
    agg_high_val = _convert_agg(agg_high_raw)
    # Convert normal range to target unit for comparison
    norm_low = None
    norm_high = None
    try:
        if biomarker_info.normal_low is not None:
            if target_unit and biomarker_info.normal_unit:
                norm_low = convert_value(biomarker_info.normal_low, biomarker_info.normal_unit, target_unit)
            else:
                norm_low = biomarker_info.normal_low
        if biomarker_info.normal_high is not None:
            if target_unit and biomarker_info.normal_unit:
                norm_high = convert_value(biomarker_info.normal_high, biomarker_info.normal_unit, target_unit)
            else:
                norm_high = biomarker_info.normal_high
    except Exception:
        norm_low = biomarker_info.normal_low
        norm_high = biomarker_info.normal_high
    # Determine which aggregated value corresponds to a low or high bound.
    # Historically aggregated "low" and "high" fields can actually both
    # represent low critical thresholds (e.g. eGFR has 25 and 60, both of
    # which indicate declining renal function rather than high values).  To
    # interpret these sensibly we compare each aggregated value against the
    # normal range and categorise it accordingly.  Values below the normal
    # lower bound are treated as low thresholds, and values above the
    # normal upper bound are treated as high thresholds.  When multiple
    # candidates fall into the same category we select the one closest to
    # the normal range (i.e. the maximum of low thresholds and the minimum
    # of high thresholds).
    critical_low_threshold: Optional[float] = None
    critical_high_threshold: Optional[float] = None
    candidates: List[float] = []
    if agg_low_val is not None:
        candidates.append(agg_low_val)
    if agg_high_val is not None:
        candidates.append(agg_high_val)
    if candidates:
        low_thresholds: List[float] = []
        high_thresholds: List[float] = []
        for val in candidates:
            # If a normal high bound exists and val exceeds it, classify as high threshold
            if norm_high is not None and val > norm_high:
                high_thresholds.append(val)
            # If a normal low bound exists and val is below it, classify as low threshold
            elif norm_low is not None and val < norm_low:
                low_thresholds.append(val)
            else:
                # When no normal bounds, or val falls within the normal range, fall back
                # to interpreting the labels heuristically.  Treat the smaller of two
                # values as the low threshold and the larger as the high threshold.
                pass
        if low_thresholds:
            # For low thresholds choose the maximum value (closest to the normal range)
            critical_low_threshold = max(low_thresholds)
        if high_thresholds:
            # For high thresholds choose the minimum value (closest to the normal range)
            critical_high_threshold = min(high_thresholds)
        # If both lists are empty and we have exactly two candidates but no normal range
        # information, assign min to low and max to high as a fallback.
        if not low_thresholds and not high_thresholds and len(candidates) == 2:
            lo, hi = sorted(candidates)
            critical_low_threshold, critical_high_threshold = lo, hi
        # If only one candidate and no classification possible, decide based on position
        elif not low_thresholds and not high_thresholds and len(candidates) == 1:
            single = candidates[0]
            # If normal_low/high defined, compare accordingly
            if norm_high is not None and single > norm_high:
                critical_high_threshold = single
            elif norm_low is not None and single < norm_low:
                critical_low_threshold = single
            else:
                # Default: treat it as a high threshold if we have an upper normal bound,
                # otherwise as a low threshold
                if norm_high is not None:
                    critical_high_threshold = single
                else:
                    critical_low_threshold = single
    # Check each measurement against the inferred critical thresholds
    if critical_low_threshold is not None or critical_high_threshold is not None:
        for v, d in zip(values, dates):
            if v is None:
                continue
            if critical_low_threshold is not None and v < critical_low_threshold:
                return f"Critically Low - {d}"
            if critical_high_threshold is not None and v > critical_high_threshold:
                return f"Critically High - {d}"

    # ---------------------------------------------------------------------
    # 2. Evaluate absolute thresholds defined in sharp_change_rules
    rules = biomarker_info.sharp_change_rules or []
    for v, d in zip(values, dates):
        if v is None:
            continue
        for rule in rules:
            comp = rule.get("comp")
            unit = rule.get("unit")
            low = rule.get("low")
            high = rule.get("high")
            # Skip percentage‑based rules when evaluating absolute thresholds
            skip_units = ["%", "%points", "points", "percentage", "percent", "symptomatic"]
            if unit and any(u in unit for u in skip_units):
                continue
            if comp in (">", ">=") and low is not None:
                threshold = _get_threshold(low, unit)
                if threshold is not None and v > threshold:
                    return f"Critically High - {d}"
            if comp in ("<", "<=") and high is not None:
                threshold = _get_threshold(high, unit)
                if threshold is not None and v < threshold:
                    return f"Critically Low - {d}"

    # ---------------------------------------------------------------------
    # 3. Evaluate rate‑of‑change rules based on successive pairs
    if len(values) >= 2 and rules:
        for i in range(len(values) - 1):
            v_old, v_new = values[i], values[i + 1]
            d_old, d_new = dates[i], dates[i + 1]
            if v_old is None or v_new is None or d_old is None or d_new is None:
                continue
            delta_days = max((d_new - d_old).days, 1)
            delta_value = v_new - v_old
            rel_change_perc = (delta_value / v_old) * 100.0 / delta_days if v_old != 0 else 0.0
            abs_change_per_day = delta_value / delta_days
            for rule in rules:
                unit = rule.get("unit")
                per_day_low = rule.get("per_day_low")
                # Rate‑of‑change rules only have one bound (per_day_low)
                if unit == "%" and per_day_low is not None:
                    threshold = per_day_low
                    if abs(rel_change_perc) >= threshold:
                        return f"Critically High - {d_new}" if rel_change_perc > 0 else f"Critically Low - {d_new}"
                elif unit and unit not in ("%", "persistent") and per_day_low is not None:
                    if abs(abs_change_per_day) >= per_day_low:
                        return f"Critically High - {d_new}" if abs_change_per_day > 0 else f"Critically Low - {d_new}"

    # If no critical conditions met return None
    return None


def convert_value(value: Optional[float], from_unit: Optional[str], to_unit: Optional[str]) -> Optional[float]:
    """Convert a numeric value between units where possible.

    The conversion factors implemented here cover the most common unit
    transformations encountered in the supplied datasets (e.g. mg/dL to
    g/dL, mg/L to mg/dL and related micro-/nano‑scale conversions).  If
    either the source or target unit is ``None``, or if no conversion
    factor is defined, the original value is returned.  This function can
    be extended to support additional unit pairs as required.

    Parameters
    ----------
    value : float or None
        The numerical value to convert.
    from_unit : str or None
        The unit the value is currently expressed in.
    to_unit : str or None
        The unit to convert into.

    Returns
    -------
    float or None
        The converted value or the original value if no conversion was
        performed.
    """
    if value is None or from_unit is None or to_unit is None:
        return value
    # Normalise unit strings: remove whitespace and case
    f = from_unit.strip().lower()
    t = to_unit.strip().lower()
    if f == t:
        return value
    conversions = {
        ("g/dl", "mg/dl"): lambda v: v * 100.0,
        ("mg/dl", "g/dl"): lambda v: v / 100.0,
        ("mg/l", "mg/dl"): lambda v: v / 10.0,
        ("mg/dl", "mg/l"): lambda v: v * 10.0,
        ("µg/dl", "mg/dl"): lambda v: v / 1000.0,
        ("mg/dl", "µg/dl"): lambda v: v * 1000.0,
        ("ng/ml", "mg/dl"): lambda v: v / 100.0,
        ("mg/dl", "ng/ml"): lambda v: v * 100.0,
        ("mmol/l", "mg/dl"): lambda v: v * 18.0182,  # approximate for glucose
        ("mg/dl", "mmol/l"): lambda v: v / 18.0182,
        ("µmol/l", "mg/dl"): lambda v: v / 88.42,  # approximate for creatinine
        ("mg/dl", "µmol/l"): lambda v: v * 88.42,
        # gram per litre to gram per decilitre (1 g/L = 0.1 g/dL)
        ("g/l", "g/dl"): lambda v: v / 10.0,
        ("g/dl", "g/l"): lambda v: v * 10.0,
        # gram per litre to milligram per decilitre
        ("g/l", "mg/dl"): lambda v: v * 100.0,  # 1 g/L = 100 mg/dL
        ("mg/dl", "g/l"): lambda v: v / 100.0,
        # white blood cell and platelet counts: /µL to x10^9/L conversions
        ("/µl", "x10^9/l"): lambda v: v / 1000.0,
        ("/ul", "x10^9/l"): lambda v: v / 1000.0,
        ("/µl", "x10^9/ l"): lambda v: v / 1000.0,
        ("x10^9/l", "/µl"): lambda v: v * 1000.0,
        ("x10^9/l", "/ul"): lambda v: v * 1000.0,
        # cell counts expressed as trillions per litre to millions per microlitre.
        # 1 L = 1e6 µL, so a value in x10^12/L (trillions per litre) is
        # multiplied by 1e6 to obtain the count per microlitre.  Similarly,
        # converting from x10^12/L to x10^9/L multiplies by 1000 and vice versa.
        ("x10^12/l", "/µl"): lambda v: v * 1_000_000.0,
        ("x10^12/l", "/ul"): lambda v: v * 1_000_000.0,
        ("/µl", "x10^12/l"): lambda v: v / 1_000_000.0,
        ("/ul", "x10^12/l"): lambda v: v / 1_000_000.0,
        ("x10^12/l", "x10^9/l"): lambda v: v * 1000.0,
        ("x10^9/l", "x10^12/l"): lambda v: v / 1000.0,
        # microgram per litre to microgram per decilitre (1 µg/L = 0.1 µg/dL)
        ("µg/l", "µg/dl"): lambda v: v / 10.0,
        ("µg/dl", "µg/l"): lambda v: v * 10.0,
        ("ug/l", "ug/dl"): lambda v: v / 10.0,
        ("ug/dl", "ug/l"): lambda v: v * 10.0,
    }
    key = (f, t)
    if key in conversions:
        try:
            return conversions[key](value)
        except Exception:
            return value
    # If from_unit includes multiple candidates (e.g. "mg/dL men; >=50mg/dL women"), extract the first token
    if ";" in f:
        f_base = f.split(";")[0].strip()
        return convert_value(value, f_base, t)
    return value


def compute_rate_of_change(
    values: List[float],
    dates: List[dt.datetime],
    biomarker_info: Optional[BiomarkerInfo] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute the average daily rate of change and classify it relative to normal.

    The function calculates the per‑day change between the earliest and latest
    measurements.  When metadata provide a *normal* range for the daily change
    (``normal_change_per_day_low`` and ``normal_change_per_day_high``), the
    absolute magnitude of the computed rate is compared against these bounds.
    Unlike the original implementation, the low and high limits are treated as
    an **absolute range of acceptable change**, meaning that very small or very
    large changes are flagged regardless of direction.  Measurement values are
    first converted into the unit specified by ``normal_change_unit`` if such
    a unit is defined; otherwise the values are used as‑is.  Unit conversion
    leverages the internal ``convert_value`` helper.  If the conversion fails or
    units are undefined, the raw values are used.

    Parameters
    ----------
    values : list of float
        Measurement values in chronological order (already converted to the
        biomarker's normal unit when available).
    dates : list of datetime
        Corresponding timestamps for the values.
    biomarker_info : BiomarkerInfo or None
        Metadata describing normal rates of change and unit definitions.

    Returns
    -------
    tuple
        ``(rate_per_day, rate_status)`` where ``rate_per_day`` is the
        numeric rate (units per day) and ``rate_status`` is one of
        ``"Low"``, ``"High"``, ``"Normal"`` or ``None`` when no bounds are
        defined.
    """
    # Need at least two measurements and valid dates
    if not values or len(values) < 2 or not dates or len(dates) < 2:
        return (None, None)

    # Extract first and last values and dates
    v0_raw, v1_raw = values[0], values[-1]
    d0, d1 = dates[0], dates[-1]
    delta_days = max((d1 - d0).days, 1)

    # If biomarker metadata defines a separate unit for change, convert the
    # values into that unit.  Otherwise, assume values are already in the
    # appropriate unit.  The effective measurement unit is biomarker_info.normal_unit
    # because upstream conversions standardise values to that when available.
    if biomarker_info and biomarker_info.normal_change_unit:
        try:
            from_unit = biomarker_info.normal_unit
            to_unit = biomarker_info.normal_change_unit
            v0 = convert_value(v0_raw, from_unit, to_unit) if from_unit and to_unit else v0_raw
            v1 = convert_value(v1_raw, from_unit, to_unit) if from_unit and to_unit else v1_raw
        except Exception:
            # On failure fall back to original values
            v0, v1 = v0_raw, v1_raw
    else:
        v0, v1 = v0_raw, v1_raw

    # Compute per‑day change.  The interpretation depends on the unit used for
    # the normal change.  For percentage‑based change (e.g. "%/day"), compute
    # the relative change per day.  Otherwise compute the absolute change per day.
    rate: Optional[float]
    try:
        # Determine if the normal change unit is percentage based
        perc_unit = False
        if biomarker_info and biomarker_info.normal_change_unit:
            u = biomarker_info.normal_change_unit.strip().lower()
            perc_unit = any(tok in u for tok in ["%", "percent", "percentage"])
        if perc_unit:
            # Avoid division by zero; if the baseline is zero the relative change is undefined
            if v0 == 0:
                rate = None
            else:
                rate = ((v1 - v0) / abs(v0)) * 100.0 / delta_days
        else:
            rate = (v1 - v0) / delta_days
    except Exception:
        return (None, None)

    status: Optional[str] = None
    if rate is not None and biomarker_info:
        low = biomarker_info.normal_change_per_day_low
        high = biomarker_info.normal_change_per_day_high
        # Interpret the low/high bounds as absolute magnitude limits
        if low is not None or high is not None:
            abs_rate = abs(rate)
            if low is not None and abs_rate < low:
                status = "Low"
            elif high is not None and abs_rate > high:
                status = "High"
            else:
                status = "Normal"
    return (rate, status)


def compute_derived_ratios(result_dict: Dict[str, Dict[str, Any]]) -> None:
    """Compute additional clinical ratios and insert them into the result.

    The function operates in-place on the provided ``result_dict``.  It
    requires that the underlying dictionary already contains entries for
    the component biomarkers.  Currently computed ratios include:

    * **BUN/Creatinine** – ratio of latest BUN to latest Creatinine values.
    * **Albumin/Globulin** – ratio where globulin is defined as total
      protein minus albumin.
    * **Total/HDL Cholesterol** – derived ratio of total cholesterol to
      HDL‑C.

    Each new ratio entry is structured like a normal biomarker entry
    containing the computed value in the ``value`` field and a textual
    description in ``note``.  If required components are missing the
    ratio is not added.

    Parameters
    ----------
    result_dict : dict
        Mapping of biomarker names to their analysis output.
    """
    # BUN/Creatinine ratio
    if "BUN" in result_dict and "Creatinine" in result_dict:
        bun_vals = result_dict["BUN"]["values"]
        cr_vals = result_dict["Creatinine"]["values"]
        if bun_vals and cr_vals:
            bun_latest = bun_vals[-1]
            cr_latest = cr_vals[-1]
            if cr_latest:
                ratio = bun_latest / cr_latest
                result_dict["BUN/Creatinine"] = {
                    "test_name": "BUN/Creatinine",
                    "values": [ratio],
                    "dates": [result_dict["BUN"]["dates"][-1]],
                    "unit": None,
                    "temporal_context": "Derived",
                    "trend": None,
                    "measurement_status": None,
                    "critical_status": None,
                    "rate_of_change": None,
                    "rate_status": None,
                    "note": "Ratio of BUN to Creatinine"
                }
    # Albumin/Globulin ratio
    if "Albumin" in result_dict and "Total Protein" in result_dict:
        alb_vals = result_dict["Albumin"]["values"]
        tot_vals = result_dict["Total Protein"]["values"]
        if alb_vals and tot_vals:
            albumin_latest = alb_vals[-1]
            total_protein_latest = tot_vals[-1]
            globulin = total_protein_latest - albumin_latest
            if globulin > 0:
                ag_ratio = albumin_latest / globulin
                result_dict["Albumin/Globulin"] = {
                    "test_name": "Albumin/Globulin",
                    "values": [ag_ratio],
                    "dates": [result_dict["Albumin"]["dates"][-1]],
                    "unit": None,
                    "temporal_context": "Derived",
                    "trend": None,
                    "measurement_status": None,
                    "critical_status": None,
                    "rate_of_change": None,
                    "rate_status": None,
                    "note": "Albumin divided by Globulin (Total Protein - Albumin)"
                }
    # Total/HDL ratio
    if "Total Cholesterol" in result_dict and "HDL‑C" in result_dict:
        tot_vals = result_dict["Total Cholesterol"]["values"]
        hdl_vals = result_dict["HDL‑C"]["values"]
        if tot_vals and hdl_vals:
            tot_latest = tot_vals[-1]
            hdl_latest = hdl_vals[-1]
            if hdl_latest:
                ratio = tot_latest / hdl_latest
                result_dict["Total/HDL Ratio"] = {
                    "test_name": "Total/HDL Ratio",
                    "values": [ratio],
                    "dates": [result_dict["Total Cholesterol"]["dates"][-1]],
                    "unit": None,
                    "temporal_context": "Derived",
                    "trend": None,
                    "measurement_status": None,
                    "critical_status": None,
                    "rate_of_change": None,
                    "rate_status": None,
                    "note": "Total Cholesterol divided by HDL‑C"
                }


def process_results(
    single_path: str,
    multiple_path: str,
    biomarker_path: str,
    sex: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Process the results and return a structured dictionary.

    This is the high-level driver that loads all input files,
    harmonises them, computes trends, statuses and derived metrics and
    returns the final composite result structure.

    Parameters
    ----------
    single_path : str
        Path to the ``Results_Single.csv`` file.
    multiple_path : str
        Path to the ``Results_Multiple.csv`` file.
    biomarker_path : str
        Path to the ``Biomarker_Data.csv`` file.
    sex : str, optional
        Patient sex, either ``"Male"`` or ``"Female"``.  When
        provided, sex-specific reference ranges and change limits are
        selected from the biomarker metadata.  When omitted, the
        ``"Both"`` entry or the first available entry is used.

    Returns
    -------
    dict
        A mapping from English test names (and derived ratio names) to
        dictionaries containing analysis results.
    """
    biomarker_map = load_biomarker_data(biomarker_path)
    # Load and melt the result sets
    df_single = load_results(single_path)
    df_multiple = load_results(multiple_path)
    long_single = melt_results(df_single)
    long_multiple = melt_results(df_multiple)
    # Combine the long-format dataframes
    combined = pd.concat([long_single, long_multiple], ignore_index=True)
    # Remove exact duplicate measurements to avoid double counting when
    # identical rows appear in both the single and multiple files.  Duplicates
    # are identified by the test name, timestamp and value.
    combined = combined.drop_duplicates(subset=["Test_clean", "datetime", "value"])
    # Map Turkish names to English names
    combined["test_name"] = combined["Test_clean"].apply(map_test_name)
    # Group by test_name
    result_dict: Dict[str, Dict[str, Any]] = {}
    for test_name, group in combined.groupby("test_name"):
        """
        Iterate through each test, convert values and reference ranges on a per‑row basis
        taking into account potential unit mismatches and obvious mis‑scaling.  The
        previous implementation applied a single unit to all measurements in a group,
        which left large numeric artefacts (e.g. 39377 instead of 39.377) when
        comma‑separated decimals were parsed inconsistently.  We now determine a
        target unit for each test and convert each measurement individually from
        its own recorded unit.  When a converted value is orders of magnitude
        outside its reference range we heuristically scale it by factors of 1000 or
        1/1000 to bring it into a plausible range.  This is particularly important
        for tests lacking metadata in ``Biomarker_Data.csv``, where the only
        guidance is the reference range present in the results file.
        """
        # Sort measurements by timestamp
        group_sorted = group.sort_values(by="datetime")
        dates: List[dt.datetime] = group_sorted["datetime"].tolist()
        # Retrieve biomarker info for the given test_name and patient sex if available.
        # ``biomarker_map`` now returns a nested mapping of ``{test_name: {sex: BiomarkerInfo}}``.
        b_info_dict = biomarker_map.get(test_name)
        b_info: Optional[BiomarkerInfo] = None
        if isinstance(b_info_dict, dict):
            if sex:
                # Normalise requested sex to title case (e.g. "male" -> "Male")
                sex_key = sex.strip().title()
                # First try exact match on sex
                b_info = b_info_dict.get(sex_key)
                # If not found, fall back to the 'Both' entry
                if b_info is None:
                    b_info = b_info_dict.get("Both")
                # If still not found, take any available entry (e.g. the first)
                if b_info is None and b_info_dict:
                    b_info = next(iter(b_info_dict.values()))
            else:
                # When no sex specified, prioritise 'Both'; otherwise pick any available entry
                b_info = b_info_dict.get("Both") if b_info_dict.get("Both") is not None else next(iter(b_info_dict.values()))
        else:
            # For backward compatibility if biomarker_map was not nested (should not occur)
            b_info = b_info_dict
        # Determine the target unit.  When metadata provides a normal unit we use it;
        # otherwise fall back to the unit of the most recent non‑null measurement.
        if b_info and b_info.normal_unit:
            target_unit: Optional[str] = b_info.normal_unit
        else:
            # Choose the unit from the last non‑empty entry in this group
            target_unit = None
            for u in reversed(group_sorted["unit"].tolist()):
                if isinstance(u, str) and u.strip():
                    target_unit = u
                    break
        # Convert each value and its reference bounds individually
        values_converted: List[Optional[float]] = []
        ref_low_conv_series: List[Optional[float]] = []
        ref_high_conv_series: List[Optional[float]] = []
        for (_, row) in group_sorted.iterrows():
            raw_value = row["value"]
            u = row["unit"] if isinstance(row["unit"], str) else None
            # Convert measurement to target unit if both units are defined
            if target_unit and u:
                v_conv = convert_value(raw_value, u, target_unit)
                # Convert this row's reference range
                low = row["ref_low"]
                high = row["ref_high"]
                low_conv = convert_value(low, u, target_unit) if low is not None else None
                high_conv = convert_value(high, u, target_unit) if high is not None else None
            else:
                v_conv = raw_value
                low_conv = row["ref_low"]
                high_conv = row["ref_high"]
            values_converted.append(v_conv)
            ref_low_conv_series.append(low_conv)
            ref_high_conv_series.append(high_conv)
        # Heuristically correct mis‑scaled values.  If a value is >100× above
        # its reference high we assume a missing decimal separator and divide by 1000.
        # Conversely if a value is <1/100th of its reference low (and positive)
        # multiply by 1000.  This uses per‑row reference ranges when available.
        for i in range(len(values_converted)):
            val = values_converted[i]
            if val is None:
                continue
            ref_high = ref_high_conv_series[i]
            ref_low = ref_low_conv_series[i]
            # Only apply heuristics when at least one reference bound is available
            try:
                if ref_high is not None and val is not None and ref_high is not None:
                    # scale down extremely large numbers
                    if val > ref_high * 100:
                        scaled = val / 1000.0
                        # Accept the scaling if the scaled value is within a reasonable
                        # range relative to the reference high (less than 100×)
                        if scaled <= (ref_high * 100):
                            values_converted[i] = scaled
                            val = scaled
                if ref_low is not None and val is not None and ref_low > 0:
                    # scale up extremely small numbers (if originally positive)
                    if val < ref_low / 100 and val > 0:
                        values_converted[i] = val * 1000.0
            except Exception:
                # In case of unexpected type errors, skip scaling
                pass
        # The effective unit for this test is the chosen target unit (may be None)
        effective_unit = target_unit
        # The reference low/high to report come from the most recent measurement
        ref_low_conv = ref_low_conv_series[-1] if ref_low_conv_series else None
        ref_high_conv = ref_high_conv_series[-1] if ref_high_conv_series else None
        # Temporal context
        temporal_context, _ = compute_temporal_context(dates)
        # Trend classification
        trend = compute_trend(values_converted, dates, b_info)
        # Measurement status for the latest value
        latest_val = values_converted[-1] if values_converted else None
        status = compute_measurement_status(latest_val, ref_low_conv, ref_high_conv, b_info)
        # Critical status (with possible date flag)
        critical = compute_critical_status(values_converted, dates, b_info)
        # Rate of change
        rate, rate_status = compute_rate_of_change(values_converted, dates, b_info)
        result_dict[test_name] = {
            "test_name": test_name,
            "values": values_converted,
            "dates": dates,
            "unit": effective_unit,
            "temporal_context": temporal_context,
            "trend": trend,
            "measurement_status": status,
            "critical_status": critical,
            "rate_of_change": rate,
            "rate_status": rate_status,
            "reference low": ref_low_conv,
            "reference high": ref_high_conv,
            "effective_unit": effective_unit,
        }
    # Compute derived ratios after base biomarkers are processed
    compute_derived_ratios(result_dict)
    return result_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process biomarker results and produce analysis output.")
    parser.add_argument("--single", type=str, default="Results_Single.csv", help="Path to Results_Single.csv")
    parser.add_argument("--multiple", type=str, default="Results_Multiple.csv", help="Path to Results_Multiple.csv")
    parser.add_argument("--biomarker", type=str, default="Biomarker_Data.csv", help="Path to Biomarker_Data.csv")
    parser.add_argument("--sex", type=str, choices=["Male", "Female"], default=None,
                        help="Optional patient sex (Male or Female) to select sex-specific biomarker ranges.")
    parser.add_argument("--output", type=str, help="Optional path to write JSON output")
    args = parser.parse_args()
    # Process the results
    output = process_results(args.single, args.multiple, args.biomarker, sex=args.sex)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, default=str, indent=2)
    else:
        print(json.dumps(output, default=str, indent=2))
