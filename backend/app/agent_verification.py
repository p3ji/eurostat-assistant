from __future__ import annotations

import re
from typing import Any

# Regex to match Markdown links: [text](url) -> keep text only
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Regex to match code spans: `code`
_CODE_SPAN = re.compile(r"`[^`]+`")
# Numbered lists: "1. ", "2. "
_LIST_NUMBER = re.compile(r"(?:^|\n)\s*\d+\.\s+")
# Common dataset identifiers with alphanumeric digits: nama_10_gdp, naio_10_cp1700, EU27, EA19, C10-12, Y15-74
_DATASET_OR_CATEGORY_TOKEN = re.compile(r"\b[A-Za-z]+[0-9]+[A-Za-z0-9_-]*\b|\b[A-Za-z0-9_-]+[0-9]+[A-Za-z]+\b")
# Chemical or scientific formulas like CO2, CO₂
_CHEMICAL_FORMULA = re.compile(r"\bCO[2₂]\b", re.IGNORECASE)
# Index base notation: e.g. "2005=100", "=100"
_INDEX_BASE = re.compile(r"=\s*100\b")
# 4-digit years (1900-2099)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
# Ordinals: 1st, 2nd, 3rd, 4th, etc.
_ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
# Candidate numeric values: integer or float, optionally with % or commas
_NUMERIC_CLAIM = re.compile(
    r"(?<![\w])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)(?![\w])"
)


def extract_prose_numerical_claims(text: str) -> list[str]:
    """Extract candidate statistical numerical claims from prose.

    Filters out structural numbers (markdown links, code, list enumerators,
    4-digit years, chemical formulas, and dataset/category codes).
    """
    if not text:
        return []

    # Replace markdown links with their label only so URLs are not scanned
    cleaned = _MD_LINK.sub(r"\1", text)
    # Remove code blocks and inline code
    cleaned = _CODE_SPAN.sub(" ", cleaned)
    # Remove index base references (=100)
    cleaned = _INDEX_BASE.sub(" ", cleaned)
    # Remove chemical formulas like CO2
    cleaned = _CHEMICAL_FORMULA.sub(" ", cleaned)
    # Remove list items: 1. , 2.
    cleaned = _LIST_NUMBER.sub("\n", cleaned)
    # Remove category/dataset codes with digits: e.g. C10-12, EU27, nama_10_gdp
    cleaned = _DATASET_OR_CATEGORY_TOKEN.sub(" ", cleaned)

    claims: list[str] = []
    for match in _NUMERIC_CLAIM.finditer(cleaned):
        token = match.group(1).strip()
        # Skip years
        bare_token = token.rstrip("%")
        if _YEAR.fullmatch(bare_token):
            continue
        # Skip ordinals
        if _ORDINAL.fullmatch(bare_token):
            continue
        claims.append(token)
    return claims


def _parse_claim_value(token: str) -> float | None:
    raw = token.rstrip("%").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_row_observations(rows: list[dict[str, Any]]) -> set[float]:
    """Extract all valid observation floats from structured dataset rows."""
    observations: set[float] = set()
    for row in rows:
        for key, val in row.items():
            if key in {"value", "x_value", "y_value"} or (
                isinstance(val, (int, float)) and not isinstance(val, bool) and key not in {"time", "year"}
            ):
                try:
                    observations.add(round(float(val), 4))
                except (ValueError, TypeError):
                    continue
    return observations


def _is_grounded(
    claim_val: float,
    observations: set[float],
    metadata_numbers: set[float],
) -> bool:
    """Check if a claim value matches any observation or known metadata count."""
    if claim_val in metadata_numbers:
        return True

    for obs in observations:
        # Match if within 0.05 absolute tolerance or 1% relative tolerance
        if abs(claim_val - obs) < 0.05:
            return True
        if obs != 0 and abs((claim_val - obs) / obs) < 0.01:
            return True
        # Also check if user rounded or converted percent/ratio (e.g., 0.142 vs 14.2)
        if abs(claim_val - (obs * 100)) < 0.05:
            return True
        if abs((claim_val / 100) - obs) < 0.0005:
            return True
    return False


def verify_answer_grounding(
    answer: str,
    result: dict[str, Any] | None,
    confidence: str,
    tool_truncated: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Audit the model's text answer against the structured observations.

    If ungrounded numerical claims are detected, append an explanatory warning
    to the result and calibrate the confidence level.
    """
    if not result or not isinstance(result.get("rows"), list):
        return result, confidence

    rows = result["rows"]
    if not rows:
        return result, confidence

    claims = extract_prose_numerical_claims(answer)
    if not claims:
        return result, confidence

    observations = extract_row_observations(rows)

    # Allow metadata counts such as row count, time period count, etc.
    metadata_numbers: set[float] = {
        float(len(rows)),
        float(len(result.get("columns", []))),
    }
    filters = result.get("filters") or {}
    for key, values in filters.items():
        if key in {"lastTimePeriod", "time"}:
            for val in values:
                try:
                    metadata_numbers.add(float(val))
                except ValueError:
                    pass

    ungrounded: list[str] = []
    for claim in claims:
        val = _parse_claim_value(claim)
        if val is None:
            continue
        if not _is_grounded(val, observations, metadata_numbers):
            ungrounded.append(claim)

    if ungrounded:
        warnings = list(result.get("warnings") or [])
        sample_str = ", ".join(ungrounded[:3])
        if len(ungrounded) > 3:
            sample_str += f" and {len(ungrounded) - 3} more"
        warnings.append(
            f"Notice: The text response mentions ungrounded numerical values ({sample_str}) "
            "that could not be verified against the returned Eurostat data. "
            "The table and chart below are the authoritative source."
        )
        if tool_truncated:
            warnings.append(
                "Observations were truncated in the model's working context, "
                "which may increase the risk of transcription or hallucination errors."
            )
        result["warnings"] = warnings

        # Calibrate confidence: an answer asserting unverified numbers cannot be high confidence
        if confidence == "high":
            confidence = "medium"

    return result, confidence
