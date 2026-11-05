"""Redact common PII patterns before text is embedded or sent to a model.

Pattern-based redaction is a first line of defence, not a guarantee. Regulated
workloads should add a trained detector (e.g. Amazon Comprehend, Presidio).
"""

from __future__ import annotations

import re

PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS.items():
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts
