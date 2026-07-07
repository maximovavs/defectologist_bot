from __future__ import annotations

"""Targeted reliability overrides for llm_generator.

The project historically keeps generation and validation logic in one large module.
These overrides isolate narrowly scoped production safeguards while preserving the
existing public API and avoiding unrelated changes to the generator.
"""

import re
from types import ModuleType
from typing import Dict, List, Set, Tuple


_SIMPLE_NUMBER_WORDS: Dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
}

_NUMERIC_DETAIL_RE = re.compile(
    r"\b(?P<number>\d+|one|two|three|four|five|один|одна|одно|два|две|три|четыре|пять)"
    r"\s*[-–—]?\s*"
    r"(?P<unit>"
    r"секунд\w*|seconds?|sec|"
    r"минут\w*|minutes?|min|"
    r"раз(?:а|ов)?|повтор\w*|times?|repetitions?|"
    r"карточ\w*|cards?|"
    r"предмет\w*|objects?"
    r")\b",
    flags=re.IGNORECASE,
)


def _normalize_numeric_unit(unit: str) -> str:
    probe = re.sub(r"\s+", " ", (unit or "").strip()).replace("ё", "е").lower()
    if re.match(r"^(секунд|second|sec)", probe):
        return "seconds"
    if re.match(r"^(минут|minute|min)", probe):
        return "minutes"
    if re.match(r"^(раз|повтор|time|repetition)", probe):
        return "repetitions"
    if re.match(r"^(карточ|card)", probe):
        return "cards"
    if re.match(r"^(предмет|object)", probe):
        return "objects"
    return probe


def _normalize_simple_number(value: str) -> int:
    probe = re.sub(r"\s+", " ", (value or "").strip()).replace("ё", "е").lower()
    if probe.isdigit():
        return int(probe)
    return _SIMPLE_NUMBER_WORDS[probe]


def _extract_numeric_concrete_details(text: str) -> Set[Tuple[int, str]]:
    details: Set[Tuple[int, str]] = set()
    for raw_line in (text or "").replace("\r\n", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line.strip())
        low = line.replace("ё", "е").lower()
        if low.startswith("👶 возраст:") or low.startswith("возраст:"):
            continue
        for match in _NUMERIC_DETAIL_RE.finditer(line):
            details.add(
                (
                    _normalize_simple_number(match.group("number")),
                    _normalize_numeric_unit(match.group("unit")),
                )
            )
    return details


def _normalize_scan_segments(text: str) -> List[str]:
    """Normalize local lines/sentences without losing myth/fact boundaries."""
    segments: List[str] = []
    for raw_line in (text or "").replace("\r\n", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line.strip()).replace("ё", "е").lower()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        segments.extend(part.strip() for part in parts if part.strip())
    return segments


def apply_overrides(module: ModuleType) -> None:
    """Patch the three validators through the module's existing public API."""

    def validate_evidence_grounding(
        output_text: str,
        evidence_text: str,
        rubric_format: str = "",
    ) -> Tuple[bool, str]:
        """Conservatively reject risky mechanism claims in short posts.

        Even when one source mentions a mechanism, a short source extract is not
        enough to establish clinical validity or causality. The post should describe
        observable actions and outcomes instead.
        """
        del evidence_text, rubric_format
        out = module._normalize_scan_text(output_text)
        if not out:
            return True, "ok"

        for pattern, label, _evidence_terms in module.RISKY_MECHANISM_CLAIMS:
            if re.search(pattern, out, flags=re.IGNORECASE):
                return False, f"unsupported_mechanism_claim:{label}"

        return True, "ok"

    def validate_pro_concrete_details(
        output_text: str,
        evidence_text: str,
    ) -> Tuple[bool, str]:
        out = module._normalize_scan_text(output_text)
        if not out:
            return True, "ok"

        evidence = module._normalize_scan_text(evidence_text)
        for pattern, label, evidence_aliases in module.PRO_CONCRETE_DETAIL_PATTERNS:
            has_evidence_concept = any(
                all(term in evidence for term in alias_terms)
                for alias_terms in evidence_aliases
            )
            if re.search(pattern, out, flags=re.IGNORECASE) and not has_evidence_concept:
                return False, f"pro_unsupported_concrete_detail:{label}"

        evidence_numeric = _extract_numeric_concrete_details(evidence_text)
        for value, unit in sorted(_extract_numeric_concrete_details(output_text)):
            if (value, unit) not in evidence_numeric:
                return False, f"pro_unsupported_numeric_detail:{value}_{unit}"

        return True, "ok"

    def _has_parent_specific_risk(text: str) -> bool:
        specific_patterns = [
            r"\bмой\s+реб[её]нок.{0,60}мало\s+говор",
            r"\bреб[её]нок.{0,60}мало\s+говор",
            r"\bреб[её]нок.{0,60}перестал\w*\s+говор",
            r"\bперестал\w*\s+говор",
            r"\b(он|она).{0,40}потерял\w*.{0,40}навык",
            r"\bреб[её]нок.{0,60}потерял\w*.{0,40}навык",
            r"\bпотерял\w*.{0,20}(уже\s+)?появивш\w*.{0,30}навык",
            r"\bреб[её]нок.{0,60}не\s+понимает.{0,40}(бытов\w*\s+просьб|реч)",
        ]
        general_exclusions = [
            r"^\W*миф\s*:",
            r"не\s+вызыва\w*.{0,20}задержк\w*\s+реч",
            r"статья\s+рассматривает.{0,40}задержк\w*\s+реч",
            r"задержк\w*\s+реч\w*\s+может\s+иметь\s+разн\w*\s+причин",
        ]

        for segment in _normalize_scan_segments(text):
            if any(
                re.search(pattern, segment, flags=re.IGNORECASE)
                for pattern in general_exclusions
            ):
                continue
            if any(
                re.search(pattern, segment, flags=re.IGNORECASE)
                for pattern in specific_patterns
            ):
                return True
        return False

    def _validate_parent_safety_output(text: str) -> Tuple[bool, str]:
        blob = module._normalize_scan_text(text)
        blanket = module._contains_any_fragment(blob, module.BLANKET_REASSURANCE)
        if blanket:
            return False, "blanket_reassurance"

        if not _has_parent_specific_risk(text):
            return True, "ok"

        if not module._contains_any_fragment(blob, module.PARENT_SAFETY_ACTIONS):
            return False, "missing_parent_safety_note"

        return True, "ok"

    module.validate_evidence_grounding = validate_evidence_grounding
    module.validate_pro_concrete_details = validate_pro_concrete_details
    module._has_parent_specific_risk = _has_parent_specific_risk
    module._validate_parent_safety_output = _validate_parent_safety_output
