"""Registry of the legal documents in the corpus and their metadata."""

from __future__ import annotations

from dataclasses import dataclass

CENTRAL = "India"


@dataclass(frozen=True)
class SourceInfo:
    title: str
    state: str
    document_type: str
    unit: str  # what a numbered provision is called: "Section" (Acts) or "Rule" (Rules)


# Keyed by lower-case filename stem. Add a line here when adding a PDF to data/raw.
REGISTRY: dict[str, SourceInfo] = {
    "motor_vehicles": SourceInfo("Motor Vehicles Act, 1988", CENTRAL, "Central Act", "Section"),
    "cmvr": SourceInfo("Central Motor Vehicles Rules, 1989", CENTRAL, "Central Rules", "Rule"),
    "haryana": SourceInfo("Haryana Motor Vehicles Rules, 1993", "Haryana", "State Rules", "Rule"),
    "up": SourceInfo(
        "Uttar Pradesh Motor Vehicles Rules, 1998", "Uttar Pradesh", "State Rules", "Rule"
    ),
}


def get_source_info(filename: str) -> SourceInfo:
    stem = filename.rsplit(".", 1)[0].lower()
    if stem in REGISTRY:
        return REGISTRY[stem]
    return SourceInfo(stem.replace("_", " ").title(), CENTRAL, "Motor Vehicle Law", "Rule")


def known_states() -> list[str]:
    return sorted({info.state for info in REGISTRY.values() if info.state != CENTRAL})
