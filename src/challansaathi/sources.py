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
    # How current the PDF's text is. Laws are amended often (the 2019 amendment to the Act
    # raised most fines), so this is shown to the LLM and to users next to every citation.
    text_as_of: str = "unknown"


# Keyed by lower-case filename stem. Add a line here when adding a PDF to data/raw.
REGISTRY: dict[str, SourceInfo] = {
    "motor_vehicles": SourceInfo(
        "Motor Vehicles Act, 1988",
        CENTRAL,
        "Central Act",
        "Section",
        "Oct 2018 (before the Motor Vehicles (Amendment) Act, 2019)",
    ),
    "cmvr": SourceInfo(
        "Central Motor Vehicles Rules, 1989", CENTRAL, "Central Rules", "Rule", "2008"
    ),
    "haryana": SourceInfo(
        "Haryana Motor Vehicles Rules, 1993", "Haryana", "State Rules", "Rule", "2021"
    ),
    "up": SourceInfo(
        "Uttar Pradesh Motor Vehicles Rules, 1998", "Uttar Pradesh", "State Rules", "Rule", "2023"
    ),
}


def get_source_info(filename: str) -> SourceInfo:
    stem = filename.rsplit(".", 1)[0].lower()
    if stem in REGISTRY:
        return REGISTRY[stem]
    return SourceInfo(stem.replace("_", " ").title(), CENTRAL, "Motor Vehicle Law", "Rule")


def known_states() -> list[str]:
    return sorted({info.state for info in REGISTRY.values() if info.state != CENTRAL})
