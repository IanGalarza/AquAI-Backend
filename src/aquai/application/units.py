"""Unidades de presentación; no modifica los metadatos ni convierte valores."""

import re


def is_ph_name(name: str) -> bool:
    normalized = re.sub(r"[\s_\[\]().-]+", "", name).casefold()
    return normalized in {"ph", "phuph", "phudeph", "phunidadesdeph"}


def is_ph_field(field) -> bool:
    return bool(field.sources) and all(is_ph_name(ref.column) for ref in field.sources)


def presentation_unit_status(field):
    return "dimensionless" if is_ph_field(field) else field.unit_status
