from aquai.application.Rules.variables import NAME_SEPARATORS, PH_ALIASES


def is_ph_name(name: str) -> bool:
    normalized = NAME_SEPARATORS.sub("", name).casefold()

    return normalized in PH_ALIASES


def is_ph_field(field) -> bool:
    return bool(field.sources) and all(is_ph_name(ref.column) for ref in field.sources)


def presentation_unit_status(field):
    return "dimensionless" if is_ph_field(field) else field.unit_status
