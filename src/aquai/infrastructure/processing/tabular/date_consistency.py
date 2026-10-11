"""Contrasta fechas originales con componentes y asociaciones corroboradas."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Number

from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter

from aquai.application.dtos.tabular import TabularFinding
from aquai.application.dtos.tabular_policies import (
    AssociationDiagnostic,
    DateColumnPolicy,
    PreparationPolicies,
    TemporalDateTreatment,
)
from aquai.infrastructure.processing.tabular.normalization import parse_explicit_date, trace_value
from aquai.infrastructure.processing.tabular.preparation import cell_metadata, headers_match, strict_missing
from aquai.infrastructure.processing.tabular.reader import SheetData


_MONTH_GROUPS = (
    ("enero", "ene", "january", "jan"),
    ("febrero", "feb", "february"),
    ("marzo", "mar", "march"),
    ("abril", "abr", "april", "apr"),
    ("mayo", "may"),
    ("junio", "jun", "june"),
    ("julio", "jul", "july"),
    ("agosto", "ago", "august", "aug"),
    ("septiembre", "setiembre", "sep", "sept", "set", "september"),
    ("octubre", "oct", "october"),
    ("noviembre", "nov", "november"),
    ("diciembre", "dic", "december", "dec"),
)
_MONTHS = {name: number for number, names in enumerate(_MONTH_GROUPS, 1) for name in names}


@dataclass(frozen=True)
class _DateParts:
    year: int
    month: int
    day: int | None


def _integer(value: object) -> int | None:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        try:
            integer = int(value)
            return integer if value == integer else None
        except (ValueError, OverflowError, TypeError):
            return None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def _month(value: object) -> int | None:
    numeric = _integer(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        name = "".join(character for character in unicodedata.normalize("NFKD", value.casefold())
                       if not unicodedata.combining(character)).strip().rstrip(".")
        return _MONTHS.get(name)
    return None


def _date_parts(value: object, policy: DateColumnPolicy) -> _DateParts | None:
    parsed = parse_explicit_date(value, policy.text_formats)
    if parsed.conflict:
        return None
    prepared = parsed.value
    if isinstance(prepared, datetime):
        return _DateParts(prepared.year, prepared.month, prepared.day)
    if isinstance(prepared, date):
        return _DateParts(prepared.year, prepared.month, prepared.day)
    if isinstance(prepared, str):
        try:
            native = datetime.fromisoformat(prepared.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _DateParts(native.year, native.month, native.day)
    return None


def _code_parts(value: object, formats: tuple[str, ...]) -> _DateParts | None:
    """Un código MMYYYY no aporta un día, aunque exista otra fecha completa."""
    numeric = _integer(value) if not isinstance(value, str) else None
    if numeric is not None and numeric < 0:
        return None
    text = str(numeric) if numeric is not None else value.strip() if isinstance(value, str) else ""
    if not text.isdigit():
        return None
    # Numeric cells can have lost one leading zero. Never expand a six-digit
    # month/year code into a full eight-digit day/month/year code.
    if numeric is not None and len(text) == 7 and "%d%m%Y" in formats:
        text = text.zfill(8)
    elif numeric is not None and len(text) == 5 and "%m%Y" in formats:
        text = text.zfill(6)
    for explicit_format in formats:
        if explicit_format == "%d%m%Y" and len(text) == 8:
            day, month, year = int(text[:2]), int(text[2:4]), int(text[4:])
        elif explicit_format == "%m%Y" and len(text) == 6:
            day, month, year = None, int(text[:2]), int(text[2:])
        else:
            continue
        try:
            date(year, month, day if day is not None else 1)
        except ValueError:
            continue
        return _DateParts(year, month, day)
    return None


def _same_date(left: _DateParts, right: _DateParts) -> bool:
    return (left.year == right.year and left.month == right.month
            and (left.day is None or right.day is None or left.day == right.day))


def _address(row: int, column: int) -> str:
    return f"{get_column_letter(column)}{row}"


def _original_date_evidence(sheet: SheetData, row: int, column: int) -> str:
    metadata = cell_metadata(sheet, row, column)
    value = trace_value(sheet.frame.iat[row - 1, column - 1])
    evidence = f"{sheet.name}!{_address(row, column)}={value!r} ({metadata.source_kind}, formato={metadata.number_format!r}"
    if metadata.raw_value is not None:
        evidence += f", valor XML={metadata.raw_value!r}"
    return evidence + ")"


def date_consistency_findings(
    sheets: list[SheetData],
    policies: PreparationPolicies,
    associations: list[AssociationDiagnostic],
    *,
    temporal_dates: list[TemporalDateTreatment] | None = None,
) -> list[TabularFinding]:
    """Inspecciona toda la fuente sin modificar fechas, componentes ni filas."""
    sources = {sheet.name: sheet for sheet in sheets}
    findings: list[TabularFinding] = []
    seen = set()
    # Each policy explicitly declares which original columns contain dates.
    # Field names align corresponding dates across an already exact association.
    fields: dict[str, dict[str, list[tuple[int, DateColumnPolicy]]]] = {}
    comparable_dates: dict[tuple[str, int, int], _DateParts] = {}
    selected_dates = {
        (item.sheet_name, item.source_row, coordinate_to_tuple(item.date_cell)[1]): item
        for item in temporal_dates or ()
    }
    for policy in policies.date_columns:
        sheet = sources.get(policy.sheet_name)
        if sheet is None:
            continue
        if not headers_match(sheet, policy.header_row, policy.expected_headers):
            key = ("date_consistency_policy_mismatch", sheet.name, policy.rule_id)
            if key not in seen:
                seen.add(key)
                findings.append(TabularFinding(
                    code=key[0], sheet_name=sheet.name,
                    message=f"Los encabezados originales no corroboran la política de consistencia de fechas {policy.rule_id}; se conservan los originales.",
                    requires_review=True,
                ))
            continue
        for column in policy.column_indices:
            label = policy.expected_headers[column].strip().casefold()
            fields.setdefault(sheet.name, {}).setdefault(label, []).append((column, policy))
            for row in range(policy.header_row + 1, len(sheet.frame) + 1):
                selected = selected_dates.get((sheet.name, row, column))
                if policy.temporal_priority is not None and selected is not None:
                    if selected.normalized_date is not None:
                        chosen = date.fromisoformat(selected.normalized_date)
                        comparable_dates[(sheet.name, row, column)] = _DateParts(chosen.year, chosen.month, chosen.day)
                    # The selector has already retained each original and
                    # localized its resolved or unresolved discrepancy.
                    continue
                value = sheet.frame.iat[row - 1, column - 1]
                if strict_missing(value):
                    continue
                parts = _date_parts(value, policy)
                if parts is None:
                    # The preparation parser reports unresolved date values;
                    # they cannot substantiate a contradiction by themselves.
                    continue
                comparable = parts
                discrepancies = []
                for component, related_column in policy.related_date_columns.items():
                    related = sheet.frame.iat[row - 1, related_column - 1]
                    if strict_missing(related) or isinstance(related, str) and not related.strip():
                        continue
                    label_evidence = (f"{_address(row, related_column)} "
                                      f"{policy.expected_headers[related_column]}={trace_value(related)!r}")
                    if component == "date_code":
                        if not policy.related_date_code_formats:
                            continue
                        code = _code_parts(related, policy.related_date_code_formats)
                        if code is None:
                            discrepancies.append(label_evidence + " no coincide con los formatos de código autorizados")
                        elif not _same_date(parts, code):
                            discrepancies.append(label_evidence + " difiere de la fecha original")
                        elif code.day is None:
                            comparable = _DateParts(parts.year, parts.month, None)
                    else:
                        actual = _month(related) if component == "month" else _integer(related)
                        expected = getattr(parts, component)
                        if actual != expected:
                            discrepancies.append(label_evidence + " difiere del componente de la fecha original")
                comparable_dates[(sheet.name, row, column)] = comparable
                key = ("date_components_conflict", sheet.name, row, column)
                if discrepancies and key not in seen:
                    seen.add(key)
                    findings.append(TabularFinding(
                        code=key[0], sheet_name=sheet.name, cell=_address(row, column),
                        message=(f"Contradicción interna: {_original_date_evidence(sheet, row, column)}; "
                                 + "; ".join(discrepancies)
                                 + ". Se conservan la fecha y los componentes originales sin invertir día y mes."),
                        requires_review=True,
                    ))
    for association in associations:
        if association.status != "matched" or len(association.target_rows) != 1:
            continue
        source = sources.get(association.source_sheet)
        target = sources.get(association.target_sheet)
        if source is None or target is None:
            continue
        target_row = association.target_rows[0]
        source_labels = fields.get(source.name, {})
        target_labels = fields.get(target.name, {})
        source_columns = {column for group in source_labels.values() for column, _ in group}
        target_columns = {column for group in target_labels.values() for column, _ in group}
        for label, source_fields in source_labels.items():
            target_fields = target_labels.get(label, ())
            if not target_fields and len(source_columns) == len(target_columns) == 1:
                # A single declared date on each side needs no guessed header
                # alias; both columns were already corroborated by their policy.
                target_fields = next(iter(target_labels.values()))
            for source_column, source_policy in source_fields:
                left = comparable_dates.get((source.name, association.source_row, source_column))
                if left is None:
                    continue
                for target_column, target_policy in target_fields:
                    right = comparable_dates.get((target.name, target_row, target_column))
                    key = ("linked_date_conflict", source.name, association.source_row, source_column,
                           target.name, target_row, target_column)
                    if right is None or key in seen:
                        continue
                    left_selection = selected_dates.get((source.name, association.source_row, source_column))
                    right_selection = selected_dates.get((target.name, target_row, target_column))
                    resolved_by_priority = _same_date(left, right) and (left_selection is not None or right_selection is not None)
                    if _same_date(left, right):
                        original_left = _date_parts(source.frame.iat[association.source_row - 1, source_column - 1], source_policy)
                        original_right = _date_parts(target.frame.iat[target_row - 1, target_column - 1], target_policy)
                        if (not resolved_by_priority or original_left is None or original_right is None
                                or _same_date(original_left, original_right)):
                            continue
                    seen.add(key)
                    selected_evidence = ""
                    for chosen_sheet, selected in ((source, left_selection), (target, right_selection)):
                        if selected is not None:
                            selected_evidence += (f" Fecha normalizada de {chosen_sheet.name}: {selected.normalized_date} "
                                                  f"desde {selected.source_cell}, por {selected.rule_id} v{selected.version}.")
                    findings.append(TabularFinding(
                        code="linked_date_discrepancy_resolved" if resolved_by_priority else key[0],
                        sheet_name=source.name, cell=_address(association.source_row, source_column),
                        message=(("Diferencia original entre registros asociados resuelta por la prioridad temporal del usuario: "
                                  if resolved_by_priority else "Fechas elegidas de registros asociados incompatibles: ")
                                 + f"{_original_date_evidence(source, association.source_row, source_column)} frente a "
                                 f"{_original_date_evidence(target, target_row, target_column)}, "
                                 f"mediante {source.name}!{association.source_cell}={association.source_value!r}. "
                                 + selected_evidence
                                 + " Se conservan ambos registros, las fechas originales, identificadores y enlaces; no se invierte una fecha nativa."
                                 + (" La prioridad no confirma científicamente la veracidad del código." if resolved_by_priority else "")),
                        requires_review=not resolved_by_priority,
                    ))
    return findings
