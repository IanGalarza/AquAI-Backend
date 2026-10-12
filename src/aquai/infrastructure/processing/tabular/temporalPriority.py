"""Selección temporal explícita que conserva los campos originales del registro."""

import re
from datetime import date

from openpyxl.utils.cell import get_column_letter

from aquai.application.dtos.tabular import SheetRecipe, TabularFinding
from aquai.application.dtos.tabularPolicies import (
    DateColumnPolicy,
    PreparationPolicies,
    TemporalDateTreatment,
    TemporalDiscrepancy,
    TemporalSourceEvidence,
)
from aquai.infrastructure.processing.tabular.dateConsistency import (
    _code_parts,
    _date_parts,
    _integer,
    _month,
)
from aquai.infrastructure.processing.tabular.normalization import trace_value
from aquai.infrastructure.processing.tabular.preparation import cell_metadata, headers_match, strict_missing
from aquai.infrastructure.processing.tabular.reader import SheetData


def _missing(value: object) -> bool:
    return strict_missing(value) or isinstance(value, str) and not value.strip()


def _full_code(value: object, metadata, policy: DateColumnPolicy):
    """Sólo interpreta ocho dígitos; no fabrica un día para un código mes/año."""
    if _missing(value):
        return None, "missing"
    if isinstance(value, str):
        text = value.strip()
        interpretation = "exact_8_digits"
    else:
        numeric = _integer(value)
        if numeric is None or numeric < 0:
            return None, "invalid"
        text = str(numeric)
        interpretation = "exact_8_digits"
        # A numeric value has no literal leading zeros. An explicit eight-zero
        # display format is evidence of its eight-digit representation; neither
        # arbitrary seven-digit text nor a month/year code is padded.
        if (len(text) == 7 and metadata.source_kind == "native_number"
                and metadata.number_format == "00000000"):
            text = text.zfill(8)
            interpretation = "numeric_leading_zero"
    if not re.fullmatch(r"[0-9]{8}", text):
        partial = _code_parts(value, ("%m%Y",)) if "%m%Y" in policy.related_date_code_formats else None
        return None, "partial" if partial is not None else "invalid"
    try:
        selected = date(int(text[4:]), int(text[2:4]), int(text[:2]))
    except ValueError:
        return None, "invalid"
    return selected, interpretation


def _source_evidence(sheet: SheetData, row: int, column: int) -> TemporalSourceEvidence:
    metadata = cell_metadata(sheet, row, column)
    return TemporalSourceEvidence(
        cell=f"{get_column_letter(column)}{row}",
        original_value=trace_value(sheet.frame.iat[row - 1, column - 1]),
        source_kind=metadata.source_kind, number_format=metadata.number_format,
        raw_value=metadata.raw_value, excel_epoch=metadata.excel_epoch,
    )


def select_temporal_dates(
    sheet: SheetData,
    recipe: SheetRecipe,
    policies: PreparationPolicies,
    source_rows: list[int],
) -> tuple[list[TemporalDateTreatment], list[TabularFinding]]:
    """Produce un resultado lateral para cada observación efectivamente retenida."""
    treatments, findings = [], []
    if recipe.role != "observations":
        return treatments, findings
    seen = set()
    for policy in policies.date_columns:
        if policy.sheet_name != sheet.name or policy.temporal_priority is None:
            continue
        if not headers_match(sheet, policy.header_row, policy.expected_headers):
            # The generic preparation/consistency guards report this mismatch.
            continue
        for column in policy.column_indices:
            if not recipe.data_start_column <= column <= recipe.data_end_column:
                continue
            for row in source_rows:
                if (row, column) in seen:
                    continue
                seen.add((row, column))
                columns = {"date": column, **policy.related_date_columns}
                evidence = {field: _source_evidence(sheet, row, position)
                            for field, position in columns.items()}
                original = {field: item.original_value for field, item in evidence.items()}
                date_original = sheet.frame.iat[row - 1, column - 1]
                code_column = columns["date_code"]
                code_original = sheet.frame.iat[row - 1, code_column - 1]
                code_date, interpretation = _full_code(
                    code_original, cell_metadata(sheet, row, code_column), policy,
                )
                date_parts = _date_parts(date_original, policy)
                date_date = (date(date_parts.year, date_parts.month, date_parts.day)
                             if date_parts is not None else None)
                # An uncached formula cannot substantiate a candidate. A cached
                # formula retains its expression elsewhere and its cache here.
                chosen = code_date if code_date is not None else date_date
                source_field = "date_code" if code_date is not None else "date" if date_date is not None else None
                discrepancies = []

                def discrepancy(code, message, fields):
                    discrepancies.append(TemporalDiscrepancy(
                        code=code, message=message, fields=fields,
                        resolved_by_policy=chosen is not None,
                    ))

                if interpretation in {"invalid", "partial"}:
                    reason = ("sólo representa mes/año y no determina un día"
                              if interpretation == "partial" else "no representa una fecha DDMMAAAA completa y válida")
                    discrepancy("date_code_" + interpretation,
                                f"{evidence['date_code'].cell}={original['date_code']!r} {reason}; se evalúa Fecha como alternativa.",
                                ("date_code",))
                    if interpretation == "partial" and chosen is not None:
                        partial = _code_parts(code_original, ("%m%Y",))
                        if partial is not None and (partial.year, partial.month) != (chosen.year, chosen.month):
                            discrepancy("date_code_partial_disagrees",
                                        f"{evidence['date_code'].cell}={original['date_code']!r} representa mes/año {partial.month:02d}/{partial.year:04d}, distinto de {chosen.isoformat()} seleccionado desde Fecha; el código parcial no aporta un día ni reemplaza esa fecha.",
                                        ("date_code", "date"))
                if date_date is None and not _missing(date_original):
                    discrepancy("date_field_invalid",
                                f"{evidence['date'].cell}={original['date']!r} no representa una fecha válida con los formatos autorizados.",
                                ("date",))
                elif chosen is not None and date_date is not None and chosen != date_date:
                    discrepancy("date_field_disagrees",
                                f"{evidence['date'].cell}={original['date']!r} difiere de {chosen.isoformat()}, seleccionado desde {evidence[source_field].cell}.",
                                ("date", "date_code"))
                for component in ("month", "year", "day"):
                    if component not in columns or _missing(sheet.frame.iat[row - 1, columns[component] - 1]):
                        continue
                    value = sheet.frame.iat[row - 1, columns[component] - 1]
                    actual = _month(value) if component == "month" else _integer(value)
                    if chosen is not None and actual != getattr(chosen, component):
                        discrepancy(component + "_disagrees",
                                    f"{evidence[component].cell}={original[component]!r} difiere del componente de {chosen.isoformat()}; es un campo de contraste.",
                                    (component, source_field))
                if chosen is None:
                    discrepancy("no_valid_temporal_date",
                                "Ni Código Fecha ni Fecha determinan una fecha completa y válida; Mes y Año no sustituyen la fecha.",
                                ("date_code", "date"))
                treatment = TemporalDateTreatment(
                    sheet_name=sheet.name, block_id=recipe.block_id, source_row=row,
                    date_cell=evidence["date"].cell,
                    normalized_date=chosen.isoformat() if chosen is not None else None,
                    source_field=source_field,
                    source_cell=evidence[source_field].cell if source_field is not None else None,
                    rule_id=policy.rule_id, version=policy.version, origin=policy.origin,
                    original_values=original, original_evidence=evidence,
                    discrepancies=discrepancies,
                    resolution="user_priority" if chosen is not None else "review_required",
                    code_interpretation=interpretation,
                )
                treatments.append(treatment)
                if discrepancies:
                    prefix = (f"Fecha normalizada {chosen.isoformat()} desde {treatment.source_cell}, por {policy.rule_id} v{policy.version}. "
                              if chosen is not None else "Fecha temporal no resuelta. ")
                    findings.append(TabularFinding(
                        code="temporal_priority_discrepancies_resolved" if chosen is not None else "temporal_date_unresolved",
                        sheet_name=sheet.name, block_id=recipe.block_id, cell=treatment.date_cell,
                        message=(prefix + "; ".join(item.message for item in discrepancies)
                                 + " Se conservan todos los originales, identificadores y enlaces. La prioridad es una política del usuario, no una confirmación científica."),
                        requires_review=chosen is None,
                    ))
    return treatments, findings
