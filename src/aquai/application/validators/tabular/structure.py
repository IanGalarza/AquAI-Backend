"""Validate structural roles, measurement labels and record identity."""

import re
import unicodedata
from collections import Counter
from numbers import Number

from aquai.application.contracts.tabular.policies import PreparationPolicies, RecordIdentityClarification
from aquai.application.contracts.tabular.recipe import PreparationRecipe, RenameColumns, SheetRecipe
from aquai.application.contracts.tabular.results import AuxiliaryRowTreatment, ExecutionResult, TabularFinding
from aquai.application.contracts.tabular.structure import SheetSummary, StructuralRegion, StructuralSummary
from aquai.application.validators.tabular.preparation import AuxiliaryRows, SourceHeaders, cell_position, policy_headers_match


def validate_record_overlaps(
    summary: StructuralSummary, execution: ExecutionResult,
) -> list[TabularFinding]:
    """Expone relaciones de contenido sin anexar ni deduplicar observaciones."""
    findings = []
    source_relations = Counter(relation.model_dump_json() for relation in summary.record_overlaps)
    executed_relations = Counter(relation.model_dump_json() for relation in execution.record_overlaps)
    if source_relations != executed_relations:
        findings.append(TabularFinding(
            code="record_overlap_evidence_mismatch",
            message="La ejecución no conserva las relaciones de registros observadas en la fuente.",
            severity="error",
        ))
    for relation in summary.record_overlaps:
        if not relation.matches:
            if relation.ambiguous_repeated_rows:
                findings.append(TabularFinding(
                    code="source_repeated_record_content",
                    message=(
                        f"No hay registros compartidos entre '{relation.left_sheet}' y "
                        f"'{relation.right_sheet}', pero se informa contenido repetido dentro de "
                        "al menos uno de los orígenes. Conservar registros y procedencias; "
                        "confirmar identidad antes de anexar o deduplicar."
                    ),
                ))
            continue
        rows = [match.left_row for match in relation.matches] + relation.left_only_rows
        findings.append(TabularFinding(
            code="source_record_overlap",
            message=(
                f"Hay {len(relation.matches)} registros de contenido exacto compartido con "
                f"'{relation.right_sheet}'; {len(relation.left_only_rows)} registros exclusivos "
                f"de esta hoja y {len(relation.right_only_rows)} de la otra. Conservar ambas "
                "procedencias sin sumar coincidencias como observaciones independientes."
                + (" La fuente repite contenido dentro de una hoja: confirmar identidad antes de deduplicar."
                   if relation.ambiguous_repeated_rows else "")
            ),
            sheet_name=relation.left_sheet,
            range=(
                f"{column_label(relation.left_start_column)}{min(rows)}:"
                f"{column_label(relation.left_end_column)}{max(rows)}"
            ),
            requires_review=relation.ambiguous_repeated_rows,
        ))
    return findings


def validate_record_identity_notes(
    summary: StructuralSummary,
    clarifications: tuple[RecordIdentityClarification, ...] = (),
) -> list[TabularFinding]:
    """Las dudas explícitas de la fuente se conservan como evidencia para revisión."""
    findings = []
    seen = set()
    uncertainty = re.compile(
        r"\b(?:no (?:se(?: sabe)?|sabemos) si|posibles?|possible|potential)\b.{0,100}"
        r"\b(?:repetid\w*|duplicad\w*|duplicates?|repeated)\b", re.IGNORECASE,
    )
    for sheet in summary.sheets:
        samples = [(sheet.sample, sheet.column_indices)] + [
            (region.sample, region.column_indices) for region in sheet.regions
        ]
        for sample, positions in samples:
            if not positions:
                continue
            for row in sample:
                for column, value in zip(positions, row.values):
                    if not isinstance(value, str):
                        continue
                    normalized = _normalized_note(value)
                    key = (sheet.name, row.row_number, column)
                    if key in seen or not uncertainty.search(normalized):
                        continue
                    seen.add(key)
                    cell = f"{column_label(column)}{row.row_number}"
                    clarification = next((item for item in clarifications
                        if _normalized_note(item.original_note) == normalized
                        and (item.sheet_name is None or item.sheet_name == sheet.name)
                        and (item.cell is None or item.cell == cell)), None)
                    if clarification is not None:
                        findings.append(TabularFinding(
                            code="source_record_identity_clarified",
                            message=(
                                f"La nota original {value!r} se conserva como antecedente. "
                                f"Aclarada por {clarification.confirmed_by}: {clarification.statement} "
                                f"Regla {clarification.rule_id}; fuente: {clarification.source_reference}."
                            ),
                            sheet_name=sheet.name,
                            cell=cell,
                            requires_review=False,
                        ))
                        continue
                    findings.append(TabularFinding(
                        code="source_record_identity_uncertainty",
                        message=(
                            "La nota original declara posibles repeticiones. Se conservan los registros "
                            "y sus notas; confirmar su identidad antes de anexar o deduplicar."
                        ),
                        sheet_name=sheet.name,
                        cell=cell,
                    ))
    return findings


def _normalized_note(value: str) -> str:
    normalized = "".join(
        character for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(normalized.split())


def validate_regions(
    summary: StructuralSummary, recipe: PreparationRecipe, empty_names: set[str] | None = None,
    preparation_policies: PreparationPolicies | None = None,
    auxiliary_rows: AuxiliaryRows | None = None,
    source_headers: SourceHeaders | None = None,
) -> list[TabularFinding]:
    findings: list[TabularFinding] = []
    sources = {sheet.name: sheet for sheet in summary.sheets}
    expected_roles = {
        "total": "total", "legend": "legend", "derived": "derived",
        "summary": "summary", "repeated_header": "header", "header": "header",
    }
    for block in recipe.sheets:
        if block.sheet_name in (empty_names or set()):
            continue
        source = sources.get(block.sheet_name)
        if source is None:
            continue
        if (
            block.data_start_row > source.rows
            or block.data_end_row > source.rows
            or block.data_start_column > source.columns
            or block.data_end_column > source.columns
            or any(row > source.rows for row in block.header_rows)
        ):
            findings.append(TabularFinding(
                code="block_out_of_bounds",
                message="El bloque declara coordenadas fuera de la hoja de origen.",
                sheet_name=block.sheet_name,
                block_id=block.block_id,
                range=block_range(block, source),
                severity="error",
            ))
        if block.role == "unknown":
            findings.append(TabularFinding(
                code="unresolved_block_role",
                message="El bloque conserva un rol sin resolver.",
                sheet_name=block.sheet_name,
                block_id=block.block_id,
                range=block_range(block, source),
            ))
        for region in source.regions:
            if not _intersects(block, source, region):
                continue
            expected_role = expected_roles.get(region.kind)
            if expected_role is not None and block.role != expected_role:
                if region.kind == "derived" and _derived_region_supported(
                    source, region, block, preparation_policies, auxiliary_rows, source_headers,
                ):
                    continue
                if _candidate_contradicted(source, region, block):
                    continue
                findings.append(TabularFinding(
                    code="structural_role_conflict",
                    message=(
                        f"El bloque de rol '{block.role}' incluye evidencia candidata de '{region.kind}': "
                        f"{region.reason} Requiere contrastar sus muestras y separar funciones distintas."
                    ),
                    sheet_name=block.sheet_name,
                    block_id=block.block_id,
                    region_id=region.region_id,
                    range=block_range(block, source),
                ))
                continue
            match (region.kind, block.role):
                case ("unknown", "unknown" | "observations"):
                    findings.append(TabularFinding(
                        code="unresolved_structural_region",
                        message="Una región de estructura incierta requiere confirmar su tratamiento antes de usarla como observaciones.",
                        sheet_name=block.sheet_name,
                        block_id=block.block_id,
                        region_id=region.region_id,
                        range=block_range(block, source),
                    ))
                case ("observations", "derived" | "summary"):
                    findings.append(TabularFinding(
                        code="observations_in_nonobservational_block",
                        message=(
                            f"El bloque de rol '{block.role}' incluye una región compatible con "
                            "observaciones; requiere verificar la separación de magnitudes."
                        ),
                        sheet_name=block.sheet_name,
                        block_id=block.block_id,
                        region_id=region.region_id,
                        range=block_range(block, source),
                    ))
    return findings


def _derived_region_supported(
    source: SheetSummary, region: StructuralRegion, block: SheetRecipe,
    policies: PreparationPolicies | None, auxiliary_rows: AuxiliaryRows | None,
    source_headers: SourceHeaders | None,
) -> bool:
    match block.role:
        case "legend":
            treatments = (auxiliary_rows or {}).get((block.sheet_name, block.block_id), [])
            return _derived_auxiliary_legend(region, block, treatments)
        case "observations" if policies is not None:
            identity_columns = {
                column for policy in policies.observation_rows
                if policy.sheet_name == source.name and policy_headers_match(
                    source, policy.expected_headers, source_headers, policy.header_row,
                )
                for column in policy.auxiliary_column_indices
            }
            represented_columns = set(range(block.data_start_column, block.data_end_column + 1))
            region_columns = set(range(region.start_column, region.end_column + 1))
            return (
                region_columns.issubset(identity_columns)
                and identity_columns.issubset(represented_columns)
                and bool(represented_columns - identity_columns)
            )
        case _:
            return False


def _derived_auxiliary_legend(
    region: StructuralRegion, block: SheetRecipe, treatments: list[AuxiliaryRowTreatment],
) -> bool:
    """Sólo una intersección enteramente probada como fórmula auxiliar puede ser leyenda."""
    first_row, last_row = max(block.data_start_row, region.start_row), min(block.data_end_row, region.end_row)
    first_column = max(block.data_start_column, region.start_column)
    last_column = min(block.data_end_column, region.end_column)
    proofs = {treatment.row_number: treatment for treatment in treatments
              if not treatment.removed_from_observations}
    for row in range(first_row, last_row + 1):
        treatment = proofs.get(row)
        if treatment is None:
            return False
        positions = {cell_position(cell.cell) for cell in treatment.auxiliary_cells}
        if not all((row, column) in positions for column in range(first_column, last_column + 1)):
            return False
    return first_row <= last_row and first_column <= last_column


def _candidate_contradicted(
    source: SheetSummary, region: StructuralRegion, block: SheetRecipe,
) -> bool:
    """Contrasta candidatos con muestras; una etiqueta heurística no impone una partición."""
    positions = region.column_indices or list(range(region.start_column, region.end_column + 1))
    values = [
        value for row in region.sample
        if block.data_start_row <= row.row_number <= block.data_end_row
        for column, value in zip(positions, row.values)
        if block.data_start_column <= column <= block.data_end_column and value is not None
    ]
    labels = [value.strip().casefold() for value in values if isinstance(value, str)]
    match region.kind:
        case "total" if values:
            return not any(re.fullmatch(r"(?:grand )?totals?|totales|total general|suma", label) for label in labels)
        case "header" | "repeated_header" if values:
            # Una cabecera candidata numérica no desplaza registros de una tabla sin cabecera.
            return all(_numeric(value) for value in values)
        case "legend" if values:
            # La baja densidad sola no convierte mediciones numéricas dispersas en leyenda.
            return all(_numeric(value) for value in values)
        case "derived" if "fórmulas" in region.reason.casefold():
            return source.formula_count == 0
        case _:
            return False


def _numeric(value: object) -> bool:
    if isinstance(value, Number) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
            return True
        except ValueError:
            pass
    return False


def _intersects(block: SheetRecipe, source: SheetSummary, region: StructuralRegion) -> bool:
    return (
        max(block.data_start_row, region.start_row) <= min(block.data_end_row, region.end_row)
        and max(block.data_start_column, region.start_column)
        <= min(block.data_end_column, region.end_column)
    )


def block_range(block: SheetRecipe, source: SheetSummary) -> str:
    return (
        f"{column_label(block.data_start_column)}{block.data_start_row}:"
        f"{column_label(block.data_end_column)}{block.data_end_row}"
    )


def column_label(column: int) -> str:
    label = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        label = chr(65 + remainder) + label
    return label


def validate_unit_labels(
    summary: StructuralSummary, recipe: PreparationRecipe,
) -> list[TabularFinding]:
    findings: list[TabularFinding] = []
    sources = {sheet.name: sheet for sheet in summary.sheets}
    for block in recipe.sheets:
        source = sources.get(block.sheet_name)
        if source is None:
            continue
        for transformation in block.transformations:
            if not isinstance(transformation, RenameColumns):
                continue
            for original, renamed in transformation.mapping.items():
                original_units = _unit_labels(original)
                renamed_units = _unit_labels(renamed)
                if original_units != renamed_units:
                    findings.append(TabularFinding(
                        code="unit_label_changed",
                        message=(
                            f"El renombrado de '{original}' a '{renamed}' cambia una etiqueta "
                            "explícita de unidad o magnitud; requiere verificar su significado."
                        ),
                        sheet_name=block.sheet_name,
                        block_id=block.block_id,
                        range=block_range(block, source),
                    ))
    return findings


def _unit_labels(label: str) -> set[str]:
    normalized = "".join(
        character for character in unicodedata.normalize("NFKD", label)
        if not unicodedata.combining(character)
    )
    count_labels = {"celulas", "cells", "colonias", "colonies", "individuos", "individuals"}
    labels = set()
    for match in re.finditer(r"[^\W\d_]+(?:[23²³])?\s*/\s*[^\W\d_]+(?:[23²³])?", normalized):
        numerator, denominator = re.sub(r"\s+", "", match.group()).split("/")
        if sum(character.isalpha() for character in denominator) > 3:
            continue
        if numerator.casefold() in count_labels:
            numerator = numerator.casefold()
        labels.add(f"{numerator}/{denominator}")
    if "%" in normalized or re.search(r"\b(?:percent|percentage|porcentaje)\b", normalized, re.IGNORECASE):
        labels.add("%")
    return labels
