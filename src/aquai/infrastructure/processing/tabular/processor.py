"""Inspección y ejecución determinística de recetas tabulares."""

import math
import re
from collections import Counter, defaultdict, deque
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from numbers import Number

import pandas as pd
import numpy as np
from openpyxl.formula.tokenizer import Tokenizer, TokenizerError
from openpyxl.utils.cell import column_index_from_string, get_column_letter
from openpyxl.worksheet.cell_range import CellRange

from aquai.application.dtos.tabular import (
    ConvertType,
    DropEmptyColumns,
    DropEmptyRows,
    EmptySheetTreatment,
    ExecutionResult,
    FormulaSample,
    PreparationRecipe,
    RecordOverlap,
    RenameColumns,
    SampleRow,
    SheetExecution,
    SheetContentEvidence,
    SheetRecipe,
    SheetSummary,
    SheetCoverage,
    StructuralRegion,
    TabularFinding,
    StripStrings,
    StructuralSummary,
    SourceRowMatch,
)
from aquai.infrastructure.processing.tabular.reader import SheetData, TableReader
from aquai.application.dtos.tabular_policies import PreparationPolicies, SourceHeaderEvidence
from aquai.infrastructure.processing.tabular.date_consistency import date_consistency_findings
from aquai.infrastructure.processing.tabular.temporal_priority import select_temporal_dates
from aquai.infrastructure.processing.tabular.normalization import trace_value
from aquai.infrastructure.processing.tabular.preparation import association_diagnostics, headers_match, prepare_frame, strict_missing


def is_empty(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def json_value(value: object) -> object:
    if is_empty(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value[:200] if isinstance(value, str) else value
    return str(value)[:200]


def infer_type(series: pd.Series) -> str:
    values = series[~series.map(is_empty)]
    if values.empty:
        return "empty"
    inferred = pd.api.types.infer_dtype(values, skipna=True)
    if inferred == "string":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            return "integer" if (numeric % 1 == 0).all() else "floating"
    return inferred


class PandasTabularProcessor:
    def __init__(
        self,
        max_cells: int = 2_000_000,
        max_sheets: int = 30,
        max_expanded_bytes: int = 100 * 1024 * 1024,
        sample_rows: int = 12,
    ) -> None:
        self.reader = TableReader(max_cells, max_sheets, max_expanded_bytes, sample_rows)
        self.sample_rows = sample_rows

    def inspect(self, file_name: str, content: bytes) -> StructuralSummary:
        table = self.reader.read(file_name, content)
        summaries = [self._summarize(sheet) for sheet in table.sheets]
        return StructuralSummary(
            file_name=file_name,
            file_type=table.file_type,
            file_size_bytes=len(content),
            sheets=summaries,
            record_overlaps=self._record_overlaps(table.sheets, summaries),
            warnings=table.warnings,
            csv_encoding=table.csv_encoding,
            csv_delimiter=table.csv_delimiter,
        )

    def execute(
        self,
        file_name: str,
        content: bytes,
        recipe: PreparationRecipe,
        summary: StructuralSummary,
        *,
        policies: PreparationPolicies | None = None,
    ) -> ExecutionResult:
        table = self.reader.read(file_name, content)
        sheets = {sheet.name: sheet for sheet in table.sheets}
        selected = {sheet.sheet_name for sheet in recipe.sheets}
        content_evidence = {
            sheet.name: self._content_evidence(sheet) for sheet in table.sheets
        }
        empty_sheets = [
            EmptySheetTreatment(
                sheet_name=sheet.name,
                evidence=content_evidence[sheet.name],
                skipped_block_ids=[
                    block.block_id for block in recipe.sheets if block.sheet_name == sheet.name
                ],
            )
            for sheet in table.sheets if content_evidence[sheet.name].is_empty
        ]
        executions = []
        normalizations, auxiliary_rows, temporal_dates = [], [], []
        enabled_policies = policies if policies is not None and policies.enabled else None
        source_sha256 = sha256(content).hexdigest() if enabled_policies is not None else None
        for sheet_recipe in recipe.sheets:
            sheet = sheets.get(sheet_recipe.sheet_name)
            if sheet is None:
                executions.append(SheetExecution(
                    sheet_name=sheet_recipe.sheet_name,
                    block_id=sheet_recipe.block_id,
                    role=sheet_recipe.role,
                    data_start_row=sheet_recipe.data_start_row,
                    data_end_row=sheet_recipe.data_end_row,
                    data_start_column=sheet_recipe.data_start_column,
                    data_end_column=sheet_recipe.data_end_column,
                    source_rows=0,
                    source_columns=0,
                    rows_before=0,
                    rows_after=0,
                    columns_before=[],
                    columns_after=[],
                    errors=["La hoja indicada por la receta no existe en el archivo."],
                ))
            elif not content_evidence[sheet.name].is_empty:
                if enabled_policies is None:
                    executions.append(self._execute_sheet(sheet, sheet_recipe))
                else:
                    executions.append(self._execute_sheet(
                        sheet, sheet_recipe, policies=enabled_policies, source_sha256=source_sha256,
                        normalizations=normalizations, auxiliary_rows=auxiliary_rows,
                        temporal_dates=temporal_dates,
                    ))
        associations, association_findings = (association_diagnostics(table.sheets, enabled_policies)
                                               if enabled_policies is not None else ([], []))
        consistency_findings = (date_consistency_findings(table.sheets, enabled_policies, associations, temporal_dates=temporal_dates)
                                if enabled_policies is not None else [])
        identity_findings = (self._record_identity_findings(table.sheets, enabled_policies, executions)
                             if enabled_policies is not None else [])
        return ExecutionResult(
            sheets=executions,
            skipped_sheets=[name for name in sheets if name not in selected],
            coverage=[self._coverage(sheet, recipe) for sheet in table.sheets],
            findings=([finding for execution in executions for finding in execution.findings]
                      + association_findings + consistency_findings + identity_findings),
            record_overlaps=self._record_overlaps(table.sheets, [
                item for item in summary.sheets
                if item.name in content_evidence and not content_evidence[item.name].is_empty
            ]),
            sheet_content_evidence=list(content_evidence.values()),
            empty_sheets=empty_sheets,
            normalizations=normalizations,
            auxiliary_rows=auxiliary_rows,
            temporal_dates=temporal_dates,
            associations=associations,
            source_headers=(self._source_headers(table.sheets, enabled_policies)
                            if enabled_policies is not None else []),
        )

    def _source_headers(
        self, sheets: list[SheetData], policies: PreparationPolicies,
    ) -> list[SourceHeaderEvidence]:
        """Prueba filas completas con posiciones originales, sin compactar encabezados."""
        sources = {sheet.name: sheet for sheet in sheets}
        selected = {(policy.sheet_name, policy.header_row) for policy in (
            *policies.date_columns, *policies.number_columns, *policies.confirmed_cells,
            *policies.observation_rows,
        )}
        for policy in policies.associations:
            selected.update(((policy.source_sheet, policy.header_row),
                             (policy.target_sheet, policy.header_row)))
        evidence = []
        for name, row in sorted(selected):
            sheet = sources.get(name)
            if sheet is None or not 1 <= row <= len(sheet.frame) or not len(sheet.frame.columns):
                continue
            evidence.append(SourceHeaderEvidence(
                sheet_name=name, row_number=row, inspected_columns=len(sheet.frame.columns),
                values={column: (None if strict_missing(sheet.frame.iat[row - 1, column - 1])
                                 else trace_value(sheet.frame.iat[row - 1, column - 1]))
                        for column in range(1, len(sheet.frame.columns) + 1)},
                formula_columns=tuple(column for column in range(1, len(sheet.frame.columns) + 1)
                                      if (row, column) in sheet.formula_cells),
            ))
        return evidence

    def _value_key(self, value: object) -> tuple:
        """Compara valores completos y tipos sin recortar textos, códigos ni espacios."""
        if value is None or value is pd.NA or value is pd.NaT:
            return ("missing",)
        if not isinstance(value, str) and bool(pd.isna(value)):
            return ("missing",)
        if hasattr(value, "item"):
            value = value.item()
        return (type(value).__qualname__, value)

    def _cell_key(self, sheet: SheetData, row: int, column: int) -> tuple:
        formula = sheet.formula_cells.get((row, column))
        source_type = "excel_error" if (row, column) in sheet.error_positions else "value"
        if formula is not None:
            return ("formula", formula.expression, source_type, self._value_key(formula.cached_value))
        return ("literal", source_type, self._value_key(sheet.frame.iat[row - 1, column - 1]))

    def _record_overlaps(
        self, sheets: list[SheetData], summaries: list[SheetSummary],
    ) -> list[RecordOverlap]:
        """Relaciona multiconjuntos de filas originales; no fusiona, suma ni elimina datos."""
        sources = {sheet.name: sheet for sheet in sheets}
        compatible = defaultdict(list)
        for summary in summaries:
            sheet = sources.get(summary.name)
            if sheet is None:
                continue
            bands = defaultdict(list)
            for region in summary.regions:
                if region.kind == "observations":
                    bands[(region.start_row, region.end_row)].append(region)
            for (start_row, end_row), regions in bands.items():
                first = min(region.start_column for region in regions)
                last = max(region.end_column for region in regions)
                prior = [region for region in summary.regions
                         if region.kind in {"header", "repeated_header"} and region.end_row < start_row
                         and max(first, region.start_column) <= min(last, region.end_column)]
                if not prior:
                    continue
                header = max(prior, key=lambda region: region.end_row)
                header_rows = list(range(header.start_row, header.end_row + 1))
                header_key = tuple(tuple(self._cell_key(sheet, row, column)
                                        for column in range(first, last + 1)) for row in header_rows)
                compatible[header_key].append((sheet, header_rows, start_row, end_row, first, last))
        overlaps = []
        for bands in compatible.values():
            if len(bands) < 2:
                continue
            keyed_bands = []
            for sheet, headers, start, end, first, last in bands:
                records = [(row, tuple(self._cell_key(sheet, row, column)
                                       for column in range(first, last + 1)))
                           for row in range(start, end + 1)]
                keyed_bands.append((sheet, headers, first, last, records))
            for index, (left, left_headers, left_first, left_last, left_records) in enumerate(keyed_bands):
                for right, right_headers, right_first, right_last, right_records in keyed_bands[index + 1:]:
                    if left.name == right.name:
                        continue
                    right_by_value = defaultdict(deque)
                    for row, key in right_records:
                        right_by_value[key].append(row)
                    matches, left_only = [], []
                    for row, key in left_records:
                        candidates = right_by_value[key]
                        if candidates:
                            matches.append(SourceRowMatch(left_row=row, right_row=candidates.popleft()))
                        else:
                            left_only.append(row)
                    right_only = sorted(row for rows in right_by_value.values() for row in rows)
                    repeated = any(count > 1 for count in Counter(key for _, key in left_records).values()) or any(
                        count > 1 for count in Counter(key for _, key in right_records).values()
                    )
                    overlaps.append(RecordOverlap(
                        left_sheet=left.name, right_sheet=right.name,
                        left_header_rows=left_headers, right_header_rows=right_headers,
                        left_start_column=left_first, left_end_column=left_last,
                        right_start_column=right_first, right_end_column=right_last,
                        left_record_count=len(left_records), right_record_count=len(right_records),
                        matches=matches, left_only_rows=left_only, right_only_rows=right_only,
                        comparison="exact_values_headers_and_formulas",
                        policy="preserve_sources_do_not_add_matches", ambiguous_repeated_rows=repeated,
                    ))
        return overlaps

    def _occupied_cells(self, sheet: SheetData) -> np.ndarray:
        occupied = ~sheet.frame.map(is_empty).to_numpy(dtype=bool)
        for row, column in sheet.formula_cells:
            occupied[row - 1, column - 1] = True
        return occupied

    def _content_evidence(self, sheet: SheetData) -> SheetContentEvidence:
        """Verifica toda la hoja; un literal que contiene espacios es un valor.

        El lector conserva todas las celdas y expresiones, incluso las fórmulas
        sin caché. Ni las muestras ni las dimensiones de formato prueban que
        una hoja esté vacía: se cuentan los valores efectivos de todas las
        celdas y las expresiones completas antes de decidirlo.
        """
        return SheetContentEvidence(
            sheet_name=sheet.name,
            inspected_cells=int(sheet.frame.size),
            value_cells=int(sheet.frame.count().sum()),
            formula_cells=len(sheet.formula_cells),
        )

    def _coverage(self, sheet: SheetData, recipe: PreparationRecipe) -> SheetCoverage:
        occupied = self._occupied_cells(sheet)
        covered = np.zeros(occupied.shape, dtype=bool)
        selected = np.zeros(occupied.shape, dtype=np.uint16)
        for block in recipe.sheets:
            if block.sheet_name != sheet.name:
                continue
            if self._range_errors(sheet, block):
                continue
            row_slice = slice(block.data_start_row - 1, block.data_end_row)
            column_slice = slice(block.data_start_column - 1, block.data_end_column)
            covered[row_slice, column_slice] = True
            selected[row_slice, column_slice] += 1
            for header_row in block.header_rows:
                covered[header_row - 1, column_slice] = True
        uncovered = occupied & ~covered
        return SheetCoverage(
            sheet_name=sheet.name,
            nonempty_cells=int(occupied.sum()),
            covered_nonempty_cells=int((occupied & covered).sum()),
            uncovered_nonempty_cells=int(uncovered.sum()),
            overlapping_nonempty_cells=int((occupied & (selected > 1)).sum()),
            uncovered_sample=[
                f"{get_column_letter(int(column) + 1)}{int(row) + 1}"
                for row, column in np.argwhere(uncovered)[:20]
            ],
        )

    def _sample(
        self, frame: pd.DataFrame, column_positions: list[int] | None = None,
    ) -> list[SampleRow]:
        size = len(frame)
        if not size:
            return []
        count = min(self.sample_rows, size)
        head_count = min(6, count)
        positions = set(range(head_count))
        remaining = count - head_count
        for offset in range(remaining):
            positions.add(head_count + (size - head_count - 1) * (offset + 1) // remaining)
        return [
            SampleRow(
                row_number=int(frame.index[position]) + 1,
                values=[json_value(value) for value in (
                    frame.iloc[position, column_positions]
                    if column_positions is not None else frame.iloc[position, :50]
                )],
            )
            for position in sorted(positions)
        ]

    def _summarize(self, sheet: SheetData) -> SheetSummary:
        frame = sheet.frame
        content_evidence = self._content_evidence(sheet)
        empty = frame.map(is_empty)
        empty_rows = [
            int(index) + 1
            for index in frame.index[empty.all(axis=1)]
            if int(index) + 1 not in sheet.formula_rows
        ]
        empty_columns = [
            int(index) + 1 for index in frame.columns[empty.all(axis=0)]
            if int(index) + 1 not in sheet.formula_columns
        ]
        column_positions = list(range(len(frame.columns)))
        if len(column_positions) > 16:
            populated_columns = [
                int(index) for index in frame.columns
                if int(index) + 1 not in empty_columns or int(index) + 1 in sheet.formula_columns
            ]
            column_positions = sorted(set(column_positions[:12] + populated_columns[-4:]))
        warnings = []
        if content_evidence.is_empty:
            warnings.append("La hoja está vacía: se verificó que no contiene valores ni fórmulas.")
        if len(empty_rows) > 100 or len(empty_columns) > 100:
            warnings.append(
                "Los índices de filas y columnas vacías se limitaron a los primeros 100; "
                "los totales son completos."
            )
        if len(sheet.merged_cells) > 100:
            warnings.append(
                f"La hoja contiene {len(sheet.merged_cells)} rangos combinados; "
                "se muestran los primeros 100."
            )
        if len(column_positions) < len(frame.columns):
            warnings.append(
                "Las muestras y los metadatos por columna están acotados; "
                "column_indices indica sus posiciones originales. "
                "Las dimensiones y los totales incluyen toda la hoja."
            )
        if sheet.missing_formula_cache_count:
            warnings.append(
                f"Hay {sheet.missing_formula_cache_count} fórmulas sin valor almacenado; "
                "se conserva la expresión original sin recalcularla."
            )
        if sheet.error_cell_count:
            warnings.append(
                f"Hay {sheet.error_cell_count} celdas con errores de Excel; "
                "sus valores se conservaron."
            )
        long_texts = frame.map(lambda value: isinstance(value, str) and len(value) > 200)
        long_text_count = int(long_texts.to_numpy(dtype=bool).sum())
        if long_text_count:
            coordinates = ", ".join(
                f"{get_column_letter(int(column) + 1)}{int(row) + 1}"
                for row, column in np.argwhere(long_texts.to_numpy(dtype=bool))[:6]
            )
            warnings.append(
                f"Las muestras muestran como máximo 200 caracteres de {long_text_count} textos "
                f"largos ({coordinates}); los valores originales se conservan íntegros en memoria."
            )
        header_candidates = []
        for index in range(min(25, len(frame))):
            values = [value for value in frame.iloc[index] if not is_empty(value)]
            if values and sum(isinstance(value, str) for value in values) >= len(values) / 2:
                header_candidates.append(index + 1)
                if len(header_candidates) == 5:
                    break
        initial = [
            str(json_value(value)) if not is_empty(value) else f"column_{index + 1}"
            for index, value in zip(column_positions, frame.iloc[0, column_positions], strict=True)
        ] if len(frame) else [f"column_{index + 1}" for index in column_positions]
        return SheetSummary(
            name=sheet.name,
            rows=len(frame),
            columns=len(frame.columns),
            column_indices=[index + 1 for index in column_positions],
            initial_columns=initial,
            inferred_types=[infer_type(frame[column]) for column in column_positions],
            null_counts=[int(empty[column].sum()) for column in column_positions],
            sample=self._sample(frame, column_positions),
            merged_cells=sheet.merged_cells[:100],
            formula_count=sheet.formula_count,
            missing_formula_cache_count=sheet.missing_formula_cache_count,
            formula_sample=[
                FormulaSample(
                    cell=str(item["cell"]), formula=str(item["formula"]),
                    cached_value=json_value(item["cached_value"]),
                    cache_available=item["cached_value"] is not None,
                )
                for item in sheet.formulas
            ],
            error_cell_count=sheet.error_cell_count,
            error_cells=sheet.error_cells,
            empty_rows=empty_rows[:100],
            empty_columns=empty_columns[:100],
            empty_row_count=len(empty_rows),
            empty_column_count=len(empty_columns),
            possible_header_rows=header_candidates,
            regions=self._regions(sheet, warnings),
            warnings=warnings,
            content_evidence=content_evidence,
        )

    def _regions(self, sheet: SheetData, warnings: list[str]) -> list[StructuralRegion]:
        occupied = self._occupied_cells(sheet)
        if not occupied.any():
            return []
        frame = sheet.frame
        counts = occupied.sum(axis=1)
        first_row = int(np.flatnonzero(counts)[0])
        initial = frame.iloc[first_row]
        header_columns = [index for index, value in enumerate(initial) if not is_empty(value)]
        initial_headers = set(self._initial_header_rows(sheet))
        baseline = max(1, int(np.median(counts[counts > 0])))
        formula_mask = np.zeros(occupied.shape, dtype=bool)
        for row, column in sheet.formula_cells:
            formula_mask[row - 1, column - 1] = True
        kinds: list[str | None] = []
        for index, count in enumerate(counts):
            if not count:
                kinds.append(None)
                continue
            values = frame.iloc[index]
            labels = [str(value).strip().casefold() for value in values if isinstance(value, str)]
            if index + 1 in initial_headers:
                kind = "header"
            elif kinds and kinds[-1] in {"header", "repeated_header"} and self._is_unit_row(sheet, index):
                kind = kinds[-1]
            elif index == first_row:
                header_text_count = sum(
                    isinstance(value, str) and not self._is_numeric(value)
                    for value in values if not is_empty(value)
                )
                kind = "header" if header_text_count >= count / 2 else "observations"
            elif labels and re.fullmatch(r"(?:grand )?totals?|totales|total general|suma", labels[0]):
                kind = "total"
            elif kinds[first_row] == "header" and len(header_columns) >= 2 and sum(
                self._same_value(values.iloc[column], initial.iloc[column])
                for column in header_columns
            ) >= max(2, sum(not is_empty(values.iloc[column]) for column in header_columns) * 0.8):
                kind = "repeated_header"
            elif index > first_row + 1 and count < baseline * 0.3 and any(
                previous is None or previous == "repeated_header"
                for previous in kinds[max(first_row + 1, index - 3):]
            ):
                kind = "legend"
            elif kinds and kinds[-1] == "legend" and count < baseline * 0.3:
                kind = "legend"
            else:
                kind = "observations"
            kinds.append(kind)
        regions = []
        start = 0
        while start < len(kinds):
            kind = kinds[start]
            end = start + 1
            while end < len(kinds) and kinds[end] == kind:
                end += 1
            if kind is not None:
                column_kinds = self._column_kinds(
                    sheet, occupied[start:end], formula_mask[start:end], initial, kind,
                )
                column = 0
                while column < len(column_kinds):
                    column_kind = column_kinds[column]
                    last_column = column + 1
                    while last_column < len(column_kinds) and column_kinds[last_column] == column_kind:
                        last_column += 1
                    if column_kind is not None:
                        regions.append(self._region(
                            sheet, occupied, start, end, column, last_column, column_kind, warnings,
                        ))
                    column = last_column
            start = end
        return regions

    def _is_unit_label(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        label = value.strip().casefold().replace("µ", "u").replace("μ", "u")
        label = label.replace("⁻", "-").replace("¹", "1").replace("²", "2").replace("³", "3")
        bare = r"(?:%|°\s*[cfk]|(?:[munpk]?g|[mun]?mol|[mun]?s|(?:[mun]?l)|cm|mm|m|ppm|ppb|unt|ntu))"
        ratio = (r"\b(?:[munpk]?g|[mun]?mol|[mun]?s|cells?|cel(?:ulas)?|colonias?|ind(?:ividuos)?)"
                 r"(?:\s+[a-z]+[0-9]*)?\s*(?:/|\s)\s*(?:[mun]?l|[cm]?m|kg|g)"
                 r"(?:\s*\^?\s*-?[123])?\b")
        return bool(re.fullmatch(r"\(?\s*" + bare + r"\s*\)?", label)
                    or re.search(ratio, label) or re.search(r"°\s*[cfk]\b", label))

    def _initial_header_rows(self, sheet: SheetData) -> list[int]:
        """Reconoce unidades iniciales por su evidencia textual, sin tratar códigos como unidades."""
        occupied = self._occupied_cells(sheet)
        nonempty_rows = np.flatnonzero(occupied.any(axis=1))
        if not len(nonempty_rows):
            return []
        first = int(nonempty_rows[0])
        initial = [value for value in sheet.frame.iloc[first] if not is_empty(value)]
        if sum(isinstance(value, str) and not self._is_numeric(value) for value in initial) < len(initial) / 2:
            return []
        headers = [first + 1]
        for index in range(first + 1, min(first + 6, len(sheet.frame))):
            if not self._is_unit_row(sheet, index):
                break
            headers.append(index + 1)
        return headers

    def _is_unit_row(self, sheet: SheetData, index: int) -> bool:
        values = [value for value in sheet.frame.iloc[index] if not is_empty(value)]
        return (bool(values) and index + 1 not in sheet.formula_rows
                and all(isinstance(value, str) for value in values)
                and sum(self._is_unit_label(value) for value in values) >= max(1, math.ceil(len(values) * 0.6)))

    def _same_value(self, left: object, right: object) -> bool:
        if is_empty(left) or is_empty(right):
            return is_empty(left) and is_empty(right)
        return str(left).strip() == str(right).strip()

    def _column_kinds(
        self, sheet: SheetData, occupied: np.ndarray, formulas: np.ndarray,
        initial: pd.Series, row_kind: str,
    ) -> list[str | None]:
        counts = occupied.sum(axis=0)
        formula_counts = formulas.sum(axis=0)
        populated = np.flatnonzero(counts)
        if row_kind != "observations":
            first, last = int(populated[0]), int(populated[-1])
            return [row_kind if first <= column <= last else None for column in range(len(counts))]
        presence_columns = {
            column for column, value in enumerate(initial)
            if isinstance(value, str) and re.search(r"\b(?:presencias?|presence(?:s)?)\b", value, re.IGNORECASE)
        }
        percentage_columns = {
            column - 1 for (_, column), formula in sheet.formula_cells.items()
            if re.search(r"\*\s*100\s*/|/\s*[^+*-]+\*\s*100", formula.expression)
            and (re.search(r"\b(?:SUM|COUNT(?:IF|IFS|A)?|SUMPRODUCT|AVERAGE)\s*\(", formula.expression, re.IGNORECASE)
                 or any(re.search(
                     rf"(?<![A-Za-z0-9_])\$?{get_column_letter(presence + 1)}\$?\d+", formula.expression,
                     re.IGNORECASE,
                 ) for presence in presence_columns))
        }
        narrow_summary = len(sheet.frame.columns) <= 5 and bool(presence_columns)
        kinds = []
        for column, count in enumerate(counts):
            if not count:
                kinds.append(None)
            elif (narrow_summary or column in percentage_columns or column in presence_columns
                  or (str(initial.iloc[column]).strip() == "%" and any(
                      abs(column - presence) == 1 for presence in presence_columns))):
                kinds.append("summary")
            elif self._is_record_identifier_column(sheet, column, initial.iloc[column]):
                kinds.append("observations")
            elif formula_counts[column] >= count * 0.6:
                kinds.append("derived")
            else:
                kinds.append("observations")
        return kinds

    def _is_identifier_header(self, header: object) -> bool:
        return isinstance(header, str) and bool(re.fullmatch(
            r"(?:id|identifier|identificador|(?:sample|record|muestra|registro)[ _-]*id|id[ _-]*(?:sample|record|muestra|registro))",
            header.strip(), re.IGNORECASE,
        ))

    def _identifier_expression(self, expression: str) -> tuple[bool, set[int], bool, bool]:
        without_literals = re.sub(r'"(?:[^\"]|\"\")*"', "", expression)
        aggregate = bool(re.search(
            r"\b(?:SUM(?:IF|IFS|PRODUCT)?|COUNT(?:IF|IFS|A)?|AVERAGE(?:IF|IFS)?|MIN|MAX|SUBTOTAL|AGGREGATE|SUMA|PROMEDIO|CONTAR|SUBTOTALES)\s*\(",
            without_literals, re.IGNORECASE,
        ))
        construction = not aggregate and bool("&" in without_literals or re.search(
            r"\b(?:CONCAT(?:ENATE|ENAR)?|TEXTJOIN|UNIRCADENAS)\s*\(", without_literals, re.IGNORECASE,
        ))
        referenced_rows = {int(value) for token in self._identifier_reference_tokens(expression)
                           for value in re.findall(r"\$?[A-Za-z]{1,3}\$?([0-9]+)", token)}
        return construction, referenced_rows, "!" in without_literals, aggregate

    def _identifier_reference_tokens(self, expression: str) -> list[str]:
        try:
            return [token.value for token in Tokenizer(expression).items
                    if token.type == "OPERAND" and token.subtype == "RANGE"]
        except TokenizerError:
            return []

    def _identifier_operand(self, sheet: SheetData, row: int, column: int) -> tuple[object, bool]:
        if not (1 <= row <= len(sheet.frame) and 1 <= column <= len(sheet.frame.columns)):
            return None, False
        formula = sheet.formula_cells.get((row, column))
        value = formula.cached_value if formula is not None else sheet.frame.iat[row - 1, column - 1]
        known = ((formula is None or not strict_missing(formula.cached_value))
                 and (row, column) not in sheet.error_positions)
        return value, known

    def _identifier_operand_values_match(self, left: object, right: object) -> bool:
        if (isinstance(left, Number) and isinstance(right, Number)
                and not isinstance(left, bool) and not isinstance(right, bool)
                and not strict_missing(left) and not strict_missing(right)):
            return bool(left == right)
        return self._value_key(left) == self._value_key(right)

    def _record_has_other_data(self, sheet: SheetData, row: int, column: int) -> bool:
        return (any(not is_empty(value) for index, value in enumerate(sheet.frame.iloc[row - 1]) if index != column)
                or any(formula_row == row and formula_column != column + 1
                       for formula_row, formula_column in sheet.formula_cells))

    def _is_record_identifier_column(self, sheet: SheetData, column: int, header: object) -> bool:
        if not self._is_identifier_header(header):
            return False
        formulas = [(row, formula.expression) for (row, original_column), formula in sheet.formula_cells.items()
                    if original_column == column + 1]
        if not formulas:
            return False
        matching_records = 0
        for row, expression in formulas:
            construction, referenced_rows, external, aggregate = self._identifier_expression(expression)
            if aggregate:
                return False
            matching_records += (construction and referenced_rows == {row} and not external
                                 and self._record_has_other_data(sheet, row, column))
        return matching_records >= len(formulas) * 0.6

    def _record_identity_findings(
        self, sheets: list[SheetData], policies: PreparationPolicies, executions: list[SheetExecution],
    ) -> list[TabularFinding]:
        """Separa anomalías de identificadores del rol de su columna, sin reconstruirlos."""
        sources = {sheet.name: sheet for sheet in sheets}
        findings, seen = [], set()
        for policy in policies.observation_rows:
            sheet = sources.get(policy.sheet_name)
            if sheet is None or not headers_match(sheet, policy.header_row, policy.expected_headers):
                continue
            for column in policy.auxiliary_column_indices:
                if not self._is_record_identifier_column(sheet, column - 1, policy.expected_headers[column]):
                    continue
                for (row, formula_column), formula in sheet.formula_cells.items():
                    if formula_column != column or row <= policy.header_row:
                        continue
                    construction, referenced_rows, external, _ = self._identifier_expression(formula.expression)
                    if (not construction or (not external and (not referenced_rows or referenced_rows == {row}))
                            or not self._record_has_other_data(sheet, row, column - 1)
                            or (sheet.name, row, column) in seen):
                        continue
                    seen.add((sheet.name, row, column))
                    cell = f"{get_column_letter(column)}{row}"
                    comparisons, requires_review = [], external
                    for token in self._identifier_reference_tokens(formula.expression):
                        reference = re.fullmatch(r"\$?([A-Za-z]{1,3})\$?([0-9]+)", token)
                        if reference is None:
                            requires_review = True
                            comparisons.append(f"referencia {token!r} sin comparación inequívoca")
                            continue
                        operand_column = column_index_from_string(reference.group(1))
                        operand_row = int(reference.group(2))
                        if operand_row == row:
                            continue
                        referenced, referenced_known = self._identifier_operand(sheet, operand_row, operand_column)
                        own, own_known = self._identifier_operand(sheet, row, operand_column)
                        equivalent = (referenced_known and own_known
                                      and self._identifier_operand_values_match(referenced, own))
                        requires_review |= not equivalent
                        own_cell = f"{get_column_letter(operand_column)}{row}"
                        comparisons.append(
                            f"{token}={trace_value(referenced)!r} frente a {own_cell}={trace_value(own)!r} "
                            f"({'igualdad comprobada' if equivalent else 'valores distintos' if referenced_known and own_known else 'evidencia o caché insuficiente'})"
                        )
                    findings.append(TabularFinding(
                        code="record_identifier_cross_row_reference", sheet_name=sheet.name, cell=cell,
                        message=(f"El identificador corroborado contiene referencias externas a su fila: {'; '.join(comparisons)}; "
                                 f"se conservan la fórmula original {formula.expression!r} y su valor almacenado "
                                 f"{trace_value(formula.cached_value)!r}. Control {policy.rule_id}, versión {policy.version}."),
                        requires_review=requires_review,
                    ))
                    for execution in executions:
                        if (execution.sheet_name == sheet.name
                                and execution.data_start_row <= row <= execution.data_end_row
                                and execution.data_start_column <= column <= execution.data_end_column
                                and cell not in {item.cell for item in execution.formula_sample}):
                            execution.formula_sample.append(FormulaSample(
                                cell=cell, formula=formula.expression, cached_value=trace_value(formula.cached_value),
                                cache_available=not strict_missing(formula.cached_value),
                            ))
        return findings

    def _region(
        self, sheet: SheetData, occupied: np.ndarray,
        start: int, end: int, first_column: int, last_column: int, kind: str,
        warnings: list[str],
    ) -> StructuralRegion:
        header_anchors = set()
        if kind in {"header", "repeated_header"}:
            for reference in sheet.merged_cells:
                merged = CellRange(reference)
                if (start < merged.min_row <= end and first_column < merged.min_col <= last_column):
                    last_column = max(last_column, merged.max_col)
                    if occupied[merged.min_row - 1, merged.min_col - 1]:
                        header_anchors.add(merged.min_col - 1)
        reasons = {
            "header": "Primera fila con valores; posible encabezado.",
            "repeated_header": "Los valores de la fila coinciden con el encabezado inicial en al menos un 80% de las posiciones compartidas.",
            "total": "La primera etiqueta textual identifica explícitamente un total.",
            "legend": "Sección después de un separador con una densidad distinta a la tabla principal.",
            "derived": "Al menos el 60% de las celdas con información de estas columnas son fórmulas.",
            "summary": "El encabezado indica presencia o la fórmula calcula un porcentaje; se separa de las abundancias.",
            "observations": "Rango continuo con valores y estructura compatible con registros.",
        }
        positions = list(range(first_column, last_column))
        if len(positions) > 6:
            positions = positions[:3] + positions[-3:]
        if header_anchors:
            populated = [int(column) + first_column for column in np.flatnonzero(
                occupied[start:end, first_column:last_column].any(axis=0),
            )]
            edges = populated if len(populated) <= 6 else populated[:3] + populated[-3:]
            anchors = sorted(header_anchors)
            if len(anchors) > 16:
                positions = [anchors[index] for index in self._sample_positions(len(anchors), 16)]
                warnings.append(
                    f"El encabezado {get_column_letter(first_column + 1)}{start + 1}:"
                    f"{get_column_letter(last_column)}{end} contiene {len(anchors)} textos de "
                    "celdas combinadas; la muestra incluye 16 anclas distribuidas en el rango "
                    "y conserva sus índices originales."
                )
            else:
                additional = [column for column in edges if column not in header_anchors]
                remaining = 16 - len(anchors)
                sampled = [additional[index] for index in self._sample_positions(len(additional), remaining)]
                positions = sorted(anchors + sampled)
                if len(sampled) < len(additional):
                    warnings.append(
                        f"La muestra del encabezado {get_column_letter(first_column + 1)}{start + 1}:"
                        f"{get_column_letter(last_column)}{end} está limitada a 16 columnas; "
                        f"se priorizaron sus {len(anchors)} anclas combinadas y se omitieron "
                        f"{len(additional) - len(sampled)} etiquetas de los extremos."
                    )
        row_positions = sorted({start, (start + end - 1) // 2, end - 1})
        longest_text_row = max(
            range(start, end),
            key=lambda row: max(
                (len(value) for value in sheet.frame.iloc[row, first_column:last_column]
                 if isinstance(value, str)), default=0,
            ),
        )
        row_positions = sorted(set(row_positions) | {longest_text_row})
        return StructuralRegion(
            region_id=f"r{start + 1}_{end}_c{first_column + 1}_{last_column}",
            start_row=start + 1, end_row=end,
            start_column=first_column + 1, end_column=last_column,
            kind=kind, reason=reasons[kind],
            nonempty_cells=int(occupied[start:end, first_column:last_column].sum()),
            column_indices=[column + 1 for column in positions],
            sample=[
                SampleRow(
                    row_number=row + 1,
                    values=[json_value(sheet.frame.iat[row, column]) for column in positions],
                )
                for row in row_positions
            ],
        )

    def _column_names(
        self, sheet: SheetData, header_rows: list[int], warnings: list[str],
        start_column: int, end_column: int,
    ) -> list[str]:
        if not header_rows:
            return [f"column_{index}" for index in range(start_column, end_column + 1)]
        headers = sheet.frame.iloc[[row - 1 for row in header_rows]].copy()
        for merged_reference in sheet.merged_cells:
            merged = CellRange(merged_reference)
            value = sheet.frame.iat[merged.min_row - 1, merged.min_col - 1]
            for row in header_rows:
                if merged.min_row <= row <= merged.max_row:
                    headers.loc[row - 1, merged.min_col - 1:merged.max_col - 1] = value
        headers = headers.iloc[:, start_column - 1:end_column]
        names = []
        used = set()
        missing_headers = []
        for column in headers.columns:
            parts = list(dict.fromkeys(
                str(value).strip() for value in headers[column] if not is_empty(value)
            ))
            base = " ".join(parts) or f"column_{column + 1}"
            if not parts:
                missing_headers.append(column + 1)
            name = base
            suffix = 2
            while name in used:
                name = f"{base}__{suffix}"
                suffix += 1
            if name != base:
                warnings.append(f"El encabezado {base} está repetido; se asignó {name}.")
            names.append(name)
            used.add(name)
        if len(missing_headers) == 1:
            column = missing_headers[0]
            warnings.append(f"La columna {column} no tiene encabezado; se asignó column_{column}.")
        elif missing_headers:
            sample = ", ".join(get_column_letter(column) for column in missing_headers[:12])
            warnings.append(
                f"Hay {len(missing_headers)} columnas sin encabezado; se asignaron nombres column_N "
                f"respetando la posición original. Muestra de columnas: {sample}."
            )
        return names

    def _range_errors(self, sheet: SheetData, recipe: SheetRecipe) -> list[str]:
        """Rechaza coordenadas inválidas antes de seleccionar celdas o contar cobertura."""
        bounds = (
            recipe.data_start_row, recipe.data_end_row,
            recipe.data_start_column, recipe.data_end_column,
        )
        if any(type(value) is not int or value < 1 for value in bounds):
            return ["Todos los límites del bloque deben ser enteros positivos explícitos."]
        if (recipe.data_start_row > recipe.data_end_row
                or recipe.data_start_column > recipe.data_end_column):
            return ["El inicio del rango debe ser menor o igual a su final."]
        rows, columns = sheet.frame.shape
        errors = []
        if recipe.data_end_row > rows:
            errors.append("La receta selecciona filas que exceden las dimensiones de la hoja.")
        if recipe.data_end_column > columns:
            errors.append("La receta selecciona columnas que exceden las dimensiones de la hoja.")
        if (any(type(row) is not int or not 1 <= row <= rows for row in recipe.header_rows)
                or recipe.header_rows != sorted(set(recipe.header_rows))):
            errors.append("Los encabezados deben ser filas válidas, únicas y ordenadas de la hoja.")
        elif recipe.header_rows and recipe.header_rows[-1] >= recipe.data_start_row:
            errors.append("Los encabezados deben preceder al rango de datos del bloque.")
        return errors

    def _execute_sheet(
        self, sheet: SheetData, recipe: SheetRecipe, *,
        policies: PreparationPolicies | None = None, source_sha256: str | None = None,
        normalizations: list | None = None, auxiliary_rows: list | None = None,
        temporal_dates: list | None = None,
    ) -> SheetExecution:
        rows, columns = sheet.frame.shape
        end_row = recipe.data_end_row
        end_column = recipe.data_end_column
        result = SheetExecution(
            sheet_name=sheet.name,
            block_id=recipe.block_id,
            role=recipe.role,
            data_start_row=recipe.data_start_row,
            data_end_row=end_row,
            data_start_column=recipe.data_start_column,
            data_end_column=end_column,
            source_rows=rows,
            source_columns=columns,
            rows_before=0,
            rows_after=0,
            columns_before=[],
            columns_after=[],
        )
        result.errors.extend(self._range_errors(sheet, recipe))
        if result.errors:
            return result
        if recipe.role == "unknown" and recipe.transformations:
            result.errors.append("Un bloque unknown debe conservarse sin transformaciones.")
            return result
        names = self._column_names(
            sheet, recipe.header_rows, result.warnings, recipe.data_start_column, end_column,
        )
        frame = sheet.frame.iloc[
            recipe.data_start_row - 1:end_row, recipe.data_start_column - 1:end_column,
        ].copy()
        frame.columns = names
        result.rows_before = len(frame)
        result.columns_before = names
        column_positions = {
            name: column for column, name in enumerate(names, recipe.data_start_column)
        }
        selected_formulas = {
            position: formula for position, formula in sheet.formula_cells.items()
            if recipe.data_start_row <= position[0] <= end_row
            and recipe.data_start_column <= position[1] <= end_column
        }
        formula_positions: dict[str, set[int]] = {}
        for row, column in selected_formulas:
            name = names[column - recipe.data_start_column]
            formula_positions.setdefault(name, set()).add(row)
        missing_formulas = [
            f"{get_column_letter(column)}{row}"
            for (row, column), formula in selected_formulas.items() if formula.cached_value is None
        ]
        result.unresolved_formula_count = len(missing_formulas)
        result.unresolved_formula_cells = missing_formulas[:100]
        result.formula_count = len(selected_formulas)
        formula_items = list(selected_formulas.items())
        formula_indices = self._sample_positions(len(formula_items), min(self.sample_rows, 12))
        result.formula_sample = [
            FormulaSample(
                cell=f"{get_column_letter(column)}{row}", formula=formula.expression,
                cached_value=json_value(formula.cached_value),
                cache_available=formula.cached_value is not None,
            )
            for index in formula_indices
            for (row, column), formula in [formula_items[index]]
        ]
        error_positions = [
            (row, column) for row, column in sorted(sheet.error_positions)
            if recipe.data_start_row <= row <= end_row
            and recipe.data_start_column <= column <= end_column
        ]
        result.error_cell_count = len(error_positions)
        result.error_cells = [f"{get_column_letter(column)}{row}" for row, column in error_positions[:100]]
        if missing_formulas and policies is None:
            result.warnings.append(
                f"Hay {len(missing_formulas)} fórmulas sin valor almacenado en el bloque; "
                "se conserva su expresión original sin recalcularla."
            )
        if error_positions:
            result.warnings.append(
                f"Hay {len(error_positions)} errores de Excel en el bloque; se conservaron sus valores."
            )
        occupied = self._occupied_cells(sheet)
        result.source_nonempty_cells = int(occupied[
            recipe.data_start_row - 1:end_row, recipe.data_start_column - 1:end_column,
        ].sum())
        row_numbers = sheet.frame.index + 1
        outside = ~row_numbers.isin(recipe.header_rows) & (
            (row_numbers < recipe.data_start_row) | (row_numbers > end_row)
        )
        excluded = sheet.frame.loc[outside]
        nonempty = ~excluded.map(is_empty).all(axis=1)
        nonempty |= (excluded.index + 1).isin(sheet.formula_rows)
        result.excluded_nonempty_rows = int(nonempty.sum())
        for row in recipe.header_rows:
            values = [
                value for value in sheet.frame.iloc[row - 1, recipe.data_start_column - 1:end_column]
                if not is_empty(value)
            ]
            if any(self._is_numeric(value) for value in values):
                result.suspicious_header_rows.append(row)
        if result.suspicious_header_rows:
            result.warnings.append(
                "Los encabezados seleccionados contienen valores numéricos; "
                "podrían ser registros."
            )
        if recipe.role == "unknown":
            result.findings.append(TabularFinding(
                code="unresolved_block_role", sheet_name=sheet.name, block_id=recipe.block_id,
                range=(f"{get_column_letter(recipe.data_start_column)}{recipe.data_start_row}:"
                       f"{get_column_letter(end_column)}{end_row}"),
                message="El bloque se conserva sin transformaciones y requiere resolver su rol.",
            ))
        if policies is not None:
            frame, applied, omitted, preparation_findings = prepare_frame(
                sheet, recipe, frame, policies, source_sha256 or "",
            )
            if normalizations is not None:
                normalizations.extend(applied)
            if auxiliary_rows is not None:
                auxiliary_rows.extend(omitted)
            result.auxiliary_rows_removed = sum(item.removed_from_observations for item in omitted)
            result.findings.extend(preparation_findings)
            auxiliary_row_numbers = {item.row_number for item in omitted}
            missing_formulas = [
                f"{get_column_letter(column)}{row}" for (row, column), formula in selected_formulas.items()
                if row - 1 in frame.index and row not in auxiliary_row_numbers and formula.cached_value is None
            ]
            result.unresolved_formula_count = len(missing_formulas)
            result.unresolved_formula_cells = missing_formulas[:100]
            if missing_formulas:
                result.warnings.append(
                    f"Hay {len(missing_formulas)} fórmulas sin valor almacenado en el bloque; "
                    "se conserva su expresión original sin recalcularla."
                )
        for transformation in recipe.transformations:
            try:
                effective_transformation = transformation
                if policies is not None and isinstance(transformation, ConvertType) and transformation.target_type == "datetime":
                    protected = {
                        column for policy in policies.number_columns
                        if policy.sheet_name == sheet.name
                        and headers_match(sheet, policy.header_row, policy.expected_headers)
                        for column in policy.column_indices
                    }
                    denied = [name for name in transformation.columns if column_positions.get(name) in protected]
                    for name in denied:
                        letter = get_column_letter(column_positions[name])
                        result.findings.append(TabularFinding(
                            code="rule_policy_type_conflict", sheet_name=sheet.name, block_id=recipe.block_id,
                            range=f"{letter}{recipe.data_start_row}:{letter}{recipe.data_end_row}",
                            message=f"No se aplica la conversión a fecha de {name}: la política corroborada define esta columna como abundancia numérica.",
                        ))
                    allowed = [name for name in transformation.columns if name not in denied]
                    if not allowed:
                        continue
                    effective_transformation = transformation.model_copy(update={"columns": allowed})
                frame = self._transform(
                    frame, effective_transformation, result, formula_positions, column_positions,
                )
            except (ValueError, TypeError, OverflowError) as error:
                result.errors.append(str(error))
                break
        result.rows_after = len(frame)
        if policies is not None:
            selected_dates, temporal_findings = select_temporal_dates(
                sheet, recipe, policies, [int(index) + 1 for index in frame.index],
            )
            if temporal_dates is not None:
                temporal_dates.extend(selected_dates)
            result.findings.extend(temporal_findings)
        result.columns_after = list(frame.columns)
        result.empty_columns = [
            name for name in frame.columns
            if name not in formula_positions and frame[name].map(is_empty).all()
        ]
        retained = ~frame.map(is_empty)
        for name, formula_rows in formula_positions.items():
            if name in retained.columns:
                retained.loc[(retained.index + 1).isin(formula_rows), name] = True
        result.preserved_nonempty_cells = int(retained.to_numpy(dtype=bool).sum())
        result.inferred_types = {name: infer_type(frame[name]) for name in frame.columns}
        result.sample = self._sample(frame)
        return result

    def _sample_positions(self, size: int, count: int) -> list[int]:
        if not size:
            return []
        count = min(count, size)
        if count <= 1:
            return [0] if count else []
        return sorted({index * (size - 1) // (count - 1) for index in range(count)})

    def _is_numeric(self, value: object) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, Number):
            return True
        if isinstance(value, str):
            try:
                return Decimal(value.strip()).is_finite()
            except InvalidOperation:
                return False
        return False

    def _transform(
        self,
        frame: pd.DataFrame,
        transformation: object,
        result: SheetExecution,
        formula_positions: dict[str, set[int]],
        column_positions: dict[str, int],
    ) -> pd.DataFrame:
        if isinstance(transformation, DropEmptyRows):
            empty = frame.map(is_empty).all(axis=1)
            formula_rows = set().union(*formula_positions.values()) if formula_positions else set()
            empty &= ~(frame.index + 1).isin(formula_rows)
            result.dropped_empty_rows += int(empty.sum())
            return frame.loc[~empty].copy()
        if isinstance(transformation, DropEmptyColumns):
            empty_columns = [
                name for name in frame.columns
                if name not in formula_positions and frame[name].map(is_empty).all()
            ]
            result.dropped_empty_columns.extend(empty_columns)
            return frame.drop(columns=empty_columns)
        if isinstance(transformation, RenameColumns):
            self._require_columns(frame, list(transformation.mapping))
            names = [transformation.mapping.get(name, name) for name in frame.columns]
            if any(not name.strip() for name in names) or len(names) != len(set(names)):
                raise ValueError("El renombrado produce nombres vacíos o columnas duplicadas.")
            for mapping in (formula_positions, column_positions):
                renamed = {transformation.mapping.get(name, name): value for name, value in mapping.items()}
                mapping.clear()
                mapping.update(renamed)
            return frame.rename(columns=transformation.mapping)
        if isinstance(transformation, StripStrings):
            selected = transformation.columns if transformation.columns is not None else list(frame.columns)
            self._require_columns(frame, selected)
            for name in selected:
                frame[name] = frame[name].map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
            return frame
        if isinstance(transformation, ConvertType):
            self._require_columns(frame, transformation.columns)
            for name in transformation.columns:
                original = frame[name].copy()
                converted = self._convert(original, transformation.target_type)
                invalid = ~original.map(is_empty) & converted.isna()
                failures = int(invalid.sum())
                if failures:
                    result.conversion_failures[name] = result.conversion_failures.get(name, 0) + failures
                    for row in original.index[invalid]:
                        coordinate = f"{get_column_letter(column_positions[name])}{int(row) + 1}"
                        result.findings.append(TabularFinding(
                            code="conversion_preserved", sheet_name=result.sheet_name,
                            block_id=result.block_id, cell=coordinate,
                            message=(f"No se convirtió {coordinate} a {transformation.target_type}; "
                                     "se conservó el valor original."),
                        ))
                converted = converted.astype(object)
                unchanged = converted.isna()
                converted.loc[unchanged] = original.loc[unchanged]
                frame[name] = converted
            return frame
        raise ValueError("La receta contiene una operación no permitida.")

    def _require_columns(self, frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"La transformación referencia columnas inexistentes: {', '.join(missing)}.")

    def _convert(self, series: pd.Series, target_type: str) -> pd.Series:
        if target_type == "number":
            converted = pd.to_numeric(series.dropna(), errors="coerce").astype(object).reindex(series.index)
            values = []
            for original, value in zip(series, converted, strict=True):
                if (is_empty(value) or not math.isfinite(value)
                        or isinstance(original, (bool, np.bool_)) or self._is_padded_code(original)):
                    values.append(None)
                    continue
                try:
                    source = Decimal(str(original).strip())
                    precise = source == Decimal(str(value))
                except InvalidOperation:
                    precise = False
                values.append(value if precise else None)
            return pd.Series(values, index=series.index, dtype=object)
        if target_type == "integer":
            def to_integer(value: object) -> int | None:
                if is_empty(value):
                    return None
                if isinstance(value, (bool, np.bool_)) or self._is_padded_code(value):
                    return None
                try:
                    number = Decimal(str(value).strip())
                    if (
                        number.is_finite()
                        and number == number.to_integral_value()
                        and -(2 ** 63) <= number < 2 ** 63
                    ):
                        return int(number)
                except InvalidOperation:
                    pass
                return None
            return pd.Series([to_integer(value) for value in series], index=series.index, dtype="Int64")
        if target_type == "datetime":
            def unambiguous_date(value: object) -> object:
                if isinstance(value, (datetime, date, pd.Timestamp)):
                    return value
                if isinstance(value, str) and re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:\d{2})?)?",
                    value.strip(),
                ):
                    return value.strip()
                return None
            unambiguous = series.map(unambiguous_date)
            return pd.to_datetime(unambiguous, errors="coerce", format="mixed", utc=True)
        if target_type == "boolean":
            values = {
                "true": True, "false": False,
            }
            def to_boolean(value: object) -> bool | None:
                if is_empty(value):
                    return None
                if isinstance(value, bool):
                    return value
                return values.get(str(value).strip().casefold())
            return pd.Series([to_boolean(value) for value in series], index=series.index, dtype="boolean")
        raise ValueError("El tipo de conversión no está permitido.")

    def _is_padded_code(self, value: object) -> bool:
        return isinstance(value, str) and bool(re.fullmatch(
            r"[+-]?0\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value.strip(),
        ))
