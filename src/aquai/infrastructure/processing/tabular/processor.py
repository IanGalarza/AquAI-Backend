"""Inspección y ejecución determinística de recetas tabulares."""

import math
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from numbers import Number

import pandas as pd
from openpyxl.worksheet.cell_range import CellRange

from aquai.application.dtos.tabular import (
    ConvertType,
    DropEmptyColumns,
    DropEmptyRows,
    ExecutionResult,
    FormulaSample,
    PreparationRecipe,
    RenameColumns,
    SampleRow,
    SheetExecution,
    SheetRecipe,
    SheetSummary,
    StripStrings,
    StructuralSummary,
)
from aquai.infrastructure.processing.tabular.reader import SheetData, TableReader


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
        return StructuralSummary(
            file_name=file_name,
            file_type=table.file_type,
            file_size_bytes=len(content),
            sheets=[self._summarize(sheet) for sheet in table.sheets],
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
    ) -> ExecutionResult:
        table = self.reader.read(file_name, content)
        sheets = {sheet.name: sheet for sheet in table.sheets}
        selected = {sheet.sheet_name for sheet in recipe.sheets}
        executions = []
        for sheet_recipe in recipe.sheets:
            sheet = sheets.get(sheet_recipe.sheet_name)
            if sheet is None:
                executions.append(SheetExecution(
                    sheet_name=sheet_recipe.sheet_name,
                    source_rows=0,
                    source_columns=0,
                    rows_before=0,
                    rows_after=0,
                    columns_before=[],
                    columns_after=[],
                    errors=["La hoja indicada por la receta no existe en el archivo."],
                ))
            else:
                executions.append(self._execute_sheet(sheet, sheet_recipe))
        return ExecutionResult(
            sheets=executions,
            skipped_sheets=[name for name in sheets if name not in selected],
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
        if not frame.size or len(empty_rows) == len(frame):
            warnings.append("La hoja no contiene valores para procesar.")
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
                "se leerán como nulos."
            )
        if sheet.error_cell_count:
            warnings.append(
                f"Hay {sheet.error_cell_count} celdas con errores de Excel; "
                "sus valores se conservaron."
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
            formula_sample=[
                FormulaSample(cell=str(item["cell"]), formula=str(item["formula"]))
                for item in sheet.formulas
            ],
            error_cell_count=sheet.error_cell_count,
            error_cells=sheet.error_cells,
            empty_rows=empty_rows[:100],
            empty_columns=empty_columns[:100],
            empty_row_count=len(empty_rows),
            empty_column_count=len(empty_columns),
            possible_header_rows=header_candidates,
            warnings=warnings,
        )

    def _column_names(self, sheet: SheetData, header_rows: list[int], warnings: list[str]) -> list[str]:
        if not header_rows:
            return [f"column_{index + 1}" for index in range(len(sheet.frame.columns))]
        headers = sheet.frame.iloc[[row - 1 for row in header_rows]].copy()
        for merged_reference in sheet.merged_cells:
            merged = CellRange(merged_reference)
            value = sheet.frame.iat[merged.min_row - 1, merged.min_col - 1]
            for row in header_rows:
                if merged.min_row <= row <= merged.max_row:
                    headers.loc[row - 1, merged.min_col - 1:merged.max_col - 1] = value
        names = []
        used = set()
        for index, column in enumerate(headers.columns):
            parts = list(dict.fromkeys(
                str(value).strip() for value in headers[column] if not is_empty(value)
            ))
            base = " ".join(parts) or f"column_{index + 1}"
            if not parts:
                warnings.append(f"La columna {index + 1} no tiene encabezado; se asignó {base}.")
            name = base
            suffix = 2
            while name in used:
                name = f"{base}__{suffix}"
                suffix += 1
            if name != base:
                warnings.append(f"El encabezado {base} está repetido; se asignó {name}.")
            names.append(name)
            used.add(name)
        return names

    def _execute_sheet(self, sheet: SheetData, recipe: SheetRecipe) -> SheetExecution:
        rows, columns = sheet.frame.shape
        result = SheetExecution(
            sheet_name=sheet.name,
            source_rows=rows,
            source_columns=columns,
            rows_before=0,
            rows_after=0,
            columns_before=[],
            columns_after=[],
            unresolved_formula_cells=sheet.missing_formula_cells,
            error_cells=sheet.error_cells,
        )
        if sheet.missing_formula_cache_count:
            result.warnings.append(
                f"Hay {sheet.missing_formula_cache_count} fórmulas sin valor almacenado; "
                "no se calculan automáticamente."
            )
        if sheet.error_cell_count:
            result.warnings.append(
                f"Hay {sheet.error_cell_count} celdas con errores de Excel; "
                "sus valores se conservaron."
            )
        if any(row > rows for row in recipe.header_rows) or recipe.data_start_row > rows + 1:
            result.errors.append("La receta selecciona filas que exceden las dimensiones de la hoja.")
            return result
        if recipe.data_end_row is not None and recipe.data_end_row > rows:
            result.errors.append("data_end_row excede las dimensiones de la hoja.")
            return result
        end_row = recipe.data_end_row if recipe.data_end_row is not None else rows
        names = self._column_names(sheet, recipe.header_rows, result.warnings)
        frame = sheet.frame.iloc[recipe.data_start_row - 1:end_row].copy()
        frame.columns = names
        result.rows_before = len(frame)
        result.columns_before = names
        row_numbers = sheet.frame.index + 1
        outside = ~row_numbers.isin(recipe.header_rows) & (
            (row_numbers < recipe.data_start_row) | (row_numbers > end_row)
        )
        excluded = sheet.frame.loc[outside]
        nonempty = ~excluded.map(is_empty).all(axis=1)
        nonempty |= (excluded.index + 1).isin(sheet.formula_rows)
        result.excluded_nonempty_rows = int(nonempty.sum())
        for row in recipe.header_rows:
            values = [value for value in sheet.frame.iloc[row - 1] if not is_empty(value)]
            if any(self._is_numeric(value) for value in values):
                result.suspicious_header_rows.append(row)
        if result.suspicious_header_rows:
            result.warnings.append(
                "Los encabezados seleccionados contienen valores numéricos; "
                "podrían ser registros."
            )
        formula_columns = {names[column - 1] for column in sheet.formula_columns}
        for transformation in recipe.transformations:
            try:
                frame = self._transform(
                    frame, transformation, result, sheet.formula_rows, formula_columns,
                )
            except (ValueError, TypeError, OverflowError) as error:
                result.errors.append(str(error))
                break
        result.rows_after = len(frame)
        result.columns_after = list(frame.columns)
        result.empty_columns = [
            name for name in frame.columns
            if name not in formula_columns and frame[name].map(is_empty).all()
        ]
        result.inferred_types = {name: infer_type(frame[name]) for name in frame.columns}
        result.sample = self._sample(frame)
        return result

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
        formula_rows: set[int],
        formula_columns: set[str],
    ) -> pd.DataFrame:
        if isinstance(transformation, DropEmptyRows):
            empty = frame.map(is_empty).all(axis=1)
            empty &= ~(frame.index + 1).isin(formula_rows)
            result.dropped_empty_rows += int(empty.sum())
            return frame.loc[~empty].copy()
        if isinstance(transformation, DropEmptyColumns):
            empty_columns = [
                name for name in frame.columns
                if name not in formula_columns and frame[name].map(is_empty).all()
            ]
            result.dropped_empty_columns.extend(empty_columns)
            return frame.drop(columns=empty_columns)
        if isinstance(transformation, RenameColumns):
            self._require_columns(frame, list(transformation.mapping))
            names = [transformation.mapping.get(name, name) for name in frame.columns]
            if any(not name.strip() for name in names) or len(names) != len(set(names)):
                raise ValueError("El renombrado produce nombres vacíos o columnas duplicadas.")
            renamed_formulas = {transformation.mapping.get(name, name) for name in formula_columns}
            formula_columns.clear()
            formula_columns.update(renamed_formulas)
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
                original = frame[name].map(lambda value: None if is_empty(value) else value)
                converted = self._convert(original, transformation.target_type)
                failures = int((original.notna() & converted.isna()).sum())
                if failures:
                    result.conversion_failures[name] = result.conversion_failures.get(name, 0) + failures
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
                if is_empty(value) or not math.isfinite(value):
                    values.append(None)
                    continue
                try:
                    source_text = str(int(original)) if isinstance(original, bool) else str(original).strip()
                    source = Decimal(source_text)
                    precise = source == Decimal(str(value))
                except InvalidOperation:
                    precise = False
                values.append(value if precise else None)
            return pd.Series(values, index=series.index, dtype=object)
        if target_type == "integer":
            def to_integer(value: object) -> int | None:
                if is_empty(value):
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
            unambiguous = series.map(lambda value: None if isinstance(value, Number) else value)
            return pd.to_datetime(unambiguous, errors="coerce", format="mixed", utc=True)
        if target_type == "boolean":
            values = {
                "true": True, "false": False,
                "1": True, "0": False,
                "yes": True, "no": False,
                "si": True, "sí": True,
            }
            def to_boolean(value: object) -> bool | None:
                if is_empty(value):
                    return None
                if isinstance(value, bool):
                    return value
                if isinstance(value, Number) and value in (0, 1):
                    return bool(value)
                return values.get(str(value).strip().casefold())
            return pd.Series([to_boolean(value) for value in series], index=series.index, dtype="boolean")
        raise ValueError("El tipo de conversión no está permitido.")
