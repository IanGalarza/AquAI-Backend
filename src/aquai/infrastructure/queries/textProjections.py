import re
from decimal import Decimal

from sqlalchemy import Column, MetaData, Numeric, Table, Text, case, cast, func, select

from aquai.infrastructure.processing.tabular.normalization import (
    measurement_column,
    numeric_projection,
)
from aquai.infrastructure.processing.tabular.processor import projection_column, projection_name

_PLAIN_NUMBER = r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_STORED_NUMBER = re.compile(_PLAIN_NUMBER)
_SQL_NUMBER = f"^{_PLAIN_NUMBER}$"


#
# Devuelve las proyecciones que la regla de ingesta da a las mediciones TEXT guardadas sin una, sin escribir nada.
#
def text_projections(session, dataset) -> list[dict]:
    columns = dataset.columns_metadata
    pending = _pending_columns(columns)
    if not pending:
        return []

    table = Table(
        dataset.physical_table_name,
        MetaData(),
        *(Column(column["sql_name"], Text) for column in pending),
    )
    rows = session.execute(select(*table.c)).all()

    projections = []
    for column, values in zip(pending, zip(*rows)):
        projection = _read_projection(column, list(values), [*columns, *projections])
        if projection is not None:
            projections.append(projection)

    return projections


#
# Devuelve la advertencia que deja constancia de una proyección calculada al leer.
#
def projection_warning(projection: dict) -> str:
    return (
        f"{projection['original_name']}: calculada al leer; {projection['excluded_count']} valores no "
        "convertibles conservados en el texto original y excluidos de cálculos."
    )


#
# Agrega a una tabla autorizada las proyecciones de sus columnas que no son físicas.
#
def with_text_projections(table, dataset):
    virtual = [
        column
        for column in dataset.columns_metadata
        if column.derived_from and column.sql_name not in table.c
    ]
    if not virtual:
        return table

    return select(
        table,
        *(_numeric_text(table.c[column.derived_from]).label(column.sql_name) for column in virtual),
    ).subquery()


#
# Devuelve las columnas TEXT con encabezado de medición que todavía no tienen proyección.
#
def _pending_columns(columns: list[dict]) -> list[dict]:
    projected = {column.get("derived_from") for column in columns}

    return [
        column
        for column in columns
        if column["sql_type"] == "TEXT"
        and column["sql_name"] not in projected
        and not column.get("header_missing")
        and measurement_column(column["original_name"])
    ]


#
# Construye la proyección de una columna si la lectura SQL reproduce exactamente la regla de ingesta.
#
def _read_projection(column: dict, values: list, columns: list[dict]) -> dict | None:
    projection = numeric_projection(column["original_name"], values)
    if projection is None:
        return None

    numbers, excluded = projection
    if numbers != [_stored_number(value) for value in values]:
        return None

    return projection_column(column, projection_name(column["sql_name"], columns), numbers, excluded)


#
# Devuelve el número que la lectura SQL obtiene de un texto guardado, o None si no es un número simple.
#
def _stored_number(value: str | None) -> Decimal | None:
    text = (value or "").strip(" ")

    return Decimal(text) if _STORED_NUMBER.fullmatch(text) else None


#
# Construye la lectura numérica de una columna de texto: NULL si el valor no es un número simple.
#
def _numeric_text(field):
    text = func.pg_catalog.btrim(field, type_=Text)

    return case((text.regexp_match(_SQL_NUMBER), cast(text, Numeric)))
