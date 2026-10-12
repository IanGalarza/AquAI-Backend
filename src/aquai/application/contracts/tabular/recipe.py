from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aquai.application.contracts.tabular.structure import BlockRole
from aquai.application.validators.tabular.recipeValidation import (
    validate_preparation_recipe,
    validate_sheet_recipe,
)


def _sheet_role_schema(schema: dict) -> None:
    properties = schema.pop("properties")
    common = {key: schema.pop(key) for key in ("type", "required", "additionalProperties")}
    branches = []
    for role in ("header", "unknown", "content"):
        branch_properties = deepcopy(properties)
        match role:
            case "header" | "unknown":
                branch_properties["role"].pop("enum")
                branch_properties["role"]["const"] = role
                empty_field = {"header": "header_rows", "unknown": "transformations"}[role]
                branch_properties[empty_field]["maxItems"] = 0
            case "content":
                branch_properties["role"]["enum"] = [
                    value for value in properties["role"]["enum"] if value not in {"header", "unknown"}
                ]
        branches.append({**common, "properties": branch_properties})
    schema["oneOf"] = branches


def _preparation_status_schema(schema: dict, ref_template: str) -> dict:
    definitions = schema["$defs"]
    block_branches = definitions["SheetRecipe"]["oneOf"]
    definitions["ResolvedSheetRecipe"] = {"oneOf": [block_branches[0], block_branches[2]]}
    properties = schema.pop("properties")
    common = {key: schema.pop(key) for key in ("type", "required", "additionalProperties")}
    branches = []
    for status in properties["status"]["enum"]:
        branch_properties = deepcopy(properties)
        branch_properties["status"].pop("enum")
        branch_properties["status"]["const"] = status
        match status:
            case "ready":
                branch_properties["sheets"]["minItems"] = 1
                branch_properties["warnings"]["maxItems"] = 0
                branch_properties["requires_human_review"]["const"] = False
            case "review_required":
                branch_properties["warnings"]["minItems"] = 1
                branch_properties["requires_human_review"]["const"] = True
            case "failed":
                branch_properties["warnings"]["minItems"] = 1
        if status != "review_required":
            branch_properties["sheets"]["items"] = {"$ref": ref_template.format(model="ResolvedSheetRecipe")}
        branches.append({**common, "properties": branch_properties})
    schema["oneOf"] = branches
    return schema


class RecipeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DropEmptyRows(RecipeContract):
    operation: Literal["drop_empty_rows"]


class DropEmptyColumns(RecipeContract):
    operation: Literal["drop_empty_columns"]


class StripStrings(RecipeContract):
    operation: Literal["strip_strings"]
    columns: list[str] | None = None


class RenameColumns(RecipeContract):
    operation: Literal["rename_columns"]
    mapping: dict[str, str] = Field(min_length=1)


class ConvertType(RecipeContract):
    operation: Literal["convert_type"]
    columns: list[str] = Field(min_length=1)
    target_type: Literal["number", "integer", "datetime", "boolean"]


Transformation = Annotated[
    DropEmptyRows | DropEmptyColumns | StripStrings | RenameColumns | ConvertType,
    Field(discriminator="operation"),
]


class SheetRecipe(RecipeContract):
    model_config = ConfigDict(json_schema_extra=_sheet_role_schema)

    sheet_name: str = Field(min_length=1, max_length=255, pattern=r"\S")
    block_id: str = Field(
        min_length=1, max_length=100, pattern=r"\S",
        description="Identificador explícito, no vacío y único en toda la receta.",
    )
    role: BlockRole = Field(description=(
        "Función de las celdas dentro del rango de datos. header conserva únicamente "
        "celdas de encabezado como contenido; no representa una copia de los registros."
    ))
    header_rows: list[Annotated[int, Field(ge=1)]] = Field(description=(
        "Filas originales usadas para nombrar columnas del bloque, antes de data_start_row. "
        "No requieren otro bloque header. Para role=header esta lista debe estar vacía."
    ))
    data_start_row: int = Field(ge=1, description="Primera fila de contenido del bloque, original e inclusiva.")
    data_end_row: int = Field(ge=1, description="Última fila de contenido del bloque, original e inclusiva.")
    data_start_column: int = Field(ge=1, description="Primera columna de contenido del bloque, original e inclusiva.")
    data_end_column: int = Field(ge=1, description="Última columna de contenido del bloque, original e inclusiva.")
    transformations: list[Transformation] = Field(max_length=50)

    @model_validator(mode="after")
    def validate_row_ranges(self) -> "SheetRecipe":
        validate_sheet_recipe(self)
        return self


class PreparationRecipe(RecipeContract):
    status: Literal["ready", "review_required", "failed"]
    sheets: list[SheetRecipe] = Field(max_length=120)
    warnings: list[Annotated[str, Field(min_length=1, max_length=512, pattern=r"\S")]] = Field(max_length=20)
    requires_human_review: bool

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        schema = super().model_json_schema(*args, **kwargs)
        ref_template = args[1] if len(args) > 1 else kwargs.get("ref_template", "#/$defs/{model}")
        return _preparation_status_schema(schema, ref_template)

    @model_validator(mode="after")
    def validate_sheets(self) -> "PreparationRecipe":
        validate_preparation_recipe(self)
        return self
