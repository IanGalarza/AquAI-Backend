import json
from dataclasses import dataclass

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import ValidationError

from aquai.application.dtos.tabular import PreparationRecipe, ResponseFormatMetrics


@dataclass
class RecipeParseResult:
    recipe: PreparationRecipe
    response_format: ResponseFormatMetrics


class RecipeParseError(ValueError):
    def __init__(self, message: str, response_format: ResponseFormatMetrics) -> None:
        super().__init__(message)
        self.response_format = response_format


class _InvalidJSON(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise _InvalidJSON(f"La respuesta JSON repite la clave '{key}'.")
        document[key] = value
    return document


def _reject_constant(value: str) -> None:
    raise _InvalidJSON(f"La respuesta JSON contiene el valor no permitido '{value}'.")


def _decode_document(content: str, metrics: ResponseFormatMetrics) -> object:
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    try:
        document = decoder.decode(content)
    except json.JSONDecodeError:
        pass
    except _InvalidJSON as exc:
        raise RecipeParseError(str(exc), metrics) from exc
    except RecursionError as exc:
        raise RecipeParseError("La respuesta JSON tiene demasiados niveles de anidación.", metrics) from exc
    else:
        metrics.json_valid = True
        metrics.strict_json = True
        return document

    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            document, _ = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            continue
        except _InvalidJSON as exc:
            raise RecipeParseError(str(exc), metrics) from exc
        except RecursionError as exc:
            raise RecipeParseError("La respuesta JSON tiene demasiados niveles de anidación.", metrics) from exc
        if not isinstance(document, dict) or not {"status", "sheets"}.intersection(document):
            continue
        metrics.json_valid = True
        metrics.cleanup_required = True
        return document
    raise RecipeParseError("La respuesta del modelo no contiene una receta JSON válida.", metrics)


def _schema_error_detail(error: SchemaValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path) or "receta"
    descriptions = {
        "type": f"el tipo debe cumplir {error.validator_value}",
        "required": "faltan campos obligatorios",
        "additionalProperties": "contiene campos no permitidos",
        "enum": f"el valor debe pertenecer a {error.validator_value}",
        "const": f"el valor debe ser {error.validator_value}",
        "minimum": f"el valor debe ser mayor o igual a {error.validator_value}",
        "maximum": f"el valor debe ser menor o igual a {error.validator_value}",
        "minLength": f"debe contener al menos {error.validator_value} caracteres",
        "maxLength": f"no puede superar {error.validator_value} caracteres",
        "minItems": f"debe contener al menos {error.validator_value} elementos",
        "maxItems": f"no puede superar {error.validator_value} elementos",
        "minProperties": f"debe contener al menos {error.validator_value} propiedades",
        "oneOf": "no coincide con una única operación permitida",
        "anyOf": "no coincide con ninguna estructura permitida",
    }
    description = descriptions.get(error.validator, "no cumple una restricción del contrato")
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = [field for field in error.validator_value if field not in error.instance]
        description += f": {', '.join(missing)}"
    return f"{location}: {description}"


def parse_preparation_recipe_with_metrics(content: str) -> RecipeParseResult:
    metrics = ResponseFormatMetrics(recipe_executable=False)
    document = _decode_document(content, metrics)
    if isinstance(document, dict) and isinstance(document.get("status"), str):
        metrics.proposed_status = document["status"]
    validator = Draft202012Validator(PreparationRecipe.model_json_schema())
    schema_errors = sorted(validator.iter_errors(document), key=lambda error: str(error.absolute_path))
    metrics.schema_valid = not schema_errors
    if schema_errors:
        details = "; ".join(_schema_error_detail(error) for error in schema_errors)
        raise RecipeParseError(f"La receta no cumple el JSON Schema: {details}", metrics)

    try:
        recipe = PreparationRecipe.model_validate(document)
    except ValidationError as exc:
        metrics.recipe_valid = False
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'receta'}: "
            f"{error.get('ctx', {}).get('error', 'no cumple el contrato estricto de Pydantic')}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise RecipeParseError(f"La receta no cumple el contrato Pydantic: {details}", metrics) from exc
    metrics.recipe_valid = True
    return RecipeParseResult(recipe=recipe, response_format=metrics)


def parse_preparation_recipe(content: str) -> PreparationRecipe:
    return parse_preparation_recipe_with_metrics(content).recipe
