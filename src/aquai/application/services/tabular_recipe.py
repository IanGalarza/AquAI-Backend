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
        "oneOf": "no coincide con una única variante permitida",
        "anyOf": "no coincide con ninguna estructura permitida",
    }
    description = descriptions.get(error.validator, "no cumple una restricción del contrato")
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = [field for field in error.validator_value if field not in error.instance]
        description += f": {', '.join(missing)}"
    return f"{location}: {description}"


def _schema_error_details(error: SchemaValidationError) -> list[str]:
    """Explica la variante declarada sin alterar el documento ni el rechazo del esquema."""
    if error.validator in {"oneOf", "anyOf"} and error.context:
        branches: dict[object, list[SchemaValidationError]] = {}
        for child in error.context:
            branch = next(iter(child.schema_path), None)
            branches.setdefault(branch, []).append(child)
        candidates = []
        for children in branches.values():
            wrong_discriminator = any(
                child.validator in {"const", "enum"}
                and list(child.absolute_path) in [
                    [*error.absolute_path, "status"], [*error.absolute_path, "role"],
                ]
                for child in children
            )
            if not wrong_discriminator:
                candidates.append(children)
        if len(candidates) == 1:
            return [detail for child in candidates[0] for detail in _schema_error_details(child)]
        if candidates:
            required_details = [
                {_schema_error_detail(child) for child in children if child.validator == "required"}
                for children in candidates
            ]
            shared = set.intersection(*required_details)
            if shared:
                return sorted(shared)
    return [_schema_error_detail(error)]


def parse_preparation_recipe_with_metrics(content: str) -> RecipeParseResult:
    metrics = ResponseFormatMetrics(recipe_executable=False)
    document = _decode_document(content, metrics)
    if isinstance(document, dict) and isinstance(document.get("status"), str):
        metrics.proposed_status = document["status"]
    validator = Draft202012Validator(PreparationRecipe.model_json_schema())
    schema_errors = sorted(validator.iter_errors(document), key=lambda error: str(error.absolute_path))
    metrics.schema_valid = not schema_errors
    if schema_errors:
        details = "; ".join(dict.fromkeys(
            detail for error in schema_errors for detail in _schema_error_details(error)
        ))
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
