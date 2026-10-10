import json

from pydantic import ValidationError

from aquai.application.dtos.tabular import PreparationRecipe


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"La respuesta JSON repite la clave '{key}'.")
        document[key] = value
    return document


def parse_preparation_recipe(content: str) -> PreparationRecipe:
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            document, _ = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            continue
        except RecursionError as exc:
            raise ValueError("La respuesta JSON tiene demasiados niveles de anidación.") from exc
        if "status" not in document and "sheets" not in document:
            continue
        try:
            return PreparationRecipe.model_validate(document)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False, include_url=False)
            )
            raise ValueError(f"La receta no cumple el contrato JSON: {details}") from exc
    raise ValueError("La respuesta del modelo no contiene una receta JSON válida.")
