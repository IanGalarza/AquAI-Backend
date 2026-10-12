from aquai.application.contracts.tabular.recipe import PreparationRecipe


def build_recipe_schema() -> dict:
    """Comparte las variantes de roles y concentra descripciones para la gramática de Ollama."""
    schema = PreparationRecipe.model_json_schema()
    definitions = schema["$defs"]
    definitions.pop("ResolvedSheetRecipe")
    block_schema = definitions["SheetRecipe"]
    branches = block_schema.pop("oneOf")
    descriptions = [
        f"{name}: {definition['description']}"
        for name, definition in branches[0]["properties"].items() if "description" in definition
    ]
    references = []
    for name, branch in zip(
        ("SheetRecipeHeader", "SheetRecipeUnknown", "SheetRecipeContent"), branches, strict=True,
    ):
        for definition in branch["properties"].values():
            definition.pop("description", None)
        definitions[name] = branch
        references.append({"$ref": f"#/$defs/{name}"})
    definitions["SheetRecipe"] = {
        "description": " ".join(descriptions), **block_schema, "oneOf": references,
    }
    definitions["ResolvedSheetRecipe"] = {"oneOf": [references[0], references[2]]}
    return schema
