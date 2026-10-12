"""Apply scoped review corrections without waiving recipe validation."""

from copy import deepcopy

from aquai.application.contracts.tabular.policies import RecordIdentityClarification
from aquai.application.contracts.tabular.recipe import PreparationRecipe, SheetRecipe
from aquai.application.contracts.tabular.results import TabularEvaluation, TabularFinding
from aquai.application.contracts.tabular.structure import StructuralSummary
from aquai.application.errors import SourceDecisionError
from aquai.application.services.sourceReviewOptions import original_note


def clarify_record_identity(
    review: dict, finding: TabularFinding, values: dict,
    result: TabularEvaluation, snapshot: dict,
) -> None:
    _check_keys(values, {"identity", "statement"})
    statement = values["statement"]
    if values["identity"] != "distinct" or not isinstance(statement, str) or not statement.strip():
        raise SourceDecisionError("Debe confirmar registros distintos y explicar la aclaración.")
    if len(statement) > 4000:
        raise SourceDecisionError("La aclaración no puede superar 4000 caracteres.")
    note = original_note(result.structural_summary, finding)
    if note is None:
        raise SourceDecisionError("No se encontró la nota original en la ubicación del hallazgo.")
    clarification = RecordIdentityClarification(
        rule_id=f"source_review:{review['id']}", original_note=note,
        statement=statement.strip(), confirmed_by="source_review",
        source_reference=f"source:{review['source_id']}:review:{review['id']}",
        sheet_name=finding.sheet_name, cell=finding.cell,
    )
    snapshot.setdefault("identity_clarifications", []).append(clarification.model_dump(mode="json"))


def apply_recipe_correction(
    action: str, current_recipe: PreparationRecipe | None, finding: TabularFinding,
    values: dict, summary: StructuralSummary,
) -> PreparationRecipe:
    if current_recipe is None:
        raise SourceDecisionError("No hay una receta vigente que pueda corregirse.")
    recipe = current_recipe.model_dump(mode="json")
    match action:
        case "correct_transformations":
            _correct_transformations(recipe, finding, values)
        case "replace_block":
            _replace_block(recipe, finding, values, summary)
        case "add_block":
            _add_block(recipe, finding, values, summary)
        case _:
            raise SourceDecisionError("La acción no está admitida para esta revisión.")
    return PreparationRecipe.model_validate(recipe)


def _correct_transformations(recipe: dict, finding: TabularFinding, values: dict) -> None:
    _check_keys(values, {"transformations"})
    if not isinstance(values["transformations"], list):
        raise SourceDecisionError("transformations debe ser una lista de operaciones admitidas.")
    index = _block_index(recipe, finding)
    previous = deepcopy(recipe["sheets"][index])
    recipe["sheets"][index]["transformations"] = values["transformations"]
    candidate = SheetRecipe.model_validate(recipe["sheets"][index])
    if candidate.model_dump(mode="json") == previous:
        raise SourceDecisionError("La decisión no modifica la propuesta afectada.")


def _candidate_block(finding: TabularFinding, values: dict, summary: StructuralSummary) -> SheetRecipe:
    _check_keys(values, {"block"})
    candidate = SheetRecipe.model_validate(values["block"])
    if candidate.sheet_name != finding.sheet_name:
        raise SourceDecisionError("El bloque debe pertenecer a la hoja afectada.")
    _check_bounds(candidate, summary)
    return candidate


def _replace_block(recipe: dict, finding: TabularFinding, values: dict, summary: StructuralSummary) -> None:
    candidate = _candidate_block(finding, values, summary)
    if candidate.block_id != finding.block_id:
        raise SourceDecisionError("La corrección debe conservar el identificador del bloque afectado.")
    index = _block_index(recipe, finding)
    block = candidate.model_dump(mode="json")
    if block["transformations"] != recipe["sheets"][index]["transformations"]:
        raise SourceDecisionError("Esta acción sólo corrige la estructura; las transformaciones requieren su revisión específica.")
    if block == recipe["sheets"][index]:
        raise SourceDecisionError("La decisión no modifica la propuesta afectada.")
    recipe["sheets"][index] = block


def _add_block(recipe: dict, finding: TabularFinding, values: dict, summary: StructuralSummary) -> None:
    candidate = _candidate_block(finding, values, summary)
    recipe["sheets"].append(candidate.model_dump(mode="json"))


def _check_keys(values: dict, expected: set[str]) -> None:
    if set(values) != expected:
        raise SourceDecisionError("La decisión contiene valores faltantes o no admitidos.")


def _block_index(recipe: dict, finding: TabularFinding) -> int:
    for index, block in enumerate(recipe["sheets"]):
        if block["sheet_name"] == finding.sheet_name and block["block_id"] == finding.block_id:
            return index
    raise SourceDecisionError("No se encontró el bloque afectado en la receta vigente.")


def _check_bounds(block: SheetRecipe, summary: StructuralSummary) -> None:
    sheet = next((sheet for sheet in summary.sheets if sheet.name == block.sheet_name), None)
    if sheet is None:
        raise SourceDecisionError("Los límites propuestos deben estar dentro de la hoja original.")
    if block.data_end_row > sheet.rows or block.data_end_column > sheet.columns:
        raise SourceDecisionError("Los límites propuestos deben estar dentro de la hoja original.")
    if any(row > sheet.rows for row in block.header_rows):
        raise SourceDecisionError("Los límites propuestos deben estar dentro de la hoja original.")
