"""Scoped corrections to persisted recipes; a review never waives validation."""

from copy import deepcopy
from hashlib import sha256
import json
import re
from uuid import UUID

from pydantic import ValidationError

from aquai.application.contracts.tabular.recipe import PreparationRecipe, SheetRecipe
from aquai.application.contracts.tabular.policies import RecordIdentityClarification
from aquai.application.contracts.tabular.structure import StructuralSummary
from aquai.application.contracts.tabular.results import TabularEvaluation, TabularFinding
from aquai.application.errors import SourceDecisionError


class SourceReviewService:
    _block_findings = {
        "block_out_of_bounds", "unresolved_block_role", "structural_role_conflict",
        "unresolved_structural_region", "observations_in_nonobservational_block",
        "rule_profile_role_conflict", "rule_policy_identifier_separation", "suspicious_header_rows",
    }
    _transformation_findings = {
        "unit_label_changed", "rule_profile_columns_changed", "rule_profile_categorical_conversion",
        "rule_policy_type_conflict", "conversion_failures", "duplicate_column_names", "unnamed_columns",
        "nonempty_value_loss",
    }
    _add_block_findings = {"uncovered_source_cells", "unhandled_source_sheet"}
    _technical_findings = {
        "source_missing_formula_cache", "unresolved_formula_cache", "source_excel_error", "excel_error_cells",
        "execution_range_mismatch", "row_count_mismatch", "column_count_mismatch", "unexplained_row_loss",
        "source_header_evidence_mismatch", "auxiliary_row_evidence_mismatch",
        "empty_sheet_evidence_mismatch", "temporal_date_evidence_mismatch", "association_evidence_mismatch",
    }

    def build_reviews(
        self, source_id: UUID, evaluation: TabularEvaluation, rules: dict | None,
        *, revision: int = 0,
    ) -> list[dict]:
        reviews = []
        seen = set()
        for finding in evaluation.validation.findings:
            if not finding.requires_review:
                continue
            options = self._options(finding, evaluation)
            if not options and self.is_technical_finding(finding):
                continue
            identity = json.dumps({
                "source_id": str(source_id), "revision": revision,
                "finding": finding.model_dump(mode="json"),
            }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            review_id = sha256(identity.encode()).hexdigest()[:32]
            if review_id in seen:
                continue
            seen.add(review_id)
            reviews.append({
                "id": review_id, "review_id": review_id, "source_id": str(source_id),
                "revision": revision, "code": finding.code, "message": finding.message,
                "kind": "review",
                "sheet_name": finding.sheet_name, "block_id": finding.block_id,
                "cell": finding.cell, "range": self._finding_range(finding, evaluation), "region_id": finding.region_id,
                "options": options, "finding": finding.model_dump(mode="json"),
                "blocked_reason": None if options else "No hay una corrección estructurada admitida para este hallazgo; se conserva pendiente de revisión.",
            })
        return reviews

    @classmethod
    def is_technical_finding(cls, finding: TabularFinding | dict) -> bool:
        finding = finding.model_dump() if isinstance(finding, TabularFinding) else finding
        code = finding.get("code")
        actionable = cls._block_findings | cls._transformation_findings | cls._add_block_findings
        return code in cls._technical_findings or (finding.get("severity") == "error" and code not in actionable)

    @staticmethod
    def _finding_range(finding: TabularFinding, evaluation: TabularEvaluation) -> str | None:
        if finding.range or finding.cell:
            return finding.range or finding.cell
        if evaluation.recipe is None:
            return None
        block = next((block for block in evaluation.recipe.sheets
            if block.sheet_name == finding.sheet_name and block.block_id == finding.block_id), None)
        def coordinate(row, column):
            letters = ""
            while column:
                column, remainder = divmod(column - 1, 26)
                letters = chr(65 + remainder) + letters
            return f"{letters}{row}"

        if block is not None:
            return f"{coordinate(block.data_start_row, block.data_start_column)}:{coordinate(block.data_end_row, block.data_end_column)}"
        sheet = next((sheet for sheet in evaluation.structural_summary.sheets
                      if sheet.name == finding.sheet_name), None)
        if sheet is not None and sheet.rows > 0 and sheet.columns > 0:
            return f"A1:{coordinate(sheet.rows, sheet.columns)}"
        return None

    def apply(
        self, review: dict, action: str, values: dict,
        evaluation: TabularEvaluation, rules: dict | None,
    ) -> tuple[TabularEvaluation, dict, str]:
        try:
            finding = TabularFinding.model_validate(review["finding"])
        except (KeyError, ValidationError) as exc:
            raise SourceDecisionError("La revisión guardada no es válida.") from exc
        if not any(item == finding and item.requires_review for item in evaluation.validation.findings):
            raise SourceDecisionError("La revisión ya no corresponde a un hallazgo vigente.")
        supported = {option["action"] for option in self._options(finding, evaluation)}
        if action not in supported:
            raise SourceDecisionError("La acción no está admitida para esta revisión.")
        if not isinstance(values, dict):
            raise SourceDecisionError("Los valores de la decisión deben ser un objeto.")
        result = evaluation.model_copy(deep=True)
        snapshot = deepcopy(rules) if rules is not None else {
            "metadata": {}, "prompt_rules": {}, "findings": [], "review_reasons": [],
            "identity_clarifications": [], "preparation_policies": None,
        }
        try:
            match action:
                case "clarify_record_identity":
                    self._clarify_identity(review, finding, values, result, snapshot)
                    restart = "validation"
                case "correct_transformations" | "replace_block" | "add_block":
                    if result.recipe is None:
                        raise SourceDecisionError("No hay una receta vigente que pueda corregirse.")
                    recipe = result.recipe.model_dump(mode="json")
                    corrections = {
                        "correct_transformations": self._correct_transformations,
                        "replace_block": self._replace_block,
                        "add_block": self._add_block,
                    }
                    corrections[action](recipe, finding, values, result.structural_summary)
                    result.recipe = PreparationRecipe.model_validate(recipe)
                    restart = "execution"
                case _:
                    raise SourceDecisionError("La acción no está admitida para esta revisión.")
        except (ValidationError, KeyError, TypeError) as exc:
            raise SourceDecisionError("Los valores de la decisión no cumplen el contrato de la receta.") from exc
        snapshot.setdefault("source_decisions", []).append({
            "review_id": review["id"], "action": action, "values": deepcopy(values),
            "scope": {key: review.get(key) for key in ("sheet_name", "block_id", "cell", "range", "region_id")},
        })
        return result, snapshot, restart

    def _clarify_identity(
        self, review: dict, finding: TabularFinding, values: dict,
        result: TabularEvaluation, snapshot: dict,
    ) -> None:
        self._check_keys(values, {"identity", "statement"})
        statement = values["statement"]
        if values["identity"] != "distinct" or not isinstance(statement, str) or not statement.strip():
            raise SourceDecisionError("Debe confirmar registros distintos y explicar la aclaración.")
        if len(statement) > 4000:
            raise SourceDecisionError("La aclaración no puede superar 4000 caracteres.")
        original_note = self._original_note(result.structural_summary, finding)
        if original_note is None:
            raise SourceDecisionError("No se encontró la nota original en la ubicación del hallazgo.")
        clarification = RecordIdentityClarification(
            rule_id=f"source_review:{review['id']}", original_note=original_note,
            statement=statement.strip(), confirmed_by="source_review",
            source_reference=f"source:{review['source_id']}:review:{review['id']}",
            sheet_name=finding.sheet_name, cell=finding.cell,
        )
        snapshot.setdefault("identity_clarifications", []).append(clarification.model_dump(mode="json"))

    def _correct_transformations(
        self, recipe: dict, finding: TabularFinding, values: dict, summary: StructuralSummary,
    ) -> None:
        self._check_keys(values, {"transformations"})
        if not isinstance(values["transformations"], list):
            raise SourceDecisionError("transformations debe ser una lista de operaciones admitidas.")
        index = self._block_index(recipe, finding)
        previous = deepcopy(recipe["sheets"][index])
        recipe["sheets"][index]["transformations"] = values["transformations"]
        candidate = SheetRecipe.model_validate(recipe["sheets"][index])
        if candidate.model_dump(mode="json") == previous:
            raise SourceDecisionError("La decisión no modifica la propuesta afectada.")

    def _candidate_block(
        self, finding: TabularFinding, values: dict, summary: StructuralSummary,
    ) -> SheetRecipe:
        self._check_keys(values, {"block"})
        candidate = SheetRecipe.model_validate(values["block"])
        if candidate.sheet_name != finding.sheet_name:
            raise SourceDecisionError("El bloque debe pertenecer a la hoja afectada.")
        self._check_bounds(candidate, summary)
        return candidate

    def _replace_block(
        self, recipe: dict, finding: TabularFinding, values: dict, summary: StructuralSummary,
    ) -> None:
        candidate = self._candidate_block(finding, values, summary)
        if candidate.block_id != finding.block_id:
            raise SourceDecisionError("La corrección debe conservar el identificador del bloque afectado.")
        index = self._block_index(recipe, finding)
        block = candidate.model_dump(mode="json")
        if block["transformations"] != recipe["sheets"][index]["transformations"]:
            raise SourceDecisionError("Esta acción sólo corrige la estructura; las transformaciones requieren su revisión específica.")
        if block == recipe["sheets"][index]:
            raise SourceDecisionError("La decisión no modifica la propuesta afectada.")
        recipe["sheets"][index] = block

    def _add_block(
        self, recipe: dict, finding: TabularFinding, values: dict, summary: StructuralSummary,
    ) -> None:
        candidate = self._candidate_block(finding, values, summary)
        recipe["sheets"].append(candidate.model_dump(mode="json"))

    def _options(self, finding: TabularFinding, evaluation: TabularEvaluation) -> list[dict]:
        if evaluation.recipe is None:
            return []
        block = next((block for block in evaluation.recipe.sheets
            if block.sheet_name == finding.sheet_name and block.block_id == finding.block_id), None)
        if finding.code in self._block_findings and block is not None:
            return [{
                "action": "replace_block", "description": "Corregir el rol, los encabezados o los límites del bloque afectado.",
                "values": self._block_values_schema(),
            }]
        if finding.code in self._transformation_findings and block is not None:
            schema = SheetRecipe.model_json_schema()
            transformation_schema = schema["oneOf"][0]["properties"]["transformations"]
            return [{
                "action": "correct_transformations", "description": "Corregir las operaciones del bloque y volver a verificarlas.",
                "values": {"type": "object", "required": ["transformations"], "additionalProperties": False,
                           "properties": {"transformations": transformation_schema}, "$defs": schema.get("$defs", {})},
            }]
        if finding.code in self._add_block_findings and finding.sheet_name is not None:
            return [{
                "action": "add_block", "description": "Declarar un bloque de la hoja afectada para conservar las celdas sin tratamiento.",
                "values": self._block_values_schema(),
            }]
        if finding.code == "source_record_identity_uncertainty" and self._original_note(evaluation.structural_summary, finding):
            return [{
                "action": "clarify_record_identity", "description": "Confirmar que los registros cuestionados en esta nota son distintos; conservar los datos originales.",
                "values": {"type": "object", "required": ["identity", "statement"], "additionalProperties": False,
                           "properties": {"identity": {"const": "distinct"},
                                          "statement": {"type": "string", "minLength": 1, "maxLength": 4000}}},
            }]
        return []

    @staticmethod
    def _block_values_schema() -> dict:
        block = SheetRecipe.model_json_schema()
        definitions = block.pop("$defs", {})
        return {"type": "object", "required": ["block"], "additionalProperties": False,
                "properties": {"block": block}, "$defs": definitions}

    @staticmethod
    def _check_keys(values: dict, expected: set[str]):
        if set(values) != expected:
            raise SourceDecisionError("La decisión contiene valores faltantes o no admitidos.")

    @staticmethod
    def _block_index(recipe: dict, finding: TabularFinding) -> int:
        for index, block in enumerate(recipe["sheets"]):
            if block["sheet_name"] == finding.sheet_name and block["block_id"] == finding.block_id:
                return index
        raise SourceDecisionError("No se encontró el bloque afectado en la receta vigente.")

    @staticmethod
    def _check_bounds(block: SheetRecipe, summary: StructuralSummary):
        sheet = next((sheet for sheet in summary.sheets if sheet.name == block.sheet_name), None)
        if sheet is None or block.data_end_row > sheet.rows or block.data_end_column > sheet.columns or any(
            row > sheet.rows for row in block.header_rows
        ):
            raise SourceDecisionError("Los límites propuestos deben estar dentro de la hoja original.")

    @staticmethod
    def _original_note(summary: StructuralSummary, finding: TabularFinding) -> str | None:
        if finding.sheet_name is None or finding.cell is None:
            return None
        match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", finding.cell)
        if match is None:
            return None
        column = 0
        for letter in match[1]:
            column = column * 26 + ord(letter) - ord("A") + 1
        row_number = int(match[2])
        for sheet in summary.sheets:
            if sheet.name != finding.sheet_name:
                continue
            samples = [(sheet.sample, sheet.column_indices)] + [
                (region.sample, region.column_indices) for region in sheet.regions
            ]
            for rows, columns in samples:
                if column not in columns:
                    continue
                for row in rows:
                    if row.row_number == row_number and columns.index(column) < len(row.values):
                        value = row.values[columns.index(column)]
                        return value if isinstance(value, str) and value.strip() else None
        return None
