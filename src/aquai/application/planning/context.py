"""Catálogo compacto completo por columnas, sin filas ni selección por score."""

import json
from hashlib import sha256

from aquai.application.contracts.planning import PlanningContextRecord
from aquai.application.contracts.retrieval import VerifiedDataset


def build_planning_context(question: str, datasets: list[VerifiedDataset], budget_bytes: int) -> PlanningContextRecord:
    column_fields = ["sql_name", "original_name", "sql_type", "unit", "numeric_eligible", "annotations"]
    warnings = list(dict.fromkeys(warning for dataset in datasets for warning in dataset.warnings))
    column_catalogs = {}
    projected = []
    for dataset in datasets:
        rows = []
        for column in dataset.columns_metadata:
            row = [getattr(column, field) for field in column_fields[:-1]]
            annotations = {field: getattr(column, field) for field in (
                "derived_from", "representation", "excluded_count", "header_missing", "meaning",
            ) if getattr(column, field) not in (None, False, "")}
            if annotations:
                row.append(annotations)
            rows.append(row)
        catalog_id = next((key for key, value in column_catalogs.items() if value == rows), None)
        if catalog_id is None:
            catalog_id = f"columns_{len(column_catalogs) + 1}"
            column_catalogs[catalog_id] = rows
        projected.append({
            "dataset_id": str(dataset.dataset_id), "version": dataset.verified_version,
            "source_id": str(dataset.source_id), "name": dataset.name,
            "description": dataset.description, "row_count": dataset.row_count,
            "source_locator": dataset.provenance.source_locator,
            "warning_ids": [warnings.index(warning) for warning in dataset.warnings],
            "column_catalog_id": catalog_id,
        })
    evidence = {
        "question": question,
        "column_fields": column_fields,
        "column_catalogs": column_catalogs,
        "warning_catalog": warnings,
        "datasets": projected,
        "catalog_scope": "Sólo candidatos recuperados y revalidados en PostgreSQL; no es el catálogo completo. Sin registros ni evidencia de claves únicas o cardinalidad de joins.",
    }
    compacted = []

    def serialized():
        return json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    content = serialized()
    if len(content) > budget_bytes:
        for dataset in evidence["datasets"]:
            description = dataset["description"] or ""
            if len(description) > 160:
                dataset["description"] = description[:160] + " [truncado]"
                compacted.append(f"{dataset['dataset_id']}.description")
        for key, columns in column_catalogs.items():
            for column in columns:
                annotations = column[-1] if isinstance(column[-1], dict) else {}
                meaning = annotations.get("meaning") or ""
                if len(meaning) > 100:
                    annotations["meaning"] = meaning[:100] + " [truncado]"
                    compacted.append(f"{key}.{column[0]}.meaning")
        content = serialized()
    return PlanningContextRecord(
        evidence=evidence, sha256=sha256(content).hexdigest(), size_bytes=len(content),
        budget_bytes=budget_bytes, compacted_fields=compacted,
    )
