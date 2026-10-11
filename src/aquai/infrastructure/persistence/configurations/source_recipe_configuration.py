"""Receta propia de cada fuente y sus documentos JSON de preparación y validación."""

from sqlalchemy import CheckConstraint, Column, ForeignKey, Integer, Table
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import SourceRecipe
from aquai.infrastructure.persistence.configurations.common import identity_column, timestamp_columns


def configure(mapper_registry: registry) -> None:
    table = Table(
        "SourceRecipe",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "sourceId", UUID(as_uuid=True), ForeignKey("Source.id", ondelete="RESTRICT"),
            key="source_id", nullable=False, unique=True,
        ),
        Column("recipeData", JSONB(none_as_null=True), key="recipe_data", nullable=False),
        Column("rulesSnapshot", JSONB(none_as_null=True), key="rules_snapshot", nullable=False),
        Column(
            "validationStatusId", Integer,
            ForeignKey("ValidationStatus.id", ondelete="RESTRICT"),
            key="validation_status_id", nullable=False, index=True,
        ),
        Column("validationReport", JSONB(none_as_null=True), key="validation_report", nullable=True),
        *timestamp_columns(),
        CheckConstraint('jsonb_typeof("recipeData") = \'object\'', name="recipeData_object"),
        CheckConstraint('jsonb_typeof("rulesSnapshot") = \'object\'', name="rulesSnapshot_object"),
        CheckConstraint('jsonb_typeof("validationReport") = \'object\'', name="validationReport_object"),
    )
    mapper_registry.map_imperatively(
        SourceRecipe,
        table,
        properties={
            "source": relationship("Source", back_populates="source_recipe"),
            "validation_status": relationship("ValidationStatus", back_populates="source_recipes"),
        },
    )
