"""AddSourceRecipeValidationStatus

Revisión: 5ef1c8d4e697
Revisión anterior: fefbd016121a
Fecha de creación: 2026-10-11 19:27:22.242310
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Identificadores que Alembic usa para ordenar las migraciones.
revision: str = '5ef1c8d4e697'
down_revision: str | Sequence[str] | None = 'fefbd016121a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Crea la receta por fuente y carga el catálogo fijo de validación."""
    validation_status_table = op.create_table('ValidationStatus',
        sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_ValidationStatus')),
        sa.UniqueConstraint('name', name=op.f('uq_ValidationStatus_name'))
    )

    # Los valores quedan congelados en esta revisión para conservar su historial.
    op.bulk_insert(validation_status_table, [
        {"id": 1, "name": "pending"},
        {"id": 2, "name": "ready"},
        {"id": 3, "name": "ready_with_warnings"},
        {"id": 4, "name": "review_required"},
        {"id": 5, "name": "failed"},
    ])

    op.create_table('SourceRecipe',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('sourceId', sa.UUID(), nullable=False),
        sa.Column('recipeData', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=False),
        sa.Column('rulesSnapshot', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=False),
        sa.Column('validationStatusId', sa.Integer(), nullable=False),
        sa.Column('validationReport', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint('jsonb_typeof("recipeData") = \'object\'', name=op.f('ck_SourceRecipe_recipeData_object')),
        sa.CheckConstraint('jsonb_typeof("rulesSnapshot") = \'object\'', name=op.f('ck_SourceRecipe_rulesSnapshot_object')),
        sa.CheckConstraint('jsonb_typeof("validationReport") = \'object\'', name=op.f('ck_SourceRecipe_validationReport_object')),
        sa.ForeignKeyConstraint(['sourceId'], ['Source.id'], name=op.f('fk_SourceRecipe_sourceId_Source'), ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['validationStatusId'], ['ValidationStatus.id'], name=op.f('fk_SourceRecipe_validationStatusId_ValidationStatus'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_SourceRecipe')),
        sa.UniqueConstraint('sourceId', name=op.f('uq_SourceRecipe_sourceId'))
    )
    op.create_index(op.f('ix_SourceRecipe_validationStatusId'), 'SourceRecipe', ['validationStatusId'], unique=False)


def downgrade() -> None:
    """Elimina la receta y su catálogo en orden inverso a las dependencias."""
    op.drop_index(op.f('ix_SourceRecipe_validationStatusId'), table_name='SourceRecipe')
    op.drop_table('SourceRecipe')
    op.drop_table('ValidationStatus')
