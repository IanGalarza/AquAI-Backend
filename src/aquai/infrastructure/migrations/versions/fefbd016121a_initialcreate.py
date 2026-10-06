"""InitialCreate

Revisión: fefbd016121a
Revisión anterior: ninguna
Fecha de creación: 2026-10-06 12:15:53.890083
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Identificadores que Alembic usa para ordenar las migraciones.
revision: str = 'fefbd016121a'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Crea las siete tablas de AquAI y carga los catálogos iniciales."""
    op.create_table('Conversation',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('lastMessageAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_Conversation'))
    )
    message_status_table = op.create_table('MessageStatus',
        sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_MessageStatus'))
    )
    processing_status_table = op.create_table('ProcessingStatus',
        sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_ProcessingStatus'))
    )

    # Los valores quedan congelados en esta revisión para conservar su historial.
    op.bulk_insert(processing_status_table, [
        {"id": 1, "name": "uploaded"},
        {"id": 2, "name": "processing"},
        {"id": 3, "name": "ready"},
        {"id": 4, "name": "ready_with_warnings"},
        {"id": 5, "name": "review_required"},
        {"id": 6, "name": "failed"},
    ])
    op.bulk_insert(message_status_table, [
        {"id": 1, "name": "pending"},
        {"id": 2, "name": "processing"},
        {"id": 3, "name": "completed"},
        {"id": 4, "name": "failed"},
        {"id": 5, "name": "cancelled"},
    ])

    op.create_table('Message',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('conversationId', sa.UUID(), nullable=False),
        sa.Column('sequenceNumber', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('contentData', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=False),
        sa.Column('statusId', sa.Integer(), nullable=False),
        sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['conversationId'], ['Conversation.id'], name=op.f('fk_Message_conversationId_Conversation'), ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['statusId'], ['MessageStatus.id'], name=op.f('fk_Message_statusId_MessageStatus'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_Message')),
        sa.UniqueConstraint('conversationId', 'sequenceNumber', name='uq_Message_conversationId_sequenceNumber')
    )
    op.create_index(op.f('ix_Message_statusId'), 'Message', ['statusId'], unique=False)
    op.create_table('Source',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('originalFilename', sa.String(), nullable=False),
        sa.Column('fileType', sa.String(), nullable=False),
        sa.Column('mimeType', sa.String(), nullable=False),
        sa.Column('fileSizeBytes', sa.BigInteger(), nullable=False),
        sa.Column('isActive', sa.Boolean(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('processingStatusId', sa.Integer(), nullable=False),
        sa.Column('fileContent', postgresql.BYTEA(), nullable=False),
        sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['processingStatusId'], ['ProcessingStatus.id'], name=op.f('fk_Source_processingStatusId_ProcessingStatus'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_Source'))
    )
    op.create_index(op.f('ix_Source_processingStatusId'), 'Source', ['processingStatusId'], unique=False)
    op.create_table('Dataset',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('sourceId', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('sourceLocator', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=False),
        sa.Column('physicalTableName', sa.String(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('isActive', sa.Boolean(), nullable=False),
        sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['sourceId'], ['Source.id'], name=op.f('fk_Dataset_sourceId_Source'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_Dataset')),
        sa.UniqueConstraint('id', 'sourceId', name='uq_Dataset_id_sourceId')
    )
    op.create_index(op.f('ix_Dataset_sourceId'), 'Dataset', ['sourceId'], unique=False)
    op.create_table('MessageSource',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('messageId', sa.UUID(), nullable=False),
        sa.Column('sourceId', sa.UUID(), nullable=False),
        sa.Column('datasetId', sa.UUID(), nullable=True),
        sa.Column('ReferenceOrder', sa.Integer(), nullable=False),
        sa.Column('details', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=False),
        sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['datasetId', 'sourceId'], ['Dataset.id', 'Dataset.sourceId'], name='fk_MessageSource_datasetId_sourceId_Dataset', ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['messageId'], ['Message.id'], name=op.f('fk_MessageSource_messageId_Message'), ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['sourceId'], ['Source.id'], name=op.f('fk_MessageSource_sourceId_Source'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_MessageSource'))
    )
    op.create_index(op.f('ix_MessageSource_datasetId'), 'MessageSource', ['datasetId'], unique=False)
    op.create_index(op.f('ix_MessageSource_messageId'), 'MessageSource', ['messageId'], unique=False)
    op.create_index(op.f('ix_MessageSource_sourceId'), 'MessageSource', ['sourceId'], unique=False)


def downgrade() -> None:
    """Elimina las tablas y sus datos en orden inverso a las dependencias."""
    op.drop_index(op.f('ix_MessageSource_sourceId'), table_name='MessageSource')
    op.drop_index(op.f('ix_MessageSource_messageId'), table_name='MessageSource')
    op.drop_index(op.f('ix_MessageSource_datasetId'), table_name='MessageSource')
    op.drop_table('MessageSource')
    op.drop_index(op.f('ix_Dataset_sourceId'), table_name='Dataset')
    op.drop_table('Dataset')
    op.drop_index(op.f('ix_Source_processingStatusId'), table_name='Source')
    op.drop_table('Source')
    op.drop_index(op.f('ix_Message_statusId'), table_name='Message')
    op.drop_table('Message')
    op.drop_table('ProcessingStatus')
    op.drop_table('MessageStatus')
    op.drop_table('Conversation')
