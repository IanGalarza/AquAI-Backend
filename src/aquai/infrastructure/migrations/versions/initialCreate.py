"""Esquema inicial de fuentes, datasets y entidades de chat de AquAI."""

import re

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "9b5a29c81d40"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Crea el esquema consolidado; no interpreta ni transforma originales."""
    op.create_table('Conversation',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('lastMessageAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_Conversation'))
    )
    op.create_table('MessageStatus',
    sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_MessageStatus'))
    )
    op.create_table('Source',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('originalFilename', sa.String(), nullable=False),
    sa.Column('fileType', sa.String(), nullable=False),
    sa.Column('mimeType', sa.String(), nullable=False),
    sa.Column('fileSizeBytes', sa.BigInteger(), nullable=False),
    sa.Column('isActive', sa.Boolean(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('contentHash', sa.String(length=64), nullable=False),
    sa.Column('processingStatus', sa.String(length=32), nullable=False),
    sa.Column('processingReport', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
    sa.Column('semanticIndexStatus', sa.String(length=16), nullable=False),
    sa.Column('fileContent', sa.LargeBinary(), nullable=False),
    sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint('"fileSizeBytes" >= 0', name=op.f('ck_Source_file_size_nonnegative')),
    sa.CheckConstraint('"processingStatus" IN (\'uploaded\', \'processing\', \'ready\', \'ready_with_warnings\', \'failed\')', name=op.f('ck_Source_processing_status_values')),
    sa.CheckConstraint('"semanticIndexStatus" IN (\'pending\', \'processing\', \'ready\', \'failed\')', name=op.f('ck_Source_semantic_index_status_values')),
    sa.CheckConstraint('length("contentHash") = 64', name=op.f('ck_Source_content_hash_length')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_Source')),
    sa.UniqueConstraint('contentHash', name=op.f('uq_Source_contentHash'))
    )
    op.create_index(op.f('ix_Source_processingStatus'), 'Source', ['processingStatus'], unique=False)
    op.create_index(op.f('ix_Source_semanticIndexStatus'), 'Source', ['semanticIndexStatus'], unique=False)
    op.create_table('Dataset',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('sourceId', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('sourceLocator', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
    sa.Column('physicalTableName', sa.String(length=63), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('isActive', sa.Boolean(), nullable=False),
    sa.Column('rowCount', sa.BigInteger(), nullable=False),
    sa.Column('columnsMetadata', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
    sa.Column('queryable', sa.Boolean(), nullable=False),
    sa.Column('qualityWarnings', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
    sa.Column('semanticDescription', sa.Text(), nullable=True),
    sa.Column('semanticPointId', sa.UUID(), nullable=False),
    sa.Column('CreatedAt', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('UpdatedAt', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint('"rowCount" >= 0', name=op.f('ck_Dataset_row_count_nonnegative')),
    sa.CheckConstraint('NOT queryable OR "rowCount" > 0', name=op.f('ck_Dataset_queryable_has_rows')),
    sa.CheckConstraint('version >= 1', name=op.f('ck_Dataset_version_positive')),
    sa.ForeignKeyConstraint(['sourceId'], ['Source.id'], name=op.f('fk_Dataset_sourceId_Source'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_Dataset')),
    sa.UniqueConstraint('id', 'sourceId', name='uq_Dataset_id_sourceId'),
    sa.UniqueConstraint('physicalTableName', name=op.f('uq_Dataset_physicalTableName')),
    sa.UniqueConstraint('semanticPointId', name=op.f('uq_Dataset_semanticPointId'))
    )
    op.create_index(op.f('ix_Dataset_sourceId'), 'Dataset', ['sourceId'], unique=False)
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

    op.bulk_insert(sa.table("MessageStatus", sa.column("id", sa.Integer()), sa.column("name", sa.String())), [
        {"id": 1, "name": "pending"},
        {"id": 2, "name": "processing"},
        {"id": 3, "name": "completed"},
        {"id": 4, "name": "failed"},
        {"id": 5, "name": "cancelled"},
    ])


def downgrade() -> None:
    """Retira tablas físicas registradas y luego el catálogo según sus dependencias."""
    if not context.is_offline_mode():
        physical_tables = op.get_bind().scalars(sa.text('SELECT "physicalTableName" FROM "Dataset"')).all()
        if any(not re.fullmatch(r"dataset_[0-9a-f]{32}", name) for name in physical_tables):
            raise RuntimeError("El catálogo contiene una tabla física no reconocida.")
        for name in physical_tables:
            op.drop_table(name)
    op.drop_index(op.f('ix_MessageSource_sourceId'), table_name='MessageSource')
    op.drop_index(op.f('ix_MessageSource_messageId'), table_name='MessageSource')
    op.drop_index(op.f('ix_MessageSource_datasetId'), table_name='MessageSource')
    op.drop_table('MessageSource')
    op.drop_index(op.f('ix_Message_statusId'), table_name='Message')
    op.drop_table('Message')
    op.drop_index(op.f('ix_Dataset_sourceId'), table_name='Dataset')
    op.drop_table('Dataset')
    op.drop_index(op.f('ix_Source_semanticIndexStatus'), table_name='Source')
    op.drop_index(op.f('ix_Source_processingStatus'), table_name='Source')
    op.drop_table('Source')
    op.drop_table('MessageStatus')
    op.drop_table('Conversation')
