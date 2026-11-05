import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "4e7c1d9a2b63"
down_revision = "9b5a29c81d40"
branch_labels = None
depends_on = None


#
# Crea las tablas de documentos, secciones, elementos y fragmentos.
#
def upgrade() -> None:
    op.create_table(
        "Document",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("sourceId", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("pageCount", sa.Integer(), nullable=True),
        sa.Column("textHash", sa.String(length=64), nullable=False),
        sa.Column("duplicateOfSourceId", sa.UUID(), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("CreatedAt", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("UpdatedAt", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            '"pageCount" IS NULL OR "pageCount" >= 1',
            name=op.f("ck_Document_page_count_positive"),
        ),
        sa.CheckConstraint('length("textHash") = 64', name=op.f("ck_Document_text_hash_length")),
        sa.ForeignKeyConstraint(
            ["duplicateOfSourceId"],
            ["Source.id"],
            name=op.f("fk_Document_duplicateOfSourceId_Source"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["sourceId"],
            ["Source.id"],
            name=op.f("fk_Document_sourceId_Source"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_Document")),
        sa.UniqueConstraint("sourceId", name=op.f("uq_Document_sourceId")),
    )
    op.create_index(op.f("ix_Document_textHash"), "Document", ["textHash"], unique=False)
    op.create_table(
        "DocumentSection",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("documentId", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "path",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("pageStart", sa.Integer(), nullable=True),
        sa.Column("pageEnd", sa.Integer(), nullable=True),
        sa.Column("CreatedAt", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("UpdatedAt", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            '"pageEnd" IS NULL OR "pageEnd" >= "pageStart"',
            name=op.f("ck_DocumentSection_page_end_ordered"),
        ),
        sa.CheckConstraint(
            '"pageStart" IS NULL OR "pageStart" >= 1',
            name=op.f("ck_DocumentSection_page_start_positive"),
        ),
        sa.CheckConstraint("position >= 0", name=op.f("ck_DocumentSection_position_nonnegative")),
        sa.ForeignKeyConstraint(
            ["documentId"],
            ["Document.id"],
            name=op.f("fk_DocumentSection_documentId_Document"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_DocumentSection")),
        sa.UniqueConstraint(
            "documentId", "position", name="uq_DocumentSection_documentId_position"
        ),
    )
    op.create_table(
        "DocumentElement",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("documentId", sa.UUID(), nullable=False),
        sa.Column("sectionId", sa.UUID(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column(
            "cells",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite"
            ),
            nullable=True,
        ),
        sa.Column("pageStart", sa.Integer(), nullable=True),
        sa.Column("pageEnd", sa.Integer(), nullable=True),
        sa.Column(
            "locations",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("CreatedAt", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("UpdatedAt", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint("kind IN ('table', 'figure')", name=op.f("ck_DocumentElement_kind_values")),
        sa.CheckConstraint(
            '"pageEnd" IS NULL OR "pageEnd" >= "pageStart"',
            name=op.f("ck_DocumentElement_page_end_ordered"),
        ),
        sa.CheckConstraint(
            '"pageStart" IS NULL OR "pageStart" >= 1',
            name=op.f("ck_DocumentElement_page_start_positive"),
        ),
        sa.CheckConstraint("position >= 0", name=op.f("ck_DocumentElement_position_nonnegative")),
        sa.ForeignKeyConstraint(
            ["documentId"],
            ["Document.id"],
            name=op.f("fk_DocumentElement_documentId_Document"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["sectionId"],
            ["DocumentSection.id"],
            name=op.f("fk_DocumentElement_sectionId_DocumentSection"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_DocumentElement")),
        sa.UniqueConstraint(
            "documentId", "position", name="uq_DocumentElement_documentId_position"
        ),
    )
    op.create_index(
        op.f("ix_DocumentElement_sectionId"), "DocumentElement", ["sectionId"], unique=False
    )
    op.create_table(
        "DocumentChunk",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("documentId", sa.UUID(), nullable=False),
        sa.Column("sectionId", sa.UUID(), nullable=True),
        sa.Column("elementId", sa.UUID(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("tokenCount", sa.Integer(), nullable=False),
        sa.Column("pageStart", sa.Integer(), nullable=True),
        sa.Column("pageEnd", sa.Integer(), nullable=True),
        sa.Column(
            "locations",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "mentions",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("CreatedAt", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("UpdatedAt", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('text', 'table', 'figure', 'references')",
            name=op.f("ck_DocumentChunk_kind_values"),
        ),
        sa.CheckConstraint(
            '"pageEnd" IS NULL OR "pageEnd" >= "pageStart"',
            name=op.f("ck_DocumentChunk_page_end_ordered"),
        ),
        sa.CheckConstraint(
            '"pageStart" IS NULL OR "pageStart" >= 1',
            name=op.f("ck_DocumentChunk_page_start_positive"),
        ),
        sa.CheckConstraint("position >= 0", name=op.f("ck_DocumentChunk_position_nonnegative")),
        sa.CheckConstraint(
            '"tokenCount" >= 0', name=op.f("ck_DocumentChunk_token_count_nonnegative")
        ),
        sa.ForeignKeyConstraint(
            ["documentId"],
            ["Document.id"],
            name=op.f("fk_DocumentChunk_documentId_Document"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["elementId"],
            ["DocumentElement.id"],
            name=op.f("fk_DocumentChunk_elementId_DocumentElement"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["sectionId"],
            ["DocumentSection.id"],
            name=op.f("fk_DocumentChunk_sectionId_DocumentSection"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_DocumentChunk")),
        sa.UniqueConstraint("documentId", "position", name="uq_DocumentChunk_documentId_position"),
    )
    op.create_index(op.f("ix_DocumentChunk_elementId"), "DocumentChunk", ["elementId"], unique=False)
    op.create_index(op.f("ix_DocumentChunk_sectionId"), "DocumentChunk", ["sectionId"], unique=False)


#
# Elimina las tablas documentales; las fuentes y sus originales se conservan.
#
def downgrade() -> None:
    op.drop_index(op.f("ix_DocumentChunk_sectionId"), table_name="DocumentChunk")
    op.drop_index(op.f("ix_DocumentChunk_elementId"), table_name="DocumentChunk")
    op.drop_table("DocumentChunk")
    op.drop_index(op.f("ix_DocumentElement_sectionId"), table_name="DocumentElement")
    op.drop_table("DocumentElement")
    op.drop_table("DocumentSection")
    op.drop_index(op.f("ix_Document_textHash"), table_name="Document")
    op.drop_table("Document")
