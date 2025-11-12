from __future__ import annotations

from sqlalchemy import (
	BigInteger,
	CheckConstraint,
	DateTime,
	ForeignKey,
	Integer,
	String,
	Text,
	UniqueConstraint,
	text,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
	pass


class File(Base):
	__tablename__ = "file"

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	type: Mapped[str] = mapped_column(String(255), nullable=False)
	path: Mapped[str] = mapped_column(String(255), nullable=False)
	fsspec_type: Mapped[str] = mapped_column(String(255), nullable=False)
	fsspec_kwargs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
	fsspec_nickname: Mapped[str | None] = mapped_column(String(255))


class Document(Base):
	__tablename__ = "document"

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	filepath: Mapped[int] = mapped_column(ForeignKey("file.id"), nullable=False)
	created_at: Mapped[str] = mapped_column(
		DateTime(timezone=True), nullable=False, server_default=text("now()")
	)
	last_modified_at: Mapped[str] = mapped_column(
		DateTime(timezone=True), nullable=False, server_default=text("now()")
	)
	filename: Mapped[str | None] = mapped_column(Text)
	author: Mapped[str | None] = mapped_column(Text)
	title: Mapped[str | None] = mapped_column(Text)


class Page(Base):
	__tablename__ = "page"
	__table_args__ = (
		UniqueConstraint("document_id", "page_num", name="uq_page_per_doc"),
	)

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	page_num: Mapped[int] = mapped_column(Integer, nullable=False)
	document_id: Mapped[int] = mapped_column(
		ForeignKey("document.id", ondelete="CASCADE"), nullable=False
	)
	image_path: Mapped[int | None] = mapped_column(ForeignKey("file.id"))
	page_metadata: Mapped[dict | None] = mapped_column(JSONB)


class Caption(Base):
	__tablename__ = "caption"

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	page_id: Mapped[int] = mapped_column(
		ForeignKey("page.id", ondelete="CASCADE"), nullable=False
	)
	contents: Mapped[str] = mapped_column(Text, nullable=False)
	module_name: Mapped[str | None] = mapped_column(String(255))
	module_kwargs: Mapped[dict | None] = mapped_column(JSONB)


class Chunk(Base):
	__tablename__ = "chunk"

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	parent_caption: Mapped[int] = mapped_column(
		ForeignKey("caption.id", ondelete="CASCADE"), nullable=False
	)
	contents: Mapped[str] = mapped_column(Text, nullable=False)
	chunk_method: Mapped[str | None] = mapped_column(String(255))
	chunk_kwargs: Mapped[dict | None] = mapped_column(JSONB)


class ImageChunk(Base):
	__tablename__ = "image_chunk"

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	parent_page: Mapped[int] = mapped_column(
		ForeignKey("page.id", ondelete="CASCADE"), nullable=False
	)
	image_path: Mapped[int] = mapped_column(ForeignKey("file.id"), nullable=False)
	chunk_method: Mapped[str | None] = mapped_column(String(255))
	chunk_kwargs: Mapped[dict | None] = mapped_column(JSONB)


class CaptionChunkRelation(Base):
	__tablename__ = "caption_chunk_relation"

	caption_id: Mapped[int] = mapped_column(
		ForeignKey("caption.id", ondelete="CASCADE"), primary_key=True
	)
	chunk_id: Mapped[int] = mapped_column(
		ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True
	)


class Query(Base):
	__tablename__ = "query"

	id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
	query: Mapped[str] = mapped_column(Text, nullable=False)
	generation_gt: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)


class RetrievalRelation(Base):
	__tablename__ = "retrieval_relation"
	__table_args__ = (
		CheckConstraint(
			"(chunk_id IS NULL) <> (image_chunk_id IS NULL)", name="ck_rr_one_only"
		),
	)

	query_id: Mapped[int] = mapped_column(
		ForeignKey("query.id", ondelete="CASCADE"), primary_key=True
	)
	group_index: Mapped[int] = mapped_column(Integer, primary_key=True)
	group_order: Mapped[int] = mapped_column(Integer, primary_key=True)
	chunk_id: Mapped[int | None] = mapped_column(
		ForeignKey("chunk.id", ondelete="CASCADE"), nullable=True
	)
	image_chunk_id: Mapped[int | None] = mapped_column(
		ForeignKey("image_chunk.id", ondelete="CASCADE"), nullable=True
	)
