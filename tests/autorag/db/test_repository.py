from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from autorag.db import schema as m
from autorag.db.repository import (
	FileRepository,
	DocumentRepository,
	PageRepository,
	CaptionRepository,
	ChunkRepository,
	ImageChunkRepository,
	QueryRepository,
	RetrievalRelationRepository,
)


def test_file_repository_crud(session):
	file_repo = FileRepository(session)

	file_row = m.File(
		type="raw",
		path="/tmp/a.txt",
		fsspec_type="local",
		fsspec_kwargs=None,
		fsspec_nickname=None,
	)
	file_repo.add(file_row)
	# Persist to assign PK and validate
	session.flush()

	fetched = file_repo.get_by_id(file_row.id)
	assert fetched is not None
	assert fetched.path == "/tmp/a.txt"

	listed = file_repo.list()
	assert any(r.id == file_row.id for r in listed)

	file_repo.delete(file_row)
	session.flush()
	assert file_repo.get_by_id(file_row.id) is None


def _seed_minimal_doc_hierarchy(session):
	"""Create File -> Document -> Page -> Caption -> Chunk and ImageChunk."""
	file_repo = FileRepository(session)
	doc_repo = DocumentRepository(session)
	page_repo = PageRepository(session)
	caption_repo = CaptionRepository(session)
	chunk_repo = ChunkRepository(session)
	img_chunk_repo = ImageChunkRepository(session)

	file_row = m.File(
		type="raw",
		path="/tmp/doc.pdf",
		fsspec_type="local",
		fsspec_kwargs=None,
	)
	file_repo.add(file_row)
	session.flush()

	doc = m.Document(filepath=file_row.id, filename="doc.pdf")
	doc_repo.add(doc)
	session.flush()

	page = m.Page(page_num=1, document_id=doc.id)
	page_repo.add(page)
	session.flush()

	caption = m.Caption(page_id=page.id, contents="hello")
	caption_repo.add(caption)
	session.flush()

	chunk = m.Chunk(parent_caption=caption.id, contents="world")
	chunk_repo.add(chunk)
	session.flush()

	img_file = m.File(
		type="image",
		path="/tmp/img.png",
		fsspec_type="local",
	)
	file_repo.add(img_file)
	session.flush()

	img_chunk = m.ImageChunk(parent_page=page.id, image_path=img_file.id)
	img_chunk_repo.add(img_chunk)
	session.flush()

	return {
		"file": file_row,
		"doc": doc,
		"page": page,
		"caption": caption,
		"chunk": chunk,
		"img_file": img_file,
		"img_chunk": img_chunk,
	}


def test_page_unique_per_document(session):
	file_repo = FileRepository(session)
	doc_repo = DocumentRepository(session)
	page_repo = PageRepository(session)

	f = m.File(type="raw", path="/tmp/u.pdf", fsspec_type="local")
	file_repo.add(f)
	session.flush()

	doc = m.Document(filepath=f.id, filename="u.pdf")
	doc_repo.add(doc)
	session.flush()

	page1 = m.Page(page_num=1, document_id=doc.id)
	page_repo.add(page1)
	session.flush()

	page2 = m.Page(page_num=1, document_id=doc.id)
	page_repo.add(page2)
	with pytest.raises(IntegrityError):
		# Violates UniqueConstraint(document_id, page_num)
		session.flush()
	# Clean up failed transaction for later tests
	session.rollback()


def test_retrieval_relation_xor_constraint(session):
	seed = _seed_minimal_doc_hierarchy(session)
	query_repo = QueryRepository(session)
	rr_repo = RetrievalRelationRepository(session)

	q = m.Query(query="what?", generation_gt=["a"])
	query_repo.add(q)
	session.flush()

	# Case 1: both null -> error
	rr_invalid_both_null = m.RetrievalRelation(
		query_id=q.id, group_index=0, group_order=0, chunk_id=None, image_chunk_id=None
	)
	rr_repo.add(rr_invalid_both_null)
	with pytest.raises(IntegrityError):
		session.flush()
	session.rollback()

	# Case 2: both non-null -> error
	rr_invalid_both_set = m.RetrievalRelation(
		query_id=q.id,
		group_index=1,
		group_order=0,
		chunk_id=seed["chunk"].id,
		image_chunk_id=seed["img_chunk"].id,
	)
	rr_repo.add(rr_invalid_both_set)
	with pytest.raises(IntegrityError):
		session.flush()
	session.rollback()

	# Case 3: only chunk_id -> ok
	rr_only_chunk = m.RetrievalRelation(
		query_id=q.id,
		group_index=2,
		group_order=0,
		chunk_id=seed["chunk"].id,
		image_chunk_id=None,
	)
	rr_repo.add(rr_only_chunk)
	session.flush()

	# Case 4: only image_chunk_id -> ok
	rr_only_image = m.RetrievalRelation(
		query_id=q.id,
		group_index=3,
		group_order=0,
		chunk_id=None,
		image_chunk_id=seed["img_chunk"].id,
	)
	rr_repo.add(rr_only_image)
	session.flush()


def test_retrieval_relation_pk_triplet_unique(session):
	seed = _seed_minimal_doc_hierarchy(session)
	query_repo = QueryRepository(session)
	rr_repo = RetrievalRelationRepository(session)

	q = m.Query(query="q", generation_gt=["x"])
	query_repo.add(q)
	session.flush()

	first = m.RetrievalRelation(
		query_id=q.id,
		group_index=0,
		group_order=0,
		chunk_id=seed["chunk"].id,
		image_chunk_id=None,
	)
	rr_repo.add(first)
	session.flush()

	dup = m.RetrievalRelation(
		query_id=q.id,
		group_index=0,
		group_order=0,
		chunk_id=None,
		image_chunk_id=seed["img_chunk"].id,
	)
	rr_repo.add(dup)
	with pytest.raises(IntegrityError):
		session.flush()
	session.rollback()
