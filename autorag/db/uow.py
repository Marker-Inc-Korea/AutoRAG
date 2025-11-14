from __future__ import annotations

from contextlib import AbstractContextManager

from sqlalchemy.orm import Session, sessionmaker

from autorag.db.repository import (
	FileRepository,
	DocumentRepository,
	PageRepository,
	CaptionRepository,
	ChunkRepository,
	ImageChunkRepository,
	CaptionChunkRelationRepository,
	QueryRepository,
	RetrievalRelationRepository,
)


class UnitOfWork(AbstractContextManager):
	def __init__(self, session_factory: sessionmaker):
		self._session_factory = session_factory
		self.session: Session | None = None

	def __enter__(self):
		self.session = self._session_factory()
		if self.session is None:
			raise RuntimeError("Failed to create a database session")
		self.files = FileRepository(self.session)
		self.documents = DocumentRepository(self.session)
		self.pages = PageRepository(self.session)
		self.captions = CaptionRepository(self.session)
		self.chunks = ChunkRepository(self.session)
		self.image_chunks = ImageChunkRepository(self.session)
		self.caption_chunk_relations = CaptionChunkRelationRepository(self.session)
		self.queries = QueryRepository(self.session)
		self.retrieval_relations = RetrievalRelationRepository(self.session)
		return self

	def __exit__(self, exc_type, exc, tb):
		try:
			if exc_type is not None and self.session is not None:
				self.session.rollback()
		finally:
			if self.session is not None:
				self.session.close()

	def commit(self):
		if self.session is None:
			raise RuntimeError("UoW session is not initialized")
		self.session.commit()

	def rollback(self):
		if self.session is None:
			return
		self.session.rollback()
