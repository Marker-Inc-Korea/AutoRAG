from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from autorag.db import schema as m


class GenericRepository:
	def __init__(self, session: Session, model_cls: type):
		self.session = session
		self.model_cls = model_cls

	def add(self, entity):
		self.session.add(entity)
		return entity

	def get_by_id(self, id: int):
		return self.session.get(self.model_cls, id)

	def list(self):
		stmt = select(self.model_cls)
		return list(self.session.execute(stmt).scalars())

	def delete(self, entity):
		self.session.delete(entity)


class FileRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.File)


class DocumentRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.Document)


class PageRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.Page)


class CaptionRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.Caption)


class ChunkRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.Chunk)


class ImageChunkRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.ImageChunk)


class CaptionChunkRelationRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.CaptionChunkRelation)


class QueryRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.Query)


class RetrievalRelationRepository(GenericRepository):
	def __init__(self, session: Session):
		super().__init__(session, m.RetrievalRelation)
