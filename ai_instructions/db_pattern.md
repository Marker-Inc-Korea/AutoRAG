SQLAlchemy + PostgreSQL을 사용하는 AutoRAG 프레임워크에서 반복적인 비즈니스 로직을 효율적으로 처리하기 위한 추천 패턴은 **Generic Repository + Unit of Work + Service Layer** 조합입니다.[^3_1][^3_2][^3_3]

## 추천 패턴 조합

### Generic Repository Pattern

반복적인 CRUD 로직을 제네릭으로 추상화하여 코드 중복을 최소화합니다.[^3_4][^3_5]

**구현 예시**:

```python
from typing import TypeVar, Generic, Type, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

T = TypeVar('T')

class GenericRepository(Generic[T]):
    def __init__(self, session: Session, model_cls: Type[T]):
        self.session = session
        self.model_cls = model_cls

    def add(self, entity: T) -> T:
        self.session.add(entity)
        return entity

    def get_by_id(self, id: int) -> Optional[T]:
        return self.session.get(self.model_cls, id)

    def get_all(self) -> List[T]:
        return self.session.query(self.model_cls).all()

    def delete(self, entity: T) -> None:
        self.session.delete(entity)
```

**AutoRAG 특화 Repository**:

```python
class CorpusRepository(GenericRepository[Corpus]):
    def find_by_query_id(self, query_id: str) -> List[Corpus]:
        """반복적으로 사용되는 비즈니스 로직"""
        return self.session.query(self.model_cls)\
            .join(CorpusQueryRelation)\
            .filter(CorpusQueryRelation.query_id == query_id)\
            .all()

    def find_by_relevance_score(self, min_score: float) -> List[Corpus]:
        """또 다른 반복 로직"""
        return self.session.query(self.model_cls)\
            .filter(self.model_cls.relevance_score >= min_score)\
            .all()
```


### Unit of Work Pattern

여러 Repository 작업을 하나의 트랜잭션으로 묶어 데이터 일관성을 보장합니다.[^3_2][^3_3]

**구현 예시**:

```python
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker

class UnitOfWork:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def __enter__(self):
        self.session = self.session_factory()
        self.corpus_repo = CorpusRepository(self.session, Corpus)
        self.query_repo = QueryRepository(self.session, Query)
        self.relation_repo = RelationRepository(self.session, CorpusQueryRelation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()
```

**사용 예시**:

```python
def add_corpus_with_queries(corpus_data, query_ids):
    with UnitOfWork(session_factory) as uow:
        corpus = Corpus(**corpus_data)
        uow.corpus_repo.add(corpus)

        for query_id in query_ids:
            relation = CorpusQueryRelation(
                corpus_id=corpus.id,
                query_id=query_id
            )
            uow.relation_repo.add(relation)

        uow.commit()  # 모든 작업이 원자적으로 처리
```


### Service Layer Pattern

반복적인 비즈니스 로직을 캡슐화하여 재사용성을 극대화합니다.[^3_6][^3_7]

**구현 예시**:

```python
class CorpusService:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def retrieve_relevant_corpus(
        self,
        query_id: str,
        min_score: float = 0.7
    ) -> List[Corpus]:
        """반복적으로 사용되는 검색 로직"""
        with UnitOfWork(self.session_factory) as uow:
            return uow.corpus_repo.find_by_query_id(query_id)

    def batch_add_corpus_with_relations(
        self,
        corpus_list: List[dict],
        query_mapping: dict
    ) -> None:
        """배치 추가 로직 - 반복 사용"""
        with UnitOfWork(self.session_factory) as uow:
            for corpus_data in corpus_list:
                corpus = Corpus(**corpus_data)
                uow.corpus_repo.add(corpus)

                if corpus.id in query_mapping:
                    for query_id in query_mapping[corpus.id]:
                        relation = CorpusQueryRelation(
                            corpus_id=corpus.id,
                            query_id=query_id
                        )
                        uow.relation_repo.add(relation)

            uow.commit()

    def update_relevance_scores(
        self,
        corpus_query_pairs: List[tuple]
    ) -> None:
        """점수 업데이트 - 반복 로직"""
        with UnitOfWork(self.session_factory) as uow:
            for corpus_id, query_id, score in corpus_query_pairs:
                relation = uow.relation_repo.find_by_corpus_and_query(
                    corpus_id, query_id
                )
                if relation:
                    relation.relevance_score = score
            uow.commit()
```


## Session 관리 Best Practices

### Context Manager 사용

SQLAlchemy Session을 안전하게 관리하기 위해 반드시 context manager를 사용합니다.[^3_8]

**Session Factory 설정**:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@localhost/autorag"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 연결 상태 확인
    pool_size=10,         # 커넥션 풀 크기
    max_overflow=20       # 추가 커넥션
)

SessionFactory = sessionmaker(bind=engine)
```

**Scoped Session (멀티스레드 환경)**:

```python
from sqlalchemy.orm import scoped_session

session_factory = scoped_session(SessionFactory)
```


## AutoRAG 적용 아키텍처

```python
# 1. 모델 정의 (도메인 레이어)
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Corpus(Base):
    __tablename__ = "corpus"
    id = Column(Integer, primary_key=True)
    content = Column(String)
    metadata = Column(String)

class Query(Base):
    __tablename__ = "query"
    id = Column(Integer, primary_key=True)
    text = Column(String)

class CorpusQueryRelation(Base):
    __tablename__ = "corpus_query_relation"
    id = Column(Integer, primary_key=True)
    corpus_id = Column(Integer, ForeignKey("corpus.id"))
    query_id = Column(Integer, ForeignKey("query.id"))
    relevance_score = Column(Float)

# 2. Repository 레이어
# (위의 GenericRepository + 특화 Repository 사용)

# 3. Service 레이어
# (위의 CorpusService 사용)

# 4. API/사용자 인터페이스
def evaluate_rag_pipeline():
    service = CorpusService(SessionFactory)

    # 반복적으로 사용되는 로직을 간단히 호출
    results = service.retrieve_relevant_corpus(
        query_id="q123",
        min_score=0.8
    )

    return results
```


## 핵심 이점

**코드 재사용성**: Generic Repository로 기본 CRUD를 재사용하고, Service Layer에서 반복 로직을 메서드화합니다.[^3_3][^3_6]

**트랜잭션 안전성**: Unit of Work로 복잡한 다중 테이블 작업을 원자적으로 처리합니다.[^3_2]

**테스트 용이성**: Repository를 Mock으로 대체하여 단위 테스트가 가능합니다.[^3_1]

**성능 최적화**: Session 관리를 중앙화하여 커넥션 풀을 효율적으로 사용합니다.[^3_8]

**확장성**: 새로운 데이터 모델 추가 시 Generic Repository를 상속받아 빠르게 구현 가능합니다.[^3_4]

이 패턴 조합은 SQLAlchemy의 강력한 기능(Session, Query API)을 최대한 활용하면서도 비즈니스 로직을 깔끔하게 분리하여 AutoRAG 프레임워크의 유지보수성과 확장성을 보장합니다.[^3_3][^3_1]
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_9]</span>

<div align="center">⁂</div>

[^3_1]: https://www.cosmicpython.com/book/chapter_02_repository.html

[^3_2]: https://www.cosmicpython.com/book/chapter_06_uow.html

[^3_3]: https://cursorrules.org/article/sqlalchemy-cursor-mdc-file

[^3_4]: https://jakpentest.tistory.com/entry/Python의-Generic을-활용한-Repository-Pattern-만들기-feat-PEP-560

[^3_5]: https://www.codeproject.com/articles/Generic-Repository-with-SQLAlchemy-and-Python

[^3_6]: https://databoom.tistory.com/entry/FastAPI-디자인-패턴-서비스-레이어-패턴11-5

[^3_7]: https://brotherdan.tistory.com/41

[^3_8]: https://www.pingcap.com/article/10-essential-tips-mastering-sqlalchemy-python/

[^3_9]: https://www.gdsanadevlog.com/frameworks/flask/real-python-flask-python에서-jparepository-like-repository-구현하기/

[^3_10]: https://www.reddit.com/r/Python/comments/1aiyken/factory_and_repository_pattern_with_sqlalchemy/

[^3_11]: https://stackoverflow.com/questions/49234978/what-is-the-best-practice-for-find-in-the-python-repository-pattern

[^3_12]: https://dev.to/manukanne/a-python-implementation-of-the-unit-of-work-and-repository-design-pattern-using-sqlmodel-3mb5

[^3_13]: https://github.com/topics/unit-of-work-pattern

[^3_14]: https://www.reddit.com/r/Python/comments/9wbk8k/repository_pattern_with_sqlalchemy/

[^3_15]: https://www.gdsanadevlog.com/frameworks/flask/real-python-flask-sqlalchemy-에서-session-commit-후-저장된-모델-객체-얻어오/

[^3_16]: https://wikidocs.net/226764

[^3_17]: https://binaryflavor.com/python과-fastapi로-지속-성장-가능한-웹서비스-개발하기-2.-mysql-sqlalchemy/

[^3_18]: https://stackoverflow.com/questions/70338102/how-to-pass-a-sqlalchemy-orm-model-to-python-generict-class
