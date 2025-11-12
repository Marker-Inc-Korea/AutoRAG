-- Auto-generated from your design (PostgreSQL 16+)

-- 1) File table
CREATE TABLE IF NOT EXISTS file (
  id              BIGSERIAL PRIMARY KEY,
  type            VARCHAR(255) NOT NULL,         -- raw, image, audio, video
  path            VARCHAR(255) NOT NULL,
  fsspec_type     VARCHAR(255) NOT NULL,
  fsspec_kwargs   JSONB,
  fsspec_nickname VARCHAR(255)
);

-- 2) Document table
CREATE TABLE IF NOT EXISTS document (
  id               BIGSERIAL PRIMARY KEY,
  filepath         BIGINT NOT NULL REFERENCES file(id),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  filename         TEXT,
  author           TEXT,
  title            TEXT
);

-- 3) Page table
CREATE TABLE IF NOT EXISTS page (
  id          BIGSERIAL PRIMARY KEY,
  page_num    INT NOT NULL,
  document_id BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
  image_path  BIGINT REFERENCES file(id),
  metadata    JSONB,
  CONSTRAINT uq_page_per_doc UNIQUE (document_id, page_num)
);

-- 4) Caption table
CREATE TABLE IF NOT EXISTS caption (
  id            BIGSERIAL PRIMARY KEY,
  page_id       BIGINT NOT NULL REFERENCES page(id) ON DELETE CASCADE,
  contents      TEXT NOT NULL,
  module_name   VARCHAR(255),
  module_kwargs JSONB
);

-- 5) Chunk table
CREATE TABLE IF NOT EXISTS chunk (
  id             BIGSERIAL PRIMARY KEY,
  parent_caption BIGINT NOT NULL REFERENCES caption(id) ON DELETE CASCADE,
  contents       TEXT NOT NULL,
  chunk_method   VARCHAR(255),
  chunk_kwargs   JSONB
);

-- 6) ImageChunk table
CREATE TABLE IF NOT EXISTS image_chunk (
  id           BIGSERIAL PRIMARY KEY,
  parent_page  BIGINT NOT NULL REFERENCES page(id) ON DELETE CASCADE,
  image_path   BIGINT NOT NULL REFERENCES file(id),
  chunk_method VARCHAR(255),
  chunk_kwargs JSONB
);

-- 7) CaptionChunkRelation (M2M)
CREATE TABLE IF NOT EXISTS caption_chunk_relation (
  caption_id BIGINT NOT NULL REFERENCES caption(id) ON DELETE CASCADE,
  chunk_id   BIGINT NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
  PRIMARY KEY (caption_id, chunk_id)
);

-- 8) Query table
CREATE TABLE IF NOT EXISTS query (
  id            BIGSERIAL PRIMARY KEY,
  query         TEXT NOT NULL,
  generation_gt TEXT[] NOT NULL
);

-- 9) RetrievalRelation table (chunk_id XOR image_chunk_id)
CREATE TABLE IF NOT EXISTS retrieval_relation (
  query_id       BIGINT NOT NULL REFERENCES query(id) ON DELETE CASCADE,
  group_index    INT NOT NULL,
  group_order    INT NOT NULL,
  chunk_id       BIGINT REFERENCES chunk(id) ON DELETE CASCADE,
  image_chunk_id BIGINT REFERENCES image_chunk(id) ON DELETE CASCADE,
  PRIMARY KEY (query_id, group_index, group_order),
  CONSTRAINT ck_rr_one_only CHECK (
    (chunk_id IS NULL) <> (image_chunk_id IS NULL)
  )
);

-- 10) last_modified_at trigger for document
CREATE OR REPLACE FUNCTION set_last_modified_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.last_modified_at := now();
  RETURN NEW;
END $$;

DO $$
BEGIN
  IF NOT EXISTS(
    SELECT 1
    FROM information_schema.triggers
    WHERE event_object_table = 'document'
    AND trigger_name = 'tr_document_set_last_modified'
  )
  THEN
    CREATE TRIGGER tr_document_set_last_modified
    BEFORE UPDATE ON document
    FOR EACH ROW EXECUTE FUNCTION set_last_modified_at();
  END IF;
END $$;
