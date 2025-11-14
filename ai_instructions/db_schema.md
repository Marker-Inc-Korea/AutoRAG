Table Document {
  id bigserial [pk]
  filepath bigint [ref: - File.id, not null]
  created_at datetime [not null]
  last_modified_at datetime [not null]
  filename text
  author text
  title text
}
Table Page {
  id bigserial [pk]
  page_num int [not null]
  document_id bigint [ref: > Document.id, not null]
  image_path bigint [ref: - File.id]
  page_metadata jsonb
  indexes {
    (document_id, page_num) [unique]
  }
}
Table File {
  id bigserial [pk]
  type varchar(255) [not null] // raw, image, audio, video
  path varchar(255) [not null]
  fsspec_type varchar(255) [not null]
  fsspec_kwargs jsonb
  fsspec_nickname varchar(255)
}
Table Caption {
  id bigserial [pk]
  page_id bigint [ref: > Page.id, not null]
  contents text [not null]
  module_name varchar(255) // raw, vlm, etc...
  module_kwargs jsonb
}
Table Chunk {
  id bigserial [pk]
  parent_caption bigint [ref: > Caption.id, not null]
  contents text [not null]
  chunk_method varchar(255) // recursive, semantic, etc...
  chunk_kwargs jsonb
}
Table ImageChunk {
  id bigserial [pk]
  parent_page bigint [ref: > Page.id, not null]
  image_path bigint [ref: - File.id, not null]
  chunk_method varchar(255)
  chunk_kwargs jsonb
}
Table CaptionChunkRelation {
  caption_id bigint [ref: > Caption.id, pk]
  chunk_id bigint [ref: > Chunk.id, pk]
}
Table Query {
  id bigserial [pk]
  query text [not null]
  generation_gt text[] [not null]
}
Table RetrievalRelation {
  query_id bigint [ref: > Query.id, not null]
  group_index int [not null]
  group_order int [not null]
  chunk_id bigint [ref: > Chunk.id]
  image_chunk_id bigint [ref: > ImageChunk.id]
  indexes {
    (query_id, group_index, group_order) [pk]
  }
  // chunk_id, image_chunk_id 둘 중 하나만 null인 제약 추가 필요
}
// 추후 모든 AutoRAG 시스템을 pg 기반으로 migration
