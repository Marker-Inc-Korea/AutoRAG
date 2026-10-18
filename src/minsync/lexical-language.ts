/**
 * The BM25 tokenizer presets AutoRAG sets in MinSync's `[lexical].language`. MinSync also has `ko`, `ja`, `zh` and
 * `ar`, which AutoRAG never selects.
 */
export type MinSyncLexicalLanguage = "simple" | "multilingual";

/**
 * The tokenizer AutoRAG always asks MinSync for, whatever the global `languages` setting says.
 *
 * `simple` splits on whitespace only, so a Korean word with a particle (인증서를) never matches its stem (인증서).
 * `ko` fixes that by indexing only Kiwi's morphemes, but then a compound (연구목표) exists only as 연구 and 목표 and an
 * exact-term query loses to documents that merely contain both pieces. `multilingual` indexes the original
 * whitespace-separated words as well as every analyzer's tokens, so exact terms and inflected forms both match. On
 * a sample of real Korean documents (60 stem queries with inflected forms, 40 exact-term queries, BM25 only) it had the
 * best recall on both and kept most of `simple`'s top-5 precision on exact terms, where `ko` lost most of it.
 *
 * The preset needs the Kiwi native library; see `kiwi.ts`.
 */
export const MINSYNC_LEXICAL_LANGUAGE: MinSyncLexicalLanguage = "multilingual";
