/**
 * KakaoTalk datasource skill barrel.
 *
 * Re-exports the retrieval methods and skill plus the underlying client/types
 * from sibling modules. Public source paths are path-opaque virtual paths of
 * the form `/kakao/<instance-id>/chunks/<chunk-id>`.
 */

export { LazykatokClient } from "./client.ts";
export {
	LazykatokBm25Method,
	type LazykatokMethodOptions,
	type LazykatokSearchClient,
	LazykatokSemanticMethod,
} from "./methods.ts";
export { LazykatokSkill, type LazykatokSkillClient, type LazykatokSkillOptions } from "./skill.ts";
export type {
	LazykatokChunkResult,
	LazykatokDoctorResult,
	LazykatokFailureReason,
	LazykatokHit,
	LazykatokIndexResult,
	LazykatokOptions,
	LazykatokSearchMode,
	LazykatokSearchOptions,
	LazykatokSearchResult,
	LazykatokSyncResult,
} from "./types.ts";
export { DEFAULT_LAZYKATOK_OPTIONS } from "./types.ts";
