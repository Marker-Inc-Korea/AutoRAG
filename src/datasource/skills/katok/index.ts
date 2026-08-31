/**
 * KakaoTalk datasource skill barrel.
 *
 * Re-exports the retrieval methods and skill plus the underlying client/types
 * from sibling modules. Public source paths are path-opaque virtual paths of
 * the form `/kakao/<instance-id>/chunks/<chunk-id>`.
 */

export { KatokClient } from "./client.ts";
export { KatokBm25Method, type KatokMethodOptions, type KatokSearchClient, KatokSemanticMethod } from "./methods.ts";
export { KatokSkill, type KatokSkillClient, type KatokSkillOptions } from "./skill.ts";
export type {
	KatokChunkResult,
	KatokDoctorResult,
	KatokFailureReason,
	KatokHit,
	KatokIndexResult,
	KatokOptions,
	KatokSearchMode,
	KatokSearchOptions,
	KatokSearchResult,
	KatokSyncResult,
} from "./types.ts";
export { DEFAULT_KATOK_OPTIONS } from "./types.ts";
