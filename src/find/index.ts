/**
 * `autorag lite find`: a model-free whole search across the platform file-name
 * index, local document content, and every connected datasource. Pure and
 * injectable (runtime, judge); the CLI wires the real ones and prints NDJSON.
 */

export {
	type BuildGroupRunnersOptions,
	buildGroupRunners,
	type FindGroupOutcome,
	type FindGroupRunner,
	sourceHit,
} from "./groups.ts";
export {
	createRelevanceGate,
	JEV_RELEVANCE_THRESHOLD,
	type RelevanceCandidate,
	type RelevanceGateDeps,
	type SearchRelevanceGate,
} from "./relevance.ts";
export { type FindRunOptions, runFind } from "./run.ts";
export {
	capContent,
	contentSnippet,
	escapeSearchRegex,
	everythingRegexQuery,
	fsearchRegexQuery,
	matchesQueryLexically,
	SEARCH_HIT_CONTENT_LIMIT,
} from "./text.ts";
export type {
	FindDoneLine,
	FindGroupDescriptor,
	FindGroupLine,
	FindGroupStatus,
	FindGroupsLine,
	FindHit,
	FindJudge,
	FindJudgment,
	FindKind,
	FindLine,
	FindRuntime,
	FindSearchAnswer,
	FindVerdict,
} from "./types.ts";
