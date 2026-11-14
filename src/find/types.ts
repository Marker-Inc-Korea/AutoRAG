import type { Question } from "jev-use";
import type { EverythingSearchRequest, EverythingSearchResult } from "../everything/index.ts";
import type { FSearchSearchRequest, FSearchSearchResult } from "../fsearch/index.ts";
import type { RetrievalSelection } from "../retrieval/selection.ts";
import type {
	RetrievalDiagnostic,
	RetrievalOptions,
	RetrievalResult,
	RetrievalUnsearchedSurface,
} from "../retrieval/types.ts";

/** What a whole-search hit is: a file name, document content, or a datasource message. */
export type FindKind = "file" | "content" | "message";

/** One hit on one group, shared by the filename, content, and datasource surfaces. */
export interface FindHit {
	readonly id: string;
	readonly kind: FindKind;
	readonly title: string;
	readonly author: string | null;
	readonly snippet: string;
	/** Passed text, capped at {@link SEARCH_HIT_CONTENT_LIMIT} characters. */
	readonly content: string;
	/** Absolute local path for file and content hits; null for datasource messages. */
	readonly path: string | null;
	readonly line: number | null;
	/** Epoch milliseconds, when the source states one. */
	readonly timestamp: number | null;
	readonly url: string | null;
	readonly score: number | null;
}

/** The group list line, printed before any group runs. */
export interface FindGroupDescriptor {
	readonly key: string;
	readonly datasourceId?: string;
	readonly datasourceType?: string;
}

/** `off` = the surface is disabled in config; `unavailable` = its index is not built. */
export type FindGroupStatus = "ready" | "error" | "off" | "unavailable";

export interface FindGroupsLine {
	readonly type: "groups";
	readonly groups: readonly FindGroupDescriptor[];
}

export interface FindGroupLine {
	readonly type: "group";
	readonly key: string;
	readonly status: FindGroupStatus;
	readonly hits: readonly FindHit[];
	/** The group's own failure text verbatim, or null. */
	readonly error: string | null;
	readonly elapsedMs: number;
}

export interface FindDoneLine {
	readonly type: "done";
	readonly elapsedMs: number;
}

export type FindLine = FindGroupsLine | FindGroupLine | FindDoneLine;

/** The answer shape `searchSelected` returns; a structural subset of the retrieval engine's. */
export interface FindSearchAnswer {
	readonly results: readonly RetrievalResult[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	readonly unsearched: readonly RetrievalUnsearchedSurface[];
}

/**
 * The model-free runtime a whole search drives. `AutoRAGLite` satisfies it
 * structurally; tests inject a fake. Only these methods are touched, so a
 * search never enters the model-backed agent loop.
 */
export interface FindRuntime {
	listDatasources(): readonly { readonly datasourceId: string }[];
	searchSelected(query: string, selection?: RetrievalSelection, options?: RetrievalOptions): Promise<FindSearchAnswer>;
	searchFsearch(request: FSearchSearchRequest): Promise<FSearchSearchResult>;
	searchEverything(request: EverythingSearchRequest): Promise<EverythingSearchResult>;
}

/**
 * One batched `noul` judgment over relevance questions. The real implementation
 * is `createDecisionModels(...).judge`; a thrown error means "no decision model
 * can answer", and the gate fails open.
 */
export type FindJudge = (state: string, questions: Question[]) => Promise<FindJudgment>;

/** The verdicts the gate reads; a structural subset of `jev-use`'s `JudgeResult`. */
export interface FindJudgment {
	readonly verdicts: readonly FindVerdict[];
}

export interface FindVerdict {
	readonly id: string;
	readonly type?: string;
	readonly answer: number | string | null;
	readonly confidence?: number;
	readonly escalate?: boolean;
}
