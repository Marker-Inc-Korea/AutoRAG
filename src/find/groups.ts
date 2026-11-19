/**
 * The group runners behind `autorag lite find`: one model-free search per
 * surface — the platform file-name index, local document content, and each
 * connected datasource, each asked on its own so a slow one never blocks the
 * rest.
 */

import { basename } from "node:path";
import { MINSYNC_SURFACE } from "../retrieval/skip.ts";
import type { RetrievalResult } from "../retrieval/types.ts";
import { createRelevanceGate, type RelevanceCandidate, type SearchRelevanceGate } from "./relevance.ts";
import { capContent, contentSnippet, everythingRegexQuery, fsearchRegexQuery, matchesQueryLexically } from "./text.ts";
import type { FindGroupDescriptor, FindGroupStatus, FindHit, FindJudge, FindRuntime } from "./types.ts";

/** The result of running one group. */
export interface FindGroupOutcome {
	readonly status: FindGroupStatus;
	readonly hits: readonly FindHit[];
	readonly error: string | null;
}

export interface FindGroupRunner {
	readonly key: string;
	readonly descriptor: FindGroupDescriptor;
	run(): Promise<FindGroupOutcome>;
}

export interface BuildGroupRunnersOptions {
	readonly query: string;
	readonly topK: number;
	readonly runtime: FindRuntime;
	readonly judge?: FindJudge;
	readonly platform?: NodeJS.Platform;
	/** Whether the platform file-name index is enabled in config (default true). */
	readonly fileSearchEnabled?: boolean;
	/** Whether local semantic search is enabled in config (default true). */
	readonly semanticEnabled?: boolean;
	/** The enabled, registered datasources, in config order. */
	readonly datasources?: readonly FindGroupDescriptor[];
}

/** Filename-index failures that mean "no usable index" rather than a search error. */
const UNAVAILABLE_REASONS: Readonly<Record<string, true>> = {
	"binary-missing": true,
	"index-failed": true,
	"bundle-missing": true,
	"install-failed": true,
	"not-installed": true,
	"not-running": true,
	"startup-failed": true,
};

/** Chunks asked for per file shown: several chunks of one document usually rank together. */
const CHUNKS_PER_HIT = 3;

function text(value: unknown): string | null {
	return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

function timestampOf(metadata: Record<string, unknown>): number | null {
	if (typeof metadata.publishedAt === "number" && Number.isFinite(metadata.publishedAt)) return metadata.publishedAt;
	for (const key of ["date", "visitedAt", "startedAt", "timestamp"]) {
		const value = metadata[key];
		if (typeof value !== "string") continue;
		const parsed = /^\d+(\.\d+)?$/u.test(value) ? Number(value) * 1000 : Date.parse(value);
		if (Number.isFinite(parsed)) return parsed;
	}
	return null;
}

function urlOf(metadata: Record<string, unknown>): string | null {
	const url = text(metadata.url);
	return url !== null && /^https?:\/\//iu.test(url) ? url : null;
}

function lineOf(result: RetrievalResult): number | null {
	const line = result.metadata.lineNumber;
	return typeof line === "number" && Number.isInteger(line) && line > 0 ? line : null;
}

/** A retrieval result as a datasource hit; exported for tests. */
export function sourceHit(result: RetrievalResult, query: string): FindHit {
	return {
		id: result.id,
		kind: "message",
		// Each source names it differently: KakaoTalk `chatName`, Slack/Discord `channelName`, mail `subject`, pages `title`.
		title:
			text(result.metadata.subject) ??
			text(result.metadata.chatName) ??
			text(result.metadata.channelName) ??
			text(result.metadata.title) ??
			result.source,
		author:
			text(result.metadata.senderNickname) ??
			text(result.metadata.authorName) ??
			text(result.metadata.senderName) ??
			text(result.metadata.from) ??
			text(result.metadata.domain),
		snippet: contentSnippet(result.content, query),
		content: capContent(result.content),
		path: null,
		line: null,
		timestamp: timestampOf(result.metadata),
		url: urlOf(result.metadata),
		score: result.score,
	};
}

function fileHit(path: string): FindHit {
	return {
		id: path,
		kind: "file",
		title: basename(path),
		author: null,
		snippet: path,
		content: "",
		path,
		line: null,
		timestamp: null,
		url: null,
		score: null,
	};
}

function failureOutcome(reason: string, message: string): FindGroupOutcome {
	return { status: UNAVAILABLE_REASONS[reason] === true ? "unavailable" : "error", hits: [], error: message };
}

export function buildGroupRunners(options: BuildGroupRunnersOptions): FindGroupRunner[] {
	const platform = options.platform ?? process.platform;
	const topK = options.topK;
	const query = options.query;
	const gate: SearchRelevanceGate | undefined =
		options.judge === undefined ? undefined : createRelevanceGate({ judge: options.judge });

	const files: FindGroupRunner = {
		key: "files",
		descriptor: { key: "files" },
		async run() {
			if (options.fileSearchEnabled === false) return { status: "off", hits: [], error: null };
			if (platform === "win32") {
				const result = await options.runtime.searchEverything({
					query: everythingRegexQuery(query),
					regex: true,
					maxResults: topK,
				});
				if (!result.ok) return failureOutcome(result.reason, result.message);
				return {
					status: "ready",
					hits: result.results.slice(0, topK).map((entry) => fileHit(entry.path)),
					error: null,
				};
			}
			const result = await options.runtime.searchFsearch({
				query: fsearchRegexQuery(query),
				regex: true,
				maxResults: topK,
			});
			if (!result.ok) return failureOutcome(result.reason, result.message);
			return {
				status: "ready",
				hits: result.results.slice(0, topK).map((entry) => fileHit(entry.path)),
				error: null,
			};
		},
	};

	const content: FindGroupRunner = {
		key: "content",
		descriptor: { key: "content" },
		async run() {
			if (options.semanticEnabled === false) return { status: "off", hits: [], error: null };
			const answer = await options.runtime.searchSelected(
				query,
				{ datasourceIds: [], local: true, methods: ["hybrid"] },
				{ topK: topK * CHUNKS_PER_HIT },
			);
			const byFile = new Map<string, { readonly chunk: RetrievalResult; readonly lexical: boolean }>();
			for (const result of [...answer.results].sort((a, b) => b.score - a.score)) {
				const lexical = matchesQueryLexically(query, `${basename(result.source)}\n${result.content}`);
				const existing = byFile.get(result.source);
				if (existing === undefined) {
					if (byFile.size >= topK) continue;
					byFile.set(result.source, { chunk: result, lexical });
				} else if (!existing.lexical && lexical) {
					byFile.set(result.source, { chunk: result, lexical: true });
				}
			}
			const candidates = [...byFile.values()].map(({ chunk }): ContentCandidate => {
				const content = capContent(chunk.content);
				return {
					id: chunk.source,
					sourceLabel: "content",
					title: basename(chunk.source),
					subtitle: null,
					content,
					hit: {
						id: chunk.source,
						kind: "content",
						title: basename(chunk.source),
						author: null,
						snippet: contentSnippet(chunk.content, query),
						content,
						path: chunk.source,
						line: lineOf(chunk),
						timestamp: null,
						url: null,
						score: chunk.score,
					},
				};
			});
			const hits = (gate === undefined ? candidates : await gate.filter(query, candidates)).map(
				(entry) => entry.hit,
			);
			if (hits.length === 0 && answer.unsearched.some((entry) => entry.surface === MINSYNC_SURFACE)) {
				return { status: "unavailable", hits: [], error: null };
			}
			return { status: "ready", hits, error: null };
		},
	};

	const sources: FindGroupRunner[] = (options.datasources ?? []).map((descriptor) => ({
		key: descriptor.key,
		descriptor,
		async run() {
			const datasourceId = descriptor.datasourceId!;
			const answer = await options.runtime.searchSelected(
				query,
				{ datasourceIds: [datasourceId], local: false },
				{ topK },
			);
			const failed = answer.diagnostics.find((entry) => entry.severity === "error");
			if (answer.results.length === 0 && failed !== undefined) {
				return { status: "error", hits: [], error: failed.reason ?? failed.message };
			}
			const raw = [...answer.results].sort((a, b) => b.score - a.score).slice(0, topK);
			const candidates = raw.map((result): SourceCandidate => {
				const hit = sourceHit(result, query);
				return {
					id: hit.id,
					sourceLabel: datasourceId,
					title: hit.title,
					subtitle: hit.author,
					content: hit.content,
					hit,
				};
			});
			const hits = (gate === undefined ? candidates : await gate.filter(query, candidates)).map(
				(entry) => entry.hit,
			);
			if (hits.length === 0 && answer.unsearched.some((entry) => entry.surface === datasourceId)) {
				return { status: "unavailable", hits: [], error: null };
			}
			return { status: "ready", hits, error: null };
		},
	}));

	return [files, content, ...sources].map(guardRunner);
}

/** Any throwing group becomes an `error` outcome so one failure never affects the others. */
function guardRunner(runner: FindGroupRunner): FindGroupRunner {
	return {
		...runner,
		async run(): Promise<FindGroupOutcome> {
			try {
				return await runner.run();
			} catch (error) {
				return { status: "error", hits: [], error: error instanceof Error ? error.message : String(error) };
			}
		},
	};
}

interface ContentCandidate extends RelevanceCandidate {
	readonly hit: FindHit;
}
interface SourceCandidate extends RelevanceCandidate {
	readonly hit: FindHit;
}
