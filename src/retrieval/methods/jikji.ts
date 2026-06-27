import { resolve } from "node:path";
import { JikjiClient } from "../../jikji/client.ts";
import { isUnsafeReturnedPath, mapJikjiPath, planJikjiSourceRoots } from "../../jikji/path-map.ts";
import type {
	JikjiCompactCandidate,
	JikjiEvidenceItem,
	JikjiFindPayload,
	JikjiJudgeCandidate,
	JikjiNextRead,
	JikjiOptions,
} from "../../jikji/types.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export interface JikjiMethodOptions extends JikjiOptions {
	readonly root: string;
	readonly searchPaths: readonly string[];
}

type CandidateEvidence = {
	readonly path: string;
	readonly content: string;
	readonly score?: number;
	readonly why: readonly string[];
	readonly matchedTerms: readonly string[];
	readonly nextRead?: JikjiNextRead;
};

export class JikjiMethod implements RetrievalMethod {
	private readonly root: string;
	private readonly searchPaths: readonly string[];
	private readonly client: JikjiClient;
	private readonly options: JikjiMethodOptions;

	constructor(options: JikjiMethodOptions) {
		this.root = resolve(options.root);
		this.searchPaths = options.searchPaths;
		this.options = options;
		this.client = new JikjiClient(options);
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "jikji",
			type: "hybrid",
			description: "optional Jikji CLI local file-discovery retrieval over configured source directories",
			status: "active",
			capabilities: [
				"local-file-discovery",
				"cli-json",
				"fielded-search",
				"agent-handoff",
				"opaque-root-relative-paths",
			],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const results: RetrievalResult[] = [];
		const seenSources = new Set<string>();
		for (const sourceRoot of planJikjiSourceRoots(this.searchPaths)) {
			const findResult = await this.client.find(sourceRoot.rootPath, query, {
				topK: this.options.topK ?? options.topK,
				signal: options.signal,
			});
			if (!findResult.ok) continue;
			for (const candidate of candidatesFrom(findResult.payload)) {
				const source = mapJikjiPath(sourceRoot, candidate.path);
				if (source === undefined || seenSources.has(source)) continue;
				seenSources.add(source);
				results.push({
					id: `${source}:jikji`,
					content: candidate.content || candidate.path,
					source,
					score: scoreFor(findResult.payload, candidate),
					metadata: metadataFor(findResult.payload, candidate),
				});
			}
		}
		return options.topK === undefined ? results : results.slice(0, options.topK);
	}
}

function candidatesFrom(payload: JikjiFindPayload): CandidateEvidence[] {
	const preferredPaths = payload.answerPaths.length > 0 ? payload.answerPaths : payload.paths;
	if (preferredPaths.length > 0) return preferredPaths.map((path) => evidenceForPath(payload, path));
	return payload.candidates.map((candidate) => evidenceForCompactCandidate(candidate));
}

function evidenceForPath(payload: JikjiFindPayload, path: string): CandidateEvidence {
	const evidence = payload.evidencePack.find((item) => item.path === path);
	const judged = payload.judgeCandidateSlate.find((item) => item.path === path);
	if (evidence) return evidenceForPackItem(evidence, judged);
	if (judged) return evidenceForJudgeCandidate(judged);
	const compact = payload.candidates.find((item) => item.p === path);
	if (compact) return evidenceForCompactCandidate(compact);
	return { path, content: path, why: [], matchedTerms: [] };
}

function evidenceForPackItem(item: JikjiEvidenceItem, judged: JikjiJudgeCandidate | undefined): CandidateEvidence {
	return {
		path: item.path,
		content: item.evidence[0] ?? item.path,
		score: judged?.score,
		why: item.why,
		matchedTerms: item.matchedTerms,
		nextRead: item.nextRead,
	};
}

function evidenceForJudgeCandidate(item: JikjiJudgeCandidate): CandidateEvidence {
	return {
		path: item.path,
		content: item.evidence[0] ?? item.path,
		score: item.score,
		why: [],
		matchedTerms: [],
		nextRead: item.nextRead,
	};
}

function evidenceForCompactCandidate(item: JikjiCompactCandidate): CandidateEvidence {
	return {
		path: item.p,
		content: item.ev ?? item.p,
		score: item.s,
		why: item.why,
		matchedTerms: item.terms,
		nextRead: item.nextRead,
	};
}

function scoreFor(payload: JikjiFindPayload, candidate: CandidateEvidence): number {
	return candidate.score ?? payload.confidenceScore ?? confidenceScore(payload.confidence);
}

function confidenceScore(confidence: string | undefined): number {
	switch (confidence) {
		case "high":
			return 1;
		case "medium_high":
			return 0.75;
		case "medium":
			return 0.5;
		case "low":
			return 0.25;
		default:
			return 0;
	}
}

function metadataFor(payload: JikjiFindPayload, candidate: CandidateEvidence): Record<string, unknown> {
	return {
		method: "jikji",
		confidence: payload.confidence,
		handoffAction: payload.handoffAction,
		indexStatus: payload.indexStatus,
		queryType: payload.queryType,
		why: candidate.why,
		matchedTerms: candidate.matchedTerms,
		nextRead: sanitizeNextRead(candidate.nextRead),
	};
}

function sanitizeNextRead(nextRead: JikjiNextRead | undefined): JikjiNextRead | undefined {
	if (nextRead === undefined) return undefined;
	if (nextRead.path === undefined) return { kind: nextRead.kind };
	const normalized = nextRead.path.replace(/\\/g, "/");
	if (isUnsafeReturnedPath(normalized) || normalized.split("/").includes("..")) return { kind: nextRead.kind };
	return { kind: nextRead.kind, path: normalized };
}
