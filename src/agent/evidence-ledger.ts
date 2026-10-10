import { createHash } from "node:crypto";
import { statSync } from "node:fs";
import { isAbsolute } from "node:path";
import type { RetrievalResult } from "../retrieval/types.ts";
import type { AutoRAGEvidenceRef } from "./emit-results-tool.ts";

/** Stored content per evidence entry; a huge chunk must not bloat the run or the memory file. */
const MAX_CONTENT_CHARS = 2_000;

const EVIDENCE_ID_PATTERN = /^e\d+$/u;

export interface LedgerEvidenceInput {
	readonly method: string;
	readonly source: string;
	readonly content: string;
	readonly retrievalResultId?: string;
	readonly chunkIndex?: number;
	readonly lineNumber?: number;
	readonly parserType?: string;
	readonly documentType?: string;
	readonly documentArea?: string;
	readonly evidenceType?: string;
	readonly evidenceLocation?: string;
}

interface LedgerEntry extends LedgerEvidenceInput {
	readonly id: string;
	readonly retrieverMix: string[];
}

export interface ResolveEvidenceOptions {
	/** Tool name used to prefix corrective errors. */
	readonly label: string;
	/** Result number the refs belong to; named in corrective errors. */
	readonly number: number;
	/** Excerpt recorded for a local file the model opened itself (no search returned it). */
	readonly fallbackContent: string;
	/** False in remote sessions: only evidence a tool returned this run may be cited. */
	readonly allowLocalFiles: boolean;
}

function metadataString(metadata: Record<string, unknown>, key: string): string | undefined {
	const value = metadata[key];
	return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function metadataNumber(metadata: Record<string, unknown>, key: string): number | undefined {
	const value = metadata[key];
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function isRegularFile(path: string): boolean {
	try {
		return statSync(path).isFile();
	} catch {
		return false;
	}
}

function toEvidenceRef(entry: LedgerEntry): AutoRAGEvidenceRef {
	return {
		method: entry.method,
		source: entry.source,
		content: entry.content,
		retrieverMix: [...entry.retrieverMix],
		...(entry.retrievalResultId !== undefined ? { retrievalResultId: entry.retrievalResultId } : {}),
		...(entry.chunkIndex !== undefined ? { chunkIndex: entry.chunkIndex } : {}),
		...(entry.lineNumber !== undefined ? { lineNumber: entry.lineNumber } : {}),
		...(entry.parserType !== undefined ? { parserType: entry.parserType } : {}),
		...(entry.documentType !== undefined ? { documentType: entry.documentType } : {}),
		...(entry.documentArea !== undefined ? { documentArea: entry.documentArea } : {}),
		...(entry.evidenceType !== undefined ? { evidenceType: entry.evidenceType } : {}),
		...(entry.evidenceLocation !== undefined ? { evidenceLocation: entry.evidenceLocation } : {}),
	};
}

/**
 * Per-run record of every piece of evidence a retrieval tool returned.
 *
 * Tools hand the model a short id (`e1`, `e2`, ...) next to each result. The
 * model cites evidence by id; the harness resolves the id back to the exact
 * source, method, content, and chunk metadata it recorded. The model never
 * retypes a path or a chunk, so it cannot misquote or invent one, and the
 * stable evidence id stored in memory is always derived from real evidence.
 */
export class EvidenceLedger {
	private entries: LedgerEntry[] = [];
	private readonly byKey = new Map<string, LedgerEntry>();
	private readonly byId = new Map<string, LedgerEntry>();
	/**
	 * Ordinal of the last issued id. Survives {@link clear}: a hosted (TUI)
	 * conversation keeps earlier runs' tool output in context, so restarting at
	 * e1 would let a stale id silently resolve to a different chunk.
	 */
	private lastOrdinal = 0;

	/** Record evidence and return its id. The same chunk always gets the same id. */
	register(input: LedgerEvidenceInput): string {
		const content = input.content.slice(0, MAX_CONTENT_CHARS);
		// Key on the full normalized chunk, not the capped stored content: two
		// distinct chunks sharing their first MAX_CONTENT_CHARS must keep their
		// own ids and metadata instead of merging into the first one.
		const digest = createHash("sha256").update(input.content.replace(/\s+/gu, " ").trim()).digest("hex");
		const key = `${input.source}\0${digest}`;
		const existing = this.byKey.get(key);
		if (existing !== undefined) {
			if (!existing.retrieverMix.includes(input.method)) existing.retrieverMix.push(input.method);
			return existing.id;
		}
		const entry: LedgerEntry = {
			...input,
			content,
			id: `e${++this.lastOrdinal}`,
			retrieverMix: [input.method],
		};
		this.entries.push(entry);
		this.byKey.set(key, entry);
		this.byId.set(entry.id, entry);
		return entry.id;
	}

	/** Record one retrieval result as returned by `tool`; backend metadata is carried over. */
	registerResult(tool: string, result: RetrievalResult): string {
		const metadata = result.metadata;
		const chunkIndex = metadataNumber(metadata, "chunkIndex");
		const lineNumber = metadataNumber(metadata, "lineNumber");
		const parserType = metadataString(metadata, "parserType");
		const documentType = metadataString(metadata, "documentType");
		const documentArea = metadataString(metadata, "documentArea");
		const evidenceType = metadataString(metadata, "evidenceType");
		const evidenceLocation = metadataString(metadata, "evidenceLocation");
		return this.register({
			method: metadataString(metadata, "method") ?? tool,
			source: result.source,
			content: result.content,
			retrievalResultId: result.id,
			...(chunkIndex !== undefined ? { chunkIndex } : {}),
			...(lineNumber !== undefined ? { lineNumber } : {}),
			...(parserType !== undefined ? { parserType } : {}),
			...(documentType !== undefined ? { documentType } : {}),
			...(documentArea !== undefined ? { documentArea } : {}),
			...(evidenceType !== undefined ? { evidenceType } : {}),
			...(evidenceLocation !== undefined ? { evidenceLocation } : {}),
		});
	}

	/**
	 * Resolve model-supplied refs to recorded evidence. A ref is an evidence id
	 * (`e3` or `[e3]`), the exact source a tool returned (a URL, a datasource
	 * id, a path), or the absolute path of a real local file the model opened
	 * itself. Anything else throws a corrective error naming the result.
	 */
	resolve(refs: readonly string[], options: ResolveEvidenceOptions): AutoRAGEvidenceRef[] {
		const resolved: AutoRAGEvidenceRef[] = [];
		const seen = new Set<string>();
		const push = (entry: LedgerEntry): void => {
			if (seen.has(entry.id)) return;
			seen.add(entry.id);
			resolved.push(toEvidenceRef(entry));
		};
		for (const raw of refs) {
			const trimmed = raw.trim();
			const ref = trimmed.startsWith("[") && trimmed.endsWith("]") ? trimmed.slice(1, -1).trim() : trimmed;
			const byId = this.byId.get(ref);
			if (byId !== undefined) {
				push(byId);
				continue;
			}
			const bySource = this.entries.filter((entry) => entry.source === ref);
			if (bySource.length > 0) {
				for (const entry of bySource) push(entry);
				continue;
			}
			if (options.allowLocalFiles && isAbsolute(ref) && isRegularFile(ref)) {
				const key = `file\0${ref}`;
				if (seen.has(key)) continue;
				seen.add(key);
				resolved.push({
					method: "bash",
					source: ref,
					content: (options.fallbackContent.trim() || ref).slice(0, MAX_CONTENT_CHARS),
				});
				continue;
			}
			throw new Error(this.unresolvedMessage(ref, EVIDENCE_ID_PATTERN.test(ref), options));
		}
		return resolved;
	}

	/** Forget every recorded entry. Ids are never reissued, so a previous run's id stays unresolvable. */
	clear(): void {
		this.entries = [];
		this.byKey.clear();
		this.byId.clear();
	}

	private unresolvedMessage(ref: string, looksLikeId: boolean, options: ResolveEvidenceOptions): string {
		const known =
			this.entries.length === 0
				? "No search or fetch has returned evidence in this run yet."
				: `Evidence ids issued in this run: ${this.entries[0]?.id}${this.entries.length > 1 ? `..${this.entries[this.entries.length - 1]?.id}` : ""}.`;
		const local = options.allowLocalFiles ? " For a file you opened yourself with bash, pass its absolute path." : "";
		return (
			`${options.label}: result ${options.number} cites evidence "${ref}" that no tool returned in this run` +
			`${looksLikeId ? " (unknown evidence id)" : ""}. ${known} ` +
			`Cite only the evidence ids shown next to retrieved results (e.g. e3).${local} Re-emit with valid refs.`
		);
	}
}
