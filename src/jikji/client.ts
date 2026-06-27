import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import type {
	JikjiCompactCandidate,
	JikjiEvidenceItem,
	JikjiFailureReason,
	JikjiFindOptions,
	JikjiFindPayload,
	JikjiFindResult,
	JikjiJudgeCandidate,
	JikjiNextRead,
	JikjiOptions,
	JikjiParseResult,
} from "./types.ts";

const DEFAULT_BINARY = "jikji";
const DEFAULT_TOP_K = 20;
const DEFAULT_TIMEOUT_MS = 10_000;
const DEFAULT_MAX_BUFFER_BYTES = 1_048_576;
const MEDIA_ENV_KEY = "JIKJI_ENABLE_MEDIA_INDEX";

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: JikjiFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};

type BufferState = {
	readonly text: string;
	readonly bytes: number;
	readonly capped: boolean;
};

type SpawnJikjiRequest = {
	readonly options: JikjiOptions;
	readonly root: string;
	readonly query: string;
	readonly findOptions: JikjiFindOptions;
};

type FindArgsRequest = {
	readonly options: JikjiOptions;
	readonly root: string;
	readonly query: string;
	readonly topK?: number;
};

export class JikjiClient {
	private readonly options: JikjiOptions;

	constructor(options: JikjiOptions = {}) {
		this.options = options;
	}

	async find(root: string, query: string, options: JikjiFindOptions = {}): Promise<JikjiFindResult> {
		const result = await spawnJikji({ options: this.options, root, query, findOptions: options });
		if (!result.ok) {
			return {
				ok: false,
				reason: result.reason ?? "nonzero-exit",
				stdout: result.stdout,
				stderr: result.stderr,
				code: result.code,
			};
		}
		const parsed = parseJikjiFindPayload(result.stdout);
		if (!parsed.ok)
			return { ok: false, reason: parsed.reason, stdout: result.stdout, stderr: result.stderr, code: result.code };
		return {
			ok: true,
			payload: parsed.payload,
			stdout: result.stdout,
			stderr: result.stderr,
			code: result.code ?? 0,
		};
	}
}

export function parseJikjiFindPayload(text: string): JikjiParseResult {
	const parsed = parseJson(text);
	if (parsed.kind === "malformed") return { ok: false, reason: "malformed-json" };
	if (!isRecord(parsed.value)) return { ok: false, reason: "invalid-payload" };
	const payload = parseFindPayload(parsed.value);
	if (payload === undefined) return { ok: false, reason: "invalid-payload" };
	return { ok: true, payload };
}

function spawnJikji(request: SpawnJikjiRequest): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const options = request.options;
		const command = commandFor(options.binaryPath);
		const args = buildFindArgs({
			options,
			root: request.root,
			query: request.query,
			topK: request.findOptions.topK,
		});
		const child = spawn(command, args, {
			env: controlledEnv(options.env),
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: JikjiFailureReason | undefined;
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		const abortHandler = (): void => {
			finalReason = "aborted";
			terminate(child);
		};
		if (request.findOptions.signal?.aborted) abortHandler();
		request.findOptions.signal?.addEventListener("abort", abortHandler, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendBounded(stdout, chunk, options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES);
			if (stdout.capped) {
				finalReason = "stdout-too-large";
				terminate(child);
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendBounded(stderr, chunk, options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES);
			if (stderr.capped) {
				finalReason = "stderr-too-large";
				terminate(child);
			}
		});
		child.on("error", (error) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			request.findOptions.signal?.removeEventListener("abort", abortHandler);
			resolve({ ok: false, reason: "spawn-error", stdout: stdout.text, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			request.findOptions.signal?.removeEventListener("abort", abortHandler);
			if (finalReason !== undefined) {
				resolve({ ok: false, reason: finalReason, stdout: stdout.text, stderr: stderr.text, code });
				return;
			}
			resolve({ ok: code === 0, stdout: stdout.text, stderr: stderr.text, code });
		});
	});
}

function commandFor(binaryPath: string | undefined): string {
	return binaryPath === undefined || binaryPath === DEFAULT_BINARY ? DEFAULT_BINARY : binaryPath;
}

function buildFindArgs(request: FindArgsRequest): readonly string[] {
	const options = request.options;
	const args = [
		"find",
		request.root,
		request.query,
		"--json",
		"--top-k",
		String(request.topK ?? options.topK ?? DEFAULT_TOP_K),
	];
	if (options.includeHidden === true) args.push("--include-hidden");
	if (options.includeSensitive === true) args.push("--include-sensitive");
	if (options.parseTimeout !== undefined) args.push("--parse-timeout", String(options.parseTimeout));
	if (options.maxFiles !== undefined) args.push("--max-files", String(options.maxFiles));
	if (options.staleAfterSeconds !== undefined) args.push("--stale-after-seconds", String(options.staleAfterSeconds));
	for (const pattern of options.exclude ?? []) {
		args.push("--exclude", pattern);
	}
	return args;
}

function controlledEnv(configuredEnv: Readonly<Record<string, string | undefined>> | undefined): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (key !== MEDIA_ENV_KEY && value !== undefined) env[key] = value;
	}
	for (const [key, value] of Object.entries(configuredEnv ?? {})) {
		if (key !== MEDIA_ENV_KEY && value !== undefined) env[key] = value;
	}
	delete env[MEDIA_ENV_KEY];
	return env;
}

function terminate(child: ChildProcess): void {
	if (!child.killed) child.kill("SIGTERM");
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const chunkBytes = Buffer.byteLength(chunk);
	const nextBytes = state.bytes + chunkBytes;
	if (nextBytes <= maxBytes) return { text: state.text + chunk, bytes: nextBytes, capped: false };
	const remainingBytes = Math.max(maxBytes - state.bytes, 0);
	return { text: state.text + chunk.slice(0, remainingBytes), bytes: maxBytes, capped: true };
}

function parseJson(text: string): { readonly kind: "ok"; readonly value: unknown } | { readonly kind: "malformed" } {
	try {
		return { kind: "ok", value: JSON.parse(text) };
	} catch (error) {
		if (error instanceof SyntaxError) return { kind: "malformed" };
		throw error;
	}
}

function parseFindPayload(value: Record<string, unknown>): JikjiFindPayload | undefined {
	const paths = optionalStringArray(value.paths);
	const answerPaths = optionalStringArray(value.answer_paths);
	const evidencePack = optionalRecordArray(value.evidence_pack, parseEvidenceItem);
	const judgeCandidateSlate = optionalRecordArray(value.judge_candidate_slate, parseJudgeCandidate);
	const candidates = optionalRecordArray(value.candidates, parseCompactCandidate);
	if (paths === undefined || answerPaths === undefined || evidencePack === undefined) return undefined;
	if (judgeCandidateSlate === undefined || candidates === undefined) return undefined;
	return {
		mode: optionalString(value.mode),
		answerPackVersion: optionalNumber(value.answer_pack_version),
		root: optionalString(value.root),
		query: optionalString(value.query),
		queryType: optionalString(value.query_type),
		confidence: optionalString(value.confidence),
		confidenceScore: optionalNumber(value.confidence_score),
		recommendedAction: optionalString(value.recommended_action),
		handoffAction: optionalString(value.handoff_action),
		indexStatus: optionalString(value.index_status),
		command: optionalString(value.command),
		paths: paths ?? [],
		answerPaths: answerPaths ?? [],
		evidencePack: evidencePack ?? [],
		judgeCandidateSlate: judgeCandidateSlate ?? [],
		candidates: candidates ?? [],
	};
}

function parseEvidenceItem(value: Record<string, unknown>): JikjiEvidenceItem | undefined {
	const path = requiredString(value.path);
	const why = optionalStringArray(value.why);
	const matchedTerms = optionalStringArray(value.matched_terms);
	const evidence = optionalStringArray(value.evidence);
	if (path === undefined || why === undefined || matchedTerms === undefined || evidence === undefined)
		return undefined;
	return {
		path,
		why: why ?? [],
		matchedTerms: matchedTerms ?? [],
		evidence: evidence ?? [],
		nextRead: optionalNextRead(value.next_read),
	};
}

function parseJudgeCandidate(value: Record<string, unknown>): JikjiJudgeCandidate | undefined {
	const path = requiredString(value.path);
	const evidence = optionalStringArray(value.evidence);
	if (path === undefined || evidence === undefined) return undefined;
	return {
		rank: optionalNumber(value.rank),
		path,
		score: optionalNumber(value.score),
		evidence: evidence ?? [],
		nextRead: optionalNextRead(value.next_read),
	};
}

function parseCompactCandidate(value: Record<string, unknown>): JikjiCompactCandidate | undefined {
	const path = requiredString(value.p);
	const why = optionalStringArray(value.why);
	const terms = optionalStringArray(value.terms);
	if (path === undefined || why === undefined || terms === undefined) return undefined;
	return {
		p: path,
		s: optionalNumber(value.s),
		rank: optionalNumber(value.rank),
		why: why ?? [],
		terms: terms ?? [],
		ev: optionalString(value.ev),
		nextRead: optionalNextRead(value.next_read),
	};
}

function optionalNextRead(value: unknown): JikjiNextRead | undefined {
	if (value === undefined) return undefined;
	if (!isRecord(value)) return undefined;
	return { kind: optionalString(value.kind), path: optionalString(value.path) };
}

function optionalRecordArray<T>(
	value: unknown,
	parser: (item: Record<string, unknown>) => T | undefined,
): readonly T[] | undefined {
	if (value === undefined) return [];
	if (!Array.isArray(value)) return undefined;
	const parsed: T[] = [];
	for (const item of value) {
		if (!isRecord(item)) return undefined;
		const parsedItem = parser(item);
		if (parsedItem === undefined) return undefined;
		parsed.push(parsedItem);
	}
	return parsed;
}

function optionalStringArray(value: unknown): readonly string[] | undefined {
	if (value === undefined) return [];
	if (!Array.isArray(value)) return undefined;
	const strings: string[] = [];
	for (const item of value) {
		if (typeof item !== "string") return undefined;
		strings.push(item);
	}
	return strings;
}

function requiredString(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function optionalNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
