import { existsSync, readFileSync } from "node:fs";
import type { AutoRAGRefreshResult } from "../../agent/agent.ts";
import { createAutoRAGLite } from "../../core.ts";
import { detectMirrorStaleness } from "../../mirror/index.ts";
import { refreshReadinessPath } from "../../mirror/paths.ts";
import type { RetrievalDiagnostic, RetrievalResult, RetrievalUnsearchedSurface } from "../../retrieval/types.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

// ---------------------------------------------------------------------------
// Lite retrieve envelope types
// ---------------------------------------------------------------------------

export interface LiteRetrieveEnvelope {
	readonly ok: true;
	readonly query: string;
	/**
	 * True when the last refresh does not cover the current sources. Retrieval still
	 * runs; the caller decides whether to trust the result or refresh first.
	 */
	readonly stale: boolean;
	readonly results: readonly LiteRetrieveResultItem[];
	/**
	 * Retrieval surfaces that did not run for this query. A consumer that only
	 * reads `results` can check `unsearched.length > 0` to learn the answer is
	 * partial — for example local MinSync files skipped while an index sync holds
	 * the workspace lock, leaving only datasource hits.
	 */
	readonly unsearched: readonly LiteRetrieveUnsearched[];

	readonly diagnostics: readonly LiteRetrieveDiagnostic[];
}

export interface LiteRetrieveUnsearched {
	/** Surface label: "minsync" for local files, or the datasource id. */
	readonly surface: string;
	/** Retrieval method names on this surface that did not run. */
	readonly methods: readonly string[];
	/** The underlying failure verbatim, including CLI stderr and exit status. */
	readonly reason: string;
}

export interface LiteRetrieveResultItem {
	readonly number: number;
	readonly source: string;
	readonly method: string;
	readonly score: number;
	readonly metadata: Record<string, unknown>;
	readonly content: string;
}

export interface LiteRetrieveDiagnostic {
	readonly code: string;
	readonly severity: string;
	readonly message: string;
	readonly source?: string;
	/** Why a source is considered stale, e.g. `mtime-and-size-changed`. */
	readonly reason?: string;
	/** Suggested recovery action, e.g. `refresh`. */
	readonly action?: string;
}

// ---------------------------------------------------------------------------
// Internal envelope for the pre-refresh state (ok: false)
// ---------------------------------------------------------------------------

interface IndexNotReadyEnvelope {
	readonly ok: false;
	readonly query: string;
	readonly diagnostics: readonly {
		readonly code: "index-not-ready";
		readonly severity: "error";
		readonly message: string;
		readonly source?: string;
		readonly reason?: string;
		readonly action?: string;
	}[];
}

// ---------------------------------------------------------------------------
// Rejection envelope for bad-top-k etc.
// ---------------------------------------------------------------------------

interface RejectionEnvelope {
	readonly ok: false;
	readonly error: string;
}

// ---------------------------------------------------------------------------
// Parsed options
// ---------------------------------------------------------------------------

interface LiteRetrieveOptions {
	topK?: number;
	scope?: string;
	allowedTags?: readonly string[];
}

function parseIntOptional(value: string | boolean | undefined): number | undefined {
	if (typeof value !== "string" || value.trim() === "") return undefined;
	const parsed = Number(value);
	if (!Number.isFinite(parsed)) return undefined;
	return Math.trunc(parsed);
}

function parseCsvStrings(value: string | boolean | undefined): readonly string[] | undefined {
	if (typeof value !== "string" || value.trim() === "") return undefined;
	const parts = value
		.split(",")
		.map((part) => part.trim())
		.filter((part) => part !== "");
	return parts.length > 0 ? parts : undefined;
}

function buildRetrieveOptions(flags: CommandContext["flags"]): LiteRetrieveOptions {
	const options: LiteRetrieveOptions = {};
	const topK = parseIntOptional(flags["top-k"]);
	if (topK !== undefined) options.topK = topK;
	if (typeof flags.scope === "string" && flags.scope.trim() !== "") options.scope = flags.scope;
	const tags = parseCsvStrings(flags.tags);
	if (tags !== undefined) options.allowedTags = tags;
	return options;
}

/**
 * Parse --top-k with strict validation.
 * Returns the parsed value when valid, or a rejection error string.
 * Returns `undefined` when unset (pass-through).
 */
function parseTopK(
	value: string | boolean | undefined,
): { readonly kind: "ok"; readonly value: number | undefined } | { readonly kind: "reject"; readonly error: string } {
	if (value === undefined || value === true || value === false) return { kind: "ok", value: undefined };
	const parsed = Number(value);
	if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
		return { kind: "reject", error: `Invalid --top-k value: "${String(value)}". Must be a positive integer.` };
	}
	if (parsed < 1) {
		return { kind: "reject", error: `Invalid --top-k value: ${parsed}. Must be a positive integer.` };
	}
	return { kind: "ok", value: parsed };
}

// ---------------------------------------------------------------------------
// Diagnostic projection
// ---------------------------------------------------------------------------

function diagnosticProjection(d: {
	readonly code: string;
	readonly severity: string;
	readonly message: string;
	readonly source?: string;
	readonly reason?: string;
	readonly action?: string;
}): LiteRetrieveDiagnostic {
	const out: LiteRetrieveDiagnostic = {
		code: d.code,
		severity: d.severity,
		message: d.message,
	};
	if (d.source !== undefined) {
		(out as { source: string }).source = d.source;
	}
	if (d.reason !== undefined) {
		(out as { reason: string }).reason = d.reason;
	}
	if (d.action !== undefined) {
		(out as { action: string }).action = d.action;
	}
	return out;
}

/**
 * Project the retrieval pipeline's skip report into the envelope. The reason is
 * copied through untouched so the caller sees the failure the retrieval method
 * actually hit.
 */
function unsearchedProjection(entry: RetrievalUnsearchedSurface): LiteRetrieveUnsearched {
	return { surface: entry.surface, methods: [...entry.methods], reason: entry.reason };
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

function renderLiteRetrieveJson(envelope: LiteRetrieveEnvelope | IndexNotReadyEnvelope | RejectionEnvelope): string {
	return JSON.stringify(envelope, null, 2);
}

function renderLiteRetrieveHuman(envelope: LiteRetrieveEnvelope | IndexNotReadyEnvelope, debug: boolean): string {
	const lines: string[] = [];
	if (!envelope.ok) {
		lines.push(`error: index not ready`);
		for (const d of envelope.diagnostics) {
			lines.push(`  diagnostic: [${d.severity}] ${d.code}: ${d.message}`);
		}
		return lines.join("\n");
	}
	// ok: true case
	const okEnvelope = envelope;
	if (okEnvelope.stale) {
		lines.push("warning: index may be stale; run `autorag lite refresh` or pass --refresh to rebuild it");
	}
	// Skipped surfaces are reported without --debug: the results below are partial.
	for (const entry of okEnvelope.unsearched) {
		lines.push(`warning: not searched: ${entry.surface} (${entry.methods.join(", ")}): ${entry.reason}`);
	}
	if (okEnvelope.results.length === 0) {
		lines.push("retrieve: no results");
	} else {
		lines.push("results:");
		for (const r of okEnvelope.results) {
			lines.push(`  ${r.number}. source: ${r.source} | method: ${r.method} | score: ${r.score}`);
			if (r.content.length > 0) {
				lines.push(`     ${r.content.slice(0, 200)}`);
			}
			if (debug) {
				lines.push(`     metadata: ${JSON.stringify(r.metadata)}`);
			}
		}
	}
	if (debug && okEnvelope.diagnostics.length > 0) {
		for (const d of okEnvelope.diagnostics) {
			lines.push(`  diagnostic: [${d.severity}] ${d.code}: ${d.message}`);
		}
	}
	return lines.join("\n");
}

function hasCompletedParsedRefresh(workspacePath: string): boolean {
	const markerPath = refreshReadinessPath(workspacePath);
	if (!existsSync(markerPath)) return false;
	try {
		const marker: unknown = JSON.parse(readFileSync(markerPath, "utf8"));
		return (
			typeof marker === "object" &&
			marker !== null &&
			"version" in marker &&
			marker.version === 1 &&
			"completed" in marker &&
			marker.completed === true &&
			"parsed" in marker &&
			marker.parsed === true
		);
	} catch {
		return false;
	}
}

/**
 * Turn refresh's stale-source diagnostics into the retrieve envelope's shape.
 * `severity` stays a warning: stale sources are reported, not blocked.
 */
function staleSourceDiagnostics(
	stale: readonly { readonly source: string; readonly reason?: string }[],
): LiteRetrieveDiagnostic[] {
	return stale.map((entry) => ({
		code: "stale-index",
		severity: "warning",
		message: `Index may be stale because source "${entry.source}" ${(entry.reason ?? "source-changed").replaceAll("-", " ")} since the last refresh. Run \`autorag lite refresh\` or pass --refresh to rebuild it.`,
		source: entry.source,
		reason: entry.reason ?? "source-changed",
		action: "refresh",
	}));
}

/**
 * Report a MinSync refresh that did not complete during `--refresh`, quoting the
 * reason MinSync gave so the caller can act on it instead of re-running blind.
 */
function refreshFailureDiagnostic(result: AutoRAGRefreshResult): LiteRetrieveDiagnostic | undefined {
	if (result.minsync === undefined || result.minsync.ok !== false) return undefined;
	const reason = result.minsync.reason;
	return {
		code: "minsync-unavailable",
		severity: "warning",
		message:
			reason === undefined
				? "MinSync was not refreshed during this run."
				: `MinSync was not refreshed during this run: ${reason}`,
		...(reason === undefined ? {} : { reason }),
	};
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Run the `autorag lite retrieve` command.
 *
 * Returns exit code 2 for:
 *  - Empty query
 *  - Config resolution failure
 *  - Index not ready (never refreshed)
 *
 * Returns exit code 0 on successful retrieval (results may be empty).
 * Returns exit code 1 for runtime errors.
 */
export async function runLiteRetrieve(ctx: CommandContext): Promise<number> {
	const query = ctx.positionals.join(" ").trim();
	if (query.length === 0) {
		ctx.stderr(
			renderError(new Error("Usage: autorag lite retrieve <query> [--top-k N] [--scope SCOPE] [--tags tag1,tag2]"), {
				json: ctx.json,
				debug: ctx.debug,
			}),
		);
		return 2;
	}

	// Validate --top-k before spending cycles on config resolution.
	const topKResult = parseTopK(ctx.flags["top-k"]);
	if (topKResult.kind === "reject") {
		const envelope: RejectionEnvelope = { ok: false, error: topKResult.error };
		ctx.stderr(renderLiteRetrieveJson(envelope));
		return 2;
	}

	let lite: ReturnType<typeof createAutoRAGLite>;
	try {
		lite = createAutoRAGLite({ flags: ctx.flags, cwd: ctx.cwd });
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	// Check index readiness by examining persisted index artifacts on the
	// filesystem. This is the only reliable cross-process check: the parsed
	// mirror index file is created by `lite refresh` / `autorag refresh` and
	// persists across CLI process boundaries.
	const workspacePath = lite.config.workspacePath;
	const parsedReady = hasCompletedParsedRefresh(workspacePath);
	if (!parsedReady) {
		const envelope: IndexNotReadyEnvelope = {
			ok: false,
			query,
			diagnostics: [
				{
					code: "index-not-ready",
					severity: "error",
					message:
						"Index has not been refreshed. Run `autorag lite refresh` or `autorag refresh` before retrieving.",
				},
			],
		};
		ctx.stdout(renderLiteRetrieveJson(envelope));
		return 2;
	}

	// `--refresh` updates the parsed mirrors (and configured indexes) before
	// answering, so the caller can ask for a current corpus without a second
	// command. The default path never writes: staleness is reported, not acted on.
	let refreshFailure: LiteRetrieveDiagnostic | undefined;
	if (ctx.flags.refresh === true) {
		try {
			refreshFailure = refreshFailureDiagnostic(await lite.refresh(false));
		} catch (error) {
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
			return 1;
		}
	}

	const staleDiagnostics = await detectMirrorStaleness({
		root: workspacePath,
		searchPaths: lite.config.searchPaths,
		parserOptions: lite.config.parserOptions,
	});
	if (staleDiagnostics.length > 0 && ctx.flags.strict === true) {
		const envelope: IndexNotReadyEnvelope = {
			ok: false,
			query,
			diagnostics: staleDiagnostics.map((stale) => ({
				code: "index-not-ready" as const,
				severity: "error" as const,
				message: `Index is stale because source "${stale.source}" ${stale.reason?.replaceAll("-", " ") ?? "changed"}. Run \`autorag lite refresh\` or \`autorag refresh\` to rebuild it.`,
				source: stale.source,
				reason: stale.reason ?? "source-changed",
				action: "refresh",
			})),
		};
		ctx.stdout(renderLiteRetrieveJson(envelope));
		return 2;
	}

	// Perform retrieval
	const options = buildRetrieveOptions(ctx.flags);
	let retrievalResult: {
		results: RetrievalResult[];
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	};
	try {
		retrievalResult = await lite.retrieve(query, {
			topK: topKResult.value ?? options.topK,
			scope: options.scope,
			allowedTags: options.allowedTags,
		});
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}

	// Build the envelope
	const staleEnvelopeDiagnostics = staleSourceDiagnostics(staleDiagnostics);
	const diagnostics = [
		...(refreshFailure ? [refreshFailure] : []),
		...staleEnvelopeDiagnostics,
		...(retrievalResult.diagnostics ?? []).map(diagnosticProjection),
	];
	const results: LiteRetrieveResultItem[] = retrievalResult.results.map((r, i) => ({
		number: i + 1,
		source: r.source,
		method: typeof r.metadata.method === "string" ? r.metadata.method : "unknown",
		score: r.score,
		metadata: r.metadata,
		content: r.content,
	}));

	const envelope: LiteRetrieveEnvelope = {
		ok: true,
		query,
		stale: staleEnvelopeDiagnostics.length > 0,
		results,
		unsearched: retrievalResult.unsearched.map(unsearchedProjection),
		diagnostics,
	};

	if (ctx.json) {
		ctx.stdout(renderLiteRetrieveJson(envelope));
	} else {
		ctx.stdout(renderLiteRetrieveHuman(envelope, ctx.debug));
	}
	return 0;
}
