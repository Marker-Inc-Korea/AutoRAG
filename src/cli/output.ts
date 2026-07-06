import type { AutoRAGRefreshResult, AutoRAGRefreshStatus } from "../agent/agent.ts";
import type { SearchDocumentDiagnostic, SearchDocumentsResponse } from "../agent/search-documents.ts";
import type { MemorySchemaV4 } from "../memory/memory.ts";

export interface RenderOptions {
	json?: boolean;
	debug?: boolean;
}

function diagnosticProjection(d: SearchDocumentDiagnostic): {
	code: string;
	severity: string;
	message: string;
	source?: string;
} {
	const out: { code: string; severity: string; message: string; source?: string } = {
		code: d.code,
		severity: d.severity,
		message: d.message,
	};
	if (d.source !== undefined) out.source = d.source;
	return out;
}

function refreshEnvelope(result: AutoRAGRefreshResult) {
	const envelope: Record<string, unknown> = {
		ok: true,
		counts: {
			scanned: result.scanned,
			written: result.written,
			deleted: result.deleted,
			skipped: result.skipped,
		},
		diagnostics: (result.diagnostics ?? []).map(diagnosticProjection),
	};
	if (result.bm25) {
		envelope.bm25 = {
			indexedChunks: result.bm25.indexedChunks,
			readiness: result.bm25.readiness,
			engine: result.bm25.engine,
		};
	}
	if (result.datasources && result.datasources.length > 0) {
		envelope.datasources = result.datasources.map((ds) => ({
			ok: ds.ok,
			skill: ds.skill,
			instanceId: ds.instanceId,
			indexedAt: ds.indexedAt,
			diagnostics: (ds.diagnostics ?? []).map((d) => ({
				code: d.code,
				severity: d.severity,
				message: d.message,
				...(d.source !== undefined ? { source: d.source } : {}),
			})),
		}));
	}
	return envelope;
}

function renderRefreshHuman(result: AutoRAGRefreshResult, debug: boolean): string {
	const lines: string[] = [];
	lines.push("refresh: ok");
	lines.push(
		`  counts: scanned=${result.scanned} written=${result.written} deleted=${result.deleted} skipped=${result.skipped}`,
	);
	if (result.bm25) {
		lines.push(
			`  bm25: indexedChunks=${result.bm25.indexedChunks} readiness=${result.bm25.readiness} engine=${result.bm25.engine}`,
		);
	}
	if (result.datasources && result.datasources.length > 0) {
		for (const ds of result.datasources) {
			lines.push(
				`  datasource: ok=${ds.ok} skill=${ds.skill} instanceId=${ds.instanceId} indexedAt=${ds.indexedAt}`,
			);
		}
	}
	if (debug && result.diagnostics && result.diagnostics.length > 0) {
		for (const d of result.diagnostics) {
			lines.push(`  diagnostic: [${d.severity}] ${d.code}: ${d.message}`);
		}
	}
	return lines.join("\n");
}

export function renderRefresh(result: AutoRAGRefreshResult, opts: RenderOptions): string {
	if (opts.json) {
		return JSON.stringify(refreshEnvelope(result), null, 2);
	}
	return renderRefreshHuman(result, opts.debug ?? false);
}

function renderStatusHuman(status: AutoRAGRefreshStatus, debug: boolean): string {
	const lines: string[] = [];
	lines.push(`status: ${status.state}`);
	lines.push(`  inFlight: ${status.inFlight}`);
	lines.push(`  stale: ${status.stale}`);
	if (status.lastStartedAt) lines.push(`  lastStartedAt: ${status.lastStartedAt}`);
	if (status.lastFinishedAt) lines.push(`  lastFinishedAt: ${status.lastFinishedAt}`);
	if (status.counts) {
		lines.push(
			`  counts: scanned=${status.counts.scanned} written=${status.counts.written} deleted=${status.counts.deleted} skipped=${status.counts.skipped}`,
		);
	}
	const comps = status.components;
	const compParts: string[] = [];
	if (comps.bm25) compParts.push(`bm25=${comps.bm25}`);
	if (comps.minsync) compParts.push(`minsync=${comps.minsync}`);
	if (comps.jikji) compParts.push(`jikji=${comps.jikji}`);
	if (comps.datasources) compParts.push(`datasources=${comps.datasources}`);
	if (compParts.length > 0) lines.push(`  components: ${compParts.join(" ")}`);
	if (status.lastError) lines.push(`  lastError: ${status.lastError}`);
	if (debug || status.diagnostics.length > 0) {
		for (const d of status.diagnostics) {
			lines.push(`  diagnostic: [${d.severity}] ${d.code}: ${d.message}`);
		}
	}
	return lines.join("\n");
}

export function renderStatus(status: AutoRAGRefreshStatus, opts: RenderOptions): string {
	if (opts.json) {
		return JSON.stringify(status, null, 2);
	}
	return renderStatusHuman(status, opts.debug ?? false);
}

function searchEnvelope(resp: SearchDocumentsResponse, debug: boolean) {
	const results = resp.results.map((r) => {
		const base: Record<string, unknown> = {
			number: r.number,
			title: r.title,
			summary: r.summary,
		};
		if (debug) {
			base.confidence = r.confidence;
			base.evidence = r.evidence.map((e) => {
				const ev: Record<string, unknown> = { excerpt: e.excerpt };
				if (e.lineNumber !== undefined) ev.lineNumber = e.lineNumber;
				return ev;
			});
			base.feedbackId = r.feedbackId;
		}
		return base;
	});
	const envelope: Record<string, unknown> = {
		answer: resp.answer,
		results,
	};
	if (debug) {
		envelope.sessionId = resp.sessionId;
		envelope.query = resp.query;
		envelope.searched = resp.searched;
		if (resp.diagnostics) {
			envelope.diagnostics = resp.diagnostics.map(diagnosticProjection);
		}
	}
	return envelope;
}

function renderSearchHuman(resp: SearchDocumentsResponse, debug: boolean): string {
	const lines: string[] = [];
	if (resp.answer) {
		lines.push(resp.answer);
	}
	for (const r of resp.results) {
		const header = `${r.number}. ${r.title}`;
		lines.push(header);
		if (r.summary) lines.push(`   ${r.summary}`);
		if (debug) {
			lines.push(`   confidence: ${r.confidence}`);
			for (const ev of r.evidence) {
				const ln = ev.lineNumber !== undefined ? ` (line ${ev.lineNumber})` : "";
				lines.push(`   evidence${ln}: ${ev.excerpt}`);
			}
		}
	}
	if (debug) {
		if (resp.diagnostics && resp.diagnostics.length > 0) {
			for (const d of resp.diagnostics) {
				lines.push(`   diagnostic: [${d.severity}] ${d.code}: ${d.message}`);
			}
		}
		if (resp.sessionId) lines.push(`   sessionId: ${resp.sessionId}`);
	}
	return lines.join("\n");
}

export function renderSearch(resp: SearchDocumentsResponse, opts: RenderOptions): string {
	if (opts.json) {
		return JSON.stringify(searchEnvelope(resp, opts.debug ?? false), null, 2);
	}
	return renderSearchHuman(resp, opts.debug ?? false);
}

function renderMemoryHuman(schema: MemorySchemaV4, debug: boolean): string {
	const lines: string[] = [];
	lines.push("memory:");
	lines.push(
		`  counts: feedbackSignals=${schema.feedbackSignals.length} curatedResults=${schema.curatedResults.length} insights=${schema.insights.length}`,
	);
	const sd = schema.signalDefaults;
	lines.push(
		`  signalDefaults: explicitWeight=${sd.explicitWeight} followupWeight=${sd.followupWeight} retryWeight=${sd.retryWeight} implicitCap=${sd.implicitCap}`,
	);
	if (debug) {
		lines.push(`  evidenceChunks: ${schema.evidenceChunks.length}`);
		lines.push(`  warnings: ${schema.warnings.length}`);
		lines.push(`  pendingInsightSignals: ${schema.pendingInsightSignals.length}`);
	}
	return lines.join("\n");
}

export function renderMemory(schema: MemorySchemaV4, opts: RenderOptions): string {
	if (opts.json) {
		// Memory is fully path-opaque (queries, opaque evidence ids, method names,
		// signal weights) so the whole schema is safe to emit for automation.
		return JSON.stringify(schema, null, 2);
	}
	return renderMemoryHuman(schema, opts.debug ?? false);
}

export function renderFeedback(result: { applied: boolean; sessionId: string }, opts: RenderOptions): string {
	const envelope = { ok: true, applied: result.applied, sessionId: result.sessionId };
	if (opts.json) {
		return JSON.stringify(envelope, null, 2);
	}
	const lines: string[] = [];
	lines.push(`feedback: ${result.applied ? "applied" : "not applied"}`);
	lines.push(`  sessionId: ${result.sessionId}`);
	return lines.join("\n");
}

function indexEnvelope(result: { action: "reset" | "rebuild"; removed: string[]; rebuilt?: AutoRAGRefreshResult }) {
	const envelope: Record<string, unknown> = {
		ok: true,
		action: result.action,
		removed: result.removed,
	};
	if (result.rebuilt) {
		envelope.rebuilt = refreshEnvelope(result.rebuilt);
	}
	return envelope;
}

function renderIndexHuman(
	result: { action: "reset" | "rebuild"; removed: string[]; rebuilt?: AutoRAGRefreshResult },
	debug: boolean,
): string {
	const lines: string[] = [];
	lines.push(`index: ${result.action}`);
	if (result.removed.length > 0) {
		lines.push(`  removed: ${result.removed.join(", ")}`);
	}
	if (result.rebuilt) {
		lines.push(renderRefreshHuman(result.rebuilt, debug));
	}
	return lines.join("\n");
}

export function renderIndex(
	result: { action: "reset" | "rebuild"; removed: string[]; rebuilt?: AutoRAGRefreshResult },
	opts: RenderOptions,
): string {
	if (opts.json) {
		return JSON.stringify(indexEnvelope(result), null, 2);
	}
	return renderIndexHuman(result, opts.debug ?? false);
}

export function renderError(err: unknown, opts: RenderOptions): string {
	const error = err instanceof Error ? err : new Error(String(err));
	// Never emit error.stack: Node stack traces embed absolute filesystem paths,
	// which would violate the CLI path-opacity contract even under --debug. The
	// error name/message from library errors is already path-opaque.
	const message = error.message || error.name;
	if (opts.json) {
		return JSON.stringify({ ok: false, error: message }, null, 2);
	}
	return `error: ${message}`;
}
