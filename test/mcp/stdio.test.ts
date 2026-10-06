import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { Client, ProtocolError, ProtocolErrorCode } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { describe, expect, it, onTestFinished } from "vitest";
import { writeFakeDupeyExecutable } from "../helpers/fake-dupey.ts";
import { writeFakeFSearchExecutable } from "../helpers/fake-fsearch.ts";
import { writeFakeMinSyncExecutable } from "../helpers/fake-minsync.ts";

const REFUND_FILE = "refund.md";
const REFUND_TEXT = "Refund exceptions require director approval before payout.\n";

function makeFixture() {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-stdio-"));
	// Registered before any connection: even if the handshake or an assertion
	// throws, the workspace and any spawned children are torn down.
	onTestFinished(() => rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 }));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, REFUND_FILE), REFUND_TEXT);
	const stagedFiles = join(root, ".autorag", "minsync", "files");
	mkdirSync(stagedFiles, { recursive: true });
	writeFileSync(join(stagedFiles, REFUND_FILE), REFUND_TEXT);
	// `AUTORAG_CONFIG` deliberately ignores `minSync.binaryPath` (MinSync resolves
	// from PATH/the workspace cache), so the fake must sit on PATH under the
	// resolver's platform name to make the e2e independent of an installed minsync.
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	writeFakeFSearchExecutable(binDir, join(docs, REFUND_FILE));
	writeFakeMinSyncExecutable(binDir);
	const dupeyBinary = writeFakeDupeyExecutable(binDir);
	const config = join(root, "config.json");
	writeFileSync(
		config,
		JSON.stringify({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: {
				workspacePath: join(root, ".autorag", "minsync"),
				autoInstall: false,
			},
			jikji: false,
			everything: false,
			// `watch: false` keeps the fake from becoming a live daemon: the fixture
			// must never leave a `fsearch-cli watch` process (or its timer) behind.
			fsearch: {
				binaryPath: join(binDir, process.platform === "win32" ? "fsearch-cli.exe" : "fsearch-cli"),
				watch: false,
			},
			// Pass the shim explicitly: `portableSpawnCommand` only rewrites a shebang
			// script to its interpreter when it can see the file at that exact path.
			dupey: { binaryPath: dupeyBinary },
			// Spotlight needs no external install, so a configured skill yields a
			// dynamic datasource tool without touching the network or a binary.
			datasources: { spotlight: { enabled: true } },
			datasourceAccess: { allowedTags: ["spotlight"] },
		}),
	);
	return { root, docs, config, binDir };
}

interface StdioConnection {
	readonly client: Client;
	readonly close: () => Promise<void>;
}

async function closeQuietly(target: { close(): Promise<void> }): Promise<void> {
	try {
		await target.close();
	} catch {
		// Already closed or never fully connected; takeover teardown still ran.
	}
}

async function connect(
	config: string,
	binDir: string,
	envOverrides: Record<string, string> = {},
): Promise<StdioConnection> {
	// Windows spells the variable `Path`; reuse the existing key so the child
	// env has no case-duplicate entry and the fake stays first on the lookup path.
	const pathKey = Object.keys(process.env).find((key) => key.toLowerCase() === "path") ?? "PATH";
	const env = Object.fromEntries(
		Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	) as Record<string, string>;
	env.AUTORAG_CONFIG = config;
	env[pathKey] = `${binDir}${delimiter}${env[pathKey] ?? ""}`;
	for (const [key, value] of Object.entries(envOverrides)) env[key] = value;
	const transport = new StdioClientTransport({
		command: process.execPath,
		args: [join(process.cwd(), "src/mcp/index.ts")],
		env,
		stderr: "pipe",
	});
	const client = new Client({ name: "autorag-stdio-qa", version: "1.0.0" });
	let closed: Promise<void> | undefined;
	const close = (): Promise<void> => {
		closed ??= (async () => {
			await closeQuietly(client);
			await closeQuietly(transport);
		})();
		return closed;
	};
	// Register teardown BEFORE connecting so a failed handshake or assertion
	// cannot leak the stdio child process.
	onTestFinished(close);
	try {
		await client.connect(transport);
	} catch (error) {
		await close();
		throw error;
	}
	return { client, close };
}

function field(value: unknown, key: string): unknown {
	if (typeof value !== "object" || value === null) return undefined;
	return Reflect.get(value, key);
}

/** Recursively search a structured result for a `key === expected` pair, body-shape agnostic. */
function hasFieldValue(value: unknown, key: string, expected: unknown, seen = new Set<unknown>()): boolean {
	if (typeof value !== "object" || value === null) return false;
	if (seen.has(value)) return false;
	seen.add(value);
	if (field(value, key) === expected) return true;
	return Object.values(value).some((item) => hasFieldValue(item, key, expected, seen));
}

/** The tool-authored text block must be the pretty-printed form of `structuredContent`. */
function expectStructuredMatchesText(result: { content: unknown; structuredContent?: unknown }): void {
	const content = result.content;
	if (!Array.isArray(content) || content.length === 0) throw new Error("MCP result had no content blocks");
	const text = field(content[0], "text");
	expect(typeof text).toBe("string");
	expect(JSON.parse(text as string)).toEqual(result.structuredContent);
}

/** Extract a tool's `results` array, or fail loudly when the payload shape drifted. */
function resultsArray(structured: unknown): readonly unknown[] {
	const results = field(structured, "results");
	if (!Array.isArray(results)) throw new Error("MCP result did not contain a results array");
	return results;
}

/** Calls a tool that must fail at the SDK/protocol layer and returns the thrown error. */
async function expectProtocolFailure(call: Promise<unknown>): Promise<ProtocolError> {
	let error: unknown;
	try {
		await call;
	} catch (caught) {
		error = caught;
	}
	expect(error).toBeInstanceOf(ProtocolError);
	const protocolError = error as ProtocolError;
	expect(protocolError.code).toBe(ProtocolErrorCode.InvalidParams);
	return protocolError;
}

/** Assert the exact minsync evidence for the fixture corpus. */
function expectRefundEvidence(results: readonly unknown[], docs: string): void {
	expect(results).toHaveLength(1);
	const hit = results[0];
	expect(field(hit, "source")).toBe(realpathSync(join(docs, REFUND_FILE)));
	expect(field(hit, "content")).toBe(REFUND_TEXT);
}

interface PersistedMemory {
	readonly curatedResults: readonly { readonly sessionId: string; readonly number: number; readonly title: string }[];
	readonly evidenceChunks: readonly { readonly stableEvidenceId: string; readonly source: string }[];
	readonly feedbackSignals: readonly { readonly eventId: string; readonly sentiment: string }[];
}

/** Read the persisted memory file the live runtime writes and reloads across a restart. */
function readPersistedMemory(root: string): PersistedMemory {
	return JSON.parse(readFileSync(join(root, "memory.json"), "utf8")) as PersistedMemory;
}

const REFUND_EXCERPT = "Refund exceptions require director approval before payout.";

/** The `emit_autorag_results` payload an external curator submits for the fixture corpus. */
function refundReport(source: string, content: string, method: string) {
	return {
		answer: "Refund exceptions need director approval before payout [1], confirmed by a second captured result [2].",
		results: [
			{
				number: 1,
				title: "Refund approval policy",
				summary: "Director approval is mandatory before refund payout.",
				evidence: [{ excerpt: REFUND_EXCERPT, lineNumber: 1 }],
				confidence: 0.9,
			},
			{
				number: 2,
				title: "Refund policy secondary capture",
				summary: "The same rule captured from a second chunk.",
				evidence: [{ excerpt: REFUND_EXCERPT, lineNumber: 1 }],
				confidence: 0.5,
			},
		],
		mapping: [
			{
				number: 1,
				source,
				method,
				content,
				evidenceRefs: [{ method, source, excerpt: REFUND_EXCERPT, content, chunkIndex: 0, lineNumber: 1 }],
			},
			{
				number: 2,
				source,
				method,
				content,
				evidenceRefs: [{ method, source, excerpt: REFUND_EXCERPT, content, chunkIndex: 1, lineNumber: 1 }],
			},
		],
	};
}

/** Find the evidence view for a given result number in an `autorag.evidence` payload. */
function evidenceResult(structured: unknown, number: number): unknown {
	const match = resultsArray(structured).find((entry) => field(entry, "number") === number);
	if (match === undefined) throw new Error(`MCP evidence payload had no result ${number}`);
	return match;
}

function chunksOf(result: unknown): readonly unknown[] {
	const chunks = field(result, "chunks");
	if (!Array.isArray(chunks)) throw new Error("MCP evidence result had no chunks array");
	return chunks;
}

describe("AutoRAG Lite MCP stdio", () => {
	it("refreshes, searches exact fixture evidence, routes filename search to fsearch-cli, and preserves results across a restart", async () => {
		const { docs, config, binDir } = makeFixture();
		const first = await connect(config, binDir);
		const listed = await first.client.listTools();
		const names = listed.tools.map((tool) => tool.name);
		for (const tool of [
			"autorag.status",
			"autorag.search",
			"autorag.search.files",
			"autorag.datasources.list",
			"autorag.datasources.get",
			"autorag.duplicates",
			"autorag.search_datasource_spotlight",
			"autorag.refresh",
		]) {
			expect(names).toContain(tool);
		}

		const duplicatesTool = listed.tools.find((tool) => tool.name === "autorag.duplicates");
		expect(duplicatesTool?.description).toContain("Dupey");
		const dynamicTool = listed.tools.find((tool) => tool.name === "autorag.search_datasource_spotlight");
		expect(dynamicTool?.description).toContain("Spotlight");

		// Before refresh the dynamic tool must fail ready-gated without touching Spotlight.
		const dynamicNotReady = await first.client.callTool({
			name: "autorag.search_datasource_spotlight",
			arguments: { query: "refund" },
		});
		expect(dynamicNotReady.isError).toBe(true);
		expect(hasFieldValue(dynamicNotReady.structuredContent, "errorCode", "index-not-ready")).toBe(true);
		expect(hasFieldValue(dynamicNotReady.structuredContent, "datasourceId", "spotlight")).toBe(true);

		const duplicates = await first.client.callTool({ name: "autorag.duplicates", arguments: {} });
		expect(duplicates.isError).not.toBe(true);
		expectStructuredMatchesText(duplicates);
		expect(hasFieldValue(duplicates.structuredContent, "action", "review")).toBe(true);
		expect(hasFieldValue(duplicates.structuredContent, "hash", "dupey-fixture-hash")).toBe(true);

		const refresh = await first.client.callTool({ name: "autorag.refresh", arguments: {} });
		expect(refresh.isError).not.toBe(true);
		expectStructuredMatchesText(refresh);

		const search = await first.client.callTool({
			name: "autorag.search",
			arguments: { query: "Who approves refund exceptions?", topK: 3 },
		});
		expect(search.isError).not.toBe(true);
		expectStructuredMatchesText(search);
		expectRefundEvidence(resultsArray(search.structuredContent), docs);

		const files = await first.client.callTool({
			name: "autorag.search.files",
			arguments: { query: "refund" },
		});
		if (process.platform === "win32") {
			// The fixture disables Everything, so Windows routing must surface the
			// provider failure as an MCP error instead of silently using the walker.
			expect(files.isError).toBe(true);
			expect(hasFieldValue(files.structuredContent, "backend", "everything")).toBe(true);
		} else {
			expect(files.isError).not.toBe(true);
			expectStructuredMatchesText(files);
			expect(field(files.structuredContent, "backend")).toBe("fsearch-cli");
			expect(hasFieldValue(files.structuredContent, "path", realpathSync(join(docs, REFUND_FILE)))).toBe(true);
		}
		await first.close();

		// A fresh process must serve the same corpus evidence from the persisted index.
		const second = await connect(config, binDir);
		const status = await second.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
		expect(field(status.structuredContent, "state")).toBe("success");
		expect(field(status.structuredContent, "stale")).toBe(false);

		const restarted = await second.client.callTool({
			name: "autorag.search",
			arguments: { query: "Who approves refund exceptions?", topK: 3 },
		});
		expect(restarted.isError).not.toBe(true);
		expectStructuredMatchesText(restarted);
		expectRefundEvidence(resultsArray(restarted.structuredContent), docs);

		const list = await second.client.callTool({ name: "autorag.datasources.list", arguments: {} });
		expect(list.isError).not.toBe(true);
		expectStructuredMatchesText(list);

		const missing = await second.client.callTool({
			name: "autorag.datasources.get",
			arguments: { datasourceId: "missing" },
		});
		expect(missing.isError === true || field(missing.structuredContent, "ok") === false).toBe(true);
		await second.close();
	}, 120000);

	it("persists exact curated report evidence and idempotent feedback across a restart", async () => {
		const { root, docs, config, binDir } = makeFixture();
		const first = await connect(config, binDir);

		// The report is built from a real retrieved hit, so evidence must round-trip.
		const refresh = await first.client.callTool({ name: "autorag.refresh", arguments: {} });
		expect(refresh.isError).not.toBe(true);
		const search = await first.client.callTool({
			name: "autorag.search",
			arguments: { query: "Who approves refund exceptions?", topK: 3 },
		});
		expect(search.isError).not.toBe(true);
		const hits = resultsArray(search.structuredContent);
		expectRefundEvidence(hits, docs);
		const source = field(hits[0], "source");
		const content = field(hits[0], "content");
		const method = field(hits[0], "method");
		expect(typeof source).toBe("string");
		expect(typeof content).toBe("string");
		expect(typeof method).toBe("string");
		const report = refundReport(source as string, content as string, method as string);

		const created = await first.client.callTool({
			name: "autorag.report",
			arguments: { query: "Who approves refund exceptions?", report },
		});
		expect(created.isError).not.toBe(true);
		expectStructuredMatchesText(created);
		expect(field(created.structuredContent, "ok")).toBe(true);
		expect(field(created.structuredContent, "resultCount")).toBe(2);
		const sessionId = field(created.structuredContent, "sessionId");
		expect(typeof sessionId).toBe("string");

		// Evidence must expose the exact source/content lineage the curator mapped.
		const evidence = await first.client.callTool({ name: "autorag.evidence", arguments: { sessionId } });
		expect(evidence.isError).not.toBe(true);
		expectStructuredMatchesText(evidence);
		expect(field(evidence.structuredContent, "sessionId")).toBe(sessionId);
		expect(resultsArray(evidence.structuredContent)).toHaveLength(2);
		const firstChunks = chunksOf(evidenceResult(evidence.structuredContent, 1));
		expect(field(evidenceResult(evidence.structuredContent, 1), "title")).toBe("Refund approval policy");
		expect(firstChunks).toHaveLength(1);
		expect(field(firstChunks[0], "source")).toBe(source);
		expect(field(firstChunks[0], "content")).toBe(content);
		expect(field(firstChunks[0], "excerpt")).toBe(REFUND_EXCERPT);
		expect(field(firstChunks[0], "method")).toBe(method);
		expect(field(firstChunks[0], "chunkIndex")).toBe(0);
		expect(field(firstChunks[0], "lineNumber")).toBe(1);
		const stableEvidenceId = field(firstChunks[0], "stableEvidenceId");
		expect(typeof stableEvidenceId).toBe("string");
		const secondChunk = chunksOf(evidenceResult(evidence.structuredContent, 2))[0];
		expect(field(secondChunk, "chunkIndex")).toBe(1);
		expect(field(secondChunk, "stableEvidenceId")).not.toBe(stableEvidenceId);

		// A per-result filter narrows the evidence view.
		const filtered = await first.client.callTool({
			name: "autorag.evidence",
			arguments: { sessionId, resultNumber: 1 },
		});
		expect(filtered.isError).not.toBe(true);
		expect(resultsArray(filtered.structuredContent)).toHaveLength(1);
		expect(field(resultsArray(filtered.structuredContent)[0], "number")).toBe(1);

		// Feedback is applied once, then idempotently reported as not re-applied.
		const useful = await first.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId, useful: [1] },
		});
		expect(useful.isError).not.toBe(true);
		expectStructuredMatchesText(useful);
		expect(field(useful.structuredContent, "applied")).toBe(true);
		const duplicate = await first.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId, useful: [1] },
		});
		expect(duplicate.isError).not.toBe(true);
		expect(field(duplicate.structuredContent, "ok")).toBe(true);
		expect(field(duplicate.structuredContent, "applied")).toBe(false);

		// The persisted memory file proves the report and feedback actually landed.
		const afterFeedback = readPersistedMemory(root);
		expect(afterFeedback.curatedResults).toHaveLength(2);
		expect(afterFeedback.evidenceChunks).toHaveLength(2);
		expect(afterFeedback.feedbackSignals.some((signal) => signal.eventId === `${sessionId}:1:useful`)).toBe(true);
		const curatedCount = afterFeedback.curatedResults.length;

		// Malformed reports are rejected before persistence; the connection survives.
		const malformed: unknown[] = [
			{ ...report, results: report.results.slice(0, 1) }, // result numbers no longer one-to-one
			{ ...report, mapping: [{ number: 1, source, method, evidenceRefs: [] }, report.mapping[1]] }, // no content
			{ ...report, results: [{ ...report.results[0], confidence: 1.5 }, report.results[1]] }, // out of range
		];
		for (const bad of malformed) {
			const rejected = await first.client.callTool({
				name: "autorag.report",
				arguments: { query: "malformed", report: bad },
			});
			expect(rejected.isError).toBe(true);
			expect(hasFieldValue(rejected.structuredContent, "errorCode", "invalid-report")).toBe(true);
			expect(field(rejected.structuredContent, "sessionId")).toBeUndefined();
		}
		expect(readPersistedMemory(root).curatedResults).toHaveLength(curatedCount);
		const alive = await first.client.callTool({ name: "autorag.status", arguments: {} });
		expect(alive.isError).not.toBe(true);

		// Unknown sessions, unknown result numbers and disjoint feedback all fail.
		const unknownEvidence = await first.client.callTool({
			name: "autorag.evidence",
			arguments: { sessionId: "missing-session" },
		});
		expect(unknownEvidence.isError).toBe(true);
		expect(hasFieldValue(unknownEvidence.structuredContent, "errorCode", "evidence-not-found")).toBe(true);
		const unknownSession = await first.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId: "missing-session", useful: [1] },
		});
		expect(unknownSession.isError).toBe(true);
		const unknownNumber = await first.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId, useful: [99] },
		});
		expect(unknownNumber.isError).toBe(true);
		const disjoint = await first.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId, useful: [1], notUseful: [1] },
		});
		expect(disjoint.isError).toBe(true);

		await first.close();

		// A fresh process must reload evidence and prior feedback from disk.
		const second = await connect(config, binDir);
		const restartedEvidence = await second.client.callTool({ name: "autorag.evidence", arguments: { sessionId } });
		expect(restartedEvidence.isError).not.toBe(true);
		expectStructuredMatchesText(restartedEvidence);
		expect(resultsArray(restartedEvidence.structuredContent)).toHaveLength(2);
		const restartedChunk = chunksOf(evidenceResult(restartedEvidence.structuredContent, 1))[0];
		expect(field(restartedChunk, "stableEvidenceId")).toBe(stableEvidenceId);
		expect(field(restartedChunk, "source")).toBe(source);
		expect(field(restartedChunk, "content")).toBe(content);

		// Feedback for a new number still works after restart (registry rebuilt from memory).
		const postRestart = await second.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId, notUseful: [2] },
		});
		expect(postRestart.isError).not.toBe(true);
		expect(field(postRestart.structuredContent, "applied")).toBe(true);
		// The pre-restart signal persisted, so repeating it is idempotent, not an error.
		const repeated = await second.client.callTool({
			name: "autorag.feedback",
			arguments: { sessionId, useful: [1] },
		});
		expect(repeated.isError).not.toBe(true);
		expect(field(repeated.structuredContent, "ok")).toBe(true);
		expect(field(repeated.structuredContent, "applied")).toBe(false);
		const finalSignals = readPersistedMemory(root).feedbackSignals.map((signal) => signal.eventId);
		expect(finalSignals).toContain(`${sessionId}:1:useful`);
		expect(finalSignals).toContain(`${sessionId}:2:not_useful`);
		await second.close();
	}, 180000);

	it("omits refresh, report and feedback under the read-only env, keeps evidence, and rejects direct calls", async () => {
		const { config, binDir } = makeFixture();
		const connection = await connect(config, binDir, { AUTORAG_MCP_READ_ONLY: "1" });
		const names = (await connection.client.listTools()).tools.map((tool) => tool.name);
		expect(names).toContain("autorag.status");
		expect(names).toContain("autorag.search");
		expect(names).toContain("autorag.evidence");
		expect(names).not.toContain("autorag.refresh");
		expect(names).not.toContain("autorag.report");
		expect(names).not.toContain("autorag.feedback");

		const refreshError = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.refresh", arguments: {} }),
		);
		expect(refreshError.message).toContain("autorag.refresh");
		const reportError = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.report", arguments: { query: "refund", report: {} } }),
		);
		expect(reportError.message).toContain("autorag.report");
		const feedbackError = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.feedback", arguments: { sessionId: "s", useful: [1] } }),
		);
		expect(feedbackError.message).toContain("autorag.feedback");

		// The retained evidence tool is live: an unknown session is a tool error, not a protocol error.
		const unknownEvidence = await connection.client.callTool({
			name: "autorag.evidence",
			arguments: { sessionId: "missing-session" },
		});
		expect(unknownEvidence.isError).toBe(true);
		expect(hasFieldValue(unknownEvidence.structuredContent, "errorCode", "evidence-not-found")).toBe(true);

		// The read-only server still serves the tools it does expose.
		const status = await connection.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
	}, 60000);

	it("lists only allowlisted tools and rejects an omitted tool as a protocol error", async () => {
		const { config, binDir } = makeFixture();
		const connection = await connect(config, binDir, { AUTORAG_MCP_TOOLS: "autorag.status,autorag.search.files" });
		const names = (await connection.client.listTools()).tools.map((tool) => tool.name);
		expect(names).toHaveLength(2);
		expect(names).toContain("autorag.status");
		expect(names).toContain("autorag.search.files");

		const error = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.duplicates", arguments: {} }),
		);
		expect(error.message).toContain("autorag.duplicates");

		// The new report/evidence/feedback lifecycle tools honor the same allowlist.
		const reportError = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.report", arguments: { query: "refund", report: {} } }),
		);
		expect(reportError.message).toContain("autorag.report");
		const evidenceError = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.evidence", arguments: { sessionId: "s" } }),
		);
		expect(evidenceError.message).toContain("autorag.evidence");
		const feedbackError = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.feedback", arguments: { sessionId: "s", useful: [1] } }),
		);
		expect(feedbackError.message).toContain("autorag.feedback");

		const status = await connection.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
	}, 60000);

	it("reports invalid input as a tool error without poisoning the connection", async () => {
		const { config, binDir } = makeFixture();
		const connection = await connect(config, binDir);

		const invalid = await connection.client.callTool({
			name: "autorag.search.files",
			arguments: { query: "refund", maxResults: -1 },
		});
		expect(invalid.isError).toBe(true);
		const invalidText = field(Array.isArray(invalid.content) ? invalid.content[0] : undefined, "text");
		expect(typeof invalidText).toBe("string");
		expect(invalidText).toContain("Input validation error");

		// The same connection must still answer a valid call with real state.
		const status = await connection.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
		expect(field(status.structuredContent, "state")).toBe("idle");
		expect(field(status.structuredContent, "inFlight")).toBe(false);
	}, 60000);
});
