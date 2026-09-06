import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

let temp: string | undefined;

afterEach(() => {
	if (temp) rmSync(temp, { recursive: true, force: true });
	temp = undefined;
});

function result(source: string, content: string): RetrievalResult {
	return { id: source, source, content, score: 1, metadata: {} };
}

function method(results: RetrievalResult[], datasource = false): RetrievalMethod {
	return {
		describe: () => ({
			name: datasource ? "kakao-fixture" : "fixture",
			type: "posix",
			description: "fixture",
			status: "active",
			capabilities: [],
			...(datasource ? { datasourceId: "kakao", tags: ["kakao"] } : {}),
		}),
		retrieve: async (_query: string, _options: RetrievalOptions) => results,
	};
}

describe("remote-session retrieval policy", () => {
	it("filters retrieval before the remote caller and records only observed sources", async () => {
		temp = mkdtempSync(join(tmpdir(), "autorag-remote-policy-"));
		const observedSources = new Set<string>();
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			memoryPath: join(temp, "memory.json"),
			remoteSession: true,
			minSync: false,
			bm25: false,
			jikji: false,
		});
		agent
			.getMethodRegistry()
			.register(method([result("/docs/shared.md", "allowed"), result("/docs/secret.md", "never share")]));
		const resolvePolicy = (source: string) =>
			source === "/docs/shared.md"
				? { tier: "always" as const, allowed: true, shareBytes: true, redact: false }
				: { tier: "never" as const, allowed: false, shareBytes: false, redact: true };

		const retrieved = await agent.retrieveWithDiagnostics("fixture", {
			resolvePolicy,
			peerFingerprint: "peer-a",
			observedSources,
		});

		expect(retrieved.results.map((item) => item.source)).toEqual(["/docs/shared.md"]);
		expect(observedSources).toEqual(new Set(["/docs/shared.md"]));
		expect(observedSources).not.toContain("/model/forged-source.md");
	});

	it("filters datasource results by allowed and never policy tiers", async () => {
		temp = mkdtempSync(join(tmpdir(), "autorag-remote-datasource-"));
		const observedSources = new Set<string>();
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			memoryPath: join(temp, "memory.json"),
			remoteSession: true,
			minSync: false,
			bm25: false,
			jikji: false,
			datasourceAccess: { allowedTags: ["kakao"] },
		});
		agent
			.getMethodRegistry()
			.register(
				method(
					[result("kakao:shared-room/chunks/a", "allowed"), result("kakao:private-room/chunks/b", "denied")],
					true,
				),
			);
		const resolvePolicy = (source: string) =>
			source.startsWith("kakao:shared-room/")
				? { tier: "peers" as const, allowed: true, shareBytes: false, redact: true }
				: { tier: "never" as const, allowed: false, shareBytes: false, redact: true };
		const retrieved = await agent.retrieveWithDiagnostics("messages", {
			resolvePolicy,
			peerFingerprint: "peer-a",
			observedSources,
		});
		expect(retrieved.results.map((item) => item.source)).toEqual(["kakao:shared-room/chunks/a"]);
		expect(observedSources).toEqual(new Set(["kakao:shared-room/chunks/a"]));
	});

	it("defaults remote search timeout to 120 seconds", () => {
		temp = mkdtempSync(join(tmpdir(), "autorag-remote-timeout-"));
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			memoryPath: join(temp, "memory.json"),
			remoteSession: true,
			minSync: false,
			bm25: false,
			jikji: false,
		});
		expect((agent as unknown as { searchTimeoutMs: number }).searchTimeoutMs).toBe(120_000);
	});
});
