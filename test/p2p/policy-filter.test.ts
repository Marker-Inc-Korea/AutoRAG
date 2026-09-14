import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PolicyStore } from "../../src/p2p/policy.ts";
import { filterRetrievalResultsByPolicy } from "../../src/p2p/policy-filter.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

const workspaces: string[] = [];

function result(source: string, score = 1): RetrievalResult {
	return {
		id: `id:${source}`,
		content: `content for ${source}`,
		source,
		score,
		metadata: {
			origin: { source, labels: ["retrieved"] },
		},
	};
}

function makePolicy(): PolicyStore {
	const workspace = mkdtempSync(join(tmpdir(), "autorag-p2p-policy-filter-"));
	workspaces.push(workspace);
	mkdirSync(join(workspace, ".autorag", "p2p"), { recursive: true });
	writeFileSync(
		join(workspace, ".autorag", "p2p", "policy.toml"),
		`[policy."/docs/**"]
	tier = "always"

[policy."/docs/private/**"]
	tier = "never"

[policy."/kakao/personal/chunks/**"]
	tier = "peers"
peers = ["peer-a"]
`,
	);
	return new PolicyStore({ workspacePath: workspace, newFilesPublic: true });
}

afterEach(() => {
	for (const workspace of workspaces.splice(0)) rmSync(workspace, { recursive: true, force: true });
});

describe("filterRetrievalResultsByPolicy", () => {
	it("keeps allowed file and datasource sources while dropping denied sources", () => {
		const policy = makePolicy();
		const results = new Map<string, RetrievalResult[]>([
			[
				"files",
				[result("/docs/public/guide.md"), result("/docs/private/secret.md"), result("/unknown/not-listed.md")],
			],
			["datasource", [result("/kakao/personal/chunks/chunk-1"), result("/kakao/other/chunks/chunk-2")]],
		]);

		const filtered = filterRetrievalResultsByPolicy(results, policy.resolvePolicy.bind(policy), "peer-a");

		expect(filtered).toEqual(
			new Map<string, RetrievalResult[]>([
				["files", [result("/docs/public/guide.md")]],
				["datasource", [result("/kakao/personal/chunks/chunk-1")]],
			]),
		);
	});

	it("passes the raw source and peer fingerprint to the policy resolver", () => {
		const resolver = vi.fn((source: string, peerFingerprint?: string) => ({
			tier: "always" as const,
			allowed: source === "/kakao/personal/chunks/chunk-1" && peerFingerprint === "peer-a",
			shareBytes: true,
			redact: false,
		}));
		const results = new Map<string, RetrievalResult[]>([["search", [result("/kakao/personal/chunks/chunk-1")]]]);

		const filtered = filterRetrievalResultsByPolicy(results, resolver, "peer-a");

		expect(filtered.get("search")).toHaveLength(1);
		expect(resolver).toHaveBeenCalledWith("/kakao/personal/chunks/chunk-1", "peer-a");
	});

	it("returns empty arrays when every candidate is denied, including an absent policy source", () => {
		const policy = makePolicy();
		const results = new Map<string, RetrievalResult[]>([
			["files", [result("/docs/private/secret.md"), result("/unknown/not-listed.md")]],
			["datasource", [result("/kakao/personal/chunks/chunk-1")]],
		]);

		const filtered = filterRetrievalResultsByPolicy(results, policy.resolvePolicy.bind(policy), "other-peer");

		expect(filtered).toEqual(
			new Map<string, RetrievalResult[]>([
				["files", []],
				["datasource", []],
			]),
		);
	});

	it("does not mutate the input map and deeply copies returned results", () => {
		const policy = makePolicy();
		const original = result("/docs/public/guide.md");
		const results = new Map<string, RetrievalResult[]>([["files", [original]]]);
		const before = structuredClone(Array.from(results.entries()));

		const filtered = filterRetrievalResultsByPolicy(results, policy.resolvePolicy.bind(policy), "peer-a");
		const returned = filtered.get("files")?.[0];

		expect(Array.from(results.entries())).toEqual(before);
		expect(filtered).not.toBe(results);
		expect(filtered.get("files")).not.toBe(results.get("files"));
		expect(returned).not.toBe(original);
		expect(returned?.metadata).not.toBe(original.metadata);
		expect(returned?.metadata.origin).not.toBe(original.metadata.origin);

		if (returned) {
			(returned.metadata.origin as { labels: string[] }).labels.push("changed");
		}
		expect((original.metadata.origin as { labels: string[] }).labels).toEqual(["retrieved"]);
	});

	it("fails closed if a resolver reports an allowed never tier", () => {
		const results = new Map<string, RetrievalResult[]>([["search", [result("/docs/private/secret.md")]]]);
		const resolver = () => ({ tier: "never" as const, allowed: true, shareBytes: true, redact: false });

		expect(filterRetrievalResultsByPolicy(results, resolver, "peer-a").get("search")).toEqual([]);
	});
});
