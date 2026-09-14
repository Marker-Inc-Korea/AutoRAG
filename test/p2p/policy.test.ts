import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { PolicyError, PolicyStore } from "../../src/p2p/policy.ts";

let root: string;
let home: string;
let workspace: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-p2p-policy-"));
	home = join(root, "home");
	workspace = join(root, "workspace");
	mkdirSync(join(home, ".autorag"), { recursive: true });
	mkdirSync(workspace, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeGlobalConfig(config: Record<string, unknown>): void {
	writeFileSync(join(home, ".autorag", "config.json"), `${JSON.stringify(config)}\n`);
}

function writeWorkspacePolicy(contents: string): void {
	const directory = join(workspace, ".autorag", "p2p");
	mkdirSync(directory, { recursive: true });
	writeFileSync(join(directory, "policy.toml"), contents);
}

function store(): PolicyStore {
	return new PolicyStore({ workspacePath: workspace, homePath: home });
}

describe("PolicyStore", () => {
	it("defaults to private when no policy files are present", () => {
		const result = store().resolvePolicy("/docs/missing.md", "peer-a");

		expect(result).toEqual({ tier: "private", allowed: false, shareBytes: false, redact: true });
	});

	it("resolves all F3 tiers and merges workspace policy over global defaults", () => {
		writeGlobalConfig({
			p2p: {
				policy: {
					"/docs/**": { tier: "always" },
					"/docs/private/**": { tier: "private" },
					"/docs/friends/**": { tier: "peers", peers: ["peer-listed"] },
				},
			},
		});
		writeWorkspacePolicy(`
[policy."/docs/friends/**"]
tier = "peers"
peers = ["peer-workspace"]

[policy."/docs/never/**"]
tier = "never"
`);

		const policy = store();
		policy.promoteSource("/docs/public/readme.md");
		policy.promoteSource("/docs/private/secret.md");
		policy.promoteSource("/docs/friends/notes.md");
		policy.promoteSource("/docs/never/secret.md");

		expect(policy.resolvePolicy("/docs/public/readme.md", "peer-a")).toEqual({
			tier: "always",
			allowed: true,
			shareBytes: true,
			redact: false,
		});
		expect(policy.resolvePolicy("/docs/private/secret.md", "peer-a")).toEqual({
			tier: "private",
			allowed: false,
			shareBytes: false,
			redact: true,
		});
		expect(policy.resolvePolicy("/docs/friends/notes.md", "peer-workspace")).toEqual({
			tier: "peers",
			allowed: true,
			shareBytes: false,
			redact: true,
		});
		expect(policy.resolvePolicy("/docs/friends/notes.md", "peer-listed")).toEqual({
			tier: "peers",
			allowed: false,
			shareBytes: false,
			redact: true,
		});
		expect(policy.resolvePolicy("/docs/never/secret.md", "peer-workspace")).toEqual({
			tier: "never",
			allowed: false,
			shareBytes: false,
			redact: true,
		});
	});

	it("matches slash datasource identifiers as unified source strings", () => {
		writeWorkspacePolicy(`
[policy."/kakao/personal/chunks/**"]
tier = "always"

[policy."/gmail/personal/chunks/**"]
tier = "peers"
peers = ["mail-peer"]
`);
		const policy = store();
		policy.promoteSource("/kakao/personal/chunks/chunk-1");
		policy.promoteSource("/kakao/other/chunks/chunk-1");
		policy.promoteSource("/gmail/personal/chunks/message-1");

		expect(policy.resolvePolicy("/kakao/personal/chunks/chunk-1", "any").allowed).toBe(true);
		expect(policy.resolvePolicy("/kakao/other/chunks/chunk-1", "any").allowed).toBe(false);
		expect(policy.resolvePolicy("/gmail/personal/chunks/message-1", "mail-peer")).toMatchObject({
			tier: "peers",
			allowed: true,
			shareBytes: false,
			redact: true,
		});
	});

	it("lets a narrower never rule beat a broader always rule", () => {
		writeWorkspacePolicy(`
[policy."/docs/**"]
tier = "always"

[policy."/docs/secret/**"]
tier = "never"
`);
		const policy = store();
		policy.promoteSource("/docs/secret/plan.md");
		policy.promoteSource("/docs/public/plan.md");

		expect(policy.resolvePolicy("/docs/secret/plan.md", "peer").tier).toBe("never");
		expect(policy.resolvePolicy("/docs/public/plan.md", "peer").tier).toBe("always");
	});

	it("keeps an unseen source private until the indexer marks it seen", () => {
		writeWorkspacePolicy(`
[policy."/docs/**"]
tier = "always"
`);
		const policy = store();

		expect(policy.resolvePolicy("/docs/new.md", "peer")).toEqual({
			tier: "private",
			allowed: false,
			shareBytes: false,
			redact: true,
		});

		policy.markSourceSeen("/docs/new.md");
		expect(policy.resolvePolicy("/docs/new.md", "peer").allowed).toBe(false);
		policy.promoteSource("/docs/new.md");
		expect(policy.resolvePolicy("/docs/new.md", "peer").allowed).toBe(true);
	});

	it("allows unseen matching sources when p2p.newFilesPublic is enabled", () => {
		writeGlobalConfig({ p2p: { newFilesPublic: true, policy: { "/docs/**": { tier: "always" } } } });
		const policy = store();

		expect(policy.resolvePolicy("/docs/new.md", "peer")).toMatchObject({ tier: "always", allowed: true });
	});

	it("uses the quota defaults and applies workspace quota overrides", () => {
		writeGlobalConfig({
			p2p: {
				quotas: { queriesPerHour: 20, burst: 5, maxBodyBytes: 1000, maxFileBytes: 2000 },
			},
		});
		writeWorkspacePolicy(`
[quotas]
burst = 7
max_file_bytes = 3000
`);

		expect(store().quotas).toEqual({
			queriesPerHour: 20,
			burst: 7,
			maxBodyBytes: 1000,
			maxFileBytes: 3000,
		});
		expect(store().getQuotas()).toEqual(store().quotas);
	});

	it("persists source promotion for a later store instance", () => {
		writeWorkspacePolicy(`
[policy."/docs/**"]
tier = "always"
`);
		const first = store();
		first.markSourceSeen("/docs/persisted.md");

		const second = store();
		expect(second.resolvePolicy("/docs/persisted.md", "peer").allowed).toBe(false);
		second.promoteSource("/docs/persisted.md");
		expect(second.resolvePolicy("/docs/persisted.md", "peer")).toMatchObject({ tier: "always", allowed: true });
	});

	it("supports local glob wildcards without reparsing datasource identifiers", () => {
		writeWorkspacePolicy(`
[policy."/kakao/personal/*/chunk-?"]
tier = "always"
`);
		const policy = store();
		policy.promoteSource("/kakao/personal/alice/chunk-1");
		policy.promoteSource("/kakao/personal/alice/nested/chunk-1");

		expect(policy.resolvePolicy("/kakao/personal/alice/chunk-1", "peer").allowed).toBe(true);
		expect(policy.resolvePolicy("/kakao/personal/alice/nested/chunk-1", "peer").allowed).toBe(false);
	});

	it("accepts absolute local source keys and rejects unsafe control characters", () => {
		writeGlobalConfig({
			p2p: {
				policy: {
					[join(workspace, "docs/**")]: { tier: "always" },
					"/docs/\u0000secret/**": { tier: "always" },
				},
			},
		});

		expect(() => store()).toThrow(PolicyError);
	});

	it("reports malformed TOML as a typed PolicyError", () => {
		writeWorkspacePolicy(`
[policy."/docs/**"]
tier = "always
`);

		expect(() => store()).toThrow(PolicyError);
		expect(() => store()).toThrow(/policy\.toml/i);
	});
});
