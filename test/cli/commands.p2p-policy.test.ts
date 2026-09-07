import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runP2pPolicy } from "../../src/cli/commands/p2p-policy.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let root: string;
let policyPath: string;
let ctx: CommandContext;
let captured: string[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-p2p-policy-"));
	policyPath = join(root, ".autorag", "p2p", "policy.toml");
	mkdirSync(join(root, ".autorag", "p2p"), { recursive: true });
	captured = [];
	ctx = {
		positionals: [],
		flags: {},
		json: false,
		debug: false,
		cwd: root,
		stdout: (line: string) => {
			captured.push(line);
		},
		stderr: (line: string) => {
			captured.push(line);
		},
	};
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function _writePolicy(data: Record<string, unknown>): void {
	writeFileSync(policyPath, JSON.stringify(data));
}

describe("autorag p2p policy list", () => {
	it("returns empty policy when no policy.toml exists", () => {
		ctx.positionals = ["list"];
		runP2pPolicy(ctx);
		expect(captured).toHaveLength(1);
		const output = JSON.parse(captured[0]!);
		expect(output).toEqual({});
	});

	it("returns merged entries from policy.toml (virtual-path keys only)", () => {
		writeFileSync(policyPath, '[policy]\n"/docs/*" = { tier = "always" }\n');
		ctx.positionals = ["list"];
		runP2pPolicy(ctx);
		expect(captured).toHaveLength(1);
		const output = JSON.parse(captured[0]!);
		expect(output).toHaveProperty("/docs/*");
		expect(output["/docs/*"]).toEqual({ tier: "always" });
	});

	it("returns default-deny for unknown sources — shows all entries including private as seen in effective policy", () => {
		writeFileSync(
			policyPath,
			'[policy]\n"/docs/private" = { tier = "private" }\n"/docs/public" = { tier = "always" }\n',
		);
		ctx.positionals = ["list"];
		runP2pPolicy(ctx);
		const output = JSON.parse(captured[0]!);
		expect(output).toHaveProperty("/docs/private");
		expect(output["/docs/private"]).toEqual({ tier: "private" });
		expect(output).toHaveProperty("/docs/public");
		expect(output["/docs/public"]).toEqual({ tier: "always" });
	});

	it("exposes datasource globs (e.g. kakao:...) correctly", () => {
		writeFileSync(policyPath, '[policy]\n"kakao:*" = { tier = "peers", peers = ["abc123"] }\n"/docs/*" = "always"\n');
		ctx.positionals = ["list"];
		runP2pPolicy(ctx);
		const output = JSON.parse(captured[0]!);
		expect(output).toHaveProperty("kakao:*");
		expect(output["kakao:*"]).toEqual({ tier: "peers", peers: ["abc123"] });
		expect(output).toHaveProperty("/docs/*");
		expect(output["/docs/*"]).toEqual({ tier: "always" });
	});
});

describe("autorag p2p policy set", () => {
	it("sets a simple tier for a virtual path glob", () => {
		ctx.positionals = ["set", "/docs/*", "always"];
		runP2pPolicy(ctx);
		expect(captured).toHaveLength(1);
		const content = JSON.parse(captured[0]!);
		expect(content).toEqual({ ok: true, key: "/docs/*", tier: "always" });
		// Verify persisted TOML
		const toml = readFileSync(policyPath, "utf8");
		expect(toml).toContain("/docs/*");
		expect(toml).toContain("always");
	});

	it("sets a peers tier with --peer fingerprints", () => {
		ctx.positionals = ["set", "/peers/*", "peers"];
		ctx.flags = { peer: "fp1" };
		runP2pPolicy(ctx);
		const content = JSON.parse(captured[0]!);
		expect(content).toEqual({ ok: true, key: "/peers/*", tier: "peers", peers: ["fp1"] });
		const toml = readFileSync(policyPath, "utf8");
		expect(toml).toContain("peers");
		expect(toml).toContain("fp1");
	});

	it("rejects invalid tier", () => {
		ctx.positionals = ["set", "/docs/*", "superpublic"];
		expect(() => runP2pPolicy(ctx)).toThrow(/invalid/i);
		expect(existsSync(policyPath)).toBe(false);
	});

	it("rejects set without source glob arg", () => {
		ctx.positionals = ["set"];
		expect(() => runP2pPolicy(ctx)).toThrow(/source|glob/i);
	});

	it("rejects set with peers but no --peer flags", () => {
		ctx.positionals = ["set", "/peers/*", "peers"];
		expect(() => runP2pPolicy(ctx)).toThrow(/peer/i);
	});

	it("overwrites existing entry for same key", () => {
		writeFileSync(policyPath, '[policy]\n"/docs/*" = { tier = "private" }\n');
		ctx.positionals = ["set", "/docs/*", "always"];
		runP2pPolicy(ctx);
		const toml = readFileSync(policyPath, "utf8");
		// Should have just the one entry with new tier
		expect(toml).toContain("always");
		expect(toml).not.toContain("private");
	});

	it("adds new entry alongside existing ones", () => {
		writeFileSync(policyPath, '[policy]\n"/existing" = { tier = "never" }\n');
		ctx.positionals = ["set", "/new", "always"];
		runP2pPolicy(ctx);
		const toml = readFileSync(policyPath, "utf8");
		expect(toml).toContain("/existing");
		expect(toml).toContain("/new");
		expect(toml).toContain("never");
		expect(toml).toContain("always");
	});

	it("supports datasource scheme keys like kakao:*", () => {
		ctx.positionals = ["set", "kakao:*", "always"];
		runP2pPolicy(ctx);
		const toml = readFileSync(policyPath, "utf8");
		expect(toml).toContain("kakao:*");
		expect(toml).toContain("always");
	});
});

describe("autorag p2p policy unset", () => {
	it("removes a key from policy", () => {
		writeFileSync(policyPath, '[policy]\n"/docs/*" = { tier = "always" }\n"/other" = { tier = "never" }\n');
		ctx.positionals = ["unset", "/docs/*"];
		runP2pPolicy(ctx);
		const content = JSON.parse(captured[0]!);
		expect(content).toEqual({ ok: true, key: "/docs/*" });
		const toml = readFileSync(policyPath, "utf8");
		expect(toml).not.toContain("/docs/*");
		expect(toml).toContain("/other");
	});

	it("rejects unset without key arg", () => {
		ctx.positionals = ["unset"];
		expect(() => runP2pPolicy(ctx)).toThrow(/key/i);
	});
});

describe("round-trip: set then list", () => {
	it("set then list shows the entry", () => {
		ctx.positionals = ["set", "/my/docs/*", "peers"];
		ctx.flags = { peer: "abc" };
		runP2pPolicy(ctx);
		captured = [];
		ctx.positionals = ["list"];
		ctx.flags = {};
		runP2pPolicy(ctx);
		const output = JSON.parse(captured[0]!);
		expect(output).toHaveProperty("/my/docs/*");
		expect(output["/my/docs/*"]).toEqual({ tier: "peers", peers: ["abc"] });
	});

	it("set then unset shows empty", () => {
		ctx.positionals = ["set", "/tmp", "always"];
		runP2pPolicy(ctx);
		captured = [];
		ctx.positionals = ["unset", "/tmp"];
		runP2pPolicy(ctx);
		captured = [];
		ctx.positionals = ["list"];
		runP2pPolicy(ctx);
		expect(JSON.parse(captured[0]!)).toEqual({});
	});
});

describe("autorag p2p policy workspace routing", () => {
	it("writes policy under config workspacePath, not cwd", () => {
		const workspace = join(root, "configured-ws");
		mkdirSync(join(workspace, ".autorag", "p2p"), { recursive: true });
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [root],
				workspacePath: workspace,
				memoryPath: join(workspace, "memory.json"),
			}),
		);
		ctx.flags = { config: configPath };
		ctx.positionals = ["set", "/docs/*", "always"];
		runP2pPolicy(ctx);
		expect(existsSync(join(workspace, ".autorag", "p2p", "policy.toml"))).toBe(true);
		expect(existsSync(join(root, ".autorag", "p2p", "policy.toml"))).toBe(false);
	});
});
