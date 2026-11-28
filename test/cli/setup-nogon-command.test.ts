import { describe, expect, it } from "vitest";
import { commandUsage } from "../../src/cli/command-help.ts";
import { renderSetup } from "../../src/cli/commands/setup.ts";
import { parseArgs } from "../../src/cli/index.ts";
import type { SetupReport } from "../../src/cli/setup.ts";

const BASE: SetupReport = {
	ok: true,
	mode: "semantic",
	embedding: { backend: "daemon", provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 1024 },
	datasources: [{ name: "browser", state: "configured" }],
	changed: false,
};

describe("autorag setup --sync-nogon", () => {
	it("is a boolean flag, so it never swallows the next argument", () => {
		const parsed = parseArgs(["setup", "--sync-nogon", "--json"]);
		expect("error" in parsed).toBe(false);
		if ("error" in parsed) return;
		expect(parsed.flags["sync-nogon"]).toBe(true);
		expect(parsed.flags.json).toBe(true);
	});

	it("is documented in `autorag setup --help`", () => {
		expect(commandUsage("setup")).toContain("--sync-nogon");
	});
});

describe("renderSetup: Nogon", () => {
	it("prints nothing about Nogon when none is configured", () => {
		expect(renderSetup(BASE)).not.toMatch(/nogon/i);
	});

	it("names each problem and the exact command that fixes it", () => {
		const text = renderSetup({
			...BASE,
			nogon: {
				datasource: "browser",
				installed: true,
				running: false,
				reason: "connect ECONNREFUSED 127.0.0.1:4433",
				remediation: "Start it: brew services start nogon",
			},
		});
		expect(text).toContain("nogon browser: not running");
		expect(text).toContain("ECONNREFUSED");
		expect(text).toContain("brew services start nogon");
	});

	it("shows which semantic_search keys disagree, and whether they were applied", () => {
		const pending = renderSetup({
			...BASE,
			nogon: {
				datasource: "browser",
				installed: true,
				running: true,
				embedding: { inSync: false, differs: ["dimensions", "embedding_model"], applied: false },
				remediation: "Make Nogon embed with the shared model: autorag setup --sync-nogon",
			},
		});
		expect(pending).toContain("embedding: differs from the shared model (dimensions, embedding_model)");
		expect(pending).toContain("--sync-nogon");
		const done = renderSetup({
			...BASE,
			nogon: {
				datasource: "browser",
				installed: true,
				running: true,
				embedding: {
					inSync: true,
					differs: ["dimensions"],
					applied: true,
					restartRequired: false,
					reindexRequired: false,
				},
			},
		});
		expect(done).toContain("embedding: now shares the model (updated dimensions)");
		const inSync = renderSetup({
			...BASE,
			nogon: {
				datasource: "browser",
				installed: true,
				running: true,
				embedding: { inSync: true, differs: [], applied: false },
			},
		});
		expect(inSync).toContain("embedding: shares the model");
	});
});
