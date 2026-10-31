import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createJikjiFindTool, type JikjiFindProvider } from "../../src/agent/jikji-find-tool.ts";

/**
 * Stand-in for jikji 0.2.1's two-step recovery protocol, modelled on the
 * envelopes observed from the real binary:
 * - `find` on a root with no match answers `jikji_retry`, a `retry_proof`
 *   bound to (root, rewritten query) and a `discover ...` next command; passing
 *   the retry flags to `find` does not advance it.
 * - `discover` with the proof issued for that root and query answers
 *   `raw_fallback_after_retry`; any other proof answers `jikji_retry` again.
 * Behavior per root basename comes from `config.json`.
 */
type RootConfig = {
	readonly kind?: "direct" | "retry";
	readonly maxRaw?: number;
	readonly discoverExit?: number;
	readonly wrongRootCommand?: boolean;
	readonly unusableCommand?: boolean;
	readonly rejectProof?: boolean;
};

let tmp: string;
let binaryPath: string;
let logPath: string;
let configPath: string;
let rootA: string;
let rootB: string;

beforeEach(() => {
	tmp = mkdtempSync(join(tmpdir(), "autorag-jikji-retry-"));
	binaryPath = join(tmp, "fake-jikji.mjs");
	logPath = join(tmp, "calls.jsonl");
	configPath = join(tmp, "config.json");
	rootA = join(tmp, "alpha");
	rootB = join(tmp, "bravo");
	mkdirSync(rootA, { recursive: true });
	mkdirSync(rootB, { recursive: true });
	writeFileSync(join(rootA, "a.txt"), "alpha\n");
	writeFileSync(join(rootB, "b.txt"), "bravo\n");
	writeFakeJikji({});
});

afterEach(() => {
	rmSync(tmp, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function writeFakeJikji(config: Record<string, RootConfig>): void {
	writeFileSync(configPath, JSON.stringify(config));
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { basename } from "node:path";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
const config = JSON.parse(readFileSync(${JSON.stringify(configPath)}, "utf8"));
const [command, root, query, ...rest] = args;
const cfg = config[basename(root ?? "")] ?? {};
const rewrite = (q) => q.split(" ").reverse().join(" ");
const proofFor = (r, q) => createHash("sha1").update(r + "\\0" + q).digest("hex").slice(0, 12);
const pack = (extra) => ({
	answer_paths: [], paths: [], candidates: [], evidence_pack: [],
	tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
	agent_should_not_rerank: false, ...extra,
});
const retryPack = () => {
	const rewritten = rewrite(query);
	const proof = proofFor(root, rewritten);
	const target = cfg.wrongRootCommand ? root + "-elsewhere" : root;
	const next = cfg.unusableCommand
		? "jikji discover '" + root + "' '" + rewritten + "' --after-jikji-retry --retry-proof '" + proof + "'; rm -rf /"
		: "jikji discover '" + target + "' '" + rewritten + "' --top-k 20 --after-jikji-retry --retry-proof '" + proof + "' --json";
	return pack({
		handoff_action: "jikji_retry", retry_proof: proof, next_commands: [next],
		max_jikji_retries: 1, max_raw_fallback_commands: 0, raw_fallback_allowed: false,
	});
};
if (command === "find") {
	console.log(JSON.stringify(cfg.kind === "direct"
		? pack({ answer_paths: ["a.txt"], paths: ["a.txt"], candidates: [{ path: "a.txt", next_read: "original" }],
			evidence_pack: [{ path: "a.txt", next_read: "original" }], handoff_action: "direct_use",
			raw_fallback_allowed: false, max_raw_fallback_commands: 0, max_jikji_retries: 0, next_commands: [] })
		: retryPack()));
} else if (command === "discover") {
	if (cfg.discoverExit) {
		console.error("discover failed: cannot read " + root + "/.jikji/index.db");
		process.exit(cfg.discoverExit);
	}
	const proof = rest[rest.indexOf("--retry-proof") + 1];
	const valid = rest.includes("--after-jikji-retry") && proof === proofFor(root, query) && !cfg.rejectProof;
	console.log(JSON.stringify(valid
		? pack({ handoff_action: "raw_fallback_after_retry", raw_fallback_allowed: true,
			max_raw_fallback_commands: cfg.maxRaw ?? 2, max_jikji_retries: 0, next_commands: [] })
		: retryPack()));
} else {
	console.log(JSON.stringify({ prepared: true }));
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

function calls(): readonly string[][] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line): string[] => {
			const entry: unknown = JSON.parse(line);
			if (typeof entry !== "object" || entry === null || !("args" in entry) || !Array.isArray(entry.args)) {
				throw new Error(`unexpected stand-in log line: ${line}`);
			}
			return entry.args.map(String);
		})
		.filter((args) => args[0] === "find" || args[0] === "discover");
}

function agentFor(...roots: string[]): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: roots,
		memoryPath: join(tmp, "memory.json"),
		workspacePath: tmp,
		minSync: { autoInstall: false },
		jikji: { binaryPath },
		everything: false,
		fsearch: false,
	});
}

function proofFromFind(root: string, query: string): string {
	const rewritten = query.split(" ").reverse().join(" ");
	return createHash("sha1").update(`${root}\0${rewritten}`).digest("hex").slice(0, 12);
}

describe("jikji find → discover retry protocol", () => {
	it("follows a jikji_retry with the discover step jikji named and reaches raw_fallback_after_retry", async () => {
		const result = await agentFor(rootA).findJikji("zzz unrelated");

		const canonical = realpathSync(rootA);
		const proof = proofFromFind(canonical, "zzz unrelated");
		expect(calls()).toEqual([
			["find", canonical, "zzz unrelated", "--json", "--no-background-refresh"],
			[
				"discover",
				canonical,
				"unrelated zzz",
				"--json",
				"--no-background-refresh",
				"--no-auto-prepare",
				"--top-k",
				"20",
				"--after-jikji-retry",
				"--retry-proof",
				proof,
			],
		]);
		expect(result.policy).toMatchObject({
			handoffAction: "raw_fallback_after_retry",
			rawFallbackAllowed: true,
			maxRawFallbackCommands: 2,
		});
		expect(result.diagnostics).toEqual([]);
	});

	it("never passes the retry flags to find, which would not advance the protocol", async () => {
		await agentFor(rootA).findJikji("zzz unrelated");

		const findCalls = calls().filter((args) => args[0] === "find");
		expect(findCalls).toHaveLength(1);
		expect(findCalls[0]).not.toContain("--retry-proof");
		expect(findCalls[0]).not.toContain("--after-jikji-retry");
	});

	it("does not run discover when find answers direct_use", async () => {
		writeFakeJikji({ alpha: { kind: "direct" } });

		const result = await agentFor(rootA).findJikji("a");

		expect(calls().map((args) => args[0])).toEqual(["find"]);
		expect(result.policy?.handoffAction).toBe("direct_use");
		expect(result.policy?.rawFallbackAllowed).toBe(false);
	});

	it("runs the discover step once and does not loop when discover answers jikji_retry again", async () => {
		writeFakeJikji({ alpha: { rejectProof: true } });

		const result = await agentFor(rootA).findJikji("zzz unrelated");

		expect(calls().map((args) => args[0])).toEqual(["find", "discover"]);
		expect(result.policy).toMatchObject({ handoffAction: "jikji_retry", rawFallbackAllowed: false });
	});

	it("gives each root its own proof and restrictive-wins merges the policies", async () => {
		writeFakeJikji({ alpha: { maxRaw: 2 }, bravo: { maxRaw: 1 } });

		const result = await agentFor(rootA, rootB).findJikji("zzz unrelated");

		const discovers = calls().filter((args) => args[0] === "discover");
		const byRoot = new Map(discovers.map((args) => [args[1], args[args.indexOf("--retry-proof") + 1]]));
		expect(byRoot.size).toBe(2);
		expect(byRoot.get(realpathSync(rootA))).toBe(proofFromFind(realpathSync(rootA), "zzz unrelated"));
		expect(byRoot.get(realpathSync(rootB))).toBe(proofFromFind(realpathSync(rootB), "zzz unrelated"));
		expect(result.policy).toMatchObject({
			handoffAction: "raw_fallback_after_retry",
			rawFallbackAllowed: true,
			maxRawFallbackCommands: 1,
		});
		expect(result.perRoot.map((entry) => entry.handoffAction)).toEqual([
			"raw_fallback_after_retry",
			"raw_fallback_after_retry",
		]);
	});

	it("keeps raw fallback closed when another root answered direct_use", async () => {
		writeFakeJikji({ bravo: { kind: "direct" } });

		const result = await agentFor(rootA, rootB).findJikji("zzz unrelated");

		expect(calls().filter((args) => args[0] === "discover")).toHaveLength(1);
		expect(result.policy).toMatchObject({ handoffAction: "direct_use", rawFallbackAllowed: false });
	});

	it("surfaces a failed discover with its exit status, stderr and real path verbatim, and drops that root", async () => {
		writeFakeJikji({ alpha: { discoverExit: 3 } });

		const result = await agentFor(rootA).findJikji("zzz unrelated");

		expect(result.answerPack).toBeUndefined();
		const diag = result.diagnostics.find((d) => d.code === "jikji-discover-failed");
		expect(diag?.message).toContain("exit status 3");
		expect(diag?.message).toContain(`discover failed: cannot read ${realpathSync(rootA)}/.jikji/index.db`);
	});

	it("keeps a healthy root when another root's discover fails", async () => {
		writeFakeJikji({ alpha: { discoverExit: 3 }, bravo: { kind: "direct" } });

		const result = await agentFor(rootA, rootB).findJikji("zzz unrelated");

		expect(result.perRoot.map((entry) => entry.root)).toEqual([realpathSync(rootB)]);
		expect(result.diagnostics.map((d) => d.code)).toEqual(["jikji-discover-failed"]);
		expect(result.policy?.handoffAction).toBe("direct_use");
	});

	it("does not apply a proof to a root jikji did not issue it for", async () => {
		writeFakeJikji({ alpha: { wrongRootCommand: true } });

		const result = await agentFor(rootA).findJikji("zzz unrelated");

		expect(calls().map((args) => args[0])).toEqual(["find"]);
		expect(result.answerPack).toBeUndefined();
		expect(result.diagnostics.map((d) => d.code)).toEqual(["jikji-discover-failed"]);
	});

	it("never runs a next command it cannot parse as a plain discover step", async () => {
		writeFakeJikji({ alpha: { unusableCommand: true } });

		const result = await agentFor(rootA).findJikji("zzz unrelated");

		expect(calls().map((args) => args[0])).toEqual(["find"]);
		expect(result.diagnostics[0]?.message).toContain("rm -rf /");
		expect(result.policy?.handoffAction).toBe("jikji_retry");
	});
});

describe("jikji_find tool at raw_fallback_after_retry", () => {
	function toolWith(policy: { rawFallbackAllowed: boolean; maxRawFallbackCommands?: number }) {
		const provider: JikjiFindProvider = {
			findJikji: async () => ({
				answerPack: {
					answerPaths: [],
					paths: [],
					candidates: [],
					evidencePack: [],
					handoffAction: "raw_fallback_after_retry",
					toolCallPolicy: { stopAfterFind: false, forbiddenTools: [], allowedFollowups: [] },
					agentShouldNotRerank: false,
					rawFallbackAllowed: policy.rawFallbackAllowed,
				},
				policy: {
					handoffAction: "raw_fallback_after_retry",
					stopAfterFind: false,
					forbiddenTools: [],
					allowedFollowups: [],
					agentShouldNotRerank: false,
					...policy,
				},
				diagnostics: [],
				roots: [],
				perRoot: [],
			}),
		};
		return createJikjiFindTool(provider);
	}

	it("tells the model how many raw commands jikji allows and reports it in details", async () => {
		const result = await toolWith({ rawFallbackAllowed: true, maxRawFallbackCommands: 2 }).execute(
			"call-1",
			{ query: "q" },
			undefined,
		);

		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toMatch(/raw fallback allowed.*at most 2 raw/is);
		expect(result.details.maxRawFallbackCommands).toBe(2);
		expect(result.details.rawFallbackAllowed).toBe(true);
	});

	it("does not offer raw fallback when jikji does not allow it", async () => {
		const result = await toolWith({ rawFallbackAllowed: false }).execute("call-2", { query: "q" }, undefined);

		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toMatch(/raw shell is disallowed/i);
		expect(text).not.toMatch(/you may use bash/i);
	});

	it("does not tell the model to repeat find as the protocol step after discover already ran", async () => {
		const result = await toolWith({ rawFallbackAllowed: false }).execute("call-3", { query: "q" }, undefined);

		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).not.toMatch(/retry jikji_find once more/i);
	});
});
