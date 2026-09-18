import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AutoRAGRefreshOptions, AutoRAGRefreshResult } from "../../src/agent/agent.ts";
import { runLiteRefresh } from "../../src/cli/commands/lite-refresh.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { main, parseArgs } from "../../src/cli/index.ts";
import { AutoRAGLite } from "../../src/core.ts";

describe("lite refresh forwarding", () => {
	it("forwards --method and --full to the model-free runtime", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-refresh-test-"));
		writeFileSync(
			join(root, "config.json"),
			JSON.stringify({ searchPaths: [root], workspacePath: root, memoryPath: join(root, "memory.json") }),
		);
		let received: { force: boolean | undefined; options?: AutoRAGRefreshOptions } | undefined;
		vi.spyOn(AutoRAGLite.prototype, "refresh").mockImplementation(
			async (force?: boolean, options?: AutoRAGRefreshOptions): Promise<AutoRAGRefreshResult> => {
				received = { force, options };
				return { indexPath: root, scanned: 0, written: 0, deleted: 0, skipped: 0, diagnostics: [] };
			},
		);
		const output: string[] = [];
		const context: CommandContext = {
			positionals: [],
			flags: { config: join(root, "config.json"), method: "minsync,jikji,datasources", full: true },
			json: true,
			debug: false,
			cwd: root,
			stdout: (line) => output.push(line),
			stderr: (line) => output.push(line),
		};

		expect(await runLiteRefresh(context)).toBe(0);
		expect(received).toEqual({
			force: true,
			options: { methods: ["minsync", "datasources", "jikji"] },
		});
		expect(JSON.parse(output[0] ?? "{}").ok).toBe(true);
		rmSync(root, { recursive: true, force: true });
	});
});

describe("parseArgs", () => {
	it("collects positionals in order", () => {
		const parsed = parseArgs(["search", "hello", "world"]);
		expect("error" in parsed).toBe(false);
		if ("error" in parsed) return;
		expect(parsed.positionals).toEqual(["search", "hello", "world"]);
	});

	it("parses boolean and value flags", () => {
		const parsed = parseArgs(["refresh", "--force", "--config", "a.json", "--json"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags.force).toBe(true);
		expect(parsed.flags.json).toBe(true);
		expect(parsed.flags.config).toBe("a.json");
	});

	it("parses --key=value form", () => {
		const parsed = parseArgs(["search", "--top-k=5"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["top-k"]).toBe("5");
	});

	it("keeps two-word command sub-words as positionals", () => {
		const parsed = parseArgs(["memory", "inspect"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["memory", "inspect"]);
	});

	it("rejects an unknown flag", () => {
		const parsed = parseArgs(["refresh", "--nope"]);
		expect(parsed).toEqual({ error: "Unknown flag: --nope" });
	});

	it("rejects a value flag without a value", () => {
		const parsed = parseArgs(["search", "--scope"]);
		expect("error" in parsed).toBe(true);
	});

	it("accepts --method as a value flag", () => {
		const parsed = parseArgs(["refresh", "--method", "bm25,minsync"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags.method).toBe("bm25,minsync");
	});

	it("accepts embedder-* value flags", () => {
		const parsed = parseArgs([
			"init",
			"--embedder-id",
			"text-embedding-3-small",
			"--embedder-dimension",
			"1536",
			"--embedder-api-key-env",
			"OPENAI_API_KEY",
		]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["embedder-id"]).toBe("text-embedding-3-small");
		expect(parsed.flags["embedder-dimension"]).toBe("1536");
		expect(parsed.flags["embedder-api-key-env"]).toBe("OPENAI_API_KEY");
	});

	it("accepts the lite full-refresh and report-input flags", () => {
		const refresh = parseArgs(["lite", "refresh", "--full", "--method", "minsync,jikji"]);
		expect("error" in refresh).toBe(false);
		if ("error" in refresh) return;
		expect(refresh.flags.full).toBe(true);
		expect(refresh.flags.method).toBe("minsync,jikji");

		const report = parseArgs(["lite", "report", "helper query", "--input", "/tmp/report.json"]);
		expect("error" in report).toBe(false);
		if ("error" in report) return;
		expect(report.flags.input).toBe("/tmp/report.json");
	});
});

describe("main routing", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("prints usage and exits 0 for --help", async () => {
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		const code = await main(["--help"]);
		expect(code).toBe(0);
		expect(out).toHaveBeenCalled();
		expect(String(out.mock.calls[0]?.[0])).toContain("Usage: autorag");
	});

	it("prints usage and exits 0 with no command", async () => {
		vi.spyOn(process.stdout, "write").mockReturnValue(true);
		expect(await main([])).toBe(0);
	});

	it("exits 2 for an unknown command", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["frobnicate"]);
		expect(code).toBe(2);
		expect(String(err.mock.calls[0]?.[0])).toContain("Unknown command");
	});

	it("exits 2 for an unknown flag", async () => {
		vi.spyOn(process.stderr, "write").mockReturnValue(true);
		expect(await main(["refresh", "--bogus"])).toBe(2);
	});
});

describe("parseArgs health flags", () => {
	it("accepts health as a command", () => {
		const parsed = parseArgs(["health"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["health"]);
	});

	it("accepts duplicates as a command", () => {
		const parsed = parseArgs(["duplicates"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["duplicates"]);
	});

	it("accepts evidence result selection", () => {
		const parsed = parseArgs(["evidence", "session-1", "--result", "2", "--json"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["evidence", "session-1"]);
		expect(parsed.flags.result).toBe("2");
	});

	it("accepts --skip-probes as a boolean flag", () => {
		const parsed = parseArgs(["health", "--skip-probes"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["skip-probes"]).toBe(true);
	});

	it("accepts --timeout-ms as a value flag", () => {
		const parsed = parseArgs(["health", "--timeout-ms", "5000"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["timeout-ms"]).toBe("5000");
	});

	it("accepts --timeout-ms=value form", () => {
		const parsed = parseArgs(["health", "--timeout-ms=8000"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["timeout-ms"]).toBe("8000");
	});

	it("rejects --doctor (no alias added)", () => {
		const parsed = parseArgs(["doctor"]);
		// doctor is not a command; it's an unknown positional but parseArgs
		// does not reject unknown commands (only unknown flags). The main
		// router rejects it. Verify doctor is not in COMMANDS by checking
		// that main returns 2 for it.
		expect("error" in parsed).toBe(false);
	});
});

describe("main health routing", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("rejects doctor as an unknown command (no alias)", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["doctor"]);
		expect(code).toBe(2);
		expect(String(err.mock.calls[0]?.[0])).toContain("Unknown command");
	});

	it("includes health in the usage text", async () => {
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		await main(["--help"]);
		const usage = String(out.mock.calls[0]?.[0]);
		expect(usage).toContain("health");
		expect(usage).toContain("--skip-probes");
		expect(usage).toContain("--timeout-ms");
		expect(usage).toContain("duplicates");
		expect(usage).toContain("evidence <session>");
	});

	it("advertises and dispatches the lite headless namespace", async () => {
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["lite", "retrieve", "helper function", "--help"]);
		const usage = String(out.mock.calls[0]?.[0] ?? "");
		expect(code).toBe(0);
		expect(usage).toContain("lite retrieve");
		expect(usage).toContain("lite report");
		expect(usage).toContain("lite init");
		expect(usage).toContain("lite health");
		expect(String(err.mock.calls[0]?.[0] ?? "")).not.toContain("Unknown command");
	});
});

/** The version the published package reports is the one in package.json. */
const packageVersion = (
	JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")) as { version: string }
).version;

/** Every command except `lite`, which owns its own usage renderer. */
const COMMANDS_WITH_OWN_HELP = [
	"init",
	"setup",
	"refresh",
	"status",
	"search",
	"feedback",
	"evidence",
	"memory",
	"index",
	"watch",
	"health",
	"duplicates",
	"tui",
	"ui",
	"serve",
	"p2p",
	"models",
	"gateway",
] as const;

/** Tokens each command's own help must name so it cannot be the global list. */
const COMMAND_HELP_TOKENS: Record<(typeof COMMANDS_WITH_OWN_HELP)[number], readonly string[]> = {
	init: ["--search-paths", "--workspace", "--memory-path", "--force"],
	setup: ["--search-paths", "--workspace", "--profile", "--format json"],
	refresh: ["--method", "--force"],
	status: ["Usage: autorag status"],
	search: ["--top-k", "--scope", "--tags", "--fast-thinking", "--final-thinking", "--single-phase"],
	feedback: ["--useful", "--not-useful"],
	evidence: ["--result"],
	memory: ["inspect"],
	index: ["reset", "rebuild", "--method", "--yes"],
	watch: ["--once", "--immediate", "--debounce-ms"],
	health: ["--skip-probes", "--timeout-ms"],
	duplicates: ["DIR"],
	tui: ["Usage: autorag tui"],
	ui: ["--port", "--host", "--no-open", "--allow-remote"],
	serve: ["--port", "--force"],
	p2p: ["peers", "requests", "--contact-id"],
	models: ["prefetch", "import", "verify", "--profile"],
	gateway: ["status", "stop"],
};

const GLOBAL_USAGE_HEADER = "Usage: autorag <command> [args] [flags]";

function captureStdio(): { stdout: () => string; stderr: () => string } {
	const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
	const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
	return {
		stdout: () => out.mock.calls.map((call) => String(call[0] ?? "")).join(""),
		stderr: () => err.mock.calls.map((call) => String(call[0] ?? "")).join(""),
	};
}

describe("version surface", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it.each(["--version", "-V", "version"])("prints the package version for `%s`", async (flag) => {
		const io = captureStdio();
		const code = await main([flag]);
		expect(code).toBe(0);
		expect(io.stdout().trim()).toBe(packageVersion);
		expect(io.stderr()).toBe("");
	});
});

describe("per-command help", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it.each(COMMANDS_WITH_OWN_HELP)("prints `%s` usage instead of the global list", async (command) => {
		const io = captureStdio();
		const code = await main([command, "--help"]);
		expect(code).toBe(0);
		expect(io.stdout()).toContain(`Usage: autorag ${command}`);
		expect(io.stdout()).not.toContain(GLOBAL_USAGE_HEADER);
		expect(io.stderr()).not.toContain("Unknown");
	});

	it.each(COMMANDS_WITH_OWN_HELP)("names the %s flags it actually accepts", async (command) => {
		const io = captureStdio();
		expect(await main([command, "--help"])).toBe(0);
		for (const token of COMMAND_HELP_TOKENS[command]) {
			expect(io.stdout()).toContain(token);
		}
	});

	it.each(COMMANDS_WITH_OWN_HELP)("only advertises flags the parser accepts for %s", async (command) => {
		const io = captureStdio();
		expect(await main([command, "--help"])).toBe(0);
		const advertised = [...new Set(io.stdout().match(/--[a-z][a-z0-9-]*/g) ?? [])];
		expect(advertised.length).toBeGreaterThan(0);
		for (const flag of advertised) {
			const parsed = parseArgs([command, flag, "value"]);
			const reason = "error" in parsed ? parsed.error : "";
			expect(reason, `${command} --help advertises ${flag}`).not.toContain("Unknown flag");
		}
	});

	it("keeps the global list for `--help` and for no command", async () => {
		const global = captureStdio();
		expect(await main(["--help"])).toBe(0);
		expect(global.stdout()).toContain(GLOBAL_USAGE_HEADER);
		vi.restoreAllMocks();

		const bare = captureStdio();
		expect(await main([])).toBe(0);
		expect(bare.stdout()).toContain(GLOBAL_USAGE_HEADER);
	});

	it("leaves the lite namespace on its own usage renderer", async () => {
		const io = captureStdio();
		expect(await main(["lite", "--help"])).toBe(0);
		expect(io.stdout()).toContain("Usage: autorag lite <subcommand>");
		expect(io.stdout()).not.toContain(GLOBAL_USAGE_HEADER);
	});
});

describe("two-phase thinking flags", () => {
	it("accepts --single-phase as a boolean flag", () => {
		const parsed = parseArgs(["search", "q", "--single-phase"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["single-phase"]).toBe(true);
	});

	it("accepts --fast-thinking and --final-thinking as value flags", () => {
		const parsed = parseArgs(["search", "q", "--fast-thinking", "low", "--final-thinking", "max"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["fast-thinking"]).toBe("low");
		expect(parsed.flags["final-thinking"]).toBe("max");
	});

	it("accepts the two-phase flags for the tui command", () => {
		const parsed = parseArgs(["tui", "--fast-thinking=high"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["fast-thinking"]).toBe("high");
	});

	it("routes the flags past argument parsing in main", async () => {
		const io = captureStdio();
		// No config is present, so the command body fails later; what matters is
		// that the flag itself is no longer rejected as unknown.
		await main(["tui", "--single-phase", "--help"]);
		expect(io.stderr()).not.toContain("Unknown flag");
	});
});
