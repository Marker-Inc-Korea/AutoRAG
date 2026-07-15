#!/usr/bin/env node
import { realpathSync } from "node:fs";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import type { CommandContext } from "./commands/types.ts";
import { renderError } from "./output.ts";

const BOOLEAN_FLAGS = new Set(["json", "debug", "help", "force", "yes", "once", "immediate", "skip-probes"]);
const VALUE_FLAGS = new Set([
	"config",
	"search-paths",
	"workspace",
	"memory-path",
	"model-provider",
	"model-id",
	"orchestrator-model-provider",
	"orchestrator-model-id",
	"explorer-model-provider",
	"explorer-model-id",
	"top-k",
	"scope",
	"tags",
	"useful",
	"not-useful",
	"debounce-ms",
	"method",
	"embedder-id",
	"embedder-base-url",
	"embedder-api-key-env",
	"embedder-dimension",
	"embedder-query-prefix",
	"embedder-passage-prefix",
	"embedder-timeout-ms",
	"embedder-batch-size",
	"timeout-ms",
]);

const COMMANDS = ["init", "refresh", "status", "search", "feedback", "memory", "index", "watch", "health"] as const;
type CommandName = (typeof COMMANDS)[number];

interface ParsedArgs {
	positionals: string[];
	flags: Record<string, string | boolean>;
}

const USAGE = `autorag - self-evolving librarian CLI

Usage: autorag <command> [args] [flags]

Commands:
  init                 Write ~/.autorag/config.json for a local collection
	                       (--search-paths a,b  --workspace DIR  --memory-path FILE
	                        --orchestrator-model-provider P  --orchestrator-model-id ID
	                        --explorer-model-provider P  --explorer-model-id ID
	                        --embedder-id ID --embedder-base-url URL --embedder-api-key-env VAR
	                        --embedder-dimension N --embedder-batch-size N  --force)
  refresh              Parse sources and refresh indexes (--method bm25,minsync,parsed)
  watch                Watch configured roots (or --once for cron/poll tick)
  status               Show corpus freshness and index health
  search <query>       Search and curate documents (requires a configured model)
  feedback <session>   Record numbered feedback (--useful 1,3 --not-useful 2)
  memory inspect       Inspect the retrieval memory snapshot
  index reset          Remove parsed/bm25/minsync indexes under .autorag (--method)
  index rebuild        Reset then re-run a refresh (--method bm25|minsync|all)
  health               Check model/provider auth and subagent preflight (no index check)

Setup:
  autorag init --search-paths /path/to/docs,/path/to/notes   # choose folders
  autorag refresh                                            # parse + index (+ jikji prepare)
  autorag watch --once                                       # single index refresh tick (cron)
  autorag search "your question"                             # curated answer

Global flags:
  --json               Emit machine-readable JSON
  --debug              Reveal opaque internal diagnostics (never filesystem paths)
  --config <path>      Use a specific config file
  --search-paths <csv> Folders to index/search (also AUTORAG_SEARCH_PATHS)
  --orchestrator-model-provider <name>  Override the orchestrator provider
  --orchestrator-model-id <id>          Override the orchestrator model
  --explorer-model-provider <name>      Override the explorer provider
  --explorer-model-id <id>              Override the explorer model
  --once               For watch: run one refresh tick and exit (for cron)
  --immediate          For watch: refresh once before reading fs events (default true)
  --debounce-ms <n>    For watch: debounce milliseconds for fs events (default 1500)
  --method <csv>       For refresh/index: bm25,minsync,parsed,datasources,jikji,all
  --skip-probes        For health: skip network/subagent probes (auth checks still run)
  --timeout-ms <n>     For health: per-probe timeout in ms (default 10000)
  --help, -h           Show this help
`;

export function parseArgs(argv: readonly string[]): ParsedArgs | { error: string } {
	const positionals: string[] = [];
	const flags: Record<string, string | boolean> = {};
	for (let i = 0; i < argv.length; i++) {
		const token = argv[i];
		if (token === "-h") {
			flags.help = true;
			continue;
		}
		if (!token.startsWith("--")) {
			positionals.push(token);
			continue;
		}
		const body = token.slice(2);
		const eq = body.indexOf("=");
		const key = eq === -1 ? body : body.slice(0, eq);
		if (!BOOLEAN_FLAGS.has(key) && !VALUE_FLAGS.has(key)) {
			return { error: `Unknown flag: --${key}` };
		}
		if (eq !== -1) {
			flags[key] = body.slice(eq + 1);
			continue;
		}
		if (BOOLEAN_FLAGS.has(key)) {
			flags[key] = true;
			continue;
		}
		const next = argv[i + 1];
		if (next === undefined || next.startsWith("--")) {
			return { error: `Flag --${key} requires a value` };
		}
		flags[key] = next;
		i++;
	}
	return { positionals, flags };
}

function promptYesNo(question: string): Promise<boolean> {
	const rl = createInterface({ input: process.stdin, output: process.stdout });
	return new Promise((resolve) => {
		rl.question(`${question} [y/N] `, (answer) => {
			rl.close();
			resolve(/^y(es)?$/i.test(answer.trim()));
		});
	});
}

async function dispatch(command: CommandName, ctx: CommandContext): Promise<number> {
	switch (command) {
		case "init": {
			const { runInit } = await import("./commands/init.ts");
			return runInit(ctx);
		}
		case "refresh": {
			const { runRefresh } = await import("./commands/refresh.ts");
			return runRefresh(ctx);
		}
		case "status": {
			const { runStatus } = await import("./commands/status.ts");
			return runStatus(ctx);
		}
		case "search": {
			const { runSearch } = await import("./commands/search.ts");
			return runSearch(ctx);
		}
		case "feedback": {
			const { runFeedback } = await import("./commands/feedback.ts");
			return runFeedback(ctx);
		}
		case "memory": {
			const { runMemory } = await import("./commands/memory.ts");
			return runMemory(ctx);
		}
		case "index": {
			const { runIndex } = await import("./commands/index.ts");
			return runIndex(ctx);
		}
		case "watch": {
			const { runWatch } = await import("./commands/watch.ts");
			return runWatch(ctx);
		}
		case "health": {
			const { runHealth } = await import("./commands/health.ts");
			return runHealth(ctx);
		}
	}
}

export async function main(argv: readonly string[]): Promise<number> {
	const parsed = parseArgs(argv);
	if ("error" in parsed) {
		process.stderr.write(`${parsed.error}\n`);
		return 2;
	}
	const { positionals, flags } = parsed;
	const command = positionals[0];
	const json = flags.json === true;
	const debug = flags.debug === true;

	if (flags.help === true || command === undefined || command === "help") {
		process.stdout.write(USAGE);
		return 0;
	}
	if (!(COMMANDS as readonly string[]).includes(command)) {
		process.stderr.write(`Unknown command: ${command}\n\n${USAGE}`);
		return 2;
	}

	const ctx: CommandContext = {
		positionals: positionals.slice(1),
		flags,
		json,
		debug,
		cwd: process.cwd(),
		stdout: (line: string) => process.stdout.write(`${line}\n`),
		stderr: (line: string) => process.stderr.write(`${line}\n`),
		promptYesNo,
	};

	try {
		return await dispatch(command as CommandName, ctx);
	} catch (error) {
		process.stderr.write(`${renderError(error, { json, debug })}\n`);
		return 1;
	}
}

function isInvokedDirectly(): boolean {
	const entry = process.argv[1];
	if (entry === undefined) return false;
	try {
		return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
	} catch {
		return false;
	}
}

if (isInvokedDirectly()) {
	main(process.argv.slice(2)).then(
		(code) => {
			process.exitCode = code;
		},
		(error) => {
			process.stderr.write(`${renderError(error, { json: false, debug: false })}\n`);
			process.exitCode = 1;
		},
	);
}
