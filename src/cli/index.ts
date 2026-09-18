#!/usr/bin/env node
import { readFileSync, realpathSync } from "node:fs";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { commandUsage } from "./command-help.ts";
import type { CommandContext } from "./commands/types.ts";
import { renderError } from "./output.ts";

const BOOLEAN_FLAGS = new Set([
	"json",
	"debug",
	"help",
	"version",
	"force",
	"yes",
	"once",
	"immediate",
	"skip-probes",
	"no-open",
	"allow-remote",
	"full",
	"single-phase",
]);
const VALUE_FLAGS = new Set([
	"config",
	"search-paths",
	"workspace",
	"memory-path",
	"model-provider",
	"model-id",
	"top-k",
	"scope",
	"tags",
	"result",
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
	"minsync-max-chunk-size",
	"timeout-ms",
	"port",
	"host",
	"peer",
	"accept",
	"alias",
	"endpoint",
	"remove",
	"add",
	"contact-id",
	"edit",
	"show",
	"rank",
	"display-name",
	"description",
	"role",
	"org",
	"access-hint",
	"input",
	"profile",
	"format",
	"fast-thinking",
	"final-thinking",
]);

const COMMANDS = [
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
	"lite",
	"models",
	"gateway",
] as const;
type CommandName = (typeof COMMANDS)[number];

export type { CommandName };

interface ParsedArgs {
	positionals: string[];
	flags: Record<string, string | boolean>;
}

const USAGE = `autorag - self-evolving librarian CLI

Usage: autorag <command> [args] [flags]

Commands:
  init                 Write ~/.autorag/config.json for a local collection
  setup                Probe and configure local runtime and datasources
                       (--search-paths a,b  --workspace DIR  --profile ID  --format json)
  refresh              Refresh every configured index; --method narrows the run
  watch                Watch configured roots (or --once for cron/poll tick)
  status               Show corpus freshness and index health
  search <query>       Search and curate documents (requires a configured model)
  feedback <session>   Record numbered feedback (--useful 1,3 --not-useful 2)
  evidence <session>   Show persisted source/chunk evidence (--result N)
  memory inspect       Inspect the retrieval memory snapshot
  index reset          Remove parsed/minsync indexes (--method)
  index rebuild        Reset then re-run a refresh (--method minsync|all)
  health               Check model/provider auth and completion access (no index check)
  lite init|ui|refresh|watch|status|index|duplicates|health
                       Model-free setup, indexing, status, and datasource lifecycle
                       (lite refresh: --full --force --method; watch: --once)
  lite retrieve <query>
                       Retrieve documents without model curation
                       (--top-k N  --scope SCOPE  --tags A,B  --json  --debug)
  lite report <query>   Persist a structured report (--input FILE)
  duplicates [DIR]     Scan exact/near duplicate document families; never deletes files
  tui                  Open an interactive Pi-powered librarian terminal UI
  ui                   Open a local loopback page to connect and manage data sources
                       (--port N  --host 127.0.0.1  --no-open  --allow-remote)
  serve                Start the P2P peer query server over SimpleX
                       (--port N  --force)
  p2p                  SimpleX peer trust management
                       (peers [--add <alias> --contact-id <n>] [--edit <alias>]
                        [--display-name <name>] [--description <text>]
                        [--role <role>] [--org <org>] [--access-hint <csv>]
                        [--show <alias>] [--rank <query>] [--remove <alias>])
                       (requests [approve|deny <id>])
  p2p policy list      Show effective merged sharing policy (virtual-path keys)
  p2p policy set       Set a sharing rule: <source-glob> <private|never|always|peers> [--peer fp...]
  p2p policy unset     Remove a sharing rule by key
  models prefetch|import|verify
                       Manage verified embedding model cache (--profile ID)
  gateway status|stop  Inspect or stop the on-demand embedding gateway (--format json)

Setup:
  autorag init --search-paths /path/to/docs,/path/to/notes   # choose folders
  autorag refresh                                            # parsed + MinSync + datasources + Jikji
  autorag watch --once                                       # single index refresh tick (cron)
  autorag search "your question"                             # curated answer

Global flags:
  --json               Emit machine-readable JSON
  --debug              Reveal opaque internal diagnostics (never filesystem paths)
  --config <path>      Use a specific config file
  --search-paths <csv> Folders to index/search (also AUTORAG_SEARCH_PATHS)
  --model-provider <name>  Override the model provider
  --model-id <id>          Override the model
  --once               For watch: run one refresh tick and exit (for cron)
  --immediate          For watch: refresh once before reading fs events (default true)
  --debounce-ms <n>    For watch: debounce milliseconds for fs events (default 1500)
  --method <csv>       For refresh/index: minsync,parsed,datasources,jikji,all
  --skip-probes        For health: skip the network completion probe (auth checks still run)
  --timeout-ms <n>     For health: per-probe timeout in ms (default 10000)
  --port <n>           For ui: loopback port (default 8787, 0 for ephemeral)
  --host <addr>        For ui/serve: bind address (127.0.0.1, ::1, or 0.0.0.0)
  --no-open            For ui: print the URL and do not launch a browser
  --version, -V        Print the package version
  --help, -h           Show this help

Run: autorag <command> --help   for command-specific usage.
`;

/**
 * The installed package version. Read from the manifest next to the entry
 * point: `src/cli/index.ts` and the published `dist/cli/index.js` both resolve
 * `../../package.json` to the package root.
 */
function readPackageVersion(): string {
	try {
		const manifest = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")) as {
			version?: unknown;
		};
		return typeof manifest.version === "string" && manifest.version.length > 0 ? manifest.version : "0.0.0";
	} catch {
		return "0.0.0";
	}
}

export function parseArgs(argv: readonly string[]): ParsedArgs | { error: string } {
	const positionals: string[] = [];
	const flags: Record<string, string | boolean> = {};
	for (let i = 0; i < argv.length; i++) {
		const token = argv[i];
		if (token === "-h") {
			flags.help = true;
			continue;
		}
		if (token === "-V") {
			flags.version = true;
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
		case "setup": {
			const { runSetupCommand } = await import("./commands/setup.ts");
			return runSetupCommand(ctx);
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
		case "evidence": {
			const { runEvidence } = await import("./commands/evidence.ts");
			return runEvidence(ctx);
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
		case "duplicates": {
			const { runDuplicates } = await import("./commands/duplicates.ts");
			return runDuplicates(ctx);
		}
		case "tui": {
			const { runTui } = await import("./commands/tui.ts");
			return runTui(ctx);
		}
		case "ui": {
			const { runUi } = await import("./commands/ui.ts");
			return runUi(ctx);
		}
		case "serve": {
			const { runServe } = await import("./commands/serve.ts");
			return runServe(ctx);
		}
		case "p2p": {
			if (ctx.positionals[0] === "policy") {
				const { runP2pPolicy } = await import("./commands/p2p-policy.ts");
				ctx.positionals = ctx.positionals.slice(1);
				return runP2pPolicy(ctx);
			}
			const { runP2p } = await import("./commands/p2p.ts");
			return runP2p(ctx);
		}
		case "lite": {
			return dispatchLite(ctx);
		}
		case "models": {
			const { runModels } = await import("./commands/models.ts");
			return runModels(ctx);
		}
		case "gateway": {
			const { runGateway } = await import("./commands/gateway.ts");
			return runGateway(ctx);
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

	if (flags.version === true || command === "version") {
		process.stdout.write(`${readPackageVersion()}\n`);
		return 0;
	}
	if (flags.help === true || command === undefined || command === "help") {
		if (command === "lite") {
			// Defer to lite dispatch for subcommand-specific help.
		} else {
			process.stdout.write(commandUsage(command) ?? USAGE);
			return 0;
		}
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

async function dispatchLite(ctx: CommandContext): Promise<number> {
	const { runLite } = await import("./commands/lite.ts");
	return runLite(ctx);
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
