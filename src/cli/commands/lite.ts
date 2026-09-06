import { runDuplicates } from "./duplicates.ts";
import { runIndex } from "./index.ts";
import { runInit } from "./init.ts";
import { runLiteRefresh } from "./lite-refresh.ts";
import { runStatus } from "./status.ts";
import type { CommandContext } from "./types.ts";
import { runUi } from "./ui.ts";
import { runWatch } from "./watch.ts";

const LITE_USAGE = `autorag lite - model-free AutoRAG lifecycle CLI

Usage: autorag lite <subcommand> [args] [flags]

Subcommands:
  autorag lite init
                       Write the configured model-free lifecycle config
                       (--search-paths PATHS  --workspace DIR  --memory-path FILE  --force)
  autorag lite ui      Open the local datasource setup UI
                       (--port N  --host 127.0.0.1  --no-open  --allow-remote)
  autorag lite refresh Run an incremental index refresh (no model required)
                       (--full  --force  --method minsync,parsed,datasources,jikji,all)
  autorag lite watch  Watch configured roots (or --once for one refresh tick)
                       (--once  --immediate  --debounce-ms N  --force)
  autorag lite status  Show path-opaque corpus freshness and index health
  autorag lite index   Reset or rebuild parsed/MinSync indexes
                       (reset|rebuild  --yes  --method)
  autorag lite duplicates
                       Scan duplicate document families without deleting files
  autorag lite health  Show model-free index health (alias of lite status)
  autorag lite retrieve <query>
                       Retrieve documents without model curation
                       (--top-k N  --scope SCOPE  --tags A,B  --json  --debug)
  autorag lite report <query>
                       Persist a structured report from external curation
                       (--input FILE  --json  --debug)

Global flags:
  --json               Emit machine-readable JSON
  --debug              Reveal opaque internal diagnostics
  --config <path>      Use a specific config file
  --help, -h           Show this help
`;

export function liteUsage(): string {
	return LITE_USAGE;
}

export async function runLite(ctx: CommandContext): Promise<number> {
	const sub = ctx.positionals[0];
	if (ctx.flags.help === true || sub === undefined || sub === "help") {
		ctx.stdout(LITE_USAGE.trimEnd());
		return 0;
	}

	const subCtx: CommandContext = {
		...ctx,
		positionals: ctx.positionals.slice(1),
	};

	switch (sub) {
		case "init":
			return runInit(subCtx);
		case "ui":
			return runUi(subCtx);
		case "refresh":
			return runLiteRefresh(subCtx);
		case "watch":
			return runWatch(subCtx);
		case "status":
			return runStatus(subCtx);
		case "index":
			return runIndex(subCtx);
		case "duplicates":
			return runDuplicates(subCtx);
		case "health":
			return runStatus(subCtx);
		case "retrieve": {
			const { runLiteRetrieve } = await import("./lite-retrieve.ts");
			return runLiteRetrieve(subCtx);
		}
		case "report": {
			const { runReport } = await import("./report.ts");
			return runReport(subCtx);
		}
		default:
			ctx.stderr(`Unknown lite subcommand: ${sub}\n\n${LITE_USAGE}`);
			return 2;
	}
}
