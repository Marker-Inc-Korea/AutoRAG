import { runDuplicates } from "./duplicates.ts";
import { runIndex } from "./index.ts";
import { runInit } from "./init.ts";
import { runLiteRefresh } from "./lite-refresh.ts";
import { runStatus } from "./status.ts";
import type { CommandContext } from "./types.ts";
import { runWatch } from "./watch.ts";

const LITE_USAGE = `autorag lite - model-free AutoRAG lifecycle CLI

Usage: autorag lite <subcommand> [args] [flags]

Subcommands:
  autorag lite init
                       Write the configured model-free lifecycle config
                       (--search-paths PATHS  --exclude PATH  --workspace DIR  --memory-path FILE  --force)
                       --exclude may be repeated; relative paths use the current directory
  autorag lite refresh Run an incremental index refresh (no model required)
                       (--full  --force  --method minsync,parsed,datasources,jikji,everything,fsearch,all)
  autorag lite watch  Watch configured roots (or --once for one refresh tick)
                       (--once  --immediate  --debounce-ms N  --force)
  autorag lite status  Show path-opaque corpus freshness and index health
  autorag lite index   Reset or rebuild parsed/MinSync indexes
                       (reset|rebuild  --yes  --method)
  autorag lite duplicates
                       Scan duplicate document families without deleting files
  autorag lite health  Show model-free index health (alias of lite status)
  autorag lite retrieve <query>
                       Retrieve documents without model curation. A stale index is
                       reported as "stale": true plus stale-index diagnostics, and
                       the query is still answered; indexes are only rebuilt when
                       asked to.
                       (--top-k N  --scope SCOPE
                        --refresh  --strict  --json  --debug)
                       --refresh rebuilds the indexes incrementally first;
                       --strict fails with exit 2 when the index is stale instead
                       of answering from it.
  autorag lite report <query>
                       Persist a structured report from external curation
                       (--input FILE  --json  --debug)
  autorag lite find <query>
                       Whole search without a model: the platform file-name
                       index, local document content, and every connected
                       datasource. Streams NDJSON (one line per group, in
                       completion order) with --json: a "groups" line first,
                       one "group" line per surface, and a final "done" line.
                       (--top-k N  --groups key,key  --json  --debug)
                       --query TEXT  searches TEXT as the query (wins over the
                       positional), so a query beginning with "-" works.
                       --groups narrows the search to the listed group keys
                       (files, content, source:<datasourceId>).

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
		case "find": {
			const { runLiteFind } = await import("./lite-find.ts");
			return runLiteFind(subCtx);
		}
		default:
			ctx.stderr(`Unknown lite subcommand: ${sub}\n\n${LITE_USAGE}`);
			return 2;
	}
}
