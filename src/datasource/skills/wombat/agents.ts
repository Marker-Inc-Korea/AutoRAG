import { isAbsolute, join, resolve } from "node:path";
import { resolveUserHome } from "../../../config/home.ts";
import type { WombatOptions } from "./types.ts";

/** The coding agents wombat parses, by the name `--agent` and `--source <agent>=` take (wombat `AgentKind::as_str`). */
export const WOMBAT_AGENTS: readonly string[] = [
	"codex",
	"claude-code",
	"opencode",
	"pi",
	"omp",
	"hermes",
	"aside",
	"kiro",
	"droid",
	"goose",
	"vibe",
	"aider",
	"codebuddy",
	"iflow",
	"continue",
	"copilot",
	"crush",
	"qwen",
	"kilo",
	"gemini",
];

/**
 * An agent name wombat would not recognize is refused up front: the CLI exits non-zero for it on `sync`, and on
 * search it would filter every hit out, so the datasource would look empty instead of misconfigured.
 */
export function validateWombatAgents(options: Pick<WombatOptions, "agents" | "sources">): void {
	const names = [...(options.agents ?? []), ...Object.keys(options.sources ?? {})];
	for (const name of names) {
		if (!WOMBAT_AGENTS.includes(name)) {
			throw new Error(`unknown wombat agent "${name}"; expected one of: ${WOMBAT_AGENTS.join(", ")}`);
		}
	}
}

/**
 * The directory wombat parses sessions into and MinSync indexes: `<data dir>/mirror`. The data dir follows wombat's
 * own order: `--data-dir`, `WOMBAT_DATA_DIR`, then `~/.wombat`. AutoRAG uses it to keep its local-file index from
 * embedding the same conversations a second time.
 */
export function wombatMirrorDir(
	options: Pick<WombatOptions, "dataDir" | "env">,
	env: NodeJS.ProcessEnv = process.env,
): string {
	const fromEnv = options.env?.WOMBAT_DATA_DIR ?? env.WOMBAT_DATA_DIR;
	const configured = options.dataDir ?? (fromEnv !== undefined && fromEnv.length > 0 ? fromEnv : undefined);
	const dataDir = configured === undefined ? join(resolveUserHome(env), ".wombat") : configured;
	return join(isAbsolute(dataDir) ? dataDir : resolve(dataDir), "mirror");
}
