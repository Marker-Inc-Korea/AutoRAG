import { InteractiveMode } from "@earendil-works/pi-coding-agent";
import { AutoRAGAgent, type AutoRAGAgentOptions, type AutoRAGThinkingLevel } from "../../agent/agent.ts";
import { buildAgentOptions, resolveAgentModel, resolveConfig } from "../config.ts";
import { renderError } from "../output.ts";
import { checkAutoRAGUpdate, renderAutoRAGUpdateNotice } from "../update-check.ts";
import { readPackageVersion } from "../version.ts";
import type { CommandContext } from "./types.ts";

const THINKING_LEVELS: readonly AutoRAGThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/**
 * Map the two-phase thinking flags (`--fast-thinking`, `--final-thinking`) and
 * the legacy `--single-phase` switch to the agent's `thinking` option. Returns
 * `undefined` when no flag was given so the agent keeps its default two-phase
 * flow.
 */
function parseThinkingFlags(flags: CommandContext["flags"]): AutoRAGAgentOptions["thinking"] | undefined {
	if (flags["single-phase"] === true) return false;
	const parse = (value: string | boolean | undefined): AutoRAGThinkingLevel | undefined =>
		typeof value === "string" && THINKING_LEVELS.includes(value as AutoRAGThinkingLevel)
			? (value as AutoRAGThinkingLevel)
			: undefined;
	const fast = parse(flags["fast-thinking"]);
	const final = parse(flags["final-thinking"]);
	if (flags["fast-thinking"] !== undefined && fast === undefined) {
		throw new Error(`Invalid fast thinking level. Use one of: ${THINKING_LEVELS.join(", ")}.`);
	}
	if (flags["final-thinking"] !== undefined && final === undefined) {
		throw new Error(`Invalid final thinking level. Use one of: ${THINKING_LEVELS.join(", ")}.`);
	}
	if (fast === undefined && final === undefined) return undefined;
	return { ...(fast !== undefined ? { fast } : {}), ...(final !== undefined ? { final } : {}) };
}

/**
 * Build the librarian agent that hosts the interactive session.
 *
 * The model is resolved only when the CLI config explicitly declares one. When
 * it does not, the agent is constructed without a model so Pi's native
 * `/login` and `/model` flows own first-launch credential and model selection —
 * resolving here would force the local model runtime before the user can log
 * in.
 */
async function createTuiAgent(ctx: CommandContext): Promise<AutoRAGAgent> {
	const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	const options: AutoRAGAgentOptions = { ...buildAgentOptions(config) };
	// Pi reports its own version/changelog; this notice is AutoRAG's own npm
	// release check, injected into the same interactive session.
	options.updateNotice = async () =>
		renderAutoRAGUpdateNotice(await checkAutoRAGUpdate({ currentVersion: readPackageVersion() }));
	const thinking = parseThinkingFlags(ctx.flags);
	if (thinking !== undefined) options.thinking = thinking;
	if (config.model !== undefined) {
		const resolved = await resolveAgentModel(config);
		options.model = resolved.model;
		if (resolved.apiKey !== undefined) options.apiKey = resolved.apiKey;
		if (resolved.providerApiKeys !== undefined) options.providerApiKeys = resolved.providerApiKeys;
	}
	return new AutoRAGAgent(options);
}

/**
 * Run `autorag tui`: Pi's interactive mode over the AutoRAG librarian.
 *
 * Pi owns the terminal UI, `read`/`bash`/`edit`/`write`/`grep`/`find`/`ls`
 * tools, provider credentials/OAuth, model selection, session persistence and
 * resume/fork/compaction, settings, and extension loading. AutoRAG registers
 * its domain tools and streams progress, preliminary answers, and final results
 * into the same session through the agent's interactive runtime. The native
 * `/login`, `/model`, `/resume`, `/new`, `/tree`, `/compact`, and `/settings`
 * commands are therefore always available.
 */
export async function runTui(ctx: CommandContext): Promise<number> {
	try {
		const agent = await createTuiAgent(ctx);
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			const interactive = new InteractiveMode(hosted.runtime, { verbose: ctx.debug });
			// Pi is a bundled host for `autorag`, not a separate install users manage:
			// its startup checks tell them to run `pi update`, which does not apply here.
			// AutoRAG surfaces its own npm release notice instead (options.updateNotice).
			interactive.showNewVersionNotification = () => undefined;
			interactive.showPackageUpdateNotification = () => undefined;
			await interactive.init();
			await interactive.run();
			return 0;
		} finally {
			await hosted.dispose().catch(() => undefined);
		}
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
