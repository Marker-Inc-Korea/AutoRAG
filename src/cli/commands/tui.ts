import { InteractiveMode } from "@earendil-works/pi-coding-agent";
import { TuiAltScreen } from "@earendil-works/pi-tui";
import { AutoRAGAgent, type AutoRAGAgentOptions, type AutoRAGThinkingLevel } from "../../agent/agent.ts";
import { resolveInteractiveTuiMode } from "../../agent/pi-session.ts";
import { warmFileeBinary } from "../../convert/index.ts";
import { buildAgentOptions, resolveConfig, resolveQueryDecompositionModel, resolveTuiAgentModel } from "../config.ts";
import { openCitationTarget } from "../open-target.ts";
import { renderError } from "../output.ts";
import { checkAutoRAGUpdate, renderAutoRAGUpdateNotice } from "../update-check.ts";
import { readPackageVersion } from "../version.ts";
import { resolveWhatsAppExtension } from "../whatsapp-extension.ts";
import type { CommandContext } from "./types.ts";

const THINKING_LEVELS: readonly AutoRAGThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/**
 * Map the two-phase thinking flags (`--fast-thinking`, `--final-thinking`) to
 * the agent's `thinking` option. Returns `undefined` when no flag was given so
 * the agent keeps its default levels.
 */
function parseThinkingFlags(flags: CommandContext["flags"]): AutoRAGAgentOptions["thinking"] | undefined {
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
 * The model follows the CLI chain (config, Pi default, signed-in Dazzi default).
 * When none applies the agent is constructed without a model so Pi's native
 * `/login` and `/model` flows own first-launch credential and model selection —
 * the local model runtime is never forced before the user can log in.
 */
async function createTuiAgent(ctx: CommandContext): Promise<AutoRAGAgent> {
	// A session with a remote bridge is driven from a chat as well, where no terminal dialog can be answered:
	// it defaults to no gate, and an explicit --permission-mode, environment variable, or config value still wins.
	const bridged = ctx.flags.telegram === true || ctx.flags.discord === true || ctx.flags.whatsapp === true;
	const config = resolveConfig({
		flags: ctx.flags,
		cwd: ctx.cwd,
		...(bridged ? { defaultPermissionMode: "full" as const } : {}),
	});
	const options: AutoRAGAgentOptions = { ...buildAgentOptions(config) };
	// Pi reports its own version/changelog; this notice is AutoRAG's own npm
	// release check, injected into the same interactive session.
	options.updateNotice = async () =>
		renderAutoRAGUpdateNotice(await checkAutoRAGUpdate({ currentVersion: readPackageVersion() }));
	// The first screen (and `/new`) is AutoRAG's banner, not pi's logo and key-hint listing.
	options.welcomeVersion = readPackageVersion();
	// Opt-in: the bridge opens an outbound Telegram poller once `/telegram-connect` runs.
	if (ctx.flags.telegram === true) options.telegram = true;
	if (ctx.flags.discord === true) options.discordRemote = true;
	const thinking = parseThinkingFlags(ctx.flags);
	if (thinking !== undefined) options.thinking = thinking;
	const resolved = await resolveTuiAgentModel(config);
	if (resolved !== undefined) {
		options.model = resolved.model;
		if (resolved.apiKey !== undefined) options.apiKey = resolved.apiKey;
		if (resolved.providerApiKeys !== undefined) options.providerApiKeys = resolved.providerApiKeys;
	}
	const whatsapp = await resolveWhatsAppExtension({ flags: ctx.flags, cwd: ctx.cwd });
	if (whatsapp !== undefined) {
		options.piExtensionPaths = whatsapp.paths;
		options.piExtensionFlags = whatsapp.flags;
	}
	const decompositionModel = await resolveQueryDecompositionModel(config);
	if (decompositionModel !== undefined) options.queryDecomposition = decompositionModel;
	// Installed together with the agent: fetch filee-cli in the background so the first conversion is instant.
	if (options.filee !== undefined) void warmFileeBinary(options.filee);
	return new AutoRAGAgent(options);
}

/**
 * Run `autorag tui`: Pi's interactive mode over the AutoRAG librarian.
 *
 * Pi owns the terminal UI, `read`/`bash`/`edit`/`write`/`grep`/`find`/`ls`
 * tools, provider credentials/OAuth, model selection, session persistence and
 * resume/fork/compaction, settings, and extension loading. AutoRAG registers
 * its domain tools and runs the two-phase search in the same session through
 * the agent's interactive runtime; the model's progress notes, first answer, and
 * final answer are ordinary assistant messages there. The native
 * `/login`, `/model`, `/resume`, `/new`, `/tree`, `/compact`, and `/settings`
 * commands are therefore always available.
 */
export async function runTui(ctx: CommandContext): Promise<number> {
	try {
		const agent = await createTuiAgent(ctx);
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			const interactive = new InteractiveMode(hosted.runtime, {
				verbose: ctx.debug,
				tuiMode: resolveInteractiveTuiMode(hosted.runtime.session.settingsManager),
			});
			// Pi is a bundled host for `autorag`, not a separate install users manage:
			// its startup checks tell them to run `pi update`, which does not apply here.
			// AutoRAG surfaces its own npm release notice instead (options.updateNotice).
			interactive.showNewVersionNotification = () => undefined;
			interactive.showPackageUpdateNotification = () => undefined;
			// pi hands a clicked hyperlink to `open`; fullscreen mode reads `openUrl` per click, so
			// setting it there sends cited files to Dazzi (falling back to `open`) for every
			// renderer pi creates, including after a mode switch.
			const altScreen = TuiAltScreen.prototype as unknown as {
				handleSelectionMouseEvent(this: { openUrl?: (url: string) => void }, event: unknown): void;
			};
			const handleMouse = altScreen.handleSelectionMouseEvent;
			altScreen.handleSelectionMouseEvent = function (event) {
				this.openUrl = openCitationTarget;
				handleMouse.call(this, event);
			};
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
