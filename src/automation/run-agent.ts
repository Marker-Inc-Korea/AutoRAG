import { existsSync } from "node:fs";
import { type AgentSession, getAgentDir } from "@earendil-works/pi-coding-agent";
import { resolveConfigReadOnly } from "../cli/config.ts";
import { resolveAutoRAGHome, resolveUserHome } from "../config/home.ts";
import { type AgentProfile, createProfileSession, type ProfileModel } from "../profiles/agent-factory.ts";
import { getAutomationProfile } from "../profiles/automation-profiles.ts";
import { loopGuardPrompt } from "../profiles/loop-guard.ts";
import { createRetrievalToolkit, hasRetrievalTools } from "../profiles/retrieval-tools.ts";
import { resolveSessionModel } from "../profiles/session-model.ts";
import type { Automation, AutomationEvent } from "./config.ts";
import { AUTOMATION_MAX_APPEND_FILES, type AutomationRunRecord, appendAutomationRunLog } from "./run.ts";

const MAX_CAPTURE_CHARS = 16 * 1024;

export interface RunAgentAutomationOptions {
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	/** pi's agent dir (the shared credential store); defaults to pi's own. */
	readonly agentDir?: string;
	/** Resolves the model the session runs on; defaults to the AutoRAG agent's configured model. */
	readonly resolveModel?: (cwd: string) => Promise<ProfileModel>;
}

/** The profile an agent rule runs, with the loop guard appended. A profile cannot leave the guard out. */
export function buildAutomationProfile(automation: Automation): AgentProfile {
	const agent = automation.agent;
	if (agent === undefined) throw new Error(`Automation "${automation.id}" has no agent`);
	const definition = getAutomationProfile(agent.profile);
	if (definition === undefined) throw new Error(`Unknown automation profile "${agent.profile}"`);
	const profile = definition.build({ task: agent.task });
	return { ...profile, systemPrompt: `${profile.systemPrompt}\n\n${loopGuardPrompt(automation)}` };
}

/** The user message of a run: one `type: path` line per changed file, capped like `appendFiles`. */
export function eventsPrompt(events: readonly AutomationEvent[]): string {
	const shown = events.slice(0, AUTOMATION_MAX_APPEND_FILES);
	const lines = shown.map((event) => `${event.type}: ${event.path}`);
	const hidden = events.length - shown.length;
	return [
		`${events.length} file change(s) triggered this run:`,
		...lines,
		...(hidden > 0 ? [`(${hidden} more change(s) not listed)`] : []),
		"",
		"Do the task for these files.",
	].join("\n");
}

function bounded(text: string): string {
	return text.length <= MAX_CAPTURE_CHARS ? text : text.slice(text.length - MAX_CAPTURE_CHARS);
}

/** The provider's error when the last model request failed; pi reports it on the message instead of throwing. */
function modelFailure(session: AgentSession): string | undefined {
	for (let index = session.messages.length - 1; index >= 0; index -= 1) {
		const message = session.messages[index];
		if (message?.role !== "assistant") continue;
		return message.stopReason === "error" ? (message.errorMessage ?? "The model request failed.") : undefined;
	}
	return undefined;
}

/**
 * Run an agent rule: spawn its profile as an in-memory pi session, hand it the
 * changed files, and wait for its final reply. The session never touches
 * AutoRAG's retrieval memory; retrieval tools, when the rule asks for them, run
 * against a throwaway memory file.
 */
export async function runAgentAutomation(
	automation: Automation,
	events: readonly AutomationEvent[],
	options: RunAgentAutomationOptions = {},
): Promise<AutomationRunRecord> {
	const env = options.env ?? process.env;
	const home = options.home ?? resolveAutoRAGHome(env);
	const agentDir = options.agentDir ?? getAgentDir();
	const cwd = existsSync(automation.paths[0] ?? "") ? (automation.paths[0] as string) : resolveUserHome(env);
	const startedAtMs = Date.now();
	const startedAt = new Date(startedAtMs).toISOString();
	const filesOverflow = events.length > AUTOMATION_MAX_APPEND_FILES;

	let stdout = "";
	let error: string | undefined;
	let timedOut = false;
	let session: AgentSession | undefined;
	let timer: NodeJS.Timeout | undefined;
	let dispose: () => Promise<void> = async () => {};
	try {
		const profile = buildAutomationProfile(automation);
		const retrieval = automation.agent?.retrieval;
		const toolkit =
			retrieval !== undefined && hasRetrievalTools(retrieval)
				? await createRetrievalToolkit(resolveConfigReadOnly({ flags: {}, env, cwd }), retrieval)
				: undefined;
		if (toolkit !== undefined) dispose = toolkit.dispose;
		const model = await (
			options.resolveModel ??
			((dir) => resolveSessionModel(resolveConfigReadOnly({ flags: {}, env, cwd: dir }), env, dir, agentDir))
		)(cwd);
		session = await createProfileSession({
			profile: { ...profile, customTools: [...(profile.customTools ?? []), ...(toolkit?.tools ?? [])] },
			cwd,
			agentDir,
			env,
			model,
		});
		const running = session;
		timer = setTimeout(() => {
			timedOut = true;
			void running.abort();
		}, automation.timeoutMs);
		timer.unref?.();
		await session.prompt(eventsPrompt(events));
		stdout = bounded(session.getLastAssistantText() ?? "");
		if (!timedOut) error = modelFailure(session);
	} catch (cause) {
		error = cause instanceof Error ? cause.message : String(cause);
	} finally {
		clearTimeout(timer);
		session?.dispose();
		await dispose().catch(() => undefined);
	}

	const record: AutomationRunRecord = {
		id: automation.id,
		startedAt,
		durationMs: Math.max(0, Date.now() - startedAtMs),
		eventCount: events.length,
		filesOverflow,
		exitCode: timedOut ? null : error === undefined ? 0 : 1,
		signal: null,
		timedOut,
		stdout,
		stderr: error === undefined ? "" : bounded(error),
		...(error !== undefined ? { error } : {}),
	};
	appendAutomationRunLog(home, record);
	return record;
}
