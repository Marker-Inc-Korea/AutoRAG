import type { Automation, AutomationEvent } from "./config.ts";
import { type AutomationRunRecord, type RunAutomationOptions, runAutomation } from "./run.ts";
import { type RunAgentAutomationOptions, runAgentAutomation } from "./run-agent.ts";

/**
 * Agent runs the process may have in flight at once, across every rule. A run
 * holds a model session and, with retrieval tools, the configured indexes, so
 * rules wait their turn instead of running in parallel. Events that arrive
 * while a rule waits stay in its pending batch and are handled together.
 */
export const AGENT_RUN_CONCURRENCY = 1;

let activeAgentRuns = 0;
const waiting: Array<() => void> = [];

async function acquireAgentSlot(): Promise<() => void> {
	if (activeAgentRuns >= AGENT_RUN_CONCURRENCY) {
		// tsconfig lib is ES2022, which predates Promise.withResolvers.
		await new Promise<void>((resolve) => waiting.push(resolve));
	} else {
		activeAgentRuns += 1;
	}
	return () => {
		const next = waiting.shift();
		if (next === undefined) activeAgentRuns -= 1;
		else next();
	};
}

export type RunRuleOptions = RunAutomationOptions & RunAgentAutomationOptions;

/** Run one rule: its command, or its agent profile. The single entry for the daemon, the CLI and the tool. */
export async function runRule(
	automation: Automation,
	events: readonly AutomationEvent[],
	options: RunRuleOptions = {},
): Promise<AutomationRunRecord> {
	if (automation.agent === undefined) return runAutomation(automation, events, options);
	const release = await acquireAgentSlot();
	try {
		return await runAgentAutomation(automation, events, options);
	} finally {
		release();
	}
}
