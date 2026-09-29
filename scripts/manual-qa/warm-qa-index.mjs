import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildAgentOptions, resolveConfigReadOnly } from "../../src/cli/config.ts";

/**
 * Pre-warm the QA workspace's parsed mirror and MinSync index so the app's
 * baseline retrieval has evidence on the very first search. Runs the same
 * refresh the CLI/TUI would, against the isolated AUTORAG_CONFIG workspace.
 */
export async function refreshQaIndex(configPath) {
	const config = resolveConfigReadOnly({
		flags: { config: configPath },
		env: process.env,
		cwd: process.cwd(),
	});
	const agent = new AutoRAGAgent(buildAgentOptions(config));
	await agent.refresh(false);
}
