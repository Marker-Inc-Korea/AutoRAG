import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import type { PackagedAgentSkill } from "./packaged-skill.ts";
import { formatSkillInvocation } from "./skill-invocation.ts";

/**
 * Host opt-in for agent self-configuration: the Jev `config` branch, where the
 * agent edits its own settings with the full `autorag-setup` skill in hand.
 */
export interface SelfConfigOptions {
	/** The active AutoRAG config file the agent may read and edit. */
	readonly configPath: string;
	/** Override for the packaged `autorag-setup` skill file; defaults to the packaged skill. */
	readonly skillPath?: string;
	/**
	 * Check the config file after a config turn changed it. Resolves to a
	 * problem description when the file no longer works, or `undefined` when it
	 * is fine. A reported problem restores the pre-turn file.
	 */
	readonly validate?: (configPath: string) => Promise<string | undefined>;
}

export interface SelfConfigPromptInput {
	readonly query: string;
	readonly configPath: string;
	readonly agentDir: string;
	/** The loaded `autorag-setup` skill, embedded as a `<skill …>` invocation. */
	readonly skill: PackagedAgentSkill;
	/** True when `manage_automations` is available: file-triggered automations are then in scope. */
	readonly automations?: boolean;
}

/** The config file's current text, or `undefined` when it does not exist. */
export function snapshotConfigFile(configPath: string): string | undefined {
	return existsSync(configPath) ? readFileSync(configPath, "utf8") : undefined;
}

/**
 * After a config turn: when the model changed the file and the host's
 * `validate` says the result no longer works, put the pre-turn text back.
 * Returns the problem that triggered the rollback, or `undefined` when the
 * file is unchanged, valid, or the host supplied no validator. An unchanged
 * file is never touched, even if it was already invalid before the turn.
 */
export async function rollbackIfBroken(
	options: SelfConfigOptions,
	before: string | undefined,
): Promise<string | undefined> {
	if (options.validate === undefined) return undefined;
	if (snapshotConfigFile(options.configPath) === before) return undefined;
	let problem: string | undefined;
	try {
		problem = await options.validate(options.configPath);
	} catch (error) {
		problem = error instanceof Error ? error.message : String(error);
	}
	if (problem === undefined) return undefined;
	if (before === undefined) rmSync(options.configPath, { force: true });
	else writeFileSync(options.configPath, before, "utf8");
	return problem;
}

/**
 * The turn prompt for the `config` branch: the whole setup skill, where this
 * run's config lives, the safety rules for editing it, and the user's request.
 * The turn ends with the model's plain final message; there is no fast answer.
 */
export function buildSelfConfigPrompt(input: SelfConfigPromptInput): string {
	const automationRule =
		input.automations === true
			? `- Scope: edit only \`${input.configPath}\`, and handle file-triggered automations/daemon requests exclusively through \`manage_automations\` (never hand-edit rules files or OS service entries).\n`
			: `- Scope: edit only \`${input.configPath}\` and touch nothing else.\n`;
	return (
		`Goal: inspect or update AutoRAG's own configuration to fulfill the user's request. This is a configuration task, not a document search — use file/shell tools (\`bash\`, \`read\`, \`edit\`, \`write\`) and do not call retrieval, datasource, or web tools.\n\n` +
		`Active config: \`${input.configPath}\`\n` +
		`pi agent dir: \`${input.agentDir}\`\n\n` +
		`Invariants:\n` +
		automationRule +
		`- Preserve valid JSON and every unrelated field in the existing config. Never run \`autorag init --force\` or delete the config file.\n` +
		`- For custom/OpenAI-compatible providers, configure \`model\` (\`provider\`, \`id\`, \`api\`, \`baseUrl\`, \`apiKeyEnv\`) in the AutoRAG config; never create or edit \`${input.agentDir}/models.json\`.\n` +
		`- Never print, copy, or store secret values — store only env-var names (\`apiKeyEnv\`) and check availability with \`test -n "$VAR"\`.\n` +
		`- Always pass \`--config ${input.configPath}\` to \`autorag\` commands.\n` +
		`- Never guess a model id: resolve the exact \`provider\` (before the first \`/\`) and \`id\` (everything after) from \`autorag models list --available --config ${input.configPath}\`. A gateway listing a vendor's model does not mean that vendor's native provider is usable; verify each provider's own credential and health separately.\n` +
		`- Verify changes with \`autorag health --json --config ${input.configPath}\`. If verification fails, restore the previous config and report both the failure and the rollback (a missing workspace directory can be created with \`mkdir -p\` and retried).\n` +
		`- Note in your report that changes take effect on the next \`autorag\` launch.\n\n` +
		`## autorag-setup skill\n\n${formatSkillInvocation(input.skill)}\n\n` +
		`## User request\n\n${input.query}\n\n` +
		`End with a concise report (no tool call): what changed (\`field: old → new\`), how you verified it, and anything still needing the user (env-var names only).`
	);
}
