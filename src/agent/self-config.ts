import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Host opt-in for agent self-configuration: the Jev `config` branch, where the
 * agent edits its own settings with the full `autorag-setup` skill in hand.
 */
export interface SelfConfigOptions {
	/** The active AutoRAG config file the agent may read and edit. */
	readonly configPath: string;
	/** Override for the `autorag-setup` SKILL.md; defaults to the packaged skill. */
	readonly skillPath?: string;
}

const SETUP_SKILL_RELATIVE_PATH = join("skills", "autorag-setup", "SKILL.md");

/**
 * Locate the packaged setup skill by walking up from this module. The source
 * tree (`src/agent/`) and the bundled outputs (`dist/index.js`,
 * `dist/cli/index.js`) sit at different depths below the package root.
 */
function findPackagedSetupSkill(): string | undefined {
	let dir = dirname(fileURLToPath(import.meta.url));
	for (let depth = 0; depth < 6; depth += 1) {
		const candidate = join(dir, SETUP_SKILL_RELATIVE_PATH);
		if (existsSync(candidate)) return candidate;
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return undefined;
}

/** Full text of the `autorag-setup` skill with its YAML front matter removed. */
export function loadSetupSkill(skillPath?: string): string {
	const path = skillPath ?? findPackagedSetupSkill();
	if (path === undefined) {
		throw new Error(`The packaged ${SETUP_SKILL_RELATIVE_PATH} was not found next to this AutoRAG install.`);
	}
	let text: string;
	try {
		text = readFileSync(path, "utf8");
	} catch (error) {
		throw new Error(
			`Could not read the setup skill ${path}: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
	return text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n+/u, "");
}

export interface SelfConfigPromptInput {
	readonly query: string;
	readonly configPath: string;
	readonly agentDir: string;
	readonly skill: string;
}

/**
 * The turn prompt for the `config` branch: the whole setup skill, where this
 * run's config lives, the safety rules for editing it, and the user's request.
 * The turn ends with `emit_autorag_results`; there is no fast answer.
 */
export function buildSelfConfigPrompt(input: SelfConfigPromptInput): string {
	return (
		`The user is asking you to inspect or change AutoRAG's own settings. This is a configuration task, not a document search: ` +
		`do NOT call search, retrieval, or web tools. Use bash, read, edit, and write.\n\n` +
		`## Where this agent is configured\n\n` +
		`- Active AutoRAG config file: ${input.configPath}\n` +
		`- pi agent directory (auth.json, models.json, settings.json): ${input.agentDir}\n\n` +
		`## Rules for this run\n\n` +
		`- Edit only the config file above. Touch nothing else.\n` +
		`- To add a custom or OpenAI-compatible provider, set it on the config's \`model\` object (provider, id, api, model.baseUrl, apiKeyEnv) as the skill shows. Do not create or hand-write ${input.agentDir}/models.json: it has its own schema and a wrong file silently hides models.\n` +
		`- Read the current config first. Change only the fields the user asked about and keep every other field exactly as it is. The result must stay valid JSON.\n` +
		`- Pass \`--config ${input.configPath}\` to every \`autorag\` command you run, so you inspect and verify this config and no other one.\n` +
		`- Never print, copy, or store a credential value. Store only environment-variable names such as \`apiKeyEnv\`. Check that a key is set with \`test -n "$NAME"\`, never by echoing it.\n` +
		`- Never run \`autorag init --force\`, and never delete the config file.\n` +
		`- After a change, verify it: run \`autorag health --json --config ${input.configPath}\` (and \`autorag models list --available\` when choosing a provider/model). If verification fails because of your edit, restore the previous value and say so. If a command fails only because the workspace directory does not exist, create it with \`mkdir -p\` using the config's workspacePath and retry; that is not a provider failure.\n` +
		`- A provider is usable only when its own credential is present (\`test -n "$ITS_API_KEY"\`, or listed by \`autorag models list --available\`) and \`autorag health\` passes with that provider's model. A model that a gateway provider such as openrouter lists under another vendor's name does not make that vendor's native provider usable. Report each provider separately, and say "not verified" instead of guessing.\n` +
		`- The running agent keeps the model and settings it started with; changes apply to the next \`autorag\` launch. Say so in the report.\n\n` +
		`## autorag-setup skill (full text)\n\n${input.skill}\n\n` +
		`## User request\n\n${input.query}\n\n` +
		`When done, call emit_autorag_results exactly once with: \`answer\` = a short report of what you changed (field: old → new), what you verified and how, ` +
		`and anything that still needs the user (for example an unset API-key environment variable, by name only); \`results\` = []; \`mapping\` = []. ` +
		`If you changed nothing, say why.`
	);
}
