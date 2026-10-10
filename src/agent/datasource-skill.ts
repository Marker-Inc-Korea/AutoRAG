import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { DatasourceSkillManifest } from "../datasource/types.ts";

export const LOAD_DATASOURCE_SKILL_TOOL_NAME = "load_datasource_skill";

/**
 * A datasource skill as the agent sees it: name and description for the
 * system-prompt listing, full `content` loaded on demand, and an opaque
 * `datasource://` location. pi's own `Skill` is file-backed (`baseDir`,
 * `sourceInfo`) and has no `content`, so datasource skills, which have no
 * backing file, use this shape instead.
 */
export interface DatasourceAgentSkill {
	readonly name: string;
	readonly description: string;
	/** Full skill instructions, returned by `load_datasource_skill`. */
	readonly content: string;
	/** Opaque `datasource://<name>` location; never a real filesystem path. */
	readonly filePath: string;
}

function escapeXml(value: string): string {
	return value
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&apos;");
}

/** The `<available_skills>` listing, in pi's progressive-disclosure format. */
function formatSkillsListing(skills: readonly DatasourceAgentSkill[]): string {
	const lines = [
		"The following skills provide specialized instructions for specific tasks.",
		"Read the full skill file when the task matches its description.",
		"When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
		"",
		"<available_skills>",
	];
	for (const skill of skills) {
		lines.push("  <skill>");
		lines.push(`    <name>${escapeXml(skill.name)}</name>`);
		lines.push(`    <description>${escapeXml(skill.description)}</description>`);
		lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
		lines.push("  </skill>");
	}
	lines.push("</available_skills>");
	return lines.join("\n");
}

/**
 * Opaque, path-free location used as the `filePath` of a datasource skill.
 * Datasource skills have no real backing file, and AutoRAG must never
 * expose real filesystem paths, so we use a `datasource://<name>` scheme.
 */
export function datasourceSkillLocation(name: string): string {
	return `datasource://${name}`;
}

/**
 * Map a datasource skill manifest onto the agent-skill shape. The resulting
 * skill is listed in the system prompt (name/description/location) for
 * progressive disclosure and its `content` is loaded on demand.
 */
export function toDatasourceAgentSkill(manifest: DatasourceSkillManifest): DatasourceAgentSkill {
	return {
		name: manifest.name,
		description: manifest.description,
		content: manifest.content,
		filePath: datasourceSkillLocation(manifest.name),
	};
}

/**
 * Render the `<available_skills>` progressive-disclosure block for the
 * configured datasource skills, in the same format pi uses for file-backed
 * agent skills, so datasource skills sit on the same layer.
 */
export function buildDatasourceSkillsPrompt(skills: readonly DatasourceAgentSkill[]): string {
	if (skills.length === 0) return "";
	return formatSkillsListing(skills);
}

/**
 * Format the full instructions for a loaded datasource skill, matching pi's
 * `<skill …>` invocation shape but with a path-opaque location (no real
 * filesystem reference, so no "references are relative to …" line).
 */
export function formatDatasourceSkillInvocation(skill: DatasourceAgentSkill): string {
	return `<skill name="${skill.name}" location="${skill.filePath}">\n${skill.content}\n</skill>`;
}

export interface LoadDatasourceSkillDetails {
	readonly skill: string;
	readonly loaded: boolean;
}

export interface DatasourceSkillProvider {
	/**
	 * Resolve a configured datasource skill by model-visible name. Returns
	 * `undefined` when the name is unknown or not configured.
	 */
	loadDatasourceSkill(name: string): DatasourceAgentSkill | undefined;
}

const loadDatasourceSkillSchema = Type.Object({
	name: Type.String({
		description: "Datasource skill name from the available skills list to load full instructions for.",
	}),
});

export function createLoadDatasourceSkillTool(
	provider: DatasourceSkillProvider,
): AgentTool<typeof loadDatasourceSkillSchema, LoadDatasourceSkillDetails> {
	return {
		name: LOAD_DATASOURCE_SKILL_TOOL_NAME,
		label: "Load Datasource Skill",
		description:
			"Load the full instructions for a configured datasource skill by name before searching it. Unknown skill names return not-available.",
		parameters: loadDatasourceSkillSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<LoadDatasourceSkillDetails>> {
			const name = params.name.trim();
			const skill = name.length === 0 ? undefined : provider.loadDatasourceSkill(name);
			if (skill === undefined) {
				return {
					content: [{ type: "text", text: `Datasource skill "${name}" is not available.` }],
					details: { skill: name, loaded: false },
				};
			}
			return {
				content: [{ type: "text", text: formatDatasourceSkillInvocation(skill) }],
				details: { skill: skill.name, loaded: true },
			};
		},
	};
}
