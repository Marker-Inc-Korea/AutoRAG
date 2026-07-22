import type { AgentTool, AgentToolResult, Skill } from "@earendil-works/pi-agent-core";
import { formatSkillsForSystemPrompt } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { DatasourceSkillManifest } from "../datasource/types.ts";

export const LOAD_DATASOURCE_SKILL_TOOL_NAME = "load_datasource_skill";

/**
 * Opaque, path-free location used as the Pi `Skill.filePath` for a datasource
 * skill. Datasource skills have no real backing file, and AutoRAG must never
 * expose real filesystem paths, so we use a `datasource://<name>` scheme.
 */
export function datasourceSkillLocation(name: string): string {
	return `datasource://${name}`;
}

/**
 * Map a datasource skill manifest onto the Pi agent-skill layer. The resulting
 * `Skill` is injected into the system prompt (name/description/location) for
 * progressive disclosure and its `content` is loaded on demand.
 */
export function toDatasourceAgentSkill(manifest: DatasourceSkillManifest): Skill {
	return {
		name: manifest.name,
		description: manifest.description,
		content: manifest.content,
		filePath: datasourceSkillLocation(manifest.name),
	};
}

/**
 * Render the Pi `<available_skills>` progressive-disclosure block for the
 * authorized datasource skills using Pi's own formatter, so datasource skills
 * sit on the exact same layer as file-backed agent skills.
 */
export function buildDatasourceSkillsPrompt(skills: readonly Skill[]): string {
	if (skills.length === 0) return "";
	return formatSkillsForSystemPrompt([...skills]);
}

/**
 * Format the full instructions for a loaded datasource skill, matching Pi's
 * `<skill …>` invocation shape but with a path-opaque location (no real
 * filesystem reference, so no "references are relative to …" line).
 */
export function formatDatasourceSkillInvocation(skill: Skill): string {
	return `<skill name="${skill.name}" location="${skill.filePath}">\n${skill.content}\n</skill>`;
}

export interface LoadDatasourceSkillDetails {
	readonly skill: string;
	readonly loaded: boolean;
}

export interface DatasourceSkillProvider {
	/**
	 * Resolve an authorized datasource skill by model-visible name. Returns
	 * `undefined` when the name is unknown or not authorized by the trusted,
	 * server-bound access context (default-deny).
	 */
	loadDatasourceSkill(name: string): Skill | undefined;
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
			"Load the full instructions for an authorized datasource skill by name before searching it. Permission is server-bound; unknown or unauthorized skills return not-available.",
		parameters: loadDatasourceSkillSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<LoadDatasourceSkillDetails>> {
			const name = params.name.trim();
			const skill = name.length === 0 ? undefined : provider.loadDatasourceSkill(name);
			if (skill === undefined) {
				return {
					content: [{ type: "text", text: `Datasource skill "${name}" is not available or not authorized.` }],
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
