/**
 * The `<skill name="…" location="…">…</skill>` invocation block AutoRAG embeds
 * in a turn prompt: the shape pi uses for its file-backed skills, with a
 * path-opaque `location`. AutoRAG loads skills that have no path the model
 * should resolve against (datasource skills and its own packaged skills), so
 * the location names the skill without exposing a real filesystem path.
 */
export interface SkillInvocation {
	/** Skill name, as listed in the prompt's skill listing. */
	readonly name: string;
	/** Opaque, path-free location: `datasource://<name>` or `autorag-skill://<name>`. */
	readonly location: string;
	/** Full skill instructions. */
	readonly content: string;
}

/** Render one skill's full instructions in the shared invocation shape. */
export function formatSkillInvocation(skill: SkillInvocation): string {
	return `<skill name="${skill.name}" location="${skill.location}">\n${skill.content}\n</skill>`;
}
