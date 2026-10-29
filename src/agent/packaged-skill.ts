import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { SkillInvocation } from "./skill-invocation.ts";

/** The packaged `autorag-setup` skill, loaded for the Jev `config` branch. */
export const SETUP_SKILL_NAME = "autorag-setup";
/** The packaged `organize` skill, loaded for the Jev `organize` branch. */
export const ORGANIZE_SKILL_NAME = "organize";

/**
 * Directories that hold packaged agent skills, relative to the package root,
 * searched in order:
 *
 * - `skills/` is the npm package's parent-agent skill folder (shipped through
 *   `package.json`'s `files` and copied into external agents as the README
 *   describes). `autorag-setup` lives there because it is both: the librarian's
 *   config-branch skill and a skill written for parent agents.
 * - `internal-skills/` is shipped in the package too, but it is the agent's own:
 *   nothing there is a parent-agent skill, so it is never copied into one.
 *
 * A branch skill that is the librarian's own (organize) goes in
 * `internal-skills/`; a skill that doubles as a parent-agent skill stays in
 * `skills/`.
 */
const PACKAGED_SKILL_ROOTS = ["skills", "internal-skills"] as const;

/** A packaged skill as a turn prompt needs it: name, opaque location, body. */
export type PackagedAgentSkill = SkillInvocation;

/**
 * Opaque, path-free location used for a packaged agent skill. Packaged skills
 * live on disk, but AutoRAG never exposes real filesystem paths to the model.
 */
export function packagedSkillLocation(name: string): string {
	return `autorag-skill://${name}`;
}

/**
 * Locate a packaged skill by walking up from this module. The source tree
 * (`src/agent/`) and the bundled outputs (`dist/index.js`, `dist/cli/index.js`)
 * sit at different depths below the package root.
 */
function findPackagedSkillFile(name: string): string | undefined {
	let dir = dirname(fileURLToPath(import.meta.url));
	for (let depth = 0; depth < 6; depth += 1) {
		for (const root of PACKAGED_SKILL_ROOTS) {
			const candidate = join(dir, root, name, "SKILL.md");
			if (existsSync(candidate)) return candidate;
		}
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return undefined;
}

/**
 * Load one packaged agent skill: the file at `path` when given (the host's
 * override), otherwise the packaged skill named `name`. The body is returned
 * with its YAML front matter removed, so the model sees only instructions.
 * Throws when the skill cannot be found or read; callers fall back to `local`.
 */
export function loadPackagedSkill(name: string, path?: string): PackagedAgentSkill {
	const file = path ?? findPackagedSkillFile(name);
	if (file === undefined) {
		const searched = PACKAGED_SKILL_ROOTS.map((root) => join(root, name, "SKILL.md")).join(" or ");
		throw new Error(
			`The packaged ${name} skill (${searched}, relative to the package root) was not found next to this AutoRAG install.`,
		);
	}
	let text: string;
	try {
		text = readFileSync(file, "utf8");
	} catch (error) {
		throw new Error(
			`Could not read the ${name} skill ${file}: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
	return {
		name,
		location: packagedSkillLocation(name),
		content: text.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n)+/u, ""),
	};
}
