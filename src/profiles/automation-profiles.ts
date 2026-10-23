import type { AgentProfile } from "./agent-factory.ts";

/**
 * Agent profiles an automation rule can name (`"agent": { "profile": "<id>" }`).
 *
 * A rule only carries the id, never a prompt or a tool list: what an agent may
 * do is decided here, in code, so a rules file (which the librarian agent can
 * itself edit) cannot grant `bash` or `write` to an agent that was meant to read.
 * The runner appends the mandatory loop guard (`loop-guard.ts`) to every
 * profile's prompt, writing profiles included.
 */

/** What a profile builder is told about the run it serves. */
export interface AutomationProfileInput {
	/** The rule's own instruction for this run; a profile folds it into its prompt. */
	readonly task: string;
}

export interface AutomationProfileDefinition {
	readonly id: string;
	readonly description: string;
	/** True when the profile may change files (offers `bash`, `edit` or `write`). */
	readonly writes: boolean;
	build(input: AutomationProfileInput): AgentProfile;
}

const profiles = new Map<string, AutomationProfileDefinition>();

export function registerAutomationProfile(definition: AutomationProfileDefinition): void {
	if (profiles.has(definition.id)) throw new Error(`Automation profile "${definition.id}" is already registered`);
	profiles.set(definition.id, definition);
}

export function getAutomationProfile(id: string): AutomationProfileDefinition | undefined {
	return profiles.get(id);
}

export function listAutomationProfiles(): readonly AutomationProfileDefinition[] {
	return [...profiles.values()];
}

const SHARED_RULES = [
	"You run unattended, started by a change to files in a watched folder. Nobody is there to answer questions: never ask one, decide from what you can see.",
	"The files that changed arrive in the user message as `create|update|delete: <absolute path>` lines. Work only on those files and what the task names.",
	"End with a short plain reply that says what you did (or found) for each file. Only that last reply is kept in the run log.",
];

registerAutomationProfile({
	id: "file-triage",
	description: "Reads the changed files and reports on them. Cannot change anything.",
	writes: false,
	build: ({ task }) => ({
		id: "file-triage",
		systemPrompt: [
			"You look at files that just changed in a folder and report on them for the person who owns them.",
			...SHARED_RULES,
			"You can read, list, and search files; you cannot change, create, move, or delete any file.",
			`The task: ${task}`,
		].join("\n"),
		tools: ["read", "grep", "find", "ls"],
		thinking: "low",
	}),
});

registerAutomationProfile({
	id: "file-organizer",
	description: "Reads the changed files and may rename, move, copy, create, and edit files to carry out the task.",
	writes: true,
	build: ({ task }) => ({
		id: "file-organizer",
		systemPrompt: [
			"You organize files that just changed in a folder for the person who owns them: rename, move, copy, create folders, write new files, and edit files when the task asks.",
			...SHARED_RULES,
			"Never delete permanently: no `rm`, `rmdir`, `unlink`, `shred`, `find -delete`. To delete, move the file to the OS trash (macOS `/usr/bin/trash`, Linux `gio trash`).",
			"Check before a batch that no two targets share a name, and never overwrite a file you did not create unless the task says so.",
			"List every change in your final reply as old path -> new path, so it can be undone from the log alone.",
			`The task: ${task}`,
		].join("\n"),
		tools: ["read", "grep", "find", "ls", "bash", "edit", "write"],
		thinking: "low",
	}),
});
