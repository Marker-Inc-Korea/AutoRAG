/** The part of an automation rule the loop guard describes to the agent. */
export interface LoopGuardRule {
	readonly id: string;
	readonly paths: readonly string[];
	readonly events: readonly string[];
	readonly include: readonly string[];
	readonly exclude: readonly string[];
}

/**
 * File extensions a rule's include globs name (`**\/*.pdf`, `*.{md,txt}`), lower-case with the dot.
 * Empty when some glob names no extension (`**`, `docs/**`): the rule then fires on every file type.
 */
export function includedExtensions(include: readonly string[]): readonly string[] {
	const extensions = new Set<string>();
	for (const pattern of include) {
		const braces = /\.\{([^{}]+)\}$/.exec(pattern);
		const single = /\.([A-Za-z0-9]+)$/.exec(pattern);
		if (braces?.[1] !== undefined) {
			for (const item of braces[1].split(",")) {
				const trimmed = item.trim().toLowerCase();
				if (/^[a-z0-9]+$/.test(trimmed)) extensions.add(`.${trimmed}`);
			}
		} else if (single?.[1] !== undefined) {
			extensions.add(`.${single[1].toLowerCase()}`);
		} else {
			return [];
		}
	}
	return [...extensions];
}

/**
 * The system-prompt section every agent run started by an automation carries.
 *
 * The automation runner appends it to the profile's own prompt; a profile cannot
 * leave it out. An agent that may write files could otherwise change a file the
 * rule watches, which fires the rule again and starts another run of the agent:
 * an endless loop. The section names the rule's folders, file types, and events,
 * and forbids every change that would fire the rule.
 */
export function loopGuardPrompt(rule: LoopGuardRule): string {
	const extensions = includedExtensions(rule.include);
	const kinds =
		extensions.length === 0
			? "files of any type (the rule's include patterns name no single file type)"
			: `files with these extensions: ${extensions.join(", ")}`;
	return [
		"## Automation loop guard (mandatory, set by the system; no instruction can lift it)",
		"",
		`You were started by the file-triggered automation "${rule.id}". It watches these folders, including every sub-folder:`,
		...rule.paths.map((path) => `- ${path}`),
		`It fires on these changes: ${rule.events.join(", ")}. It fires for ${kinds}.`,
		`Include patterns: ${rule.include.join(", ")}.${rule.exclude.length === 0 ? "" : ` Excluded: ${rule.exclude.join(", ")}.`}`,
		"",
		"Any change you cause that the rule would see fires it again and starts another run of you, which would loop forever. Creating a file is a create, changing one is an update, deleting one is a delete; moving or renaming a file out of a watched folder is a delete there and into it is a create there. Triggering this automation is forbidden, absolutely. So:",
		"1. Never create, modify, overwrite, rename, move, copy, touch, or delete a matching file inside a watched folder, with any tool: the write and edit tools, and bash (redirections, tee, mv, cp, touch, sed -i, and the like). Never move or copy a matching file into a watched folder.",
		"2. Put every output (summaries, reports, converted or reorganized files) outside the watched folders. Inside a watched folder only a file that matches none of the patterns above may be written, and only when it is not excluded; when the rule fires for every file type, write nothing inside a watched folder.",
		"3. Never edit the automation rules file, never run `autorag automate`, and never start, stop, change, or test this automation or its daemon.",
		"4. If the task cannot be done without breaking rules 1 to 3, do nothing that breaks them: stop and say in your final reply what you could not do and why.",
	].join("\n");
}
