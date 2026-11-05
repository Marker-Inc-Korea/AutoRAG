import { DEFAULT_EXCLUDES } from "../automation/config.ts";

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

/** True when an include pattern accepts every path, so the rule fires for every file type. */
function includesEverything(include: readonly string[]): boolean {
	return include.some((pattern) => pattern === "**" || pattern === "**/*" || pattern === "*");
}

/**
 * The system-prompt section every agent run started by an automation carries.
 *
 * The automation runner appends it to the profile's own prompt; a profile cannot
 * leave it out. An agent that may write files could otherwise change a file the
 * rule watches, which fires the rule again and starts another run of the agent:
 * an endless loop.
 *
 * The section states the one invariant (no change of this run may fire the rule
 * again) and gives the agent the rule's exact decision procedure, so it can check
 * its own plan instead of obeying a list of banned operations. A rename, a move or
 * a write is allowed whenever the events it causes would not be picked up. The
 * agent refuses only when the task cannot be done without a loop, and then
 * proposes a softer variant.
 */
export function loopGuardPrompt(rule: LoopGuardRule): string {
	const extensions = includedExtensions(rule.include);
	const everything = includesEverything(rule.include);
	const kinds = everything
		? "It fires for files of any type: every path inside a watched folder matches an include pattern, unless it is excluded."
		: extensions.length > 0
			? `Its include patterns name these extensions: ${extensions.join(", ")}.`
			: "";
	return [
		"## Automation loop guard (set by the system, appended to every run of this automation)",
		"",
		`You were started by the file-triggered automation "${rule.id}". It watches these folders, including every sub-folder:`,
		...rule.paths.map((path) => `- ${path}`),
		`It fires on these changes: ${rule.events.join(", ")}.${kinds === "" ? "" : ` ${kinds}`}`,
		`Include patterns: ${rule.include.join(", ")}.${rule.exclude.length === 0 ? "" : ` Excluded: ${rule.exclude.join(", ")}.`}`,
		`Always excluded, whatever the rule says: ${DEFAULT_EXCLUDES.join(", ")}.`,
		"",
		"The one thing to prevent is an endless loop: a change you make that fires this automation again starts another run of you, which makes another change, and so on. So the invariant is: nothing you do must fire this automation again. Nothing else here limits what you do; rename, move, copy, write, and edit are all fine when they keep the invariant.",
		"",
		"How the automation decides whether a change fires it. A change fires it only when all of these hold:",
		"- the path is inside a watched folder or one of its sub-folders (paths outside are never seen);",
		`- its type is one of the events above (${rule.events.join(", ")});`,
		"- the path, taken relative to the watched folder, matches an include pattern;",
		"- the path matches no excluded pattern and none of the always-excluded patterns.",
		"",
		"The events your tools cause: creating a file or copying one to a new path is a create at that path; changing the content of an existing file, or overwriting it, is an update; deleting is a delete. A rename or move is a delete at the old path plus a create at the new path; each half counts only where it lands inside a watched folder. A file written in several chunks can show up as a create followed by updates.",
		"",
		"What to do:",
		"1. Design the change before you make it. For every path you will touch, list the event it causes and check the four conditions above. Make the change only if none of those events would fire the automation.",
		"2. Do not stop at the first change that would fire. When the obvious way to do the task fires the rule, look for the lightest change that gets the same result without firing it: a destination name that matches no include pattern or matches an excluded one, a destination outside the watched folders, or an operation whose events the rule does not listen for. Do the task that way. Do not move a file out of a watched folder only to avoid the rule: that takes the user's file away from where they keep it, and moving it back is a create that fires the rule. A different destination is fine when the task itself is to organize the file there; otherwise leave the file where it is.",
		"3. A change that would fire the automation again is acceptable only when the follow-up run provably ends, for a mechanical reason that comes from the file state and not from your intentions: the next run, started by that event, finds the file already in its final state and cannot change anything whatever it decides (an idempotent marker such as a header that is already present). The next run is a fresh agent that does not remember this one: it sees only the event and the file. A timestamp, a counter, a value that depends on the date or time of day, or a log that grows with every run never ends. If you cannot state that mechanical reason, treat the change as one that fires.",
		"4. Only when the task is structurally impossible without a change that keeps looping, with no variant of destination, name, or operation that avoids it, decline that part. Do all the rest of the task, then say in your final reply what you did not do, why it would loop, and one or two softer alternatives (a narrower include pattern, an excluded folder or name, or a different destination). Do not decline a task that has a non-firing way to be done.",
		"5. Never edit the automation rules file, never run `autorag automate`, and never start, stop, change, or test this automation or its daemon. They are the safety net against a loop; do not use them to widen what you may do.",
	].join("\n");
}
