import { resolve } from "node:path";
import {
	AUTOMATION_EVENT_TYPES,
	type Automation,
	AutomationConfigError,
	type AutomationEvent,
	type AutomationEventType,
	loadAutomations,
	matchesAutomation,
} from "../../automation/config.ts";
import { runAutomation } from "../../automation/run.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

function parseEventSpec(spec: string, cwd: string): AutomationEvent {
	const colon = spec.indexOf(":");
	if (colon <= 0 || colon === spec.length - 1) {
		throw new AutomationConfigError(`--event must be <create|update|delete>:<path> (got "${spec}")`);
	}
	const type = spec.slice(0, colon);
	if (!(AUTOMATION_EVENT_TYPES as readonly string[]).includes(type)) {
		throw new AutomationConfigError(`--event type must be create|update|delete (got "${type}")`);
	}
	const rawPath = spec.slice(colon + 1);
	return { type: type as AutomationEventType, path: resolve(cwd, rawPath) };
}

function eventMatchesAnyRoot(automation: Automation, event: AutomationEvent): boolean {
	return automation.paths.some((root) => matchesAutomation(automation, root, event));
}

export async function runAutomate(ctx: CommandContext): Promise<number> {
	try {
		const sub = ctx.positionals[0];
		if (sub !== "run") {
			ctx.stderr(
				renderError(new Error("Usage: autorag automate run <id> [--event <create|update|delete>:<path>]"), {
					json: ctx.json,
				}),
			);
			return 2;
		}
		const id = ctx.positionals[1];
		if (!id) {
			ctx.stderr(
				renderError(new Error("Usage: autorag automate run <id> [--event <type>:<path>]"), { json: ctx.json }),
			);
			return 2;
		}

		const automations = loadAutomations();
		const automation = automations.find((item) => item.id === id);
		if (!automation) {
			ctx.stderr(renderError(new AutomationConfigError(`Unknown automation id: "${id}"`), { json: ctx.json }));
			return 2;
		}

		const specs = ctx.repeatedFlags?.event ?? (typeof ctx.flags.event === "string" ? [ctx.flags.event] : []);
		const supplied = specs.map((spec) => parseEventSpec(spec, ctx.cwd));
		const matched = supplied.filter((ev) => eventMatchesAnyRoot(automation, ev));
		if (supplied.length > 0 && matched.length === 0) {
			ctx.stderr(
				renderError(
					new AutomationConfigError(
						`None of the ${supplied.length} --event value(s) matched automation "${id}" (paths/events/include/exclude)`,
					),
					{ json: ctx.json },
				),
			);
			return 2;
		}

		const record = await runAutomation(automation, matched);
		const ok = record.exitCode === 0 && !record.timedOut && record.error === undefined;
		if (ctx.json) {
			ctx.stdout(
				JSON.stringify({
					ok,
					...record,
					matchedEvents: matched.length,
					ignoredEvents: supplied.length - matched.length,
				}),
			);
		} else {
			const status = record.timedOut
				? "timed out"
				: record.error
					? `error (${record.error})`
					: `exit ${record.exitCode}`;
			ctx.stdout(`automate run ${id}: ${status} (${record.durationMs}ms, events=${matched.length})`);
			if (record.stdout.trim()) ctx.stdout(record.stdout.trimEnd());
			if (record.stderr.trim()) ctx.stderr(record.stderr.trimEnd());
		}
		return ok ? 0 : 1;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return error instanceof AutomationConfigError ? 2 : 1;
	}
}
