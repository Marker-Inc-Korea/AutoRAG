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
import { AutomationDaemonAlreadyRunningError, startAutomationDaemon } from "../../automation/daemon.ts";
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

async function runAutomateDaemon(ctx: CommandContext): Promise<number> {
	try {
		const handle = await startAutomationDaemon({
			onRun: (automation, record) => {
				if (ctx.json) {
					ctx.stdout(JSON.stringify({ event: "run", ...record }));
				} else {
					const status = record.timedOut
						? "timed out"
						: record.error
							? `error (${record.error})`
							: `exit ${record.exitCode}`;
					ctx.stdout(
						`[automate] ${automation.id}: ${status} (${record.durationMs}ms, events=${record.eventCount})`,
					);
				}
			},
			onError: (msg) => ctx.stderr(`[automate] ${msg}`),
		});
		ctx.stdout(
			ctx.json
				? JSON.stringify({ ok: true, mode: "daemon", message: "automation daemon started" })
				: "AutoRAG automation daemon started. Send SIGINT/SIGTERM to stop.",
		);
		await new Promise<void>((resolveStop) => {
			const keepAlive = setInterval(() => {}, 60_000);
			let stopping = false;
			const onSignal = (): void => {
				if (stopping) return;
				stopping = true;
				clearInterval(keepAlive);
				void handle.stop().finally(resolveStop);
			};
			process.once("SIGINT", onSignal);
			process.once("SIGTERM", onSignal);
		});
		return 0;
	} catch (error) {
		if (error instanceof AutomationDaemonAlreadyRunningError) {
			ctx.stderr(error.message);
			return 0;
		}
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

export async function runAutomate(ctx: CommandContext): Promise<number> {
	try {
		const sub = ctx.positionals[0];
		if (sub === "daemon") return runAutomateDaemon(ctx);
		if (sub !== "run") {
			ctx.stderr(
				renderError(
					new Error("Usage: autorag automate <daemon|run> [id] [--event <create|update|delete>:<path>]"),
					{
						json: ctx.json,
					},
				),
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
