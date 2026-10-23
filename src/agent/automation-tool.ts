import { join } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import {
	type Automation,
	AutomationConfigError,
	type AutomationEvent,
	automationsConfigPath,
	deleteAutomation,
	loadAutomations,
	matchesAutomation,
	saveAutomation,
} from "../automation/config.ts";
import { type AutomationRunRecord, runAutomation } from "../automation/run.ts";
import {
	type AutomationServiceOptions,
	getAutomationStatus,
	installAutomationService,
	type ServiceCommandExec,
	uninstallAutomationService,
} from "../automation/service.ts";
import { resolveAutoRAGHome } from "../config/home.ts";

export const MANAGE_AUTOMATIONS_TOOL_NAME = "manage_automations";

const eventType = Type.Union([Type.Literal("create"), Type.Literal("update"), Type.Literal("delete")]);

const manageAutomationsSchema = Type.Object({
	action: Type.Union(
		[
			Type.Literal("list"),
			Type.Literal("save"),
			Type.Literal("delete"),
			Type.Literal("run"),
			Type.Literal("install_service"),
			Type.Literal("uninstall_service"),
		],
		{
			description:
				"list: show service state and every rule with its last run. save: create or replace rule `id`. delete: remove rule `id`. run: execute rule `id` now (optionally with test `events`). install_service / uninstall_service: start or remove the background daemon that runs rules on file changes.",
		},
	),
	id: Type.Optional(
		Type.String({ description: "Rule id (letters, digits, . _ -). Required for save, delete, and run." }),
	),
	automation: Type.Optional(
		Type.Object(
			{
				paths: Type.Array(Type.String(), {
					description: "Folders to watch recursively; absolute or ~/ paths.",
				}),
				events: Type.Optional(
					Type.Array(eventType, { description: "File changes that trigger the rule. Default: all three." }),
				),
				include: Type.Optional(
					Type.Array(Type.String(), {
						description: "Globs relative to the watched folder, e.g. **/*.pdf. Default: every file.",
					}),
				),
				exclude: Type.Optional(Type.Array(Type.String(), { description: "Globs to ignore." })),
				run: Type.Array(Type.String(), {
					description:
						'Command argv, run without a shell. "autorag" runs this AutoRAG CLI and "node" runs this Node (it executes .ts files directly). Use ["sh","-c","…"] only when shell syntax is needed.',
				}),
				appendFiles: Type.Optional(
					Type.Boolean({ description: "Append changed file paths to argv (capped at 100)." }),
				),
				debounceMs: Type.Optional(
					Type.Integer({ minimum: 1, description: "Quiet period before a batch runs. Default 1000." }),
				),
				timeoutMs: Type.Optional(
					Type.Integer({ minimum: 1, description: "Kill the command after this long. Default 600000." }),
				),
			},
			{ description: "Rule body for save." },
		),
	),
	events: Type.Optional(
		Type.Array(Type.Object({ type: eventType, path: Type.String() }), {
			description: "Test events for run; only events that match the rule are passed to it.",
		}),
	),
});

export interface ManageAutomationsToolOptions {
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	readonly platform?: NodeJS.Platform;
	readonly exec?: ServiceCommandExec;
	readonly waitForRunningMs?: number;
}

export interface ManageAutomationsDetails {
	readonly action: string;
	readonly id?: string;
	readonly run?: AutomationRunRecord;
}

function requireId(action: string, id: string | undefined): string {
	if (id === undefined || id.trim().length === 0) {
		throw new AutomationConfigError(`manage_automations ${action} requires id`);
	}
	return id;
}

function describeRule(rule: Automation): string {
	const parts = [
		`paths=${rule.paths.join(", ")}`,
		`events=${rule.events.join(",")}`,
		`include=${rule.include.join(",")}`,
		...(rule.exclude.length > 0 ? [`exclude=${rule.exclude.join(",")}`] : []),
		`run=${JSON.stringify(rule.run)}`,
		...(rule.appendFiles ? ["appendFiles"] : []),
	];
	return parts.join("; ");
}

function describeRun(record: AutomationRunRecord): string {
	const status = record.timedOut
		? "timed out"
		: record.error !== undefined
			? `error (${record.error})`
			: `exit ${record.exitCode}`;
	return `${record.startedAt} ${status}, ${record.eventCount} event(s), ${record.durationMs}ms`;
}

function text(value: string, details: ManageAutomationsDetails): AgentToolResult<ManageAutomationsDetails> {
	return { content: [{ type: "text", text: value }], details };
}

export function createManageAutomationsTool(
	options: ManageAutomationsToolOptions = {},
): AgentTool<typeof manageAutomationsSchema, ManageAutomationsDetails> {
	const env = options.env ?? process.env;
	const home = options.home ?? resolveAutoRAGHome(env);
	const configPath = options.home === undefined ? automationsConfigPath(env) : join(home, "automations.json");
	const serviceOptions: AutomationServiceOptions = {
		home,
		env,
		...(options.platform !== undefined ? { platform: options.platform } : {}),
		...(options.exec !== undefined ? { exec: options.exec } : {}),
		...(options.waitForRunningMs !== undefined ? { waitForRunningMs: options.waitForRunningMs } : {}),
	};

	return {
		name: MANAGE_AUTOMATIONS_TOOL_NAME,
		label: "Manage Automations",
		description:
			"Create, list, change, delete, and test file-triggered automations, and start or stop the background daemon that runs them. A rule watches folders and runs a command when files are created, updated, or deleted there; changed files arrive on the command's stdin as JSON [{type, path}].",
		parameters: manageAutomationsSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<ManageAutomationsDetails>> {
			const { action } = params;
			switch (action) {
				case "list": {
					const report = getAutomationStatus(serviceOptions);
					const svc = report.service;
					const lines = [
						`Daemon service ${svc.id}: installed=${svc.installed}, running=${svc.running}.`,
						`Rules file: ${report.configPath}`,
						...(report.configError !== undefined ? [`Rules file error: ${report.configError}`] : []),
						...(report.automations.length === 0 ? ["No automations configured."] : []),
						...report.automations.map(
							(rule) =>
								`- ${rule.id}: ${describeRule(rule)}; last run: ${rule.lastRun ? describeRun(rule.lastRun) : "never"}`,
						),
					];
					return text(lines.join("\n"), { action });
				}
				case "save": {
					const id = requireId(action, params.id);
					if (params.automation === undefined) {
						throw new AutomationConfigError("manage_automations save requires automation");
					}
					const saved = saveAutomation(configPath, id, params.automation, env);
					const running = getAutomationStatus(serviceOptions).service.running;
					return text(
						`Saved automation ${id}: ${describeRule(saved)}.${running ? " The running daemon reloads it automatically." : " The daemon is not running; use install_service so it runs on file changes."}`,
						{ action, id },
					);
				}
				case "delete": {
					const id = requireId(action, params.id);
					deleteAutomation(configPath, id);
					return text(`Deleted automation ${id}.`, { action, id });
				}
				case "run": {
					const id = requireId(action, params.id);
					const rule = loadAutomations(configPath, env).find((item) => item.id === id);
					if (rule === undefined) throw new AutomationConfigError(`Unknown automation id: "${id}"`);
					const supplied: readonly AutomationEvent[] = params.events ?? [];
					const matched = supplied.filter((event) =>
						rule.paths.some((root) => matchesAutomation(rule, root, event)),
					);
					const record = await runAutomation(rule, matched, { home, env });
					const lines = [
						`Ran automation ${id}: ${describeRun(record)} (${supplied.length - matched.length} test event(s) did not match).`,
						...(record.stdout.trim() ? [`stdout:\n${record.stdout.trimEnd()}`] : []),
						...(record.stderr.trim() ? [`stderr:\n${record.stderr.trimEnd()}`] : []),
					];
					return text(lines.join("\n"), { action, id, run: record });
				}
				case "install_service": {
					const installed = await installAutomationService(serviceOptions);
					const state = installed.running
						? "the daemon is running"
						: `the daemon has not started yet; check ${installed.logPath}`;
					return text(`Installed daemon service ${installed.id} (${installed.definitionPath}); ${state}.`, {
						action,
					});
				}
				case "uninstall_service": {
					const desc = await uninstallAutomationService(serviceOptions);
					return text(`Stopped and removed daemon service ${desc.id}. Rules are kept in the rules file.`, {
						action,
					});
				}
			}
		},
	};
}
