import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { resolveAutoRAGHome, resolveUserHome } from "../config/home.ts";
import { type Automation, loadAutomations } from "./config.ts";
import { isAutomationDaemonRunning } from "./daemon.ts";
import { type AutomationRunRecord, readLastAutomationRun } from "./run.ts";

export interface AutomationServiceSpec {
	readonly id: string;
	readonly nodePath: string;
	readonly cliEntry: string;
	readonly envFilePath: string;
	readonly logPath: string;
	readonly home: string;
	readonly pathEnv: string;
	readonly configPath?: string;
}

export interface AutomationServiceDescriptor {
	readonly platform: NodeJS.Platform;
	readonly id: string;
	readonly definitionPath: string;
	readonly envFilePath: string;
	readonly logPath: string;
}

export interface ServiceExecResult {
	readonly code: number;
	readonly stdout: string;
	readonly stderr: string;
}

export type ServiceCommandExec = (cmd: string, args: readonly string[]) => Promise<ServiceExecResult>;

export interface AutomationServiceOptions {
	readonly platform?: NodeJS.Platform;
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	readonly selfCommand?: readonly string[];
	readonly exec?: ServiceCommandExec;
}

export interface AutomationStatusEntry extends Automation {
	readonly lastRun?: AutomationRunRecord;
}

export interface AutomationStatusReport {
	readonly service: {
		readonly platform: NodeJS.Platform;
		readonly id: string;
		readonly definitionPath: string;
		readonly logPath: string;
		readonly installed: boolean;
		readonly running: boolean;
	};
	readonly configPath: string;
	readonly configError?: string;
	readonly automations: readonly AutomationStatusEntry[];
}

function xmlEscape(value: string): string {
	return value
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&apos;");
}

function systemdQuote(value: string): string {
	return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

function winArgQuote(value: string): string {
	if (!/[\s"]/u.test(value)) return value;
	return `"${value.replaceAll('"', '\\"')}"`;
}

export function automationServiceDescriptor(options: AutomationServiceOptions = {}): AutomationServiceDescriptor {
	const env = options.env ?? process.env;
	const platform = options.platform ?? process.platform;
	const userHome = resolveUserHome(env);
	const defaultHome = resolve(join(userHome, ".autorag"));
	const home = resolve(options.home ?? resolveAutoRAGHome(env));
	const suffix = home === defaultHome ? "" : `-${createHash("sha256").update(home).digest("hex").slice(0, 8)}`;

	const automationsDir = join(home, "automations");
	const envFilePath = join(automationsDir, "service.env");
	const logPath = join(automationsDir, "daemon.log");

	if (platform === "darwin") {
		const id = `com.autorag.automate${suffix}`;
		return {
			platform,
			id,
			definitionPath: join(userHome, "Library", "LaunchAgents", `${id}.plist`),
			envFilePath,
			logPath,
		};
	}
	if (platform === "linux") {
		const id = `autorag-automate${suffix}`;
		const xdg = env.XDG_CONFIG_HOME?.trim() || join(userHome, ".config");
		return {
			platform,
			id,
			definitionPath: join(xdg, "systemd", "user", `${id}.service`),
			envFilePath,
			logPath,
		};
	}
	if (platform === "win32") {
		const id = `AutoRAG-Automate${suffix}`;
		return {
			platform,
			id,
			definitionPath: join(automationsDir, `${id}.xml`),
			envFilePath,
			logPath,
		};
	}
	throw new Error(`Unsupported platform for automation service: ${platform}`);
}

export function renderLaunchdPlist(spec: AutomationServiceSpec): string {
	const args = [spec.nodePath, `--env-file=${spec.envFilePath}`, spec.cliEntry, "automate", "daemon"];
	const envEntries: Array<[string, string]> = [
		["PATH", spec.pathEnv],
		["AUTORAG_HOME", spec.home],
		...(spec.configPath ? ([["AUTORAG_CONFIG", spec.configPath]] as Array<[string, string]>) : []),
	];
	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
		'<plist version="1.0">',
		"<dict>",
		"\t<key>Label</key>",
		`\t<string>${xmlEscape(spec.id)}</string>`,
		"\t<key>ProgramArguments</key>",
		"\t<array>",
		...args.map((arg) => `\t\t<string>${xmlEscape(arg)}</string>`),
		"\t</array>",
		"\t<key>EnvironmentVariables</key>",
		"\t<dict>",
		...envEntries.flatMap(([k, v]) => [`\t\t<key>${xmlEscape(k)}</key>`, `\t\t<string>${xmlEscape(v)}</string>`]),
		"\t</dict>",
		"\t<key>RunAtLoad</key>",
		"\t<true/>",
		"\t<key>KeepAlive</key>",
		"\t<dict>",
		"\t\t<key>SuccessfulExit</key>",
		"\t\t<false/>",
		"\t</dict>",
		"\t<key>StandardOutPath</key>",
		`\t<string>${xmlEscape(spec.logPath)}</string>`,
		"\t<key>StandardErrorPath</key>",
		`\t<string>${xmlEscape(spec.logPath)}</string>`,
		"</dict>",
		"</plist>",
		"",
	].join("\n");
}

export function renderSystemdUnit(spec: AutomationServiceSpec): string {
	const execArgs = [
		systemdQuote(spec.nodePath),
		systemdQuote(`--env-file=${spec.envFilePath}`),
		systemdQuote(spec.cliEntry),
		"automate",
		"daemon",
	].join(" ");
	const envLines = [
		`Environment=${systemdQuote(`PATH=${spec.pathEnv}`)}`,
		`Environment=${systemdQuote(`AUTORAG_HOME=${spec.home}`)}`,
		...(spec.configPath ? [`Environment=${systemdQuote(`AUTORAG_CONFIG=${spec.configPath}`)}`] : []),
	];
	return [
		"[Unit]",
		`Description=AutoRAG File Automation Daemon (${spec.id})`,
		"",
		"[Service]",
		"Type=simple",
		`ExecStart=${execArgs}`,
		"Restart=on-failure",
		"RestartSec=5",
		...envLines,
		"",
		"[Install]",
		"WantedBy=default.target",
		"",
	].join("\n");
}

export function renderWindowsTaskXml(spec: AutomationServiceSpec): string {
	const args = `${winArgQuote(`--env-file=${spec.envFilePath}`)} ${winArgQuote(spec.cliEntry)} automate daemon`;
	return [
		'<?xml version="1.0" encoding="UTF-16"?>',
		'<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">',
		"\t<Triggers>",
		"\t\t<LogonTrigger>",
		"\t\t\t<Enabled>true</Enabled>",
		"\t\t</LogonTrigger>",
		"\t</Triggers>",
		"\t<Settings>",
		"\t\t<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>",
		"\t\t<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>",
		"\t\t<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>",
		"\t\t<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>",
		"\t\t<Hidden>true</Hidden>",
		"\t\t<RestartOnFailure>",
		"\t\t\t<Interval>PT1M</Interval>",
		"\t\t\t<Count>3</Count>",
		"\t\t</RestartOnFailure>",
		"\t</Settings>",
		'\t<Actions Context="Author">',
		"\t\t<Exec>",
		`\t\t\t<Command>${xmlEscape(spec.nodePath)}</Command>`,
		`\t\t\t<Arguments>${xmlEscape(args)}</Arguments>`,
		"\t\t</Exec>",
		"\t</Actions>",
		"</Task>",
		"",
	].join("\r\n");
}

const defaultExec: ServiceCommandExec = (cmd, args) => {
	return new Promise((resolveExec) => {
		execFile(cmd, [...args], { windowsHide: true }, (error, stdout, stderr) => {
			const code = error && "code" in error && typeof error.code === "number" ? error.code : error ? 1 : 0;
			resolveExec({ code, stdout: String(stdout), stderr: String(stderr) });
		});
	});
};

function resolveSelfNodeAndCli(selfCommand?: readonly string[]): {
	readonly nodePath: string;
	readonly cliEntry: string;
} {
	if (selfCommand && selfCommand.length >= 2) {
		return { nodePath: selfCommand[0]!, cliEntry: resolve(selfCommand[1]!) };
	}
	const entry = process.argv[1];
	if (!entry) throw new Error("Cannot determine AutoRAG CLI entrypoint from process.argv[1]");
	return { nodePath: process.execPath, cliEntry: resolve(entry) };
}

async function runChecked(exec: ServiceCommandExec, cmd: string, args: readonly string[]): Promise<void> {
	const res = await exec(cmd, args);
	if (res.code !== 0) {
		throw new Error(`${cmd} ${args.join(" ")} failed (exit ${res.code}): ${(res.stderr || res.stdout).trim()}`);
	}
}

export async function installAutomationService(
	options: AutomationServiceOptions = {},
): Promise<AutomationServiceDescriptor> {
	const env = options.env ?? process.env;
	const home = resolve(options.home ?? resolveAutoRAGHome(env));
	const desc = automationServiceDescriptor({ ...options, home, env });
	const { nodePath, cliEntry } = resolveSelfNodeAndCli(options.selfCommand);
	const exec = options.exec ?? defaultExec;
	const pathEnv = env.PATH ?? "/usr/local/bin:/usr/bin:/bin";
	const configPath = env.AUTORAG_CONFIG?.trim();

	mkdirSync(dirname(desc.envFilePath), { recursive: true });
	mkdirSync(dirname(desc.definitionPath), { recursive: true });

	const envLines = [
		`AUTORAG_HOME=${home}`,
		`PATH=${pathEnv}`,
		...(configPath ? [`AUTORAG_CONFIG=${resolve(configPath)}`] : []),
	];
	writeFileSync(desc.envFilePath, `${envLines.join("\n")}\n`, "utf8");

	const spec: AutomationServiceSpec = {
		id: desc.id,
		nodePath,
		cliEntry,
		envFilePath: desc.envFilePath,
		logPath: desc.logPath,
		home,
		pathEnv,
		...(configPath ? { configPath: resolve(configPath) } : {}),
	};

	if (desc.platform === "darwin") {
		writeFileSync(desc.definitionPath, renderLaunchdPlist(spec), "utf8");
		const domain = `gui/${process.getuid?.() ?? 501}`;
		await exec("launchctl", ["bootout", domain, desc.definitionPath]);
		await runChecked(exec, "launchctl", ["bootstrap", domain, desc.definitionPath]);
	} else if (desc.platform === "linux") {
		writeFileSync(desc.definitionPath, renderSystemdUnit(spec), "utf8");
		await runChecked(exec, "systemctl", ["--user", "daemon-reload"]);
		await runChecked(exec, "systemctl", ["--user", "enable", "--now", `${desc.id}.service`]);
	} else if (desc.platform === "win32") {
		const utf16WithBom = Buffer.concat([
			Buffer.from([0xff, 0xfe]),
			Buffer.from(renderWindowsTaskXml(spec), "utf16le"),
		]);
		writeFileSync(desc.definitionPath, utf16WithBom);
		await runChecked(exec, "schtasks", ["/Create", "/TN", desc.id, "/XML", desc.definitionPath, "/F"]);
		await runChecked(exec, "schtasks", ["/Run", "/TN", desc.id]);
	}

	return desc;
}

export async function uninstallAutomationService(
	options: AutomationServiceOptions = {},
): Promise<AutomationServiceDescriptor> {
	const env = options.env ?? process.env;
	const home = resolve(options.home ?? resolveAutoRAGHome(env));
	const desc = automationServiceDescriptor({ ...options, home, env });
	const exec = options.exec ?? defaultExec;

	if (desc.platform === "darwin") {
		const domain = `gui/${process.getuid?.() ?? 501}`;
		await exec("launchctl", ["bootout", domain, desc.definitionPath]);
	} else if (desc.platform === "linux") {
		await exec("systemctl", ["--user", "disable", "--now", `${desc.id}.service`]);
		await exec("systemctl", ["--user", "daemon-reload"]);
	} else if (desc.platform === "win32") {
		await exec("schtasks", ["/End", "/TN", desc.id]);
		await exec("schtasks", ["/Delete", "/TN", desc.id, "/F"]);
	}

	rmSync(desc.definitionPath, { force: true });
	rmSync(desc.envFilePath, { force: true });
	return desc;
}

export function getAutomationStatus(options: AutomationServiceOptions = {}): AutomationStatusReport {
	const env = options.env ?? process.env;
	const home = resolve(options.home ?? resolveAutoRAGHome(env));
	const desc = automationServiceDescriptor({ ...options, home, env });
	const configPath = join(home, "automations.json");

	let automations: Automation[] = [];
	let configError: string | undefined;
	try {
		automations = loadAutomations(configPath, env);
	} catch (error) {
		configError = error instanceof Error ? error.message : String(error);
	}

	return {
		service: {
			platform: desc.platform,
			id: desc.id,
			definitionPath: desc.definitionPath,
			logPath: desc.logPath,
			installed: existsSync(desc.definitionPath),
			running: isAutomationDaemonRunning(home),
		},
		configPath,
		...(configError ? { configError } : {}),
		automations: automations.map((a) => {
			const lastRun = readLastAutomationRun(home, a.id);
			return lastRun ? { ...a, lastRun } : a;
		}),
	};
}
