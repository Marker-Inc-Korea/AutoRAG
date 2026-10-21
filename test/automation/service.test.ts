import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type AutomationDaemonHandle, startAutomationDaemon } from "../../src/automation/daemon.ts";
import {
	automationServiceDescriptor,
	getAutomationStatus,
	installAutomationService,
	renderLaunchdPlist,
	renderSystemdUnit,
	renderWindowsTaskXml,
	uninstallAutomationService,
} from "../../src/automation/service.ts";

let root: string;
let userHome: string;
let autoragHome: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-automation-service-"));
	userHome = join(root, "user");
	autoragHome = join(userHome, ".autorag");
	mkdirSync(autoragHome, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("automationServiceDescriptor", () => {
	it("uses the canonical label for the default ~/.autorag home and a suffixed label for isolated homes", () => {
		const env = { HOME: userHome };
		const def = automationServiceDescriptor({ platform: "darwin", home: autoragHome, env });
		expect(def.id).toBe("com.autorag.automate");
		expect(def.definitionPath).toBe(join(userHome, "Library", "LaunchAgents", "com.autorag.automate.plist"));

		const custom = automationServiceDescriptor({ platform: "darwin", home: join(root, "qa-home"), env });
		expect(custom.id).toMatch(/^com\.autorag\.automate-[0-9a-f]{8}$/);
	});

	it("places systemd user units under XDG_CONFIG_HOME when set", () => {
		const xdg = join(root, "xdg");
		const desc = automationServiceDescriptor({
			platform: "linux",
			home: autoragHome,
			env: { HOME: userHome, XDG_CONFIG_HOME: xdg },
		});
		expect(desc.definitionPath).toBe(join(xdg, "systemd", "user", "autorag-automate.service"));
	});
});

describe("service definition renderers", () => {
	it("escapes XML special characters in launchd plist and Windows Task XML", () => {
		const spec = {
			id: "com.autorag.automate",
			nodePath: "/opt/node & co/bin/node",
			cliEntry: "/opt/app <1>/dist/cli/index.js",
			envFilePath: "/tmp/a&b/service.env",
			logPath: "/tmp/a&b/daemon.log",
			home: "/tmp/a&b",
			pathEnv: "/usr/local/bin:/usr/bin",
		};
		const plist = renderLaunchdPlist(spec);
		expect(plist).toContain("/opt/node &amp; co/bin/node");
		expect(plist).toContain("/opt/app &lt;1&gt;/dist/cli/index.js");
		expect(plist).toContain("<key>SuccessfulExit</key>");

		const xml = renderWindowsTaskXml(spec);
		expect(xml).toContain("<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>");
		expect(xml).toContain("<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>");
		expect(xml).toContain("/opt/node &amp; co/bin/node");
	});

	it("quotes paths with spaces in systemd ExecStart and Environment", () => {
		const unit = renderSystemdUnit({
			id: "autorag-automate",
			nodePath: "/opt/my node/bin/node",
			cliEntry: "/home/u/My App/cli.js",
			envFilePath: "/home/u/My App/service.env",
			logPath: "/home/u/My App/daemon.log",
			home: "/home/u/My App",
			pathEnv: "/usr/local/bin:/usr/bin",
		});
		expect(unit).toContain(
			'ExecStart="/opt/my node/bin/node" "--env-file=/home/u/My App/service.env" "/home/u/My App/cli.js" automate daemon',
		);
		expect(unit).toContain("Restart=on-failure");
	});
});

describe("installAutomationService / uninstallAutomationService / getAutomationStatus", () => {
	it("writes the service definition, invokes the platform controller, and cleans up on uninstall", async () => {
		const calls: Array<{ cmd: string; args: readonly string[] }> = [];
		const exec = async (cmd: string, args: readonly string[]) => {
			calls.push({ cmd, args });
			return { code: 0, stdout: "", stderr: "" };
		};
		const env = { HOME: userHome, PATH: "/custom/bin:/usr/bin" };

		const installed = await installAutomationService({
			platform: "darwin",
			home: autoragHome,
			env,
			selfCommand: ["/usr/local/bin/node", "/app/dist/cli/index.js"],
			exec,
			waitForRunningMs: 0,
		});
		expect(installed.running).toBe(false);
		expect(existsSync(installed.definitionPath)).toBe(true);
		expect(existsSync(installed.envFilePath)).toBe(true);
		expect(readFileSync(installed.envFilePath, "utf8")).toContain(`AUTORAG_HOME=${autoragHome}`);
		expect(calls.map((c) => `${c.cmd} ${c.args[0]}`)).toEqual(["launchctl bootout", "launchctl bootstrap"]);

		writeFileSync(
			join(autoragHome, "automations.json"),
			JSON.stringify({ automations: [{ id: "inbox", paths: [userHome], run: ["node", "-e", "0"] }] }),
		);
		const status = getAutomationStatus({ platform: "darwin", home: autoragHome, env });
		expect(status.service.installed).toBe(true);
		expect(status.service.running).toBe(false);
		expect(status.automations).toHaveLength(1);
		expect(status.automations[0]?.id).toBe("inbox");

		await uninstallAutomationService({ platform: "darwin", home: autoragHome, env, exec });
		expect(existsSync(installed.definitionPath)).toBe(false);
		expect(existsSync(installed.envFilePath)).toBe(false);
	});

	it("reports running once the started daemon holds its lock", async () => {
		const env = { HOME: userHome, PATH: "/usr/bin" };
		let daemon: AutomationDaemonHandle | undefined;
		const exec = async (_cmd: string, args: readonly string[]) => {
			if (args[0] === "bootstrap") daemon = await startAutomationDaemon({ home: autoragHome, env });
			return { code: 0, stdout: "", stderr: "" };
		};
		try {
			const installed = await installAutomationService({
				platform: "darwin",
				home: autoragHome,
				env,
				selfCommand: ["/usr/local/bin/node", "/app/dist/cli/index.js"],
				exec,
				waitForRunningMs: 2000,
			});
			expect(installed.running).toBe(true);
		} finally {
			await daemon?.stop();
		}
	});
});
