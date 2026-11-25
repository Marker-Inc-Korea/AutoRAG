/**
 * Live E2E: OS-level read-only filesystem enforcement and binary `readonly`/`full` permission switching
 * inside real AutoRAGAgent interactive sessions (no Jev decision model involved in permission enforcement).
 *
 * Runs locally without any network call or API key (faux model + baseline Jev for routing, real OS
 * sandbox and real pi tool execution):
 *
 *   bunx vitest run test/live-e2e/permission-gate.test.ts
 */
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { PermissionMode } from "../../src/agent/permission-gate.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-live-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

const bash = (command: string): FauxResponseStep =>
	fauxAssistantMessage([fauxToolCall("bash", { command })], { stopReason: "toolUse" });
const writeTool = (path: string, content: string): FauxResponseStep =>
	fauxAssistantMessage([fauxToolCall("write", { path, content })], { stopReason: "toolUse" });
const say = (text: string): FauxResponseStep => fauxAssistantMessage(text, { stopReason: "stop" });

async function openSession(mode: PermissionMode, responses: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-live-${randomUUID()}`, models: [{ id: "live" }] });
	// Two-phase local route: fast answer first, then verification runs the scripted tool calls.
	registration.setResponses([say("fast"), ...responses]);
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	const model = registration.getModel();
	writeFileSync(
		join(agentDir, "settings.json"),
		JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
	);
	const agent = new AutoRAGAgent({
		model,
		apiKey: "unused",
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		permissionMode: mode,
		jev: baselineJev(),
	});
	const hosted = await agent.createPiInteractiveRuntime();
	const statuses: (string | undefined)[] = [];
	const notices: string[] = [];
	const noop = () => undefined;
	await hosted.runtime.session.bindExtensions({
		uiContext: {
			select: async () => undefined,
			confirm: async () => false,
			input: async () => undefined,
			notify: (message: string) => void notices.push(message),
			onTerminalInput: () => noop,
			setStatus: (_key: string, text: string | undefined) => void statuses.push(text),
			setWorkingMessage: noop,
			setWorkingVisible: noop,
			setWorkingIndicator: noop,
			setHiddenThinkingLabel: noop,
			setWidget: noop,
			setFooter: noop,
			setHeader: noop,
			setTitle: noop,
			theme: { fg: (_color: string, text: string) => text },
		} as unknown as ExtensionUIContext,
	});
	return {
		agent,
		session: hosted.runtime.session,
		statuses,
		notices,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

describe.skipIf(process.platform !== "darwin")("live E2E: OS-level read-only permission gate", () => {
	const MUTATION_ATTEMPTS: readonly { readonly name: string; readonly command: string }[] = [
		{ name: "rm", command: "rm {dir}/target.txt" },
		{ name: "trash", command: "/usr/bin/trash {dir}/target.txt" },
		{ name: "mv rename", command: "mv {dir}/target.txt {dir}/renamed.txt" },
		{ name: "redirect overwrite", command: "echo replaced > {dir}/target.txt" },
		{ name: "append >>", command: "echo extra >> {dir}/target.txt" },
		{ name: "sed -i", command: "sed -i '' 's/original/hacked/' {dir}/target.txt" },
		{
			name: "node fs.unlinkSync",
			command: `node -e 'require("fs").unlinkSync("{dir}/target.txt")'`,
		},
		{
			name: "python subprocess trash",
			command: "python3 - <<'EOF'\nimport subprocess\nsubprocess.run(['/usr/bin/trash', '{dir}/target.txt'])\nEOF",
		},
		{
			name: "python open write",
			command: "python3 -c \"open('{dir}/target.txt','w').write('hacked')\"",
		},
		{ name: "perl unlink", command: "perl -e 'unlink \"{dir}/target.txt\"'" },
		{ name: "find -delete", command: "find {dir} -name 'target.txt' -delete" },
		{ name: "shell script file", command: "sh {dir}/cleanup.sh" },
		{ name: "symlink escape through $TMPDIR", command: 'ln -s {dir}/target.txt "$TMPDIR/s" && echo x > "$TMPDIR/s"' },
	];

	for (const attempt of MUTATION_ATTEMPTS) {
		it(`blocks disk mutation in readonly mode: ${attempt.name}`, async () => {
			const target = join(root, "target.txt");
			writeFileSync(target, "original", "utf8");
			writeFileSync(join(root, "cleanup.sh"), `rm -f "${target}"\n`, "utf8");
			const t = await openSession("readonly", [bash(attempt.command.replaceAll("{dir}", root)), say("done")]);
			try {
				await t.session.prompt("Inspect target.txt", { source: "interactive" });
				await t.agent.waitForIdle();
				expect(existsSync(target)).toBe(true);
				expect(readFileSync(target, "utf8")).toBe("original");
			} finally {
				await t.dispose();
			}
		}, 30_000);
	}

	it("allows reading files and writing temporary files inside $TMPDIR in readonly mode", async () => {
		const target = join(root, "target.txt");
		writeFileSync(target, "hello-world", "utf8");
		const t = await openSession("readonly", [
			bash(`cat "${target}" && echo "temp-ok" > "$TMPDIR/scratch.txt" && cat "$TMPDIR/scratch.txt"`),
			say("done"),
		]);
		try {
			await t.session.prompt("Read target.txt", { source: "interactive" });
			await t.agent.waitForIdle();
			const bashResult = t.session.messages.find(
				(m): m is Extract<typeof m, { role: "toolResult" }> => m.role === "toolResult" && m.toolName === "bash",
			);
			expect(bashResult?.isError).toBe(false);
			expect(JSON.stringify(bashResult?.content)).toContain("hello-world");
			expect(JSON.stringify(bashResult?.content)).toContain("temp-ok");
		} finally {
			await t.dispose();
		}
	}, 30_000);

	it("blocks write tool in readonly mode and allows it after switching to full mode", async () => {
		const target = join(root, "deliverable.md");
		const t = await openSession("readonly", [
			writeTool(target, "blocked-content"),
			say("blocked"),
			writeTool(target, "allowed-content"),
			say("written"),
		]);
		try {
			expect(t.statuses.at(-1)).toBe("permissions: read-only");
			await t.session.prompt("Write deliverable.md", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(existsSync(target)).toBe(false);

			// Switch to full mode via the shared controller (same path as Shift+Tab / `/permissions full`)
			t.agent.permissionControl.set("full");
			expect(t.statuses.at(-1)).toBe("permissions: full");

			await t.session.prompt("Now write deliverable.md", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(readFileSync(target, "utf8")).toBe("allowed-content");
		} finally {
			await t.dispose();
		}
	}, 30_000);
});

describe.skipIf(process.platform !== "win32")("live E2E (Windows): OS-level Low-IL read-only permission gate", () => {
	it("blocks disk mutations via PowerShell, cmd /c del, and node while allowing reads, %TEMP% writes, and full-mode switch", async () => {
		const target = join(root, "target.txt");
		writeFileSync(target, "original-win", "utf8");
		const deliverable = join(root, "deliverable.md");

		const registration = registerFauxProvider({ api: `faux-win-live-${randomUUID()}`, models: [{ id: "live-win" }] });
		registration.setResponses([
			say("fast"),
			fauxAssistantMessage(
				[
					fauxToolCall("powershell", {
						command: `Set-Content -Path '${target}' -Value 'hacked'; Remove-Item -Force '${target}'; cmd.exe /c del /f /q "${target}"`,
					}),
				],
				{ stopReason: "toolUse" },
			),
			fauxAssistantMessage(
				[
					fauxToolCall("powershell", {
						command: `Write-Output ("READ=" + (Get-Content -Raw '${target}').Trim()); $f = Join-Path $env:TEMP 'qa.txt'; Set-Content -Path $f -Value 'tmp-ok'; Write-Output ("TMP=" + (Get-Content -Raw $f).Trim())`,
					}),
				],
				{ stopReason: "toolUse" },
			),
			writeTool(deliverable, "blocked-in-readonly"),
			say("checked"),
			writeTool(deliverable, "written-in-full"),
			say("written"),
		]);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		const model = registration.getModel();
		writeFileSync(
			join(agentDir, "settings.json"),
			JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
		);
		const agent = new AutoRAGAgent({
			model,
			apiKey: "unused",
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
			permissionMode: "readonly",
			jev: baselineJev(),
		});
		const hosted = await agent.createPiInteractiveRuntime();
		const statuses: (string | undefined)[] = [];
		const noop = () => undefined;
		await hosted.runtime.session.bindExtensions({
			uiContext: {
				select: async () => undefined,
				confirm: async () => false,
				input: async () => undefined,
				notify: noop,
				onTerminalInput: () => noop,
				setStatus: (_key: string, text: string | undefined) => void statuses.push(text),
				setWorkingMessage: noop,
				setWorkingVisible: noop,
				setWorkingIndicator: noop,
				setHiddenThinkingLabel: noop,
				setWidget: noop,
				setFooter: noop,
				setHeader: noop,
				setTitle: noop,
				theme: { fg: (_color: string, text: string) => text },
			} as unknown as ExtensionUIContext,
		});
		const active = hosted.runtime.session.getActiveToolNames();
		if (!active.includes("powershell")) {
			hosted.runtime.session.setActiveToolsByName([...active, "powershell"]);
		}
		try {
			expect(statuses.at(-1)).toBe("permissions: read-only");
			await hosted.runtime.session.prompt("Inspect target.txt", { source: "interactive" });
			await agent.waitForIdle();

			expect(existsSync(target)).toBe(true);
			expect(readFileSync(target, "utf8")).toBe("original-win");
			expect(existsSync(deliverable)).toBe(false);

			const psResults = hosted.runtime.session.messages.filter(
				(m): m is Extract<typeof m, { role: "toolResult" }> =>
					m.role === "toolResult" && m.toolName === "powershell",
			);
			const readText = JSON.stringify(psResults[1]?.content ?? "");
			expect(readText).toContain("READ=original-win");
			expect(readText).toContain("TMP=tmp-ok");

			agent.permissionControl.set("full");
			expect(statuses.at(-1)).toBe("permissions: full");
			await hosted.runtime.session.prompt("Now write deliverable.md", { source: "interactive" });
			await agent.waitForIdle();
			expect(readFileSync(deliverable, "utf8")).toBe("written-in-full");
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	}, 60_000);
});
