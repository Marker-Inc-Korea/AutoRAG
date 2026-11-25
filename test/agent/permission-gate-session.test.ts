import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createPermissionGate, PermissionController, type PermissionMode } from "../../src/agent/permission-gate.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-session-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function ui(statusLog: (string | undefined)[] = [], noticeLog: string[] = []): ExtensionUIContext {
	const noop = () => undefined;
	return {
		select: async () => undefined,
		confirm: async () => false,
		input: async () => undefined,
		notify: (message: string) => void noticeLog.push(message),
		onTerminalInput: () => noop,
		setStatus: (_key: string, text: string | undefined) => void statusLog.push(text),
		setWorkingMessage: noop,
		setWorkingVisible: noop,
		setWorkingIndicator: noop,
		setHiddenThinkingLabel: noop,
		setWidget: noop,
		setFooter: noop,
		setHeader: noop,
		setTitle: noop,
		custom: async () => undefined as never,
		pasteToEditor: noop,
		setEditorText: noop,
		getEditorText: () => "",
		editor: async () => undefined,
		addAutocompleteProvider: noop,
		setEditorComponent: noop,
		getEditorComponent: noop,
		theme: { fg: (_color: string, text: string) => text } as never,
		getAllThemes: () => [],
		getTheme: noop,
		setTheme: () => ({ success: false, error: "UI not available" }),
		getToolsExpanded: () => false,
		setToolsExpanded: noop,
	} as unknown as ExtensionUIContext;
}

function toolThenDone(toolName: string, args: Parameters<typeof fauxToolCall>[1]): FauxResponseStep[] {
	return [
		fauxAssistantMessage([fauxToolCall(toolName, args)], { stopReason: "toolUse" }),
		fauxAssistantMessage("done", { stopReason: "stop" }),
	];
}

function toolResultText(
	messages: readonly { role: string; toolName?: string; content?: unknown; isError?: boolean }[],
	toolName: string,
): { text: string; isError: boolean } {
	const result = messages.find((message) => message.role === "toolResult" && message.toolName === toolName);
	return { text: JSON.stringify(result?.content ?? ""), isError: Boolean(result?.isError) };
}

describe("permission gate inside a real pi session (headless, no Jev)", () => {
	async function run(mode: PermissionMode, steps: FauxResponseStep[], controller?: PermissionController) {
		const registration = registerFauxProvider({ api: `faux-perm-${randomUUID()}`, models: [{ id: "perm" }] });
		registration.setResponses(steps);
		const permController = controller ?? new PermissionController(mode);
		const hosted = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			getSystemPrompt: () => "system",
			customTools: [],
			extensionFactories: [createPermissionGate({ controller: permController })],
		});
		try {
			await hosted.session.prompt("go");
			return hosted.session.messages as readonly {
				role: string;
				toolName?: string;
				content?: unknown;
				isError?: boolean;
			}[];
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	}

	it("blocks write and edit tools in readonly mode without consulting any decision model", async () => {
		const existing = join(root, "keep.txt");
		const created = join(root, "new.txt");
		writeFileSync(existing, "precious", "utf8");

		const wMsgs = await run("readonly", toolThenDone("write", { path: created, content: "nope" }));
		expect(existsSync(created)).toBe(false);
		expect(toolResultText(wMsgs, "write").text).toMatch(/read-only/i);

		const eMsgs = await run(
			"readonly",
			toolThenDone("edit", { path: existing, edits: [{ oldText: "precious", newText: "hacked" }] }),
		);
		expect(readFileSync(existing, "utf8")).toBe("precious");
		expect(toolResultText(eMsgs, "edit").text).toMatch(/read-only/i);
	});

	it("allows write and edit tools in full mode", async () => {
		const existing = join(root, "edit-me.txt");
		const created = join(root, "created.txt");
		writeFileSync(existing, "before", "utf8");

		await run("full", toolThenDone("write", { path: created, content: "hello" }));
		expect(readFileSync(created, "utf8")).toBe("hello");

		await run("full", toolThenDone("edit", { path: existing, edits: [{ oldText: "before", newText: "after" }] }));
		expect(readFileSync(existing, "utf8")).toBe("after");
	});

	it.skipIf(process.platform !== "darwin")(
		"enforces OS-level read-only disk inside real bash calls in readonly mode, while allowing reads and $TMPDIR",
		async () => {
			const file = join(root, "precious.txt");
			writeFileSync(file, "original", "utf8");

			// Reading succeeds
			const readMsgs = await run("readonly", toolThenDone("bash", { command: `cat "${file}"` }));
			const readRes = toolResultText(readMsgs, "bash");
			expect(readRes.isError).toBe(false);
			expect(readRes.text).toContain("original");

			// Scratch temp write succeeds
			const tmpMsgs = await run(
				"readonly",
				toolThenDone("bash", { command: 'echo "scratch" > "$TMPDIR/note.txt" && cat "$TMPDIR/note.txt"' }),
			);
			const tmpRes = toolResultText(tmpMsgs, "bash");
			expect(tmpRes.isError).toBe(false);
			expect(tmpRes.text).toContain("scratch");

			// rm is blocked by the kernel
			const rmMsgs = await run("readonly", toolThenDone("bash", { command: `rm "${file}"` }));
			expect(toolResultText(rmMsgs, "bash").isError).toBe(true);
			expect(existsSync(file)).toBe(true);

			// node script unlink is blocked by the kernel
			const nodeMsgs = await run(
				"readonly",
				toolThenDone("bash", {
					command: `node -e 'require("fs").writeFileSync(${JSON.stringify(file)}, "hacked")'`,
				}),
			);
			expect(toolResultText(nodeMsgs, "bash").isError).toBe(true);
			expect(readFileSync(file, "utf8")).toBe("original");

			// /usr/bin/trash is blocked by the kernel
			const trashMsgs = await run("readonly", toolThenDone("bash", { command: `/usr/bin/trash "${file}"` }));
			expect(toolResultText(trashMsgs, "bash").isError).toBe(true);
			expect(existsSync(file)).toBe(true);
		},
	);

	it("allows mutations in full mode and responds to dynamic mode toggling mid-session", async () => {
		const file = join(root, "toggle.txt");
		writeFileSync(file, "v1", "utf8");
		const controller = new PermissionController("readonly");

		// First in readonly: write tool blocked
		await run("readonly", toolThenDone("write", { path: file, content: "v2" }), controller);
		expect(readFileSync(file, "utf8")).toBe("v1");

		// Toggle same controller to full: write succeeds on every OS, and on Unix bash rm also succeeds
		controller.set("full");
		await run("full", toolThenDone("write", { path: file, content: "v2" }), controller);
		expect(readFileSync(file, "utf8")).toBe("v2");
		if (process.platform !== "win32") {
			await run("full", toolThenDone("bash", { command: `rm "${file}"` }), controller);
			expect(existsSync(file)).toBe(false);
		}
	});
});

describe("permission gate inside AutoRAGAgent interactive TUI runtime", () => {
	it("shows readonly/full status in TUI footer and enforces mode on real interactive turns", async () => {
		const file = join(root, "notes.txt");
		writeFileSync(file, "safe", "utf8");

		const registration = registerFauxProvider({ api: `faux-perm-tui-${randomUUID()}`, models: [{ id: "perm-tui" }] });
		registration.setResponses([
			fauxAssistantMessage("fast answer", { stopReason: "stop" }),
			fauxAssistantMessage([fauxToolCall("write", { path: file, content: "overwritten" })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage("cannot write in read-only mode", { stopReason: "stop" }),
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
			apiKey: "test-key",
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
		const statuses: (string | undefined)[] = [];
		const notices: string[] = [];
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			await hosted.runtime.session.bindExtensions({ uiContext: ui(statuses, notices) });
			expect(statuses).toContain("permissions: read-only");
			expect(agent.permissionControl.mode).toBe("readonly");

			await hosted.runtime.session.prompt("try to edit notes.txt", { source: "interactive" });
			await agent.waitForIdle();
			expect(readFileSync(file, "utf8")).toBe("safe");

			// Switching via controller updates footer immediately
			agent.permissionControl.set("full");
			expect(statuses.at(-1)).toBe("permissions: full");
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	});
});
