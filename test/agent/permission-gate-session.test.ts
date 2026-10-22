import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	createPermissionGate,
	PERMISSION_GATE_QUESTION_ID,
	PermissionController,
	type PermissionMode,
} from "../../src/agent/permission-gate.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-session-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function ui(select: ExtensionUIContext["select"]): ExtensionUIContext {
	return {
		select,
		confirm: async () => false,
		input: async () => undefined,
		notify: () => undefined,
		onTerminalInput: () => () => undefined,
		setStatus: () => undefined,
		theme: undefined as never,
	} as unknown as ExtensionUIContext;
}

/** The model asks `bash` to run `command`, then answers `done`. */
function bashThenDone(command: string): FauxResponseStep[] {
	return [
		fauxAssistantMessage([fauxToolCall("bash", { command })], { stopReason: "toolUse" }),
		fauxAssistantMessage("done", { stopReason: "stop" }),
	];
}

/** Toolresult text of the first bash call in a finished session. */
function bashResult(messages: readonly { role: string; toolName?: string; content?: unknown }[]): string {
	const result = messages.find((message) => message.role === "toolResult" && message.toolName === "bash");
	return JSON.stringify(result?.content ?? "");
}

describe("gate inside a real pi session (headless)", () => {
	async function run(mode: PermissionMode, command: string) {
		const registration = registerFauxProvider({ api: `faux-perm-${randomUUID()}`, models: [{ id: "perm" }] });
		registration.setResponses(bashThenDone(command));
		const controller = new PermissionController(mode);
		const hosted = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			getSystemPrompt: () => "system",
			customTools: [],
			extensionFactories: [createPermissionGate({ controller })],
		});
		try {
			await hosted.session.prompt("tidy up");
			return bashResult(hosted.session.messages as never);
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	}

	it("blocks a delete of an original in ask mode and the file survives", async () => {
		const file = join(root, "keep.txt");
		writeFileSync(file, "precious");
		const result = await run("ask", `rm ${file}`);
		expect(existsSync(file)).toBe(true);
		expect(result).toMatch(/approval/i);
	});

	it("deletes it in full mode", async () => {
		const file = join(root, "gone.txt");
		writeFileSync(file, "x");
		await run("full", `rm ${file}`);
		expect(existsSync(file)).toBe(false);
	});

	it("lets a copy-then-edit workflow run in ask mode without anyone to ask", async () => {
		const file = join(root, "a.md");
		writeFileSync(file, "hello");
		const copy = join(root, "a.copy.md");
		await run("ask", `cp ${file} ${copy} && sed -i.bak 's/hello/bye/' ${copy} && rm ${copy}.bak`);
		expect(readFileSync(copy, "utf8")).toBe("bye");
		expect(readFileSync(file, "utf8")).toBe("hello");
		expect(existsSync(`${copy}.bak`)).toBe(false);
	});
});

describe("gate inside the agent's interactive runtime", () => {
	async function tui(
		mode: PermissionMode,
		responses: FauxResponseStep[],
		jev: JevBackend | undefined,
		select: ExtensionUIContext["select"],
	) {
		const registration = registerFauxProvider({ api: `faux-perm-tui-${randomUUID()}`, models: [{ id: "perm-tui" }] });
		registration.setResponses(responses);
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
			permissionMode: mode,
			...(jev === undefined ? { jev: false as const } : { jev: { backend: jev } }),
		});
		const hosted = await agent.createPiInteractiveRuntime();
		await hosted.runtime.session.bindExtensions({ uiContext: ui(select) });
		return {
			agent,
			session: hosted.runtime.session,
			dispose: async () => {
				await hosted.dispose();
				registration.unregister();
			},
		};
	}

	it("asks the user, honours Deny, and the file survives", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const asked: string[] = [];
		const t = await tui(
			"ask",
			bashThenDone(`mv ${file} ${join(root, "elsewhere.pdf")}`),
			undefined,
			async (title) => {
				asked.push(title);
				return "Deny";
			},
		);
		try {
			await t.session.prompt("move the report", { source: "extension" });
			expect(asked).toHaveLength(1);
			expect(asked[0]).toContain(file);
			expect(existsSync(file)).toBe(true);
			expect(bashResult(t.session.messages as never)).toMatch(/denied/i);
		} finally {
			await t.dispose();
		}
	});

	it("runs the operation when the user allows it", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const t = await tui("ask", bashThenDone(`rm ${file}`), undefined, async () => "Allow once");
		try {
			await t.session.prompt("delete the report", { source: "extension" });
			expect(existsSync(file)).toBe(false);
		} finally {
			await t.dispose();
		}
	});

	it("switches mode live: the same call is blocked in ask and runs after the controller moves to full", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const t = await tui(
			"ask",
			[...bashThenDone(`rm ${file}`), ...bashThenDone(`rm ${file}`)],
			undefined,
			async () => "Deny",
		);
		try {
			await t.session.prompt("delete it", { source: "extension" });
			expect(existsSync(file)).toBe(true);
			t.agent.permissionControl.set("full");
			await t.session.prompt("delete it again", { source: "extension" });
			expect(existsSync(file)).toBe(false);
		} finally {
			await t.dispose();
		}
	});

	it("auto mode: Jev approves without a dialog; a Jev rejection reaches the user", async () => {
		const approve = new MockBackend({
			[PERMISSION_GATE_QUESTION_ID]: {
				answer: "allow",
				distribution: { allow: 0.97, deny: 0.03 },
				confidence: 0.97,
			},
		});
		const first = join(root, "approved.txt");
		writeFileSync(first, "x");
		let dialogs = 0;
		const a = await tui("auto", bashThenDone(`rm ${first}`), approve, async () => {
			dialogs += 1;
			return "Deny";
		});
		try {
			await a.session.prompt("tidy", { source: "extension" });
			expect(existsSync(first)).toBe(false);
			expect(dialogs).toBe(0);
		} finally {
			await a.dispose();
		}

		const reject = new MockBackend({
			[PERMISSION_GATE_QUESTION_ID]: { answer: "deny", distribution: { allow: 0.03, deny: 0.97 }, confidence: 0.97 },
		});
		const second = join(root, "rejected.txt");
		writeFileSync(second, "x");
		const titles: string[] = [];
		const b = await tui("auto", bashThenDone(`rm ${second}`), reject, async (title) => {
			titles.push(title);
			return "Allow once";
		});
		try {
			await b.session.prompt("tidy", { source: "extension" });
			expect(titles).toHaveLength(1);
			expect(titles[0]).toMatch(/jev/i);
			expect(existsSync(second)).toBe(false);
		} finally {
			await b.dispose();
		}
	});
});
