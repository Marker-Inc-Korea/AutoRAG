import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	createPermissionGate,
	describeReadOnlyBlock,
	PermissionController,
	type PermissionMode,
} from "../../src/agent/permission-gate.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import {
	buildSubagentsPrompt,
	createSubagentExtension,
	createSubagentTool,
	MAX_SUBAGENT_OUTPUT_CHARS,
	MAX_SUBAGENT_PARALLEL_TASKS,
	mutatingSubagentNames,
	normalizeSubagents,
	SUBAGENT_TOOL_NAME,
	type SubagentDefinition,
	type SubagentTask,
} from "../../src/agent/subagents.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

let root: string;
beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-subagents-")));
});
afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function recorder(
	name: string,
	options: { mutatesFiles?: boolean; reply?: (task: SubagentTask) => string; fail?: boolean } = {},
) {
	const calls: SubagentTask[] = [];
	const definition: SubagentDefinition = {
		name,
		description: `${name} does ${name} work.`,
		guidance: `Use ${name} for ${name} jobs.`,
		...(options.mutatesFiles === undefined ? {} : { mutatesFiles: options.mutatesFiles }),
		async run(input, context) {
			calls.push(input);
			context.onUpdate?.("working");
			if (options.fail === true) throw new Error(`${name} broke`);
			return {
				output: options.reply?.(input) ?? `${name} done: ${input.task}`,
				changedFiles: options.mutatesFiles === true ? input.files : [],
			};
		},
	};
	return { definition, calls };
}

type ToolText = {
	content: Array<{ type: string; text?: string }>;
	details: { mode: string; results: Array<{ agent: string; isError: boolean }> };
};
async function call(
	definitions: readonly SubagentDefinition[],
	params: Record<string, unknown>,
	signal?: AbortSignal,
): Promise<ToolText> {
	const tool = createSubagentTool(normalizeSubagents(definitions));
	return (await tool.execute("call-1", params as never, signal)) as unknown as ToolText;
}

describe("host subagent registry", () => {
	it("sorts by name, rejects bad or duplicate names, and builds the same prompt whatever the registration order", () => {
		const a = recorder("pdf-editor", { mutatesFiles: true }).definition;
		const b = recorder("chart-maker").definition;
		expect(normalizeSubagents([a, b]).map((d) => d.name)).toEqual(["chart-maker", "pdf-editor"]);
		expect(buildSubagentsPrompt(normalizeSubagents([a, b]))).toBe(buildSubagentsPrompt(normalizeSubagents([b, a])));
		expect(buildSubagentsPrompt(normalizeSubagents([a, b]))).toContain(
			"`pdf-editor`: pdf-editor does pdf-editor work. (changes files)",
		);
		expect(buildSubagentsPrompt([])).toBe("");
		expect(() => normalizeSubagents([{ ...a, name: "PDF Editor" }])).toThrow(/Invalid subagent name/);
		expect(() => normalizeSubagents([a, { ...a }])).toThrow(/Duplicate/);
		expect([...mutatingSubagentNames([a, b])]).toEqual(["pdf-editor"]);
	});

	it("leaves the system prompt untouched when no subagents are registered", () => {
		const base = { toolNames: ["read", "bash"] };
		expect(buildSystemPrompt(base)).toBe(buildSystemPrompt({ ...base, subagentsSection: "" }));
		expect(buildSystemPrompt(base)).not.toContain("## Subagents");
		const withSection = buildSystemPrompt({
			toolNames: ["read", "bash", SUBAGENT_TOOL_NAME],
			subagentsSection: "## Subagents\n\n- x",
		});
		expect(withSection).toContain("## Subagents");
		// `subagent` is described by its own section, not listed again among "Additional tools".
		expect(withSection).not.toMatch(/Additional tools\*\*: `subagent`/);
	});
});

describe("subagent tool modes", () => {
	it("runs a single task with its files and returns the subagent's output", async () => {
		const pdf = recorder("pdf-editor", { mutatesFiles: true });
		const result = await call([pdf.definition], {
			agent: "pdf-editor",
			task: "rotate page 2",
			files: ["/tmp/a.pdf"],
		});
		expect(pdf.calls).toEqual([{ task: "rotate page 2", files: ["/tmp/a.pdf"] }]);
		expect(result.content[0]!.text).toContain("pdf-editor done: rotate page 2");
		expect(result.content[0]!.text).toContain("Changed files:\n- /tmp/a.pdf");
		expect(result.details.mode).toBe("single");
	});

	it("runs independent tasks in parallel and reports each", async () => {
		const pdf = recorder("pdf-editor");
		const tasks = [1, 2, 3].map((n) => ({ agent: "pdf-editor", task: `job ${n}` }));
		const result = await call([pdf.definition], { tasks });
		expect(pdf.calls.map((c) => c.task).sort()).toEqual(["job 1", "job 2", "job 3"]);
		expect(result.details.mode).toBe("parallel");
		expect(result.content[0]!.text).toMatch(/### 1\. pdf-editor[\s\S]*### 3\. pdf-editor/);
		const tooMany = await call([pdf.definition], {
			tasks: Array.from({ length: MAX_SUBAGENT_PARALLEL_TASKS + 1 }, () => tasks[0]),
		});
		expect(tooMany.content[0]!.text).toMatch(/Too many parallel tasks/);
	});

	it("chains steps with {previous} and stops at the first failure", async () => {
		const reader = recorder("reader", { reply: () => "page 3 has the SSN" });
		const pdf = recorder("pdf-editor");
		const ok = await call([reader.definition, pdf.definition], {
			chain: [
				{ agent: "reader", task: "find the SSN" },
				{ agent: "pdf-editor", task: "redact: {previous}" },
			],
		});
		expect(pdf.calls[0]!.task).toBe("redact: page 3 has the SSN");
		expect(ok.details.results.map((r) => r.agent)).toEqual(["reader", "pdf-editor"]);

		const broken = recorder("reader", { fail: true });
		const after = recorder("pdf-editor");
		const failed = await call([broken.definition, after.definition], {
			chain: [
				{ agent: "reader", task: "x" },
				{ agent: "pdf-editor", task: "{previous}" },
			],
		});
		expect(after.calls).toHaveLength(0);
		expect(failed.content[0]!.text).toMatch(/reader \(failed\)[\s\S]*reader broke/);
	});

	it("names the available subagents on an unknown agent or a bad mode, and honours an aborted signal", async () => {
		const pdf = recorder("pdf-editor");
		expect((await call([pdf.definition], { agent: "excel", task: "x" })).content[0]!.text).toMatch(
			/Unknown subagent "excel". Available: pdf-editor/,
		);
		expect(
			(await call([pdf.definition], { agent: "pdf-editor", task: "x", tasks: [{ agent: "pdf-editor", task: "y" }] }))
				.content[0]!.text,
		).toMatch(/exactly one mode/);
		const controller = new AbortController();
		controller.abort();
		expect(
			(await call([pdf.definition], { agent: "pdf-editor", task: "x" }, controller.signal)).content[0]!.text,
		).toBe("Cancelled.");
		expect(pdf.calls).toHaveLength(0);
	});

	it("clips very long outputs", async () => {
		const chatty = recorder("pdf-editor", { reply: () => "x".repeat(MAX_SUBAGENT_OUTPUT_CHARS + 50) });
		const text = (await call([chatty.definition], { agent: "pdf-editor", task: "x" })).content[0]!.text!;
		expect(text.endsWith("…(output truncated)")).toBe(true);
	});
});

describe("subagents inside a real pi session", () => {
	function steps(args: Parameters<typeof fauxToolCall>[1]): FauxResponseStep[] {
		return [
			fauxAssistantMessage([fauxToolCall(SUBAGENT_TOOL_NAME, args)], { stopReason: "toolUse" }),
			fauxAssistantMessage("relayed", { stopReason: "stop" }),
		];
	}

	async function run(
		mode: PermissionMode,
		definitions: readonly SubagentDefinition[],
		args: Parameters<typeof fauxToolCall>[1],
	) {
		const registration = registerFauxProvider({ api: `faux-sub-${randomUUID()}`, models: [{ id: "sub" }] });
		registration.setResponses(steps(args));
		const normalized = normalizeSubagents(definitions);
		const hosted = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			getSystemPrompt: () => "system",
			customTools: [],
			extensionFactories: [
				createPermissionGate({
					controller: new PermissionController(mode),
					mutatingSubagents: mutatingSubagentNames(normalized),
				}),
				createSubagentExtension(normalized),
			],
			extensionToolNames: [SUBAGENT_TOOL_NAME],
		});
		try {
			await hosted.session.prompt("go");
			const result = (hosted.session.messages as Array<{ role: string; toolName?: string; content?: unknown }>).find(
				(message) => message.role === "toolResult" && message.toolName === SUBAGENT_TOOL_NAME,
			);
			return JSON.stringify(result?.content ?? "");
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	}

	it("lets the orchestrator model delegate and read back the result", async () => {
		const pdf = recorder("pdf-editor", { mutatesFiles: true });
		const text = await run("full", [pdf.definition], {
			i: "Delegate the PDF edit",
			agent: "pdf-editor",
			task: "stamp page 1",
			files: ["/tmp/x.pdf"],
		});
		expect(pdf.calls).toEqual([{ task: "stamp page 1", files: ["/tmp/x.pdf"] }]);
		expect(text).toContain("pdf-editor done: stamp page 1");
	});

	it("blocks a file-changing subagent in readonly mode before it runs, but allows a read-only one", async () => {
		const pdf = recorder("pdf-editor", { mutatesFiles: true });
		const reader = recorder("reader");
		const blocked = await run("readonly", [pdf.definition, reader.definition], {
			agent: "pdf-editor",
			task: "stamp",
			files: [],
		});
		expect(pdf.calls).toHaveLength(0);
		expect(blocked).toMatch(/read-only/i);
		const mixed = await run("readonly", [pdf.definition, reader.definition], {
			tasks: [
				{ agent: "reader", task: "a" },
				{ agent: "pdf-editor", task: "b" },
			],
		});
		expect(reader.calls).toHaveLength(0);
		expect(mixed).toMatch(/read-only/i);
		await run("readonly", [pdf.definition, reader.definition], { agent: "reader", task: "summarize" });
		expect(reader.calls).toHaveLength(1);
		expect(
			describeReadOnlyBlock("subagent", { agent: "reader" }, root, new Set(), undefined, new Set(["pdf-editor"])),
		).toBeUndefined();
	});
});

describe("AutoRAGAgent with host subagents", () => {
	function agentWith(subagents?: readonly SubagentDefinition[]): string {
		const registration = registerFauxProvider({ api: `faux-agent-${randomUUID()}`, models: [{ id: "agent" }] });
		const searchDir = join(root, "docs");
		mkdirSync(searchDir, { recursive: true });
		try {
			const agent = new AutoRAGAgent({
				model: registration.getModel(),
				searchPaths: [searchDir],
				memoryPath: join(root, "memory.json"),
				jikji: false,
				minSync: { autoInstall: false },
				jev: baselineJev(),
				...(subagents === undefined ? {} : { subagents }),
			});
			return (agent as unknown as { innerAgent: { state: { systemPrompt: string } } }).innerAgent.state.systemPrompt;
		} finally {
			registration.unregister();
		}
	}

	it("adds the subagent section only when a host registers subagents, identically whatever the order", () => {
		const plain = agentWith();
		expect(plain).not.toContain("## Subagents");
		const a = recorder("pdf-editor", { mutatesFiles: true }).definition;
		const b = recorder("chart-maker").definition;
		const withAB = agentWith([a, b]);
		expect(withAB).toContain("## Subagents");
		expect(withAB).toContain("### `pdf-editor`\nUse pdf-editor for pdf-editor jobs.");
		expect(agentWith([b, a])).toBe(withAB);
		// The rest of the prompt is the same as without subagents: only the section is added.
		expect(withAB.replace(/\n\n## Subagents[\s\S]*?(?=\n\n## )/u, "")).toBe(plain);
	});
});
