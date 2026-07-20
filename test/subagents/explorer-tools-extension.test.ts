import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, isAbsolute, join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	createAgentSession,
	DefaultResourceLoader,
	type ExtensionContext,
	ModelRuntime,
	SessionManager,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createMandatorySubagentSession, EXPLORER_TOOLS_EXTENSION_PATH } from "../../src/subagents/runtime.ts";

const model: Model<"openai-responses"> = {
	id: "gpt-5.6-sol",
	name: "GPT-5.6 Sol",
	api: "openai-responses",
	provider: "test-proxy",
	baseUrl: "https://example.invalid/v1",
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400_000,
	maxTokens: 128_000,
};

const TOOL_NAMES = ["read", "grep", "find", "ls"] as const;
type ExplorerToolName = (typeof TOOL_NAMES)[number];

let tempRoot: string;
let workspace: string;
let workspaceLink: string;
let outside: string;
let agentDir: string;
let context: ExtensionContext;
let explorerSession: AgentSession;
let runtime: Awaited<ReturnType<typeof createMandatorySubagentSession>>;

function requireTool(name: ExplorerToolName): ToolDefinition {
	const tool = explorerSession.getToolDefinition(name);
	if (tool === undefined) throw new Error(`Explorer tool ${name} was not registered`);
	return tool;
}

function inputFor(name: ExplorerToolName, path: string): Record<string, unknown> {
	switch (name) {
		case "read":
			return { path };
		case "grep":
			return { pattern: "needle", path };
		case "find":
			return { pattern: "*.txt", path };
		case "ls":
			return { path };
	}
}

async function execute(name: ExplorerToolName, path: string) {
	return requireTool(name).execute(`explorer-${name}`, inputFor(name, path), undefined, undefined, context);
}

function textContent(result: Awaited<ReturnType<typeof execute>>): string {
	return result.content
		.filter((item): item is { type: "text"; text: string } => item.type === "text")
		.map((item) => item.text)
		.join("\n");
}

beforeAll(async () => {
	tempRoot = mkdtempSync(join(tmpdir(), "autorag-explorer-containment-"));
	workspace = join(tempRoot, "workspace-real");
	outside = join(tempRoot, "outside");
	agentDir = join(tempRoot, "agent");
	workspaceLink = join(tempRoot, "workspace-link");
	mkdirSync(join(workspace, "docs", "nested"), { recursive: true });
	mkdirSync(outside, { recursive: true });
	writeFileSync(join(workspace, "docs", "a.txt"), "alpha needle\n", "utf8");
	writeFileSync(join(workspace, "docs", "nested", "b.txt"), "beta needle\n", "utf8");
	writeFileSync(join(outside, "secret.txt"), "outside needle\n", "utf8");
	symlinkSync(workspace, workspaceLink, "dir");
	symlinkSync(outside, join(workspace, "outside-link"), "dir");

	const extensionAgentDir = join(tempRoot, "extension-agent");
	mkdirSync(extensionAgentDir, { recursive: true });
	const resourceLoader = new DefaultResourceLoader({
		cwd: workspaceLink,
		agentDir: extensionAgentDir,
		additionalExtensionPaths: [EXPLORER_TOOLS_EXTENSION_PATH],
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
	});
	await resourceLoader.reload();
	const extensionResult = resourceLoader.getExtensions();
	if (extensionResult.errors.length > 0) {
		throw new Error(extensionResult.errors.map((error) => error.error).join("; "));
	}
	const createdExplorerSession = await createAgentSession({
		cwd: workspaceLink,
		agentDir: extensionAgentDir,
		model,
		modelRuntime: await ModelRuntime.create({
			authPath: join(extensionAgentDir, "auth.json"),
			modelsPath: join(extensionAgentDir, "models.json"),
			allowModelNetwork: false,
		}),
		thinkingLevel: "high",
		resourceLoader,
		sessionManager: SessionManager.inMemory(workspaceLink),
	});
	explorerSession = createdExplorerSession.session;
	context = explorerSession.extensionRunner.createContext();

	runtime = await createMandatorySubagentSession({
		cwd: workspaceLink,
		agentDir,
		model,
		systemPrompt: "test prompt",
		tools: [],
	});
});

afterAll(() => {
	explorerSession?.dispose();
	runtime?.session.dispose();
	if (tempRoot !== undefined) rmSync(tempRoot, { recursive: true, force: true });
});

describe("AutoRAG explorer filesystem containment", () => {
	it("registers only read-only explorer tool overrides", () => {
		const registeredNames = explorerSession.extensionRunner
			.getAllRegisteredTools()
			.map(({ definition }) => definition.name);
		expect(registeredNames.sort()).toEqual([...TOOL_NAMES].sort());
		expect(registeredNames).not.toContain("bash");
		expect(registeredNames).not.toContain("write");
		expect(registeredNames).not.toContain("edit");
	});

	it("wires a package-stable absolute extension path into the canonical agent definition", async () => {
		const installedDefinition = readFileSync(join(agentDir, "agents", "autorag-explorer.md"), "utf8");
		expect(isAbsolute(EXPLORER_TOOLS_EXTENSION_PATH)).toBe(true);
		expect(existsSync(EXPLORER_TOOLS_EXTENSION_PATH)).toBe(true);
		expect(realpathSync(EXPLORER_TOOLS_EXTENSION_PATH)).toBe(EXPLORER_TOOLS_EXTENSION_PATH);
		expect(basename(EXPLORER_TOOLS_EXTENSION_PATH)).toBe("explorer-tools-extension.ts");
		expect(installedDefinition).toContain(`subagentOnlyExtensions: ${EXPLORER_TOOLS_EXTENSION_PATH}`);

		const subagentTool = runtime.session.getToolDefinition("subagent");
		if (subagentTool === undefined) throw new Error("The mandatory subagent tool was not registered");
		const agentResult = await subagentTool.execute(
			"get-explorer-containment",
			{ action: "get", agent: "autorag-explorer", agentScope: "user" },
			undefined,
			undefined,
			runtime.session.extensionRunner.createContext(),
		);
		expect(textContent(agentResult)).toContain(EXPLORER_TOOLS_EXTENSION_PATH);
	});

	it("allows read, grep, find, and ls inside the canonical real cwd", async () => {
		const readResult = await execute("read", "docs/a.txt");
		const grepResult = await execute("grep", "docs");
		const findResult = await execute("find", "docs");
		const lsResult = await execute("ls", "docs");

		expect(textContent(readResult)).toContain("alpha needle");
		expect(textContent(grepResult)).toContain("a.txt:1: alpha needle");
		expect(textContent(findResult)).toContain("a.txt");
		expect(textContent(findResult)).toContain("nested/b.txt");
		expect(textContent(lsResult)).toContain("a.txt");
		expect(textContent(lsResult)).toContain("nested/");
	});

	it("rejects absolute paths outside the canonical cwd for all four tools", async () => {
		for (const name of TOOL_NAMES) {
			const path = name === "read" ? join(outside, "secret.txt") : outside;
			await expect(execute(name, path)).rejects.toThrow(/outside assigned cwd/i);
		}
	});

	it("rejects parent traversal even when normalization would remain inside the cwd", async () => {
		for (const name of TOOL_NAMES) {
			const path = name === "read" ? "docs/../docs/a.txt" : "docs/../docs";
			await expect(execute(name, path)).rejects.toThrow(/parent traversal/i);
		}
	});

	it("rejects existing targets that escape through a symlink", async () => {
		for (const name of TOOL_NAMES) {
			const path = name === "read" ? "outside-link/secret.txt" : "outside-link";
			await expect(execute(name, path)).rejects.toThrow(/outside assigned cwd/i);
		}
	});

	it("rejects nonexistent targets whose nearest existing real parent escapes through a symlink", async () => {
		for (const name of TOOL_NAMES) {
			const path = name === "read" ? "outside-link/missing.txt" : "outside-link/missing-directory";
			await expect(execute(name, path)).rejects.toThrow(/outside assigned cwd/i);
		}
	});

	it("keeps the explorer session pinned when its original cwd symlink is retargeted", async () => {
		const initialRead = await execute("read", "docs/a.txt");
		expect(textContent(initialRead)).toContain("alpha needle");
		unlinkSync(workspaceLink);
		symlinkSync(outside, workspaceLink, "dir");

		const readResult = await execute("read", "docs/a.txt");
		expect(textContent(readResult)).toContain("alpha needle");
		await expect(execute("read", "secret.txt")).rejects.toThrow();
	});
});
