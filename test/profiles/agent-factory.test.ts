import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessage, SimpleStreamOptions, SystemMessage, TranscriptContext } from "@earendil-works/pi-ai";
import {
	type FauxProviderRegistration,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai/compat";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import { type AgentProfile, createProfileSession } from "../../src/profiles/agent-factory.ts";

/** What the model received on one request: the prompt, the tool names it was offered, and the auth/thinking options. */
interface Seen {
	systemPrompt: string;
	toolNames: string[];
	apiKey: string | undefined;
	reasoning: string | undefined;
}

const cleanups: Array<() => Promise<void> | void> = [];
afterEach(async () => {
	for (const cleanup of cleanups.splice(0)) await cleanup();
});

async function tempDir(prefix: string): Promise<string> {
	const dir = await mkdtemp(join(tmpdir(), prefix));
	cleanups.push(() => rm(dir, { recursive: true, force: true }));
	return dir;
}

function register(provider: string, id = "faux-model"): FauxProviderRegistration {
	const registration = registerFauxProvider({
		api: `faux-${provider}-${Math.random().toString(36).slice(2)}`,
		provider,
		models: [{ id, reasoning: true }],
	});
	cleanups.push(() => registration.unregister());
	return registration;
}

function systemText(message: SystemMessage): string {
	const content =
		typeof message.content === "string" ? message.content : message.content.map((block) => block.text).join("\n");
	const sections = Object.values(message.sections ?? {}).filter(
		(section): section is string => typeof section === "string",
	);
	return [content, ...sections].join("\n");
}

/** Replays the transcript's system messages into the prompt and tool set the provider would see. */
function requestOf(context: TranscriptContext, options: SimpleStreamOptions | undefined): Seen {
	const tools = new Set<string>();
	const prompt: string[] = [];
	for (const message of context.messages) {
		if (message.role !== "system") continue;
		prompt.push(systemText(message));
		for (const tool of message.toolsAdded ?? []) tools.add(tool.name);
		for (const tool of message.toolsRemoved ?? []) tools.delete(tool.name);
	}
	return {
		systemPrompt: prompt.join("\n"),
		toolNames: [...tools].sort(),
		apiKey: options?.apiKey,
		reasoning: options?.reasoning,
	};
}

/** Scripts the model: each step records what it was sent, then answers. */
function script(registration: FauxProviderRegistration, steps: Array<(seen: Seen) => AssistantMessage>): Seen[] {
	const seen: Seen[] = [];
	registration.setResponses(
		steps.map((step) => (context, options) => {
			const request = requestOf(context, options);
			seen.push(request);
			return step(request);
		}),
	);
	return seen;
}

const probe = defineTool({
	name: "probe",
	label: "Probe",
	description: "Records that it ran.",
	parameters: Type.Object({ note: Type.String() }),
	execute: async (_id, params) => ({ content: [{ type: "text", text: `noted ${params.note}` }], details: undefined }),
});

function profile(overrides: Partial<AgentProfile> = {}): AgentProfile {
	return { id: "test", systemPrompt: "PROFILE-PROMPT", tools: ["read", "ls"], thinking: "low", ...overrides };
}

async function workspace(): Promise<{ agentDir: string; cwd: string }> {
	return { agentDir: await tempDir("factory-agent-"), cwd: await tempDir("factory-cwd-") };
}

describe("createProfileSession", () => {
	it("offers the model exactly the profile's tools, whatever the user's pi settings default to", async () => {
		const { agentDir, cwd } = await workspace();
		await writeFile(join(agentDir, "settings.json"), JSON.stringify({ defaultTools: ["bash", "write", "edit"] }));
		const registration = register("faux-tools");
		const seen = script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile({ customTools: [probe] }),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel() },
		});
		await session.prompt("hello");
		session.dispose();

		expect(seen[0]?.toolNames).toEqual(["ls", "probe", "read"]);
	});

	it("sends the profile's system prompt and loads no AGENTS.md, skill or extension unless the profile asks", async () => {
		const { agentDir, cwd } = await workspace();
		await writeFile(join(agentDir, "AGENTS.md"), "GLOBAL-AGENTS-MARKER");
		await writeFile(join(cwd, "AGENTS.md"), "PROJECT-AGENTS-MARKER");
		await mkdir(join(agentDir, "skills", "marker"), { recursive: true });
		await writeFile(
			join(agentDir, "skills", "marker", "SKILL.md"),
			"---\nname: marker\ndescription: SKILL-MARKER\n---\nbody",
		);
		await mkdir(join(agentDir, "extensions"), { recursive: true });
		await writeFile(
			join(agentDir, "extensions", "marker.ts"),
			`export default function (pi) { pi.registerTool({ name: "ext_marker", label: "x", description: "x", parameters: { type: "object", properties: {} }, execute: async () => ({ content: [], details: undefined }) }); }`,
		);
		const registration = register("faux-prompt");
		const seen = script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile({ tools: ["read", "ext_marker"] }),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel() },
		});
		await session.prompt("hello");
		session.dispose();

		const request = seen[0];
		expect(request?.systemPrompt).toContain("PROFILE-PROMPT");
		expect(request?.systemPrompt).not.toContain("GLOBAL-AGENTS-MARKER");
		expect(request?.systemPrompt).not.toContain("PROJECT-AGENTS-MARKER");
		expect(request?.systemPrompt).not.toContain("SKILL-MARKER");
		expect(request?.toolNames).toEqual(["read"]);
	});

	it("loads project files, skills and extensions for a profile that opts in", async () => {
		const { agentDir, cwd } = await workspace();
		await writeFile(join(cwd, "AGENTS.md"), "PROJECT-AGENTS-MARKER");
		await mkdir(join(agentDir, "skills", "marker"), { recursive: true });
		await writeFile(
			join(agentDir, "skills", "marker", "SKILL.md"),
			"---\nname: marker\ndescription: SKILL-MARKER\n---\nbody",
		);
		await mkdir(join(agentDir, "extensions"), { recursive: true });
		await writeFile(
			join(agentDir, "extensions", "marker.ts"),
			`export default function (pi) { pi.registerTool({ name: "ext_marker", label: "x", description: "x", parameters: { type: "object", properties: {} }, execute: async () => ({ content: [], details: undefined }) }); }`,
		);
		const registration = register("faux-optin");
		const seen = script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile({
				tools: ["read", "ext_marker"],
				context: { projectFiles: true, skills: true, extensions: true },
			}),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel() },
		});
		await session.prompt("hello");
		session.dispose();

		const request = seen[0];
		expect(request?.systemPrompt).toContain("PROJECT-AGENTS-MARKER");
		expect(request?.systemPrompt).toContain("SKILL-MARKER");
		expect(request?.toolNames).toEqual(["ext_marker", "read"]);
	});

	it("runs a custom tool the model calls and finishes the turn", async () => {
		const { agentDir, cwd } = await workspace();
		const registration = register("faux-run");
		script(registration, [
			() => fauxAssistantMessage(fauxToolCall("probe", { note: "alpha" })),
			() => fauxAssistantMessage("finished"),
		]);

		const session = await createProfileSession({
			profile: profile({ customTools: [probe] }),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel() },
		});
		await session.prompt("go");
		const results = session.messages.filter((message) => message.role === "toolResult");
		session.dispose();

		expect(results).toHaveLength(1);
		expect(JSON.stringify(results[0])).toContain("noted alpha");
	});

	it("asks the model for the profile's thinking level", async () => {
		const { agentDir, cwd } = await workspace();
		const registration = register("faux-think");
		const seen = script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile({ thinking: "medium" }),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel() },
		});
		await session.prompt("hello");
		session.dispose();

		expect(seen[0]?.reasoning).toBe("medium");
	});

	it("keeps no session on disk", async () => {
		const { agentDir, cwd } = await workspace();
		const registration = register("faux-nodisk");
		script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile(),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel() },
		});
		await session.prompt("hello");
		session.dispose();

		// The credential store and model catalog are the shared account files; a session record would be new.
		expect(await readdir(cwd)).toEqual([]);
		expect((await readdir(agentDir)).filter((name) => name !== "auth.json" && name !== "models-store.json")).toEqual(
			[],
		);
	});

	it("authenticates through the agent dir's stored credential, the one the AutoRAG agent and the login screen use", async () => {
		const { agentDir, cwd } = await workspace();
		// A signed-in plan: its OAuth credential and catalog live in the agent dir; no key is passed in.
		const registration = register("dazzi", "plan-model");
		await writeFile(
			join(agentDir, "auth.json"),
			JSON.stringify({
				dazzi: { type: "oauth", access: "STORED-PLAN-TOKEN", refresh: "", expires: Number.MAX_SAFE_INTEGER },
			}),
		);
		const planModel = registration.getModel();
		await writeFile(
			join(agentDir, "models-store.json"),
			JSON.stringify({ dazzi: { models: [{ ...planModel, type: "chat", input: ["text"] }] } }),
		);
		const seen = script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile(),
			cwd,
			agentDir,
			env: {},
			model: { model: planModel },
		});
		await session.prompt("hello");
		session.dispose();

		expect(seen[0]?.apiKey).toBe("STORED-PLAN-TOKEN");
	});

	it("registers a model the runtime does not list and authenticates it with the resolved key", async () => {
		const { agentDir, cwd } = await workspace();
		const registration = register("faux-byok");
		const seen = script(registration, [() => fauxAssistantMessage("done")]);

		const session = await createProfileSession({
			profile: profile(),
			cwd,
			agentDir,
			env: {},
			model: { model: registration.getModel(), apiKey: "EXPLICIT-KEY" },
		});
		await session.prompt("hello");
		session.dispose();

		expect(seen[0]?.apiKey).toBe("EXPLICIT-KEY");
	});
});
