import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";
import { registerChannelKind, unregisterChannelKind } from "../../src/messaging/registry.ts";
import { assertEnvName } from "../../src/messaging/support.ts";
import { MessagingConfigError } from "../../src/messaging/types.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

let root: string;
const registrations: FauxProviderRegistration[] = [];
const sent: { channel: string; text: string; to?: readonly string[] }[] = [];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-messaging-wiring-"));
	registerChannelKind<{ kind: "echo"; tokenEnv: string }>({
		kind: "echo",
		validate(raw, path) {
			return { kind: "echo", tokenEnv: assertEnvName(raw.tokenEnv, `${path}.tokenEnv`) };
		},
		create: (id) => ({
			id,
			kind: "echo",
			addressing: "fixed",
			describe: () => "echo test channel",
			async send(message) {
				sent.push({ channel: id, text: message.text });
				return { destination: "echo-room", ids: ["e1"] };
			},
		}),
	});
});

afterEach(() => {
	unregisterChannelKind("echo");
	sent.length = 0;
	for (const registration of registrations.splice(0)) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function configFile(messaging: unknown): string {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify({ searchPaths: [root], messaging }));
	return path;
}

const resolve = (messaging: unknown) =>
	resolveConfig({ flags: { config: configFile(messaging) }, cwd: root, env: { HOME: root } });

describe("messaging config", () => {
	it("has no messaging option when the section is absent", () => {
		const config = resolve(undefined);
		expect(config.messaging).toBeUndefined();
		expect(buildAgentOptions(config).messaging).toBeUndefined();
	});

	it("validates the section on load and hands it to the agent", () => {
		const config = resolve({ channels: { ops: { kind: "echo", tokenEnv: "OPS_TOKEN" } } });
		expect(buildAgentOptions(config).messaging).toEqual({
			channels: { ops: { kind: "echo", tokenEnv: "OPS_TOKEN" } },
		});
	});

	it("reports a bad section as a config error naming the path, never the value", () => {
		const attempt = () => resolve({ channels: { ops: { kind: "echo", tokenEnv: "123:SECRET-TOKEN" } } });
		expect(attempt).toThrow(ConfigError);
		expect(attempt).toThrow(/messaging\.channels\.ops\.tokenEnv/);
		expect(attempt).not.toThrow(/SECRET-TOKEN/);
	});

	it("exports the config error as a MessagingConfigError underneath", () => {
		expect(() => resolve({ nope: 1 })).toThrow(/messaging\.nope/);
		expect(new MessagingConfigError("x")).toBeInstanceOf(Error);
	});
});

function agentWith(messaging: unknown, registration: FauxProviderRegistration) {
	return new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		webSearch: false,
		jev: baselineJev(),
		permissionMode: "full",
		...(messaging === undefined ? {} : { messaging: messaging as never }),
	});
}

function faux(): FauxProviderRegistration {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "msg" }] });
	registrations.push(registration);
	return registration;
}

const toolNames = (agent: AutoRAGAgent): string[] =>
	(agent as unknown as { tools: { name: string }[] }).tools.map((t) => t.name);

describe("agent registration", () => {
	it("registers send_message only when a channel is configured", () => {
		expect(toolNames(agentWith(undefined, faux()))).not.toContain("send_message");
		expect(toolNames(agentWith({ channels: {} }, faux()))).not.toContain("send_message");
		const agent = agentWith({ channels: { ops: { kind: "echo", tokenEnv: "T" } } }, faux());
		expect(toolNames(agent)).toContain("send_message");
	});

	it("drops a caller tool that reuses the name, because the agent owns it", () => {
		const agent = new AutoRAGAgent({
			model: faux().getModel(),
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			webSearch: false,
			jev: baselineJev(),
			messaging: { channels: { ops: { kind: "echo", tokenEnv: "T" } } } as never,
			tools: [
				{
					name: "send_message",
					label: "x",
					description: "x",
					parameters: {} as never,
					execute: async () => ({ content: [], details: {} }),
				},
			],
		});
		expect((agent as unknown as { droppedCallerToolNames: string[] }).droppedCallerToolNames).toContain(
			"send_message",
		);
	});

	it("runs a real send through the agent's pi session when the model calls the tool", async () => {
		const registration = faux();
		registration.setResponses([
			fauxAssistantMessage([fauxToolCall("send_message", { channel: "ops", text: "deploy finished" })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage("sent it", { stopReason: "stop" }),
			fauxAssistantMessage("sent it", { stopReason: "stop" }),
		]);
		const agent = agentWith({ channels: { ops: { kind: "echo", tokenEnv: "T" } } }, registration);
		await agent.searchDocuments("tell the ops channel the deploy finished");
		expect(sent).toEqual([{ channel: "ops", text: "deploy finished" }]);
	});
});

describe("system prompt", () => {
	const base = { toolNames: ["bash", "send_message"] };

	it("lists the tool and states the sending rules when send_message is registered", () => {
		const prompt = buildSystemPrompt(base);
		expect(prompt).toContain("**send_message**");
		expect(prompt).toMatch(/only when the user asks/i);
		expect(prompt).toMatch(/never retry|do not (send|retry)/i);
	});

	it("says nothing about sending when the tool is absent", () => {
		expect(buildSystemPrompt({ toolNames: ["bash"] })).not.toContain("send_message");
	});
});

describe("sends are never unconfirmed by default", () => {
	const messaging = { channels: { ops: { kind: "echo", tokenEnv: "T" } } };
	const resolveWith = (opts: {
		defaultPermissionMode?: "full";
		flags?: Record<string, string>;
		env?: Record<string, string>;
		file?: unknown;
	}) =>
		resolveConfig({
			flags: { config: configFile(messaging), ...opts.flags },
			cwd: root,
			env: { HOME: root, ...opts.env },
			...(opts.defaultPermissionMode === undefined ? {} : { defaultPermissionMode: opts.defaultPermissionMode }),
		});

	it("offers send_message in an interactive session whose mode is auto by default", () => {
		expect(buildAgentOptions(resolveWith({})).messaging).toBeDefined();
	});

	it("withholds it where `full` is only the entry point's default (search, bridged TUI)", () => {
		const config = resolveWith({ defaultPermissionMode: "full" });
		expect(config.permissionMode).toBe("full");
		expect(buildAgentOptions(config).messaging).toBeUndefined();
	});

	it("offers it there once the user chose full themselves, by flag or environment", () => {
		const byFlag = resolveWith({ defaultPermissionMode: "full", flags: { "permission-mode": "full" } });
		expect(buildAgentOptions(byFlag).messaging).toBeDefined();
		const byEnv = resolveWith({ defaultPermissionMode: "full", env: { AUTORAG_PERMISSION_MODE: "full" } });
		expect(buildAgentOptions(byEnv).messaging).toBeDefined();
	});

	it("offers it there when the user asked for ask or auto", () => {
		const config = resolveWith({ defaultPermissionMode: "full", flags: { "permission-mode": "ask" } });
		expect(buildAgentOptions(config).messaging).toBeDefined();
	});
});
