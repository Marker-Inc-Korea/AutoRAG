import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, getCurrentSystemPrompt } from "@earendil-works/pi-ai";
import { type FauxProviderRegistration, registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";

interface SeenRequest {
	systemPrompt: string;
	nonSystemMessages: string;
}

let root: string;
let identityPath: string;
const cleanups: Array<() => Promise<void> | void> = [];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-identity-session-"));
	const home = join(root, "home");
	const autoragHome = join(root, "autorag-home");
	const agentDir = join(root, "pi-agent");
	for (const directory of [home, autoragHome, agentDir, join(root, "workspace")]) {
		mkdirSync(directory, { recursive: true });
	}
	vi.stubEnv("HOME", home);
	vi.stubEnv("USERPROFILE", home);
	vi.stubEnv("AUTORAG_HOME", autoragHome);
	vi.stubEnv("AUTORAG_CONFIG", join(autoragHome, "config.json"));
	vi.stubEnv("PI_CODING_AGENT_DIR", agentDir);
	identityPath = join(autoragHome, "IDENTITY.md");
});

afterEach(async () => {
	try {
		for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
	} finally {
		vi.unstubAllEnvs();
		rmSync(root, { recursive: true, force: true });
	}
});

function provider(requestCount: number): { registration: FauxProviderRegistration; seen: SeenRequest[] } {
	const registration = registerFauxProvider({
		api: `faux-identity-${randomUUID()}`,
		models: [
			{ id: "identity-first", contextWindow: 262144 },
			{ id: "identity-second", contextWindow: 262144 },
		],
	});
	cleanups.push(() => registration.unregister());
	const seen: SeenRequest[] = [];
	registration.setResponses(
		Array.from({ length: requestCount }, () => (context) => {
			seen.push({
				// Pi persists system-message deltas; replay them as the provider does.
				systemPrompt: getCurrentSystemPrompt(context.messages),
				nonSystemMessages: JSON.stringify(context.messages.filter((message) => message.role !== "system")),
			});
			return fauxAssistantMessage("Synthetic answer.");
		}),
	);
	return { registration, seen };
}

function createAgent(registration: FauxProviderRegistration, options: Partial<AutoRAGAgentOptions> = {}) {
	return new AutoRAGAgent({
		model: registration.getModel(),
		apiKey: "test-key",
		workspacePath: join(root, "workspace"),
		searchPaths: [join(root, "workspace")],
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		memoryEmbedder: false,
		jev: false,
		piAgentDir: join(root, "pi-agent"),
		piSessionDir: join(root, "sessions"),
		piExtensionPaths: [],
		persistPiSessions: false,
		...options,
	});
}

async function interactiveAgent(registration: FauxProviderRegistration, options: Partial<AutoRAGAgentOptions> = {}) {
	const agent = createAgent(registration, options);
	const hosted = await agent.createPiInteractiveRuntime();
	let disposed = false;
	const dispose = async () => {
		if (disposed) return;
		disposed = true;
		await hosted.dispose();
	};
	cleanups.push(dispose);
	await hosted.runtime.session.bindExtensions({});
	return { runtime: hosted.runtime, dispose };
}

function occurrences(text: string, marker: string): number {
	return text.split(marker).length - 1;
}

describe("IDENTITY.md in actual Pi model requests", () => {
	it("includes identity once in both answer and verification requests of a non-interactive search", async () => {
		const raw = "IDENTITY_SEARCH_1930\nPreference: concise Mandarin answers.\n";
		writeFileSync(identityPath, raw);
		const { registration, seen } = provider(2);
		const agent = createAgent(registration);

		// No bound TUI: searchDocuments creates its own Pi session. With Jev disabled,
		// the first answer always proceeds to verification; the suite's MinSync is inert.
		const response = await agent.searchDocuments("Synthetic question");

		expect(response.answer).toBe("Synthetic answer.");
		expect(seen).toHaveLength(2);
		for (const request of seen) {
			expect(request.systemPrompt).toContain(raw);
			expect(occurrences(request.systemPrompt, "IDENTITY_SEARCH_1930")).toBe(1);
			expect(request.nonSystemMessages).not.toContain("IDENTITY_SEARCH_1930");
		}
	});

	it("preserves raw identity once on initial and follow-up requests without adding conversation messages", async () => {
		const raw = "\r\n# IDENTITY_RAW_1930\r\n姓名：小张  \r\nPrefer concise replies.\r\n\r\n";
		writeFileSync(identityPath, raw);
		const { registration, seen } = provider(2);
		const { runtime } = await interactiveAgent(registration);

		await runtime.session.prompt("First synthetic question", { source: "extension" });
		await runtime.session.prompt("Follow-up synthetic question", { source: "extension" });

		expect(seen).toHaveLength(2);
		for (const request of seen) {
			expect(request.systemPrompt).toContain(raw);
			expect(occurrences(request.systemPrompt, "IDENTITY_RAW_1930")).toBe(1);
			expect(request.nonSystemMessages).not.toContain("IDENTITY_RAW_1930");
		}
	});

	it("keeps the agent's identity snapshot after editing the file, switching models, and starting a new session", async () => {
		const original = "IDENTITY_FROZEN_1930\nPrefer Mandarin.\n";
		writeFileSync(identityPath, original);
		const { registration, seen } = provider(3);
		const { runtime } = await interactiveAgent(registration);
		await runtime.session.prompt("Before file edit", { source: "extension" });

		writeFileSync(identityPath, "IDENTITY_EDITED_1930\nPrefer English.\n");
		const nextModel = registration.getModel("identity-second");
		if (nextModel === undefined) throw new Error("expected the second faux model");
		await runtime.session.setModel(nextModel);
		await runtime.session.prompt("After model switch", { source: "extension" });
		expect((await runtime.newSession()).cancelled).toBe(false);
		await runtime.session.bindExtensions({});
		await runtime.session.prompt("After new session", { source: "extension" });

		expect(seen).toHaveLength(3);
		for (const request of seen) {
			expect(request.systemPrompt).toContain(original);
			expect(occurrences(request.systemPrompt, "IDENTITY_FROZEN_1930")).toBe(1);
			expect(request.systemPrompt).not.toContain("IDENTITY_EDITED_1930");
		}
	});

	it("uses fresh identity when resuming in a new agent and removes it from requests without erasing saved history", async () => {
		const firstIdentity = "IDENTITY_SNAPSHOT_1930_A\nPreference A.\n";
		const secondIdentity = "IDENTITY_SNAPSHOT_1930_B\nPreference B.\n";
		writeFileSync(identityPath, firstIdentity);
		const { registration, seen } = provider(3);
		const first = await interactiveAgent(registration, { persistPiSessions: true });
		await first.runtime.session.prompt("Original session", { source: "extension" });
		const sessionFile = first.runtime.session.sessionFile;
		if (sessionFile === undefined) throw new Error("expected a persisted Pi session");
		await first.dispose();
		expect(readFileSync(sessionFile, "utf8")).toContain("IDENTITY_SNAPSHOT_1930_A");

		writeFileSync(identityPath, secondIdentity);
		const second = await interactiveAgent(registration, { persistPiSessions: true });
		expect((await second.runtime.switchSession(sessionFile)).cancelled).toBe(false);
		await second.runtime.session.bindExtensions({});
		await second.runtime.session.prompt("Resume with updated identity", { source: "extension" });
		await second.dispose();

		rmSync(identityPath);
		const third = await interactiveAgent(registration, { persistPiSessions: true });
		expect((await third.runtime.switchSession(sessionFile)).cancelled).toBe(false);
		await third.runtime.session.bindExtensions({});
		await third.runtime.session.prompt("Resume without identity", { source: "extension" });
		await third.dispose();

		expect(seen.map((request) => occurrences(request.systemPrompt, "IDENTITY_SNAPSHOT_1930"))).toEqual([1, 1, 0]);
		expect(seen[0].systemPrompt).toContain(firstIdentity);
		expect(seen[1].systemPrompt).toContain(secondIdentity);
		expect(seen[1].systemPrompt).not.toContain(firstIdentity);
		expect(seen.every((request) => !request.nonSystemMessages.includes("IDENTITY_SNAPSHOT_1930"))).toBe(true);
		// Removing the source affects future requests; historical session files remain on disk.
		expect(readFileSync(sessionFile, "utf8")).toContain("IDENTITY_SNAPSHOT_1930_A");
	});

	it("passes a large identity through to the faux provider without truncating its contents", async () => {
		const raw = `IDENTITY_LARGE_START_1930\n${"Synthetic preference paragraph.\n".repeat(6000)}IDENTITY_LARGE_END_1930\n`;
		writeFileSync(identityPath, raw);
		const { registration, seen } = provider(1);
		const { runtime } = await interactiveAgent(registration);

		await runtime.session.prompt("Synthetic question", { source: "extension" });

		expect(seen).toHaveLength(1);
		expect(seen[0].systemPrompt).toContain(raw);
		expect(occurrences(seen[0].systemPrompt, "IDENTITY_LARGE_START_1930")).toBe(1);
		expect(occurrences(seen[0].systemPrompt, "IDENTITY_LARGE_END_1930")).toBe(1);
	});

	it("omits identity from real model requests when the caller explicitly disables loading", async () => {
		writeFileSync(identityPath, "IDENTITY_OPT_OUT_1930\nPrivate owner context.\n");
		const { registration, seen } = provider(1);
		const { runtime } = await interactiveAgent(registration, { loadIdentity: false });

		await runtime.session.prompt("Synthetic question", { source: "extension" });

		expect(seen).toHaveLength(1);
		expect(seen[0].systemPrompt).not.toContain("IDENTITY_OPT_OUT_1930");
		expect(seen[0].nonSystemMessages).not.toContain("IDENTITY_OPT_OUT_1930");
	});
});
