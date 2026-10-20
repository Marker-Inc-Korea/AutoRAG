import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createAutoRAGPiInteractiveRuntime } from "../../src/agent/pi-session.ts";
import { readDazziLoginToken } from "../../src/cloud/credential.ts";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";

const LOGIN_TOKEN = "dz_login_token_0123456789";

let root: string;
let agentDir: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-pi-dazzi-auth-"));
	agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	// What `/login` stores for Dazzi Models: an OAuth entry with no refresh token and no expiry.
	writeFileSync(
		join(agentDir, "auth.json"),
		JSON.stringify({
			dazzi: { type: "oauth", access: LOGIN_TOKEN, refresh: "", expires: Number.MAX_SAFE_INTEGER },
		}),
	);
	// The catalog snapshot `/login` persisted, so the provider has a model to select offline.
	const models = mapServerModels({
		object: "list",
		data: [
			{
				id: "vendor/chat",
				name: "Vendor Chat",
				context_window: 100_000,
				max_output_tokens: 4_096,
				input_modalities: ["text"],
				reasoning: false,
				pricing: { input: 1, output: 2, cache_read: 0, cache_write: 0 },
			},
		],
	});
	writeFileSync(
		join(agentDir, "models-store.json"),
		JSON.stringify({ dazzi: buildStoreEntry(models, "dazzi", "https://api.dazziapp.com/v1") }),
	);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** The model the CLI resolves for a signed-in Dazzi user, carrying the login token as its apiKey. */
function dazziModel(): Model<Api> {
	return {
		id: "vendor/chat",
		name: "Vendor Chat",
		api: "openai-responses",
		provider: "dazzi",
		baseUrl: "https://api.dazziapp.com/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 100_000,
		maxTokens: 4_096,
	};
}

describe("pi session with a Dazzi Models /login credential", () => {
	it("still resolves the stored login token when the CLI forwards it as the session apiKey", async () => {
		expect(readDazziLoginToken(agentDir)).toBe(LOGIN_TOKEN);

		const hosted = await createAutoRAGPiInteractiveRuntime({
			cwd: root,
			agentDir,
			sessionDir: join(root, "sessions"),
			model: dazziModel(),
			// resolveAgentModel hands the login token back as both `apiKey` and `providerApiKeys.dazzi`.
			apiKey: LOGIN_TOKEN,
			providerApiKeys: { dazzi: LOGIN_TOKEN },
			getSystemPrompt: () => "prompt",
			customTools: [],
			onInput: () => ({ kind: "handled" }),
		});
		try {
			const session = hosted.runtime.session;
			const model = session.model;
			expect(model?.provider).toBe("dazzi");
			// Pi's request path asks the runtime for auth; undefined here is the "No API key found for dazzi" failure.
			const auth = await session.modelRuntime.getAuth(model as Model<Api>);
			expect(auth?.auth.apiKey).toBe(LOGIN_TOKEN);
		} finally {
			await hosted.dispose();
		}
	});
});
