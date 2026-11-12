import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";
import { buildChannels, normalizeMessagingConfig } from "../../src/messaging/index.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-agentmail-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

const mail = { kind: "agentmail", apiKeyEnv: "AGENTMAIL_API_KEY", inboxId: "my-agent@agentmail.to" };

function load(messaging: unknown) {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify({ searchPaths: [root], messaging }));
	return resolveConfig({ flags: { config: path }, cwd: root, env: { HOME: root } });
}

describe("agentmail in config.json", () => {
	it("is a built-in kind: the config loads and reaches the agent options", () => {
		const options = buildAgentOptions(load({ channels: { mail } }));
		expect(options.messaging?.channels.mail).toMatchObject({
			kind: "agentmail",
			inboxId: "my-agent@agentmail.to",
			region: "us",
		});
	});

	it("builds a recipient channel from the loaded config", () => {
		const channels = buildChannels(normalizeMessagingConfig({ channels: { mail } }), { env: {}, fetch });
		expect(channels.get("mail")?.addressing).toBe("recipient");
	});

	it("rejects a pasted key as a config error that never echoes it", () => {
		const attempt = () => load({ channels: { mail: { ...mail, apiKeyEnv: "am_us_SECRET123" } } });
		expect(attempt).toThrow(ConfigError);
		expect(attempt).toThrow(/messaging\.channels\.mail\.apiKeyEnv/);
		expect(attempt).not.toThrow(/SECRET123/);
	});
});
