import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-slack-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function load(messaging: unknown) {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify({ searchPaths: [root], messaging }));
	return resolveConfig({ flags: { config: path }, cwd: root, env: { HOME: root } });
}

describe("slack in config.json", () => {
	it("is a built-in kind that reaches the agent options", () => {
		const options = buildAgentOptions(
			load({ channels: { eng: { kind: "slack", webhookEnv: "SLACK_WEBHOOK_URL" } } }),
		);
		expect(options.messaging?.channels.eng).toEqual({ kind: "slack", webhookEnv: "SLACK_WEBHOOK_URL" });
	});

	it("rejects a pasted webhook URL as a config error that never echoes it", () => {
		const attempt = () =>
			load({
				channels: { eng: { kind: "slack", webhookEnv: "https://hooks.slack.com/services/T/B/secretWEBHOOKpath" } },
			});
		expect(attempt).toThrow(ConfigError);
		expect(attempt).not.toThrow(/secretWEBHOOKpath/);
	});
});
