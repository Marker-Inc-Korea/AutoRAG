import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-discord-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function load(messaging: unknown) {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify({ searchPaths: [root], messaging }));
	return resolveConfig({ flags: { config: path }, cwd: root, env: { HOME: root } });
}

describe("discord in config.json", () => {
	it("is a built-in kind that reaches the agent options", () => {
		const options = buildAgentOptions(
			load({
				channels: { team: { kind: "discord", tokenEnv: "DISCORD_BOT_TOKEN", channelId: "1234567890123456789" } },
			}),
		);
		expect(options.messaging?.channels.team).toEqual({
			kind: "discord",
			tokenEnv: "DISCORD_BOT_TOKEN",
			channelId: "1234567890123456789",
		});
	});

	it("rejects a pasted bot token as a config error that never echoes it", () => {
		const attempt = () =>
			load({
				channels: {
					team: { kind: "discord", tokenEnv: "MTIz.secretDiscordToken", channelId: "1234567890123456789" },
				},
			});
		expect(attempt).toThrow(ConfigError);
		expect(attempt).not.toThrow(/secretDiscordToken/);
	});
});
