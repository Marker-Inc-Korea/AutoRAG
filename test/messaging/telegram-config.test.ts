import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-telegram-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function load(messaging: unknown) {
	const path = join(root, "config.json");
	writeFileSync(path, JSON.stringify({ searchPaths: [root], messaging }));
	return resolveConfig({ flags: { config: path }, cwd: root, env: { HOME: root } });
}

describe("telegram in config.json", () => {
	it("is a built-in kind that reaches the agent options", () => {
		const options = buildAgentOptions(
			load({ channels: { ops: { kind: "telegram", tokenEnv: "TELEGRAM_BOT_TOKEN", chatId: 42 } } }),
		);
		expect(options.messaging?.channels.ops).toEqual({
			kind: "telegram",
			tokenEnv: "TELEGRAM_BOT_TOKEN",
			chatId: "42",
		});
	});

	it("rejects a pasted bot token as a config error that never echoes it", () => {
		const attempt = () =>
			load({ channels: { ops: { kind: "telegram", tokenEnv: "123456:ABC-secretTOKEN", chatId: 42 } } });
		expect(attempt).toThrow(ConfigError);
		expect(attempt).not.toThrow(/secretTOKEN/);
	});
});
