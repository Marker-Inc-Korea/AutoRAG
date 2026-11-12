import { afterEach, describe, expect, it } from "vitest";
import {
	channelKinds,
	createChannels,
	normalizeMessagingConfig,
	registerChannelKind,
	unregisterChannelKind,
} from "../../src/messaging/registry.ts";
import { assertEnvName } from "../../src/messaging/support.ts";
import { MessagingConfigError, MessagingError, type OutboundChannel } from "../../src/messaging/types.ts";

interface EchoConfig {
	readonly kind: "echo";
	readonly tokenEnv: string;
}

const sent: string[] = [];

function registerEcho(): void {
	registerChannelKind<EchoConfig>({
		kind: "echo",
		validate(raw, path) {
			for (const key of Object.keys(raw)) {
				if (key !== "kind" && key !== "tokenEnv")
					throw new MessagingConfigError(`${path}.${key} is not a recognized field`);
			}
			return { kind: "echo", tokenEnv: assertEnvName(raw.tokenEnv, `${path}.tokenEnv`) };
		},
		create(id, config): OutboundChannel {
			return {
				id,
				kind: "echo",
				addressing: "fixed",
				describe: () => "echo",
				async send(message) {
					sent.push(`${config.tokenEnv}:${message.text}`);
					return { destination: "echo", ids: ["1"] };
				},
			};
		},
	});
}

afterEach(() => {
	unregisterChannelKind("echo");
	sent.length = 0;
});

describe("assertEnvName", () => {
	it("accepts an environment variable name and rejects a value that looks like a secret", () => {
		expect(assertEnvName("TELEGRAM_BOT_TOKEN", "x")).toBe("TELEGRAM_BOT_TOKEN");
		expect(() => assertEnvName("123:abc-def", "x")).toThrow(MessagingConfigError);
		expect(() => assertEnvName("am_us_AbC123", "x")).toThrow(MessagingConfigError);
		expect(() => assertEnvName("lowercase_name", "x")).toThrow(/upper-case/);
		expect(() => assertEnvName("https://hooks.slack.com/services/T/B/x", "x")).toThrow(/environment variable name/);
		expect(() => assertEnvName(undefined, "x.tokenEnv")).toThrow(/x\.tokenEnv/);
	});
});

describe("normalizeMessagingConfig", () => {
	it("treats an absent section as no channels", () => {
		expect(normalizeMessagingConfig(undefined)).toEqual({ channels: {} });
		expect(normalizeMessagingConfig({})).toEqual({ channels: {} });
		expect(normalizeMessagingConfig({ channels: {} })).toEqual({ channels: {} });
	});

	it("rejects anything that is not an object, and unknown top-level fields", () => {
		expect(() => normalizeMessagingConfig([])).toThrow(MessagingConfigError);
		expect(() => normalizeMessagingConfig("x")).toThrow(MessagingConfigError);
		expect(() => normalizeMessagingConfig({ channel: {} })).toThrow(/messaging\.channel is not a recognized field/);
		expect(() => normalizeMessagingConfig({ channels: [] })).toThrow(/messaging\.channels must be an object/);
	});

	it("rejects a channel whose kind is missing or unknown, naming the kinds that exist", () => {
		registerEcho();
		expect(() => normalizeMessagingConfig({ channels: { a: {} } })).toThrow(/messaging\.channels\.a\.kind/);
		expect(() => normalizeMessagingConfig({ channels: { a: { kind: "fax" } } })).toThrow(/"fax".*echo/);
	});

	it("validates a channel with its kind's own rules and keeps the result", () => {
		registerEcho();
		expect(normalizeMessagingConfig({ channels: { ops: { kind: "echo", tokenEnv: "OPS_TOKEN" } } })).toEqual({
			channels: { ops: { kind: "echo", tokenEnv: "OPS_TOKEN" } },
		});
		expect(() =>
			normalizeMessagingConfig({ channels: { ops: { kind: "echo", tokenEnv: "OPS_TOKEN", extra: 1 } } }),
		).toThrow(/messaging\.channels\.ops\.extra/);
		expect(() => normalizeMessagingConfig({ channels: { ops: { kind: "echo" } } })).toThrow(
			/messaging\.channels\.ops\.tokenEnv/,
		);
	});

	it("restricts channel ids to short names the model can pass back verbatim", () => {
		registerEcho();
		const ok = { kind: "echo", tokenEnv: "T" };
		expect(() => normalizeMessagingConfig({ channels: { "team-chat_2": ok } })).not.toThrow();
		for (const bad of ["", "has space", "a/b", "x".repeat(41)]) {
			expect(() => normalizeMessagingConfig({ channels: { [bad]: ok } })).toThrow(/channel id/);
		}
	});
});

describe("channel kind registry", () => {
	it("refuses to register a kind twice", () => {
		registerEcho();
		expect(() => registerEcho()).toThrow(/already registered/);
		expect(channelKinds()).toContain("echo");
	});

	it("creates one channel per configured entry, keyed by id, and they send", async () => {
		registerEcho();
		const config = normalizeMessagingConfig({
			channels: { a: { kind: "echo", tokenEnv: "A_T" }, b: { kind: "echo", tokenEnv: "B_T" } },
		});
		const channels = createChannels(config, { env: {}, fetch });
		expect([...channels.keys()]).toEqual(["a", "b"]);
		await channels.get("b")?.send({ text: "hi" });
		expect(sent).toEqual(["B_T:hi"]);
	});

	it("fails clearly when a configured kind is no longer registered", () => {
		registerEcho();
		const config = normalizeMessagingConfig({ channels: { a: { kind: "echo", tokenEnv: "A_T" } } });
		unregisterChannelKind("echo");
		expect(() => createChannels(config, { env: {}, fetch })).toThrow(MessagingError);
	});
});
