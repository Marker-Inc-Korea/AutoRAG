import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { configuredModelEnvNames, containerEnvironment, parseRunnerArgs } from "../../scripts/manual-qa/docker-qa.mjs";

const ENV_KEYS = ["QA_MODEL_ENV", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HOME"] as const;
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
	for (const key of ENV_KEYS) saved[key] = process.env[key];
});

afterEach(() => {
	for (const key of ENV_KEYS) {
		if (saved[key] === undefined) delete process.env[key];
		else process.env[key] = saved[key];
	}
});

function setEnv(entries: Record<string, string | undefined>): void {
	for (const [key, value] of Object.entries(entries)) {
		if (value === undefined) delete process.env[key];
		else process.env[key] = value;
	}
}

describe("parseRunnerArgs", () => {
	it("splits plain words", () => {
		expect(parseRunnerArgs("--evidence-dir /tmp/x --lane local")).toEqual([
			"--evidence-dir",
			"/tmp/x",
			"--lane",
			"local",
		]);
	});

	it("keeps quoted words as single argv entries", () => {
		expect(parseRunnerArgs("--evidence-dir '/tmp/my evidence' " + '"a b"')).toEqual([
			"--evidence-dir",
			"/tmp/my evidence",
			"a b",
		]);
	});

	it("produces an explicit empty argument from empty quotes", () => {
		expect(parseRunnerArgs('--flag "" --other x')).toEqual(["--flag", "", "--other", "x"]);
	});

	it("preserves shell metacharacters as data", () => {
		expect(parseRunnerArgs("$(whoami);rm -rf /")).toEqual(["$(whoami);rm", "-rf", "/"]);
	});

	it("rejects an unterminated quote", () => {
		expect(() => parseRunnerArgs("--evidence-dir '/tmp/oops")).toThrow("unterminated");
	});

	it("returns no words for empty input", () => {
		expect(parseRunnerArgs("   ")).toEqual([]);
	});
});

describe("configuredModelEnvNames", () => {
	it("falls back to the default allowlist when unset", () => {
		setEnv({ QA_MODEL_ENV: undefined });
		expect(configuredModelEnvNames()).toContain("OPENAI_API_KEY");
	});

	it("forwards nothing when the allowlist is explicitly empty", () => {
		setEnv({ QA_MODEL_ENV: "" });
		expect(configuredModelEnvNames()).toEqual([]);
		setEnv({ QA_MODEL_ENV: "   " });
		expect(configuredModelEnvNames()).toEqual([]);
	});

	it("splits a custom allowlist on whitespace", () => {
		setEnv({ QA_MODEL_ENV: "GEMINI_API_KEY  XAI_API_KEY" });
		expect(configuredModelEnvNames()).toEqual(["GEMINI_API_KEY", "XAI_API_KEY"]);
	});

	it("rejects names that are not valid environment variables", () => {
		setEnv({ QA_MODEL_ENV: "OPENAI-KEY XAI_API_KEY" });
		expect(() => configuredModelEnvNames()).toThrow("invalid names");
	});
});

describe("containerEnvironment", () => {
	it("forwards only allowlisted credentials that are actually set", () => {
		setEnv({
			QA_MODEL_ENV: "OPENAI_API_KEY ANTHROPIC_API_KEY",
			OPENAI_API_KEY: "sk-set",
			ANTHROPIC_API_KEY: undefined,
		});
		expect(containerEnvironment().modelEnv).toEqual(["OPENAI_API_KEY"]);
	});

	it("refuses to forward launcher-owned environment names", () => {
		setEnv({ QA_MODEL_ENV: "HOME AUTORAG_CONFIG", HOME: "/host" });
		expect(() => containerEnvironment()).toThrow("launcher-owned");
	});

	it("pins the fixed container home paths", () => {
		setEnv({ QA_MODEL_ENV: undefined });
		const fixed = Object.fromEntries(containerEnvironment().fixed);
		expect(fixed.HOME).toBe("/tmp/autorag-home");
		expect(fixed.AUTORAG_HOME).toBe("/tmp/autorag-home/.autorag");
		expect(fixed.AUTORAG_CONFIG).toBe("/tmp/autorag-home/.autorag/config.json");
	});
});
