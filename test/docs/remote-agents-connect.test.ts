import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { registeredMcpToolNames } from "../helpers/mcp-tool-names.ts";

const AGENTS = ["muse", "grok-bot"] as const;

/** The prompt with its hard line wraps joined, so a rule is matched as a sentence and not by where the file breaks. */
function prompt(agent: string): string {
	return readFileSync(new URL(`../../plugins/remote-agents/connect/${agent}.md`, import.meta.url), "utf8").replace(
		/\s*\n\s*/g,
		" ",
	);
}

describe.each(AGENTS)("connect prompt for %s", (agent) => {
	const text = prompt(agent);

	it("has one URL placeholder the user must replace, pointing at the /mcp endpoint", () => {
		expect(text.match(/<YOUR_PUBLIC_URL>/g)?.length ?? 0).toBeGreaterThanOrEqual(1);
		expect(text).toContain("<YOUR_PUBLIC_URL>/mcp");
		expect(text).not.toMatch(/https?:\/\/(?!<)/);
	});

	it("never asks for, accepts or stores the owner token in chat", () => {
		expect(text).toMatch(/never (?:ask|request)[^.\n]*(?:token|password|key)/i);
		expect(text).toMatch(/do not (?:store|save|write)[^.\n]*(?:token|credential)/i);
		expect(text).not.toMatch(/Bearer\s+\S{8,}/);
		expect(text).not.toMatch(/AUTORAG_MCP_TOKEN\s*=/);
	});

	it("scopes the connection to read-only search and forbids scheduled or background capture of content", () => {
		expect(text).toMatch(/read-only/i);
		expect(text).toMatch(/do not (?:create|schedule)[^.\n]*(?:task|routine|schedule|recurring)/i);
		expect(text).toMatch(/do not (?:copy|save|retain|store)[^.\n]*(?:result|content|archive|file)/i);
	});

	it("tells the agent to stop and report if the sign-in or tool list is not what is described", () => {
		expect(text).toMatch(/autorag\.search/);
		expect(text).toMatch(/stop|report/i);
	});

	it("only names tools the server really registers", async () => {
		const registered = await registeredMcpToolNames({ readOnly: false });
		for (const name of new Set(text.match(/\bautorag\.[a-z]+(?:\.[a-z]+)?\b/g) ?? [])) {
			expect(registered, name).toContain(name);
		}
	});
});

describe("docs/remote-agents.md points at the prompts", () => {
	it("links every connect prompt that exists", () => {
		const doc = readFileSync(new URL("../../docs/remote-agents.md", import.meta.url), "utf8");
		for (const agent of AGENTS) expect(doc).toContain(`plugins/remote-agents/connect/${agent}.md`);
	});
});
