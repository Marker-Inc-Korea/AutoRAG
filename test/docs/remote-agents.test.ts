import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { registeredMcpToolNames } from "../helpers/mcp-tool-names.ts";

const DOC = readFileSync(new URL("../../docs/remote-agents.md", import.meta.url), "utf8");

describe("docs/remote-agents.md matches the server", () => {
	it("names only tools the server really registers", async () => {
		const registered = await registeredMcpToolNames({ readOnly: false });
		const mentioned = [...new Set(DOC.match(/\bautorag\.[a-z]+(?:\.[a-z]+)?\b/g) ?? [])].filter(
			// `autorag.example.test` style hosts are not tool names.
			(name) => !name.endsWith(".test"),
		);
		expect(mentioned.length).toBeGreaterThan(0);
		for (const name of mentioned) expect(registered, name).toContain(name);
	});

	it("claims read-only exactly when the write tools are really absent", async () => {
		const readOnly = await registeredMcpToolNames({ readOnly: true });
		for (const writer of ["autorag.refresh", "autorag.report", "autorag.convert"]) {
			expect(DOC).toContain(writer);
			expect(readOnly, writer).not.toContain(writer);
		}
	});

	it("documents only flags the CLI accepts, and every flag it accepts", () => {
		const help = spawnSync(process.execPath, ["src/mcp/index.ts", "--help"], { encoding: "utf8" });
		expect(help.status).toBe(0);
		const accepted = new Set(help.stdout.match(/--[a-z][a-z-]+/g) ?? []);
		const documented = new Set(DOC.match(/(?<![\w-])--[a-z][a-z-]+/g) ?? []);
		// Flags of other tools the doc quotes in command examples.
		const foreign = new Set(["--url", "--header", "--bearer-token-env-var", "--transport"]);
		for (const flag of documented) if (!foreign.has(flag)) expect(accepted, flag).toContain(flag);
		for (const flag of accepted) if (flag !== "--help") expect(documented, flag).toContain(flag);
	});
});

describe("docs/remote-agents.md for Dazzi app users", () => {
	const section = (heading: string): string => {
		const start = DOC.indexOf(`## ${heading}`);
		if (start < 0) return "";
		const next = DOC.indexOf("\n## ", start + 1);
		return DOC.slice(start, next < 0 ? undefined : next);
	};

	it("has a Dazzi section that tells app users the app runs the server and the tunnel", () => {
		const text = section("Using the Dazzi app");
		expect(text).not.toBe("");
		expect(text).toMatch(/Settings/);
		expect(text).toMatch(/Cloud agents/);
		expect(text).toMatch(/tunnel/i);
		// The app does the work, so the section must not tell its reader to run the commands by hand.
		expect(text).not.toMatch(/^\s*(?:autorag-mcp|cloudflared)\b/m);
	});

	it("says this is not the app's remote access, which serves the Finder UI to a person", () => {
		const text = section("Using the Dazzi app");
		expect(text).toMatch(/Remote Access/);
		expect(text).toMatch(/separate|independent|not the same/i);
		expect(text).toMatch(/pairing code/i);
	});

	it("keeps the by-hand cloudflared steps, but labels them as for people not using the app", () => {
		const start = section("Start the server");
		expect(start).toMatch(/cloudflared tunnel/);
		expect(start).toMatch(/without the Dazzi app|not using the Dazzi app|if you do not use the Dazzi app/i);
	});

	it("warns that a Quick Tunnel address changes, which breaks an agent connected earlier", () => {
		expect(DOC).toMatch(/changes (?:on|every|each) (?:every )?(?:start|restart)/i);
		expect(DOC).toMatch(/own domain|named tunnel/i);
	});
});
