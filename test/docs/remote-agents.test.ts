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
