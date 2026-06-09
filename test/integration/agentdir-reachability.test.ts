import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { assertNoSourcePath } from "../../src/agentdir/assert-no-source-path.ts";
import { createAgentdirTools } from "../../src/agentdir/tools.ts";
import { bootstrapMappings, clearWorkspaceCache, getWorkspace } from "../../src/agentdir/workspace.ts";

let root: string;
let source: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-reach-"));
	clearWorkspaceCache();
	source = join(root, "docs");
	mkdirSync(join(source, "sub"), { recursive: true });
	writeFileSync(join(source, "sub", "report.md"), "quarterly revenue grew\nfindings here\n");
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

async function call(tool: AgentTool, params: Record<string, unknown>) {
	return tool.execute("test-call", params as never, undefined, undefined as never);
}

function textOf(res: { content: Array<{ type: string; text?: string }> }): string {
	return res.content
		.filter((c) => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text as string)
		.join("\n");
}

describe("closed-world reachability (Architect T1 / AC-4 + AC-6)", () => {
	it("every grep hit's virtual path is subsequently readable with no source leak", async () => {
		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [source]);
		const tools = new Map(createAgentdirTools(ws).map((t) => [t.name, t]));

		const grepRes = await call(tools.get("grep")!, { pattern: "revenue" });
		const sources = (grepRes.details as { sources: string[] }).sources;
		expect(sources.length).toBeGreaterThan(0);
		expect(sources).toContain("/docs/sub/report.md");

		// Every virtual path returned by grep must be readable via the read tool.
		for (const virtualPath of sources) {
			const readRes = await call(tools.get("read")!, { path: virtualPath });
			expect((readRes.details as { resultCount: number }).resultCount).toBe(1);
			expect(textOf(readRes).length).toBeGreaterThan(0);
			// No source filesystem path leaks anywhere in the read result.
			expect(() => assertNoSourcePath(readRes, [source, root])).not.toThrow();
		}
	});
});
