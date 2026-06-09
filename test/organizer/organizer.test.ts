import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { discoverAgents, findAgent } from "../../src/organizer/agents.ts";
import { createOrganizeTool, runOrganizer } from "../../src/organizer/organize-tool.ts";

let cwd: string;

beforeEach(() => {
	cwd = mkdtempSync(join(tmpdir(), "autorag-org-"));
	delete process.env.AUTORAG_E2E_SPAWN;
});

afterEach(() => {
	rmSync(cwd, { recursive: true, force: true });
	delete process.env.AUTORAG_E2E_SPAWN;
});

async function call(tool: AgentTool, params: Record<string, unknown>) {
	return tool.execute("test-call", params as never, undefined, undefined as never);
}

describe("organizer agent discovery (AC-8)", () => {
	it("discovers the bundled organizer agent definition", () => {
		const agents = discoverAgents(cwd);
		const organizer = agents.find((a) => a.name === "organizer");
		expect(organizer).toBeDefined();
		expect(organizer?.source).toBe("bundled");
		expect(organizer?.description.length).toBeGreaterThan(0);
		expect(organizer?.systemPrompt).toContain("virtual");
	});

	it("lets a project definition override the bundled one by name", () => {
		const agentsDir = join(cwd, ".autorag", "agents");
		mkdirSync(agentsDir, { recursive: true });
		writeFileSync(
			join(agentsDir, "organizer.md"),
			"---\nname: organizer\ndescription: project override\n---\nProject organizer body.\n",
		);
		const organizer = findAgent(cwd, "organizer");
		expect(organizer?.source).toBe("project");
		expect(organizer?.description).toBe("project override");
	});
});

describe("organize delegation tool (AC-8, skeleton)", () => {
	it("is registered with a well-formed, spawn-tolerant result when delegation is disabled", async () => {
		const tool = createOrganizeTool(() => cwd);
		expect(tool.name).toBe("organize");
		const res = await call(tool, { task: "group all PDFs under /reports" });
		const details = res.details as { status: string; reason?: string };
		expect(details.status).toBe("unavailable");
		expect(details.reason).toContain("AUTORAG_E2E_SPAWN");
		expect(res.content[0]).toMatchObject({ type: "text" });
	});

	it("runOrganizer never throws and returns a structured status", async () => {
		const result = await runOrganizer(cwd, "do nothing yet");
		expect(["delegated", "unavailable"]).toContain(result.status);
	});
});
