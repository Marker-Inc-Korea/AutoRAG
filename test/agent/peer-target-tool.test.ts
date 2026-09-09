import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
const workspaces: string[] = [];

afterEach(() => {
	for (const workspace of workspaces.splice(0)) rmSync(workspace, { recursive: true, force: true });
});

function writePeerRegistry(workspace: string): void {
	writeFileSync(
		join(workspace, ".autorag", "p2p", "simplex-peers.json"),
		JSON.stringify({
			alice: {
				contactId: 42,
				addedAt: "2026-01-01T00:00:00.000Z",
				displayName: "Alice",
				description: "Finance and budget specialist",
				role: "finance lead",
			},
			bob: {
				contactId: 43,
				addedAt: "2026-01-01T00:00:00.000Z",
				displayName: "Bob",
				description: "Design documents",
			},
		}),
	);
}

function registeredTools(agent: AutoRAGAgent): readonly AgentTool[] {
	return (
		agent as unknown as {
			readonly innerAgent: { readonly state: { readonly tools: readonly AgentTool[] } };
		}
	).innerAgent.state.tools;
}

describe("peer persona target tool", () => {
	it("recommends local peers with matched terms and persona details", async () => {
		const workspace = mkdtempSync(join(tmpdir(), "autorag-peer-target-"));
		workspaces.push(workspace);
		mkdirSync(join(workspace, ".autorag", "p2p"), { recursive: true });
		writePeerRegistry(workspace);
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: workspace,
			memoryPath: join(workspace, "memory.json"),
			minSync: false,
			jikji: false,
		});
		const tool = registeredTools(agent).find((candidate) => candidate.name === "recommend_peer_targets");

		expect(tool).toBeDefined();
		const result = await tool?.execute("test-call", { query: "finance budget" });

		expect(result?.details).toEqual({
			method: "recommend_peer_targets",
			resultCount: 1,
			matches: [
				{
					alias: "alice",
					matchedTerms: ["budget", "finance"],
					displayName: "Alice",
					description: "Finance and budget specialist",
				},
			],
		});
	});

	it("is not registered for remote sessions", () => {
		const workspace = mkdtempSync(join(tmpdir(), "autorag-peer-target-"));
		workspaces.push(workspace);
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			workspacePath: workspace,
			memoryPath: join(workspace, "memory.json"),
			remoteSession: true,
			minSync: false,
			jikji: false,
		});

		expect(registeredTools(agent).map((tool) => tool.name)).not.toContain("recommend_peer_targets");
	});
});
