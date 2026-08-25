import { cpSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.js";

const temporaryRoots: string[] = [];

function temporaryRoot(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-scope-"));
	temporaryRoots.push(root);
	return root;
}

afterEach(() => {
	for (const root of temporaryRoots.splice(0)) {
		rmSync(root, { recursive: true, force: true });
	}
});

describe("AutoRAG retrieval scope flow", () => {
	it("searches through the configured symlink and canonical physical roots", async () => {
		const root = temporaryRoot();
		const source = join(root, "real-docs");
		const link = join(root, "docs-link");
		const workspace = join(root, "workspace");
		mkdirSync(source, { recursive: true });
		mkdirSync(workspace, { recursive: true });
		writeFileSync(join(source, "chargebacks.txt"), "Chargeback evidence requires the payment receipt.");
		symlinkSync(source, link, process.platform === "win32" ? "junction" : "dir");

		const agent = new AutoRAGAgent({
			searchPaths: [link],
			workspacePath: workspace,
			bm25: { forceEngine: "typescript-fallback" },
			minSync: false,
		});
		await agent.refresh();

		const throughLink = await agent.retrieve("chargeback evidence", { scope: link });
		const throughCanonicalRoot = await agent.retrieve("chargeback evidence", { scope: source });
		const merged = await agent.searchAllDocuments("chargeback evidence", { scope: link });

		expect(throughLink).toHaveLength(1);
		expect(throughCanonicalRoot).toHaveLength(1);
		expect(merged.results).toHaveLength(1);
		expect(throughLink[0]?.source).toBe(realpathSync(join(source, "chargebacks.txt")));
	});

	it("keeps the prepared virtual root after a single-root workspace relocation", async () => {
		const root = temporaryRoot();
		const prepared = join(root, "organized");
		const relocated = join(root, "agentdir", "workspace");
		mkdirSync(prepared, { recursive: true });
		writeFileSync(join(prepared, "policy.txt"), "The retention policy is seven years.");

		const preparingAgent = new AutoRAGAgent({
			searchPaths: [prepared],
			workspacePath: prepared,
			bm25: { forceEngine: "typescript-fallback" },
			minSync: false,
		});
		await preparingAgent.refresh();
		cpSync(prepared, relocated, { recursive: true });

		const relocatedAgent = new AutoRAGAgent({
			searchPaths: [relocated],
			workspacePath: relocated,
			bm25: { forceEngine: "typescript-fallback" },
			minSync: false,
		});
		const results = await relocatedAgent.retrieve("retention policy", { scope: relocated });

		expect(results).toHaveLength(1);
		expect(results[0]?.source).toBe(realpathSync(join(prepared, "policy.txt")));
	});

	it("rejects an unknown physical root before retrieval", async () => {
		const root = temporaryRoot();
		const source = join(root, "docs");
		const workspace = join(root, "workspace");
		const unknown = join(root, "outside");
		mkdirSync(source, { recursive: true });
		mkdirSync(workspace, { recursive: true });
		mkdirSync(unknown, { recursive: true });
		writeFileSync(join(source, "policy.txt"), "The retention policy is seven years.");

		const agent = new AutoRAGAgent({
			searchPaths: [source],
			workspacePath: workspace,
			bm25: { forceEngine: "typescript-fallback" },
			minSync: false,
		});
		await agent.refresh();

		await expect(agent.retrieve("retention policy", { scope: unknown })).rejects.toMatchObject({
			code: "invalid-retrieval-scope",
		});
	});
});
