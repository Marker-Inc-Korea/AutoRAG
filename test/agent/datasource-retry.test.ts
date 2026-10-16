import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { DatasourceIndexResult, DatasourceSkill } from "../../src/datasource/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-agent-datasource-retry-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function flaky(script: readonly DatasourceIndexResult[]): DatasourceSkill & { readonly calls: () => number } {
	let calls = 0;
	return {
		calls: () => calls,
		describe: () => ({
			name: "kakao",
			type: "chat",
			description: "KakaoTalk",
			capabilities: [],
			tags: ["kakao"],
			status: "active",
			datasourceId: "kakao",
			instanceId: "acct-1",
		}),
		polling: () => ({ mode: "none" }),
		skillManifest: () => ({ name: "datasource-kakao", description: "kakao", content: "kakao" }),
		index: async () => script[Math.min(calls++, script.length - 1)] as DatasourceIndexResult,
		retrievalMethods: () => [],
		describeSources: () => [],
	};
}

function failed(message: string): DatasourceIndexResult {
	return {
		ok: false,
		instanceId: "acct-1",
		skill: "kakao",
		indexedAt: 1,
		error: message,
		code: "datasource-unavailable",
		message,
		diagnostics: [
			{ code: "datasource-unavailable", severity: "warning", message, instanceId: "acct-1", source: "kakao" },
		],
	};
}

const SUCCESS: DatasourceIndexResult = {
	ok: true,
	instanceId: "acct-1",
	skill: "kakao",
	chunkCount: 5,
	indexedAt: 2,
	diagnostics: [],
};

function agentWith(skill: DatasourceSkill, maxRetries?: number): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: ["test/fixtures/sample-project"],
		workspacePath: tmpDir,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
		datasourceSkills: [skill],
		datasourceRetry: { baseDelayMs: 1, maxDelayMs: 2, ...(maxRetries === undefined ? {} : { maxRetries }) },
	});
}

describe("refresh datasource retry", () => {
	it("recovers a source that fails twice within one refresh", async () => {
		const skill = flaky([failed("timeout"), failed("timeout"), SUCCESS]);
		const agent = agentWith(skill);

		const refreshed = await agent.refresh(true);
		const status = await agent.getRefreshStatus();

		expect(skill.calls()).toBe(3);
		expect(status.components.datasources).toBe("configured");
		expect(JSON.stringify(refreshed)).toContain('"attempts":3');
	});

	it("fails the source after the backoff is exhausted and still indexes the next one", async () => {
		const bad = flaky([failed("timeout a"), failed("timeout b"), failed("timeout c"), SUCCESS]);
		const good = flaky([SUCCESS]);
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [bad, good],
			datasourceRetry: { baseDelayMs: 1, maxDelayMs: 2 },
		});

		const refreshed = await agent.refresh(true);
		const status = await agent.getRefreshStatus();
		const serialized = JSON.stringify(refreshed);

		expect(bad.calls()).toBe(3);
		expect(good.calls()).toBe(1);
		expect(status.components.datasources).toBe("degraded");
		expect(serialized).toContain('"attempts":3');
		expect(serialized).toContain("timeout c");
	});

	it("leaves a first-time success unchanged (no attempts field)", async () => {
		const skill = flaky([SUCCESS]);
		const agent = agentWith(skill);

		const refreshed = await agent.refresh(true);

		expect(skill.calls()).toBe(1);
		expect(JSON.stringify(refreshed)).not.toContain("attempts");
	});
});
