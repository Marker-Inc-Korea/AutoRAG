import { describe, expect, it, vi } from "vitest";
import { DEFAULT_DATASOURCE_RETRY_POLICY, datasourceRetryDelayMs, indexWithRetry } from "../../src/datasource/retry.ts";
import type {
	DatasourceIndexFail,
	DatasourceIndexOk,
	DatasourceIndexResult,
	DatasourceSkill,
} from "../../src/datasource/types.ts";

function ok(): DatasourceIndexOk {
	return { ok: true, instanceId: "acct", skill: "demo", chunkCount: 3, indexedAt: 1, diagnostics: [] };
}

function fail(message: string): DatasourceIndexFail {
	return {
		ok: false,
		instanceId: "acct",
		skill: "demo",
		indexedAt: 1,
		diagnostics: [
			{ code: "datasource-unavailable", severity: "warning", message, instanceId: "acct", source: "demo" },
		],
		error: message,
		code: "datasource-unavailable",
		message,
	};
}

/** Skill whose `index()` plays back `script`; a function entry is thrown-through. */
function scripted(script: readonly (DatasourceIndexResult | Error)[]): DatasourceSkill & { calls: number } {
	const state = { calls: 0 };
	const skill = {
		get calls() {
			return state.calls;
		},
		describe: () => ({
			name: "demo",
			type: "chat",
			description: "demo",
			capabilities: [],
			tags: [],
			status: "active" as const,
			instanceId: "acct",
		}),
		polling: () => ({ mode: "none" as const }),
		async index(): Promise<DatasourceIndexResult> {
			const next = script[Math.min(state.calls, script.length - 1)];
			state.calls += 1;
			if (next instanceof Error) throw next;
			if (next === undefined) throw new Error("empty script");
			return next;
		},
		retrievalMethods: () => [],
		describeSources: () => [],
		skillManifest: () => ({ name: "datasource-demo", description: "demo", content: "demo" }),
	};
	return skill as unknown as DatasourceSkill & { calls: number };
}

function recorder(): { sleeps: number[]; sleep: (ms: number) => Promise<void> } {
	const sleeps: number[] = [];
	return {
		sleeps,
		sleep: async (ms) => {
			sleeps.push(ms);
		},
	};
}

describe("datasourceRetryDelayMs", () => {
	it("doubles per retry within a jitter band and caps at maxDelayMs", () => {
		const policy = { ...DEFAULT_DATASOURCE_RETRY_POLICY, baseDelayMs: 1000, maxDelayMs: 3000 };
		// random() = 0 -> lower edge (half of the exponential step); 1 -> upper edge.
		expect(datasourceRetryDelayMs(1, { ...policy, random: () => 0 })).toBe(500);
		expect(datasourceRetryDelayMs(1, { ...policy, random: () => 1 })).toBe(1000);
		expect(datasourceRetryDelayMs(2, { ...policy, random: () => 1 })).toBe(2000);
		expect(datasourceRetryDelayMs(3, { ...policy, random: () => 1 })).toBe(3000);
		expect(datasourceRetryDelayMs(10, { ...policy, random: () => 1 })).toBe(3000);
	});
});

describe("indexWithRetry", () => {
	it("returns a first-time success untouched, with no attempts field and no sleep", async () => {
		const skill = scripted([ok()]);
		const { sleep, sleeps } = recorder();

		const result = await indexWithRetry(skill, { sleep });

		expect(result).toEqual(ok());
		expect(skill.calls).toBe(1);
		expect(sleeps).toEqual([]);
	});

	it("retries failures twice then succeeds, reporting total attempts and the last error", async () => {
		const skill = scripted([fail("blip 1"), fail("blip 2"), ok()]);
		const { sleep, sleeps } = recorder();

		const result = await indexWithRetry(skill, { sleep, random: () => 1 });

		expect(skill.calls).toBe(3);
		expect(result).toMatchObject({ ok: true, attempts: 3, chunkCount: 3 });
		expect(sleeps).toEqual([1000, 2000]);
		const recovered = result.diagnostics.find((d) => d.severity === "info");
		expect(recovered?.message).toContain("3 attempts");
		expect(recovered?.message).toContain("blip 2");
	});

	it("gives up after two retries and surfaces attempts and the last error", async () => {
		const skill = scripted([fail("e1"), fail("e2"), fail("e3"), ok()]);
		const { sleep, sleeps } = recorder();

		const result = await indexWithRetry(skill, { sleep });

		expect(skill.calls).toBe(3);
		expect(sleeps).toHaveLength(2);
		expect(result).toMatchObject({ ok: false, attempts: 3 });
		expect(result.ok === false && result.message).toContain("e3");
		expect(result.ok === false && result.message).toContain("3 attempts");
		expect(result.diagnostics.at(-1)?.message).toContain("3 attempts");
	});

	it("retries a thrown index() and reports it as datasource-index-failed once exhausted", async () => {
		const skill = scripted([new Error("kaboom")]);
		const { sleep, sleeps } = recorder();

		const result = await indexWithRetry(skill, { sleep });

		expect(skill.calls).toBe(3);
		expect(sleeps).toHaveLength(2);
		expect(result).toMatchObject({
			ok: false,
			code: "datasource-index-failed",
			attempts: 3,
			instanceId: "acct",
			skill: "demo",
		});
		expect(result.ok === false && result.message).toContain("kaboom");
	});

	it("honors maxRetries: 0 as retry-disabled", async () => {
		const skill = scripted([fail("blip"), ok()]);
		const { sleep, sleeps } = recorder();

		const result = await indexWithRetry(skill, { sleep, maxRetries: 0 });

		expect(skill.calls).toBe(1);
		expect(sleeps).toEqual([]);
		expect(result).toMatchObject({ ok: false, attempts: 1 });
	});

	it("rethrows an abort during the backoff wait instead of retrying or recording a failure", async () => {
		const skill = scripted([fail("blip"), ok()]);
		const controller = new AbortController();

		const pending = indexWithRetry(skill, { baseDelayMs: 60_000, maxDelayMs: 60_000 }, controller.signal);
		await vi.waitFor(() => expect(skill.calls).toBe(1));
		controller.abort(new Error("cancelled"));

		await expect(pending).rejects.toThrow("cancelled");
		expect(skill.calls).toBe(1);
	});

	it("does not record a datasource failure when index() throws because the refresh was aborted", async () => {
		const controller = new AbortController();
		const skill = scripted([new Error("spawn killed")]);
		controller.abort(new Error("cancelled"));

		await expect(indexWithRetry(skill, { sleep: async () => undefined }, controller.signal)).rejects.toThrow(
			"cancelled",
		);
		expect(skill.calls).toBe(1);
	});
});
