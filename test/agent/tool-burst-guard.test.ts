import { describe, expect, it } from "vitest";
import { DEFAULT_TOOL_BURST_GUARD, ToolBurstGuard } from "../../src/agent/tool-burst-guard.ts";

describe("ToolBurstGuard", () => {
	it("trips on the call that reaches maxCalls inside the window", () => {
		const guard = new ToolBurstGuard({ maxCalls: 3, windowMs: 1_000 });
		expect(guard.record("bash", 0)).toBe(false);
		expect(guard.record("bash", 100)).toBe(false);
		expect(guard.record("bash", 200)).toBe(true);
	});

	it("never trips on a steady rate slower than the window allows", () => {
		const guard = new ToolBurstGuard({ maxCalls: 3, windowMs: 1_000 });
		for (let i = 0; i < 200; i++) expect(guard.record("bash", i * 600)).toBe(false);
	});

	it("drops a call that is exactly one window old", () => {
		const guard = new ToolBurstGuard({ maxCalls: 2, windowMs: 1_000 });
		expect(guard.record("bash", 0)).toBe(false);
		expect(guard.record("bash", 1_000)).toBe(false);
	});

	it("counts each tool separately", () => {
		const guard = new ToolBurstGuard({ maxCalls: 3, windowMs: 1_000 });
		expect(guard.record("bash", 0)).toBe(false);
		expect(guard.record("read", 1)).toBe(false);
		expect(guard.record("bash", 2)).toBe(false);
		expect(guard.record("read", 3)).toBe(false);
		expect(guard.record("bash", 4)).toBe(true);
	});

	it("keeps reporting every further call while the burst lasts", () => {
		const guard = new ToolBurstGuard({ maxCalls: 2, windowMs: 1_000 });
		expect(guard.record("bash", 0)).toBe(false);
		expect(guard.record("bash", 1)).toBe(true);
		expect(guard.record("bash", 2)).toBe(true);
	});

	it("stops reporting once the burst has aged out of the window", () => {
		const guard = new ToolBurstGuard({ maxCalls: 2, windowMs: 1_000 });
		guard.record("bash", 0);
		expect(guard.record("bash", 1)).toBe(true);
		expect(guard.record("bash", 5_000)).toBe(false);
	});

	it("starts clean after reset", () => {
		const guard = new ToolBurstGuard({ maxCalls: 2, windowMs: 1_000 });
		guard.record("bash", 0);
		expect(guard.record("bash", 1)).toBe(true);
		guard.reset();
		expect(guard.record("bash", 2)).toBe(false);
		expect(guard.record("bash", 3)).toBe(true);
	});

	it("never trips when disabled", () => {
		const guard = new ToolBurstGuard(false);
		for (let i = 0; i < 1_000; i++) expect(guard.record("bash", 0)).toBe(false);
	});

	it("defaults to 60 calls of one tool within 60 seconds", () => {
		expect(DEFAULT_TOOL_BURST_GUARD).toEqual({ maxCalls: 60, windowMs: 60_000 });
		const guard = new ToolBurstGuard();
		for (let i = 0; i < 59; i++) expect(guard.record("bash", i)).toBe(false);
		expect(guard.record("bash", 59)).toBe(true);
	});

	it.each([
		{ maxCalls: 0, windowMs: 1_000 },
		{ maxCalls: 1.5, windowMs: 1_000 },
		{ maxCalls: 3, windowMs: 0 },
		{ maxCalls: 3, windowMs: Number.POSITIVE_INFINITY },
	])("rejects invalid options %o", (options) => {
		expect(() => new ToolBurstGuard(options)).toThrow("toolCallBurstGuard");
	});
});
