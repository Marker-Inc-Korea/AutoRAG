import { visibleWidth } from "@earendil-works/pi-tui";
import { describe, expect, it } from "vitest";
import {
	formatElapsed,
	renderSearchProgress,
	type SearchProgressSnapshot,
	SearchProgressTracker,
} from "../../src/agent/search-progress.ts";

const plain = { accent: (s: string) => s, muted: (s: string) => s, success: (s: string) => s, error: (s: string) => s };

function clock(start = 1_000) {
	let now = start;
	return {
		now: () => now,
		advance: (ms: number) => {
			now += ms;
		},
	};
}

describe("formatElapsed", () => {
	it("shows whole seconds under a minute and m:ss after", () => {
		expect(formatElapsed(0)).toBe("0s");
		expect(formatElapsed(4_900)).toBe("4s");
		expect(formatElapsed(59_999)).toBe("59s");
		expect(formatElapsed(61_000)).toBe("1:01");
		expect(formatElapsed(600_000)).toBe("10:00");
	});
});

describe("SearchProgressTracker", () => {
	it("is inactive until a search begins, ignores steps then, and resets on every begin", () => {
		const time = clock();
		const tracker = new SearchProgressTracker(time.now);
		tracker.beginStep("route", "Routing");
		expect(tracker.snapshot().steps).toEqual([]);

		tracker.begin("question one");
		tracker.beginStep("route", "Routing");
		expect(tracker.snapshot()).toMatchObject({ active: true, query: "question one", modelStreaming: false });
		expect(tracker.snapshot().steps).toHaveLength(1);

		tracker.end();
		expect(tracker.snapshot().active).toBe(false);

		tracker.begin("question two");
		expect(tracker.snapshot().steps).toEqual([]);
		expect(tracker.snapshot().query).toBe("question two");
	});

	it("closes a step with its status, detail and end time", () => {
		const time = clock();
		const tracker = new SearchProgressTracker(time.now);
		tracker.begin("q");
		tracker.beginStep("route", "Routing");
		time.advance(2_500);
		tracker.endStep("route", "done", "local");
		expect(tracker.snapshot().steps[0]).toMatchObject({
			id: "route",
			status: "done",
			detail: "local",
			startedAt: 1_000,
			endedAt: 3_500,
		});
	});

	it("tracks concurrent work on one source as a single step that ends with the summed hit count", async () => {
		const tracker = new SearchProgressTracker(clock().now);
		tracker.begin("q");
		let releaseA: (value: string[]) => void = () => {};
		let releaseB: (value: string[]) => void = () => {};
		const a = tracker.track("minsync", "MinSync", new Promise<string[]>((r) => (releaseA = r)), {
			count: (hits) => hits.length,
		});
		const b = tracker.track("minsync", "MinSync", new Promise<string[]>((r) => (releaseB = r)), {
			count: (hits) => hits.length,
		});
		expect(tracker.snapshot().steps).toHaveLength(1);

		releaseA(["x", "y"]);
		await a;
		expect(tracker.snapshot().steps[0]?.status).toBe("running");

		releaseB(["z"]);
		await b;
		expect(tracker.snapshot().steps[0]).toMatchObject({ status: "done", detail: "3 hits" });
	});

	it("reports a failed source with the error text verbatim and still rethrows", async () => {
		const tracker = new SearchProgressTracker(clock().now);
		tracker.begin("q");
		await expect(
			tracker.track("jikji", "Jikji", Promise.reject(new Error("jikji exited 2: database is locked"))),
		).rejects.toThrow("database is locked");
		expect(tracker.snapshot().steps[0]).toMatchObject({
			status: "failed",
			detail: "jikji exited 2: database is locked",
		});
	});

	it("marks a source done when some queries failed but others returned hits", async () => {
		const tracker = new SearchProgressTracker(clock().now);
		tracker.begin("q");
		const ok = tracker.track("minsync", "MinSync", Promise.resolve(["a"]), { count: (hits) => hits.length });
		const bad = tracker.track("minsync", "MinSync", Promise.reject(new Error("boom")), {
			count: (hits: string[]) => hits.length,
		});
		await ok;
		await bad.catch(() => undefined);
		expect(tracker.snapshot().steps[0]).toMatchObject({ status: "done", detail: "1 hit, 1 failed: boom" });
	});

	it("drops a step that turned out not to belong to this search, and a late settle does not bring it back", async () => {
		const tracker = new SearchProgressTracker(clock().now);
		tracker.begin("q");
		let release: (value: string) => void = () => {};
		const work = tracker.step(
			"datasource-select",
			"Choosing datasources",
			() =>
				new Promise<string>((resolve) => {
					release = resolve;
				}),
		);
		tracker.beginStep("route", "Routing");
		expect(tracker.snapshot().steps.map((step) => step.id)).toEqual(["datasource-select", "route"]);

		tracker.discardStep("datasource-select");
		expect(tracker.snapshot().steps.map((step) => step.id)).toEqual(["route"]);

		release("none");
		await work;
		expect(tracker.snapshot().steps.map((step) => step.id)).toEqual(["route"]);

		tracker.discardStep("never-started");
		expect(tracker.snapshot().steps).toHaveLength(1);
	});

	it("notifies subscribers on every change and stops after unsubscribe", () => {
		const tracker = new SearchProgressTracker(clock().now);
		const seen: SearchProgressSnapshot[] = [];
		const unsubscribe = tracker.subscribe((snapshot) => seen.push(snapshot));
		tracker.begin("q");
		tracker.setModelStreaming(true);
		tracker.end();
		expect(seen.map((s) => [s.active, s.modelStreaming])).toEqual([
			[true, false],
			[true, true],
			[false, false],
		]);
		unsubscribe();
		tracker.begin("again");
		expect(seen).toHaveLength(3);
	});
});

describe("renderSearchProgress", () => {
	function snapshot(overrides: Partial<SearchProgressSnapshot> = {}): SearchProgressSnapshot {
		return { active: true, query: "q", startedAt: 0, modelStreaming: false, steps: [], ...overrides };
	}

	it("shows a header with elapsed time and the interrupt hint before any step exists", () => {
		const lines = renderSearchProgress(snapshot(), 7_200, plain, 80, "Esc");
		expect(lines).toHaveLength(1);
		expect(lines[0]).toContain("Searching");
		expect(lines[0]).toContain("7s");
		expect(lines[0]).toContain("Esc to stop");
	});

	it("omits the stop hint when no interrupt key is bound, instead of printing a dangling '· to stop'", () => {
		const [header] = renderSearchProgress(snapshot(), 2_000, plain, 80, "");
		expect(header).toContain("Searching… 2s");
		expect(header).not.toContain("to stop");
	});

	it("animates: the spinner frame changes with time so a stuck run is visibly different from a live one", () => {
		const first = renderSearchProgress(snapshot(), 0, plain, 80, "Esc")[0];
		const later = renderSearchProgress(snapshot(), 100, plain, 80, "Esc")[0];
		expect(first?.[0]).not.toBe(later?.[0]);
	});

	it("lists steps with a status icon, per-step elapsed time, and detail when finished", () => {
		const lines = renderSearchProgress(
			snapshot({
				steps: [
					{
						id: "route",
						label: "Routing the question",
						status: "done",
						detail: "local",
						startedAt: 0,
						endedAt: 1_000,
					},
					{ id: "minsync", label: "MinSync semantic search", status: "running", startedAt: 1_000 },
					{ id: "jikji", label: "Jikji", status: "failed", detail: "exit 2", startedAt: 1_000, endedAt: 2_000 },
				],
			}),
			4_000,
			plain,
			80,
			"Esc",
		);
		const frame = lines[0]?.[0] ?? "";
		expect(lines.slice(1)).toEqual([
			"  ✓ Routing the question — local · 1s",
			`  ${frame} MinSync semantic search · 3s`,
			"  ✗ Jikji — exit 2 · 1s",
		]);
	});

	it("drops the spinner and the interrupt hint while the model streams, since pi's own loader shows then", () => {
		const [header] = renderSearchProgress(snapshot({ modelStreaming: true }), 3_000, plain, 80, "Esc");
		expect(header).toContain("Searching");
		expect(header).toContain("3s");
		expect(header).not.toContain("Esc");
		expect(header?.trimStart()[0]).toBe("S");
	});
	it("never exceeds the width and keeps running steps when trimming to the line budget", () => {
		const steps = Array.from({ length: 12 }, (_, index) => ({
			id: `s${index}`,
			label: `Step ${index} ${"x".repeat(100)}`,
			status: index === 1 ? ("running" as const) : ("done" as const),
			startedAt: 0,
			endedAt: 1,
		}));
		const lines = renderSearchProgress(snapshot({ steps }), 5_000, plain, 40, "Esc");
		expect(lines.length).toBeLessThanOrEqual(8);
		expect(lines.every((line) => visibleWidth(line) <= 40)).toBe(true);
		expect(lines.some((line) => line.includes("Step 1 "))).toBe(true);
		expect(lines.some((line) => line.includes("Step 11 "))).toBe(true);
	});
});
