import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGRunLogger } from "../../src/observability/run-log.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-run-log-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("AutoRAGRunLogger", () => {
	it("writes the single model identity without child-agent fields", () => {
		const path = join(root, "runs.jsonl");
		new AutoRAGRunLogger(path).write({
			event: "search_started",
			timestamp: "2026-08-24T00:00:00.000Z",
			sessionId: "session",
			queryLength: 5,
			model: "single-agent",
		});
		const event = JSON.parse(readFileSync(path, "utf8"));
		expect(event.model).toBe("single-agent");
		expect(event.explorerModel).toBeUndefined();
		expect(event.successfulExplorerCalls).toBeUndefined();
		expect(event.subagentDispatchCount).toBeUndefined();
	});
});
