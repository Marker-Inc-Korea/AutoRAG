import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SettingsManager } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { resolveInteractiveTuiMode } from "../../src/agent/pi-session.ts";

let root: string;
let agentDir: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-tui-mode-"));
	agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function managerWith(global?: Record<string, unknown>, project?: Record<string, unknown>): SettingsManager {
	if (global !== undefined) writeFileSync(join(agentDir, "settings.json"), JSON.stringify(global));
	if (project !== undefined) {
		mkdirSync(join(root, ".pi"), { recursive: true });
		writeFileSync(join(root, ".pi", "settings.json"), JSON.stringify(project));
	}
	return SettingsManager.create(root, agentDir);
}

describe("resolveInteractiveTuiMode", () => {
	it("keeps the scrollback UI AutoRAG shipped with instead of pi 1.x's fullscreen default", () => {
		expect(resolveInteractiveTuiMode(managerWith())).toBe("regular");
	});

	it("honors a tuiMode the user saved in their global pi settings", () => {
		expect(resolveInteractiveTuiMode(managerWith({ tuiMode: "fullscreen" }))).toBe("fullscreen");
		expect(resolveInteractiveTuiMode(managerWith({ tuiMode: "regular" }))).toBe("regular");
	});

	it("lets a project setting win over the AutoRAG default", () => {
		expect(resolveInteractiveTuiMode(managerWith(undefined, { tuiMode: "fullscreen" }))).toBe("fullscreen");
	});
});
