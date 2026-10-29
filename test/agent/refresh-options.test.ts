import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readRefreshProgress } from "../../src/agent/refresh-progress.ts";
import {
	AutoRAGAgent,
	type AutoRAGRefreshProgress,
	type DatasourceIndexOptions,
	type DatasourceIndexResult,
	type DatasourceSkill,
} from "../../src/index.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-options-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (const name of ["a", "b", "c"]) writeFileSync(join(docs, `${name}.txt`), `${name} content\n`);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeAgent(overrides: Record<string, unknown> = {}) {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
		...overrides,
	});
}

function isAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

async function waitFor(condition: () => boolean, label: string): Promise<void> {
	const deadline = Date.now() + 15_000;
	while (Date.now() < deadline) {
		if (condition()) return;
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	throw new Error(`timed out waiting for ${label}`);
}

/**
 * A MinSync stand-in whose `sync` records its pid and then hangs until killed.
 * `stage` is delegated to the shared simulator, so the parsed mirror that runs
 * before the sync completes normally instead of failing the refresh.
 */
function writeHangingMinSync(binaryPath: string, pidPath: string): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
if (args[0] === "stage") {
  createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({
    stateDir: args[args.indexOf("--state-dir") + 1],
  });
} else if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, '[embedder]\\nid = "fixture"\\n');
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
} else if (args[0] === "sync") {
  writeFileSync(${JSON.stringify(pidPath)}, String(process.pid));
  setInterval(() => {}, 1000);
} else {
  process.exit(2);
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

function datasourceSkill(
	name: string,
	index: (options: DatasourceIndexOptions | undefined) => Promise<DatasourceIndexResult>,
): DatasourceSkill {
	return {
		describe: () => ({
			name,
			type: "chat",
			description: `${name} test datasource`,
			capabilities: [],
			tags: [],
			status: "active",
			datasourceId: name,
			instanceId: "default",
		}),
		polling: () => ({ mode: "none" }),
		skillManifest: () => ({ name: `datasource-${name}`, description: name, content: name }),
		index,
		retrievalMethods: () => [],
		describeSources: () => [],
	};
}

const okResult = (skill: string): DatasourceIndexResult => ({
	ok: true,
	instanceId: "default",
	skill,
	chunkCount: 1,
	indexedAt: 1,
	diagnostics: [],
});

describe("AutoRAGRefreshOptions.onProgress", () => {
	it("reports an advancing parsed-phase processed/total with the current file", async () => {
		const events: AutoRAGRefreshProgress[] = [];
		await makeAgent().refresh(true, { methods: ["parsed"], onProgress: (progress) => events.push({ ...progress }) });

		const parsed = events.filter((event) => event.phase === "parsed");
		expect(parsed.map((event) => event.processed)).toEqual([0, 1, 2, 3]);
		expect(parsed.every((event) => event.total === 3)).toBe(true);
		expect(parsed.slice(0, 3).map((event) => event.currentVirtualPath)).toEqual([
			expect.stringMatching(/a\.txt$/),
			expect.stringMatching(/b\.txt$/),
			expect.stringMatching(/c\.txt$/),
		]);
	});

	it("reports MinSync batch progress and persists the advancing counters", async () => {
		const binary = join(root, "progress-minsync.mjs");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
if (args[0] === "stage") {
  const eqArg = args.find((a) => a.startsWith("--state-dir="));
  const flagAt = args.indexOf("--state-dir");
  const stateDir = eqArg ? eqArg.slice("--state-dir=".length) : flagAt >= 0 ? args[flagAt + 1] : args[1];
  require(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir });
} else if (args[0] === "init") {
  mkdirSync(".minsync", { recursive: true });
  writeFileSync(".minsync/config.toml", "[embedder]\\nid = \\"openai\\"\\n");
  process.exit(0);
} else if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
else {
  console.log(JSON.stringify({ event: "start", total_files: 2 }));
  console.log(JSON.stringify({ event: "batch", processed: 1, total: 2 }));
  console.log(JSON.stringify({ event: "done", processed: 2, total: 2 }));
  writeFileSync(".minsync/cursor.json", "{}");
  console.log(JSON.stringify({ files_processed: 2 }));
}
`,
		);
		chmodSync(binary, 0o755);
		const agent = makeAgent({
			minSync: { binaryPath: binary, workspacePath: join(root, ".autorag", "minsync"), autoInstall: false },
		});
		const events: AutoRAGRefreshProgress[] = [];
		const persisted: { synced?: number; total?: number }[] = [];

		await agent.refresh(true, {
			methods: ["minsync"],
			onProgress: (progress) => {
				events.push({ ...progress });
				if (progress.phase === "minsync") persisted.push({ ...readRefreshProgress(root)?.minsync });
			},
		});

		expect(events.filter((event) => event.phase === "minsync")).toEqual([
			{ phase: "minsync", processed: 0, total: 2 },
			{ phase: "minsync", processed: 1, total: 2 },
			{ phase: "minsync", processed: 2, total: 2 },
		]);
		expect(persisted).toEqual([
			{ synced: 0, total: 2 },
			{ synced: 1, total: 2 },
			{ synced: 2, total: 2 },
		]);
	});

	it("reports one datasources event per source with processed/total sources", async () => {
		const agent = makeAgent({
			datasourceSkills: [
				datasourceSkill("one", async () => okResult("one")),
				datasourceSkill("two", async () => okResult("two")),
			],
		});
		const events: AutoRAGRefreshProgress[] = [];
		await agent.refresh(true, { methods: ["datasources"], onProgress: (progress) => events.push({ ...progress }) });

		expect(
			events.filter((event) => event.phase === "datasources").map((event) => [event.processed, event.total]),
		).toEqual([
			[0, 2],
			[1, 2],
			[2, 2],
		]);
	});
});

describe("AutoRAGRefreshOptions.signal", () => {
	it("rejects with the reason and starts nothing when already aborted", async () => {
		const reason = new Error("cancelled before start");
		const controller = new AbortController();
		controller.abort(reason);
		await expect(makeAgent().refresh(true, { signal: controller.signal })).rejects.toBe(reason);
		expect(readRefreshProgress(root)).toBeUndefined();
	});

	it("stops between files, never reaches MinSync, and records the run as failed", async () => {
		const agent = makeAgent();
		const controller = new AbortController();
		const reason = new Error("host cancelled");
		const refresh = agent.refresh(true, {
			signal: controller.signal,
			onProgress: (progress) => {
				if (progress.phase === "parsed" && progress.processed === 1) controller.abort(reason);
			},
		});

		await expect(refresh).rejects.toBe(reason);
		expect(existsSync(join(root, ".minsync"))).toBe(false);
		expect(readRefreshProgress(root)).toMatchObject({ state: "failed", phase: "parsed" });
		const status = await agent.getRefreshStatus();
		expect(status.inFlight).toBe(false);
		expect(status.state).toBe("failed");
	});

	it("kills the in-flight MinSync sync and rejects with the reason", async () => {
		const binary = join(root, "hanging-minsync.mjs");
		const pidPath = join(root, "minsync.pid");
		writeHangingMinSync(binary, pidPath);
		const agent = makeAgent({
			minSync: { binaryPath: binary, workspacePath: join(root, ".autorag", "minsync"), autoInstall: false },
		});
		const controller = new AbortController();
		const reason = new Error("stop the sync");
		const refresh = agent.refresh(true, { methods: ["minsync"], signal: controller.signal });
		const settled = refresh.then(
			() => "resolved",
			(error: unknown) => error,
		);

		await waitFor(() => existsSync(pidPath), "minsync sync to start");
		const pid = Number(readFileSync(pidPath, "utf8"));
		expect(isAlive(pid)).toBe(true);
		controller.abort(reason);

		expect(await settled).toBe(reason);
		await waitFor(() => !isAlive(pid), "minsync child to exit");
		expect((await agent.getRefreshStatus()).inFlight).toBe(false);
	});

	it("forwards the signal to datasource skills and rejects when one is aborted mid-index", async () => {
		const controller = new AbortController();
		const reason = new Error("stop datasource");
		let received: AbortSignal | undefined;
		let started!: () => void;
		const startedPromise = new Promise<void>((resolve) => {
			started = resolve;
		});
		const hung = datasourceSkill("hung", (options) => {
			received = options?.signal;
			started();
			return new Promise<DatasourceIndexResult>((resolve) => {
				options?.signal?.addEventListener("abort", () => resolve(okResult("hung")), { once: true });
			});
		});
		let secondRan = false;
		const second = datasourceSkill("second", async () => {
			secondRan = true;
			return okResult("second");
		});
		const refresh = makeAgent({ datasourceSkills: [hung, second] }).refresh(true, {
			methods: ["datasources"],
			signal: controller.signal,
		});
		const settled = refresh.then(
			() => "resolved",
			(error: unknown) => error,
		);
		await startedPromise;
		expect(received).toBe(controller.signal);
		controller.abort(reason);

		expect(await settled).toBe(reason);
		expect(secondRan).toBe(false);
	});

	it("kills an in-flight Jikji prepare and rejects with the reason", async () => {
		const binary = join(root, "hanging-jikji.mjs");
		const pidPath = join(root, "jikji.pid");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(pidPath)}, String(process.pid));
setInterval(() => {}, 1000);
`,
		);
		chmodSync(binary, 0o755);
		const controller = new AbortController();
		const reason = new Error("stop jikji");
		const refresh = makeAgent({ jikji: { binaryPath: binary, autoInstall: false } }).refresh(true, {
			methods: ["jikji"],
			signal: controller.signal,
		});
		const settled = refresh.then(
			() => "resolved",
			(error: unknown) => error,
		);

		await waitFor(() => existsSync(pidPath), "jikji prepare to start");
		const pid = Number(readFileSync(pidPath, "utf8"));
		controller.abort(reason);

		expect(await settled).toBe(reason);
		await waitFor(() => !isAlive(pid), "jikji child to exit");
	});
});

describe("refresh defaults", () => {
	it("behaves as before when no options are given", async () => {
		const result = await makeAgent().refresh(true, { methods: ["parsed"] });
		expect(result).toMatchObject({ scanned: 3, written: 3, skipped: 0 });
	});
});
