import { execFile } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { acquireRefreshLock, refreshLockPath } from "../../src/mirror/refresh-lock.ts";

/**
 * Cross-process behaviour of the refresh lock.
 *
 * The in-process guard cannot be exercised by the collision that actually happens in production:
 * the CLI builds a fresh agent in a fresh process per command, so `autorag watch` and
 * `autorag refresh` are two operating system processes sharing one index directory. These tests
 * spawn real processes, and the contention cases are paired with deterministic controls so a passing
 * assertion cannot be mistaken for an assertion that never had anything to catch.
 *
 * The racing pair never relies on scheduling to overlap. Both children race on the lock primitive
 * itself (`acquireRefreshLock`, the same call `agent.refresh()` makes), and the winner then holds
 * the lock across a file-signal handshake: the loser attempts its refresh only once the winner's
 * hold is visible on disk, and the winner releases only once the loser has reported its outcome.
 * The busy verdict is therefore structurally guaranteed, and every wait is bounded so a dead peer
 * fails the race loudly instead of hanging it.
 */

/** Corpus is kept minimal: the barrier, not content volume, guarantees the overlap. */
const DOCUMENT_COUNT = 2;
const PARAGRAPH_COUNT = 2;
/** Upper bound for every barrier wait; a peer that dies silently trips this and fails the race. */
const BARRIER_WAIT_MS = 30_000;
const BARRIER_POLL_MS = 25;

const roots: string[] = [];
const REPO_ROOT = join(import.meta.dirname, "..", "..");

function makeWorkspace(marker = "alpha"): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-xproc-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < DOCUMENT_COUNT; i += 1) {
		const body = Array.from(
			{ length: PARAGRAPH_COUNT },
			(_, p) => `Paragraph ${p} of document ${i}. Marker ${marker}.`,
		).join("\n\n");
		writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\n${body}\n`);
	}
	return { root, searchPaths: [docs] };
}

function makeAgent(root: string, searchPaths: string[]): AutoRAGAgent {
	return new AutoRAGAgent({
		workspacePath: root,
		searchPaths,
		bm25: { forceEngine: "typescript-fallback" },
		minSync: false,
	});
}

/**
 * Source lines that rewrite every corpus document so a child's refresh sees `marker`.
 */
function rewriteCorpusSource(marker: string): string {
	const bodyExpr = [
		`Array.from({ length: ${PARAGRAPH_COUNT} }, (_, p) => "Paragraph " + p + " of document " + i + ". Marker ${marker}.")`,
		`.join("\\n\\n")`,
	].join("");
	return [
		`for (let i = 0; i < ${DOCUMENT_COUNT}; i += 1) {`,
		`\tconst body = ${bodyExpr};`,
		`\twriteFileSync(join(docs, "doc-" + i + ".md"), "# Document " + i + "\\n\\n" + body + "\\n");`,
		`}`,
	].join("\n");
}

/**
 * Source for a child process that rewrites the corpus and runs one refresh.
 */
function refreshScript(root: string, docs: string, marker: string): string {
	return `
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { AutoRAGAgent } from ${JSON.stringify(join(REPO_ROOT, "src/agent/agent.ts"))};
const root = ${JSON.stringify(root)};
const docs = ${JSON.stringify(docs)};
${rewriteCorpusSource(marker)}
const agent = new AutoRAGAgent({
	workspacePath: root,
	searchPaths: [docs],
	bm25: { forceEngine: "typescript-fallback" },
	minSync: false,
});
const result = await agent.refresh(true);
process.stdout.write(JSON.stringify({ outcome: result.outcome ?? "completed" }));
`;
}

/**
 * Source for one side of a racing pair.
 *
 * Both children rewrite the corpus and then race on `acquireRefreshLock` — the same call
 * `agent.refresh()` makes — and the outcome of that race decides the role:
 *
 * - winner: writes the held signal, waits for the challenger's report, releases the lock, and only
 *   then runs the actual refresh, which must therefore complete.
 * - challenger: waits for the held signal, so its refresh attempt lands strictly inside the
 *   winner's hold and must therefore be refused as busy.
 *
 * Either spawn slot can be the winner; the race itself decides. Every wait is bounded by
 * BARRIER_WAIT_MS and fails with a descriptive error, so a dead peer surfaces as a harness
 * failure rather than a hang.
 */
function raceScript(root: string, docs: string, marker: string): string {
	return `
import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { AutoRAGAgent } from ${JSON.stringify(join(REPO_ROOT, "src/agent/agent.ts"))};
import { acquireRefreshLock } from ${JSON.stringify(join(REPO_ROOT, "src/mirror/refresh-lock.ts"))};
const root = ${JSON.stringify(root)};
const docs = ${JSON.stringify(docs)};
const heldSignal = join(root, "barrier-held");
const triedSignal = join(root, "barrier-tried");
${rewriteCorpusSource(marker)}
async function waitForSignal(path, what) {
	const deadline = Date.now() + ${BARRIER_WAIT_MS};
	while (!existsSync(path)) {
		if (Date.now() > deadline) {
			throw new Error("race barrier timeout: " + what + " never appeared at " + path + " within ${BARRIER_WAIT_MS}ms");
		}
		await new Promise((resolve) => setTimeout(resolve, ${BARRIER_POLL_MS}));
	}
}
function makeAgent() {
	return new AutoRAGAgent({
		workspacePath: root,
		searchPaths: [docs],
		bm25: { forceEngine: "typescript-fallback" },
		minSync: false,
	});
}
const lock = acquireRefreshLock(root);
if (lock) {
	writeFileSync(heldSignal, "held");
	await waitForSignal(triedSignal, "the challenger's report");
	lock.release();
	const result = await makeAgent().refresh(true);
	process.stdout.write(JSON.stringify({ role: "winner", outcome: result.outcome ?? "completed" }));
} else {
	await waitForSignal(heldSignal, "the winner's hold signal");
	const result = await makeAgent().refresh(true);
	writeFileSync(triedSignal, "tried");
	process.stdout.write(JSON.stringify({ role: "challenger", outcome: result.outcome ?? "completed" }));
}
`;
}

interface ChildResult {
	/** Present only for the racing script, which decides roles by the lock race. */
	readonly role?: "winner" | "challenger";
	readonly outcome: string;
}

/**
 * Start a child process without blocking on it.
 *
 * A synchronous spawn would run the children one after another, which is not a race at all and
 * would make the contention assertions vacuous. A non-zero exit rejects with the child's stderr so
 * a barrier timeout fails the test with its message instead of a bare exit code.
 */
function startChild(root: string, script: string): Promise<ChildResult> {
	const path = join(root, `child-${Math.random().toString(36).slice(2)}.ts`);
	writeFileSync(path, script);
	return new Promise((resolve, reject) => {
		execFile("bun", ["run", path], { cwd: REPO_ROOT, encoding: "utf8" }, (error, stdout) => {
			if (error) {
				const stderr = typeof error.stderr === "string" && error.stderr.length > 0 ? `\n${error.stderr}` : "";
				reject(new Error(`race child failed: ${error.message}${stderr}`));
			} else {
				resolve(JSON.parse(stdout) as ChildResult);
			}
		});
	});
}

/** Run one round of two racing refreshes and report what the winner and the challenger did. */
async function raceOnce(marker: string): Promise<{ winner: string; challenger: string }> {
	const { root, searchPaths } = makeWorkspace("seed");
	const docs = searchPaths[0] as string;
	const outcomes = await Promise.all([
		startChild(root, raceScript(root, docs, marker)),
		startChild(root, raceScript(root, docs, marker)),
	]);
	const winner = outcomes.find((outcome) => outcome.role === "winner");
	const challenger = outcomes.find((outcome) => outcome.role === "challenger");
	if (winner === undefined || challenger === undefined) {
		throw new Error(`race harness: expected one winner and one challenger, got ${JSON.stringify(outcomes)}`);
	}
	return { winner: winner.outcome, challenger: challenger.outcome };
}

function artifactMatchesMarker(root: string, marker: string): boolean {
	const artifact = join(root, ".autorag", "bm25", "fallback-index.json");
	if (!existsSync(artifact)) return false;
	return readFileSync(artifact, "utf8").includes(marker);
}
function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Source for a child that runs the real CLI (`main` from `src/cli/index.ts`) against a workspace
 * whose config lives in a temp home. The exit code and rendered output are the assertions, so the
 * child must not swallow either.
 */
function cliChildScript(root: string, docs: string, args: readonly string[]): string {
	return `
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { main } from ${JSON.stringify(join(REPO_ROOT, "src/cli/index.ts"))};
const root = ${JSON.stringify(root)};
const home = join(root, "home");
process.env.HOME = home;
process.env.AUTORAG_HOME = join(home, ".autorag");
mkdirSync(process.env.AUTORAG_HOME, { recursive: true });
writeFileSync(join(process.env.AUTORAG_HOME, "config.json"), JSON.stringify({
	searchPaths: [${JSON.stringify(docs)}],
	workspacePath: root,
	memoryPath: join(root, "memory.json"),
	bm25: { forceEngine: "typescript-fallback" },
	minSync: false,
}, null, 2));
process.chdir(root);
const code = await main(${JSON.stringify([...args])});
process.exit(code);
`;
}

interface CliChildResult {
	readonly code: number;
	readonly stdout: string;
	readonly stderr: string;
}

/**
 * Run a CLI child that may legitimately exit non-zero. Unlike `startChild` (whose children must
 * always succeed, so a non-zero exit rejects the race), these children are the commands under
 * test: the exit code is the assertion.
 */
function startCliChild(root: string, script: string): Promise<CliChildResult> {
	const path = join(root, `cli-child-${Math.random().toString(36).slice(2)}.ts`);
	writeFileSync(path, script);
	return new Promise((resolve) => {
		execFile("bun", ["run", path], { cwd: REPO_ROOT, encoding: "utf8" }, (error, stdout, stderr) => {
			resolve({
				code: error ? (typeof error.code === "number" ? error.code : 1) : 0,
				stdout,
				stderr,
			});
		});
	});
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("refresh lock across processes", () => {
	it("refuses a second caller while another holds the lock", async () => {
		const { root, searchPaths } = makeWorkspace();
		const agent = makeAgent(root, searchPaths);

		const held = acquireRefreshLock(root);
		expect(held).toBeDefined();

		const refused = await agent.refresh(true);
		expect(refused.outcome).toBe("busy");
		expect(refused.bm25).toBeUndefined();
		expect(refused.written).toBe(0);

		held?.release();

		// Once released, the same agent proceeds normally.
		const completed = await agent.refresh(true);
		expect(completed.outcome).toBe("completed");
	});

	it("does not honour a lock whose owner process is gone", async () => {
		const { root, searchPaths } = makeWorkspace();
		const agent = makeAgent(root, searchPaths);

		// A pid that cannot be running, with a marker old enough to be reapable.
		const lockDir = refreshLockPath(root);
		mkdirSync(lockDir, { recursive: true });
		const marker = join(lockDir, "owner-abandoned.json");
		writeFileSync(
			marker,
			`${JSON.stringify({ token: "abandoned", pid: 2 ** 30, createdAt: Date.now() - 60 * 60 * 1000 })}\n`,
		);
		const old = new Date(Date.now() - 60 * 60 * 1000);
		const fs = await import("node:fs");
		fs.utimesSync(marker, old, old);

		const result = await agent.refresh(true);
		expect(result.outcome).toBe("completed");
	});

	it("honours a lock whose owner process is alive", async () => {
		const { root, searchPaths } = makeWorkspace();
		const agent = makeAgent(root, searchPaths);

		const lockDir = refreshLockPath(root);
		mkdirSync(lockDir, { recursive: true });
		// This process is alive by definition, so the record must be honoured however old it looks.
		writeFileSync(
			join(lockDir, "owner-live.json"),
			`${JSON.stringify({ token: "live", pid: process.pid, createdAt: Date.now() - 60 * 60 * 1000 })}\n`,
		);

		const result = await agent.refresh(true);
		expect(result.outcome).toBe("busy");
	});

	it("lets exactly one of two racing processes do the work", async () => {
		// Overlap is guaranteed by the handshake in `raceScript`, not by scheduling luck: the
		// winner holds the lock until the challenger has attempted a refresh, so one completion
		// and one refusal are the only reachable outcome. Two rounds so either spawn slot gets a
		// chance to be the winner.
		for (let round = 0; round < 2; round += 1) {
			const outcome = await raceOnce(`round${round}`);
			expect(outcome).toEqual({ winner: "completed", challenger: "busy" });
		}
	}, 120_000);

	it("control: a child is busy while the lock is held and completes once it is released", async () => {
		// The race test above asserts exactly one completion, which is only meaningful if the
		// "busy" outcome is caused by the lock and not by the harness. This control removes every
		// other variable: the parent takes the lock before the child starts and keeps holding it
		// until the child has exited, so the refusal cannot come from scheduling; the same script
		// and workspace then complete after the release, so the refusal cannot come from the
		// harness either.
		const { root, searchPaths } = makeWorkspace();
		const script = refreshScript(root, searchPaths[0] as string, "ctl-held");

		const held = acquireRefreshLock(root);
		expect(held).toBeDefined();
		expect((await startChild(root, script)).outcome).toBe("busy");

		held?.release();
		expect((await startChild(root, script)).outcome).toBe("completed");
	}, 120_000);

	it("control: two children in separate workspaces both complete", async () => {
		// The race test rejects two completions. This control proves that outcome is reachable
		// with the same harness: two children with no shared lock directory cannot contend, so
		// both must complete however their execution interleaves. A harness that could only ever
		// count one completion would fail here and make the race assertion vacuous.
		const first = makeWorkspace("ctl-a");
		const second = makeWorkspace("ctl-b");
		const outcomes = await Promise.all([
			startChild(first.root, refreshScript(first.root, first.searchPaths[0] as string, "ctl-a")),
			startChild(second.root, refreshScript(second.root, second.searchPaths[0] as string, "ctl-b")),
		]);
		expect(outcomes.map((outcome) => outcome.outcome)).toEqual(["completed", "completed"]);
	}, 120_000);

	it("leaves an artifact that matches the corpus after racing processes", async () => {
		// The seed corpus carries a different marker, so the final artifact can only match if the
		// winner's refresh indexed the post-race corpus and no interleaving corrupted it.
		const { root, searchPaths } = makeWorkspace("seed");
		const docs = searchPaths[0] as string;
		const marker = "mark0";

		await Promise.all([
			startChild(root, raceScript(root, docs, marker)),
			startChild(root, raceScript(root, docs, marker)),
		]);

		// Settle with one uncontended refresh, then the artifact must carry the current marker.
		const agent = makeAgent(root, searchPaths);
		await agent.refresh(false);
		expect(artifactMatchesMarker(root, marker)).toBe(true);
	}, 120_000);
	it("an explicit CLI refresh exits 1 and reports busy while the lock is held", async () => {
		// Red-team scenario 1: `autorag refresh` refused by a held lock must not exit 0 as if the
		// index had been updated. The parent holds the lock before the child starts, so the refusal
		// is structural, not scheduling.
		const { root, searchPaths } = makeWorkspace();
		const docs = searchPaths[0] as string;

		const held = acquireRefreshLock(root);
		expect(held).toBeDefined();

		const child = await startCliChild(root, cliChildScript(root, docs, ["refresh", "--json"]));

		expect(child.code).toBe(1);
		const envelope = JSON.parse(child.stdout) as Record<string, unknown>;
		expect(envelope.outcome).toBe("busy");
		expect(envelope.ok).toBe(false);
		expect(envelope.counts).toEqual({ scanned: 0, written: 0, deleted: 0, skipped: 0 });

		held?.release();
	});

	it("fails with a clear error, not busy, when a regular file sits at the lock path, and leaves it untouched", async () => {
		// Red-team scenario 2: a regular file forged to look like a live lock owner. The old
		// behaviour read it as contention forever (permanent silent DoS with exit 0) or, once
		// stale-looking, silently unlinked the user's file via the legacy-file reaper. It must now
		// fail loudly with a fixable message and never touch the file.
		const { root, searchPaths } = makeWorkspace();
		const docs = searchPaths[0] as string;

		const lockFile = refreshLockPath(root);
		mkdirSync(join(root, ".autorag"), { recursive: true });
		const forged = `${JSON.stringify({ token: "forged", pid: process.pid, createdAt: Date.now() })}\n`;
		writeFileSync(lockFile, forged);

		const child = await startCliChild(root, cliChildScript(root, docs, ["refresh", "--json"]));

		expect(child.code).toBe(1);
		expect(child.stdout).not.toContain("busy");
		expect(child.stderr).toContain("not a lock directory");
		expect(child.stderr).toContain(".autorag/refresh.lock");
		expect(readFileSync(lockFile, "utf8")).toBe(forged);
	});

	it("index rebuild holds the lock across deletion and re-indexing", async () => {
		// Red-team scenario 3: `index rebuild` deleted the indexes, released the lock, and a
		// concurrent refresh then made the re-indexing busy — exit 0 with the indexes gone. The
		// rebuild must be one transaction: the parent watches the deletion happen (proof the
		// rebuild started) and then asserts the lock stays held until the child exits. A free lock
		// at any point inside that window is the silent data-loss bug.
		const root = mkdtempSync(join(tmpdir(), "autorag-rebuild-"));
		roots.push(root);
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		// Large enough that the re-indexing after the deletion outlasts the parent's probes.
		for (let i = 0; i < 300; i += 1) {
			writeFileSync(
				join(docs, `doc-${i}.md`),
				`# Document ${i}\n\n${Array.from({ length: 3 }, (_, p) => `Paragraph ${p} of rebuild document ${i}.`).join("\n\n")}\n`,
			);
		}
		const bm25Dir = join(root, ".autorag", "bm25");
		mkdirSync(bm25Dir, { recursive: true });
		writeFileSync(join(bm25Dir, "stale-index.json"), "{}");

		const child = startCliChild(root, cliChildScript(root, docs, ["index", "rebuild", "--yes", "--json"]));

		// The deletion is the proof that the rebuild has started.
		const deleteDeadline = Date.now() + BARRIER_WAIT_MS;
		while (existsSync(bm25Dir)) {
			if (Date.now() > deleteDeadline) throw new Error("rebuild never removed the bm25 index dir");
			await sleep(BARRIER_POLL_MS);
		}

		// From the deletion until the child exits the lock must never be acquirable.
		const heldDeadline = Date.now() + BARRIER_WAIT_MS;
		let observedHeld = false;
		while (!observedHeld && Date.now() < heldDeadline) {
			const probe = acquireRefreshLock(root);
			if (probe === undefined) {
				observedHeld = true;
			} else {
				probe.release();
				await sleep(BARRIER_POLL_MS);
			}
		}
		if (!observedHeld) {
			throw new Error("rebuild lock became acquirable before the rebuild finished");
		}

		const result = await child;
		expect(result.code).toBe(0);

		// The lock is free again and the indexes were actually rebuilt.
		const after = acquireRefreshLock(root);
		expect(after).toBeDefined();
		after?.release();
		expect(existsSync(join(bm25Dir, "fallback-index.json"))).toBe(true);
	}, 120_000);
});
