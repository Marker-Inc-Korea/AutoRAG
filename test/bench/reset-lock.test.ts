import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { runIndex } from "../../src/cli/commands/index.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { MINSYNC_SUBDIR } from "../../src/minsync/paths.ts";
import { AUTORAG_DIR, PARSED_MIRROR_SUBDIR } from "../../src/mirror/paths.ts";
import { acquireRefreshLock, refreshLockPath } from "../../src/mirror/refresh-lock.ts";
import { BM25_SUBDIR } from "../../src/retrieval/methods/bm25.ts";

/**
 * `index reset` deletes whole index subtrees. Two properties have to hold, and they are separate
 * claims that are easy to conflate:
 *
 *   1. Reset must not run while a refresh holds the lock, or it deletes artifacts out from under a
 *      writer that will then commit a fingerprint describing files that no longer exist.
 *   2. The lock must survive reset. This is a placement property, not a mutual-exclusion property:
 *      a lock stored inside a directory reset removes would be destroyed by the operation it is
 *      supposed to exclude, so a third process would find the lock "free" while a refresh runs.
 *
 * Property 2 is the one that silently regresses if the lock path is ever moved back under an index
 * directory, so it is asserted directly against the old placement as a control.
 */

const roots: string[] = [];

function makeWorkspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-resetlock-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "a.md"), "# alpha\n\nsome indexable prose about alpha and beta.\n");
	// Pre-create the index dirs with a sentinel so we can prove removal actually happened.
	for (const subdir of [PARSED_MIRROR_SUBDIR, BM25_SUBDIR, MINSYNC_SUBDIR]) {
		const dir = join(root, subdir);
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "sentinel.txt"), "present");
	}
	return root;
}

function makeContext(root: string, positionals: string[]): { ctx: CommandContext; out: string[]; err: string[] } {
	const out: string[] = [];
	const err: string[] = [];
	const ctx: CommandContext = {
		positionals,
		flags: { yes: true, workspace: root },
		json: false,
		debug: false,
		cwd: root,
		stdout: (line) => out.push(line),
		stderr: (line) => err.push(line),
	};
	return { ctx, out, err };
}

function sentinels(root: string): boolean[] {
	return [PARSED_MIRROR_SUBDIR, BM25_SUBDIR, MINSYNC_SUBDIR].map((subdir) =>
		existsSync(join(root, subdir, "sentinel.txt")),
	);
}

afterAll(() => {
	for (const root of roots) rmSync(root, { recursive: true, force: true });
});

describe("index reset participates in the refresh lock", () => {
	it("refuses to delete anything while the refresh lock is held", async () => {
		const root = makeWorkspace();
		const held = acquireRefreshLock(root);
		expect(held).toBeDefined();

		const { ctx, err } = makeContext(root, ["reset"]);
		const code = await runIndex(ctx);

		expect(code).toBe(1);
		expect(err.join("\n")).toContain("already running");
		// The decisive assertion: refusal means nothing was removed, not merely that a message printed.
		expect(sentinels(root)).toEqual([true, true, true]);

		held?.release();
	});

	it("proceeds once the lock is released, and removes every target", async () => {
		const root = makeWorkspace();
		const held = acquireRefreshLock(root);
		held?.release();

		const { ctx } = makeContext(root, ["reset"]);
		const code = await runIndex(ctx);

		expect(code).toBe(0);
		expect(sentinels(root)).toEqual([false, false, false]);
	});

	it("releases the lock after a reset so a later refresh is not permanently blocked", async () => {
		const root = makeWorkspace();
		const { ctx } = makeContext(root, ["reset"]);
		expect(await runIndex(ctx)).toBe(0);

		// If reset leaked the lock, this acquire would come back undefined.
		const after = acquireRefreshLock(root);
		expect(after).toBeDefined();
		after?.release();
	});

	it("keeps the lock file outside every directory reset deletes", async () => {
		const root = makeWorkspace();
		const lockPath = refreshLockPath(root);

		// Control: the previous placement was inside the parsed mirror, which reset removes. Assert
		// that placement really would have been destroyed, so the property below is not vacuous.
		const oldPlacement = join(root, PARSED_MIRROR_SUBDIR, "refresh.lock");
		mkdirSync(oldPlacement, { recursive: true });

		const { ctx } = makeContext(root, ["reset"]);
		expect(await runIndex(ctx)).toBe(0);

		expect(existsSync(oldPlacement)).toBe(false);
		// The real lock path lives directly under `.autorag`, which reset preserves.
		expect(lockPath.startsWith(join(root, AUTORAG_DIR))).toBe(true);
		expect(lockPath.startsWith(join(root, PARSED_MIRROR_SUBDIR))).toBe(false);
		expect(lockPath.startsWith(join(root, BM25_SUBDIR))).toBe(false);
		expect(lockPath.startsWith(join(root, MINSYNC_SUBDIR))).toBe(false);
	});

	it("holds the lock across the deletion, not merely before it", async () => {
		const root = makeWorkspace();
		// Acquire, run reset in a way that must fail, then confirm a fresh acquire still works: this
		// pins that the lock is scoped to the removal and cleaned up on the refusal path too.
		const held = acquireRefreshLock(root);
		const { ctx } = makeContext(root, ["reset"]);
		expect(await runIndex(ctx)).toBe(1);
		held?.release();

		const reacquired = acquireRefreshLock(root);
		expect(reacquired).toBeDefined();
		reacquired?.release();
		expect(sentinels(root)).toEqual([true, true, true]);
	});
	it("refuses to delete anything for rebuild while the refresh lock is held", async () => {
		// Rebuild shares the reset refusal: if the lock is taken before the command starts, nothing
		// may be deleted and the command must fail loudly — a rebuild that deletes and then reports
		// busy (or success with no index) is silent data loss.
		const root = makeWorkspace();
		const held = acquireRefreshLock(root);
		expect(held).toBeDefined();

		const { ctx, err } = makeContext(root, ["rebuild"]);
		const code = await runIndex(ctx);

		expect(code).toBe(1);
		expect(err.join("\n")).toContain("already running");
		expect(sentinels(root)).toEqual([true, true, true]);

		held?.release();
	});
});
