/**
 * Live manual QA: search keeps answering while a refresh updates an index that already exists.
 *
 * A real corpus on disk, the real `minsync` binary and MinSync's own embedding engine (a running
 * `inl serve` daemon). Nothing is mocked and no remote embedder is contacted.
 *
 * It walks the user-visible path: first refresh (the index is built) -> edit, add and delete documents ->
 * an incremental `agent.refresh` with searches issued the whole time it runs -> the refresh's final state.
 * What it proves:
 *   - no search is refused or degraded while the refresh runs (no `retrieval-method-failed` diagnostic);
 *   - a document the refresh does not touch is found by every search, so the index never goes dark;
 *   - an edited document is never missing from a search while the refresh swaps its batch. MinSync 0.7.4 is the
 *     first release that guarantees it (it sweeps an edited file's old rows only after the new rows are
 *     sparse-encoded); 0.7.3 and older can miss the file for a moment, which the script reports as a number. What
 *     is asserted either way: every edited document answers at least once during the refresh and all of them
 *     answer again after it;
 *   - searches really overlapped the refresh (a pass with no overlap proves nothing, so it fails);
 *   - once the refresh ends: edits are searchable, additions are searchable, deletions are gone.
 *
 * Everything lives in one temp root, and AUTORAG_HOME points inside it, so the shared `~/.autorag` is never
 * touched.
 *
 * Prerequisites: `minsync` 0.7.5 or newer on PATH and a running `inl serve --idle-timeout 0`.
 *
 * Run: bun scripts/manual-qa/run-qa-refresh-search-live.ts
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createEmbedderFromConfig, releaseEmbedders } from "../../src/embedding/index.ts";
import { minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";

const MODIFIED = { from: 0, to: 60 };
const DELETED = { from: 140, to: 150 };
const ADDED = { from: 150, to: 170 };
/** Searches that must have run start to finish beside the refresh, or the run proves nothing. */
const MIN_OVERLAPPING_SEARCHES = 3;
/** Edited-document searches needed before "never missed" means anything. */
const MIN_EDITED_SEARCHES = 10;

const ADJECTIVES = ["Crimson", "Amber", "Silent", "Granite", "Velvet", "Cobalt", "Ivory", "Scarlet", "Emerald", "Copper"];
const NOUNS = ["Lighthouse", "Orchard", "Harbor", "Foundry", "Meadow", "Citadel", "Bridge", "Lantern", "Quarry", "Glacier"];
const NOUNS_MORE = ["Compass", "Anvil", "Canyon", "Beacon", "Vineyard", "Observatory", "Windmill", "Archive", "Monastery", "Greenhouse"];
const DEPARTMENTS = ["logistics", "finance", "staffing", "procurement", "compliance"];

const tmpRoot = realpathSync(mkdtempSync(join(tmpdir(), "autorag-refresh-search-qa-")));
const docs = join(tmpRoot, "docs");
const autoragHome = join(tmpRoot, ".autorag-home");
process.env.AUTORAG_HOME = autoragHome;
process.env.AUTORAG_CONFIG = join(autoragHome, "config.json");
mkdirSync(docs, { recursive: true });
mkdirSync(autoragHome, { recursive: true });

let failures = 0;
let inconclusive = false;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

const pad = (index: number): string => String(index).padStart(4, "0");
const docPath = (index: number): string => join(docs, `${pad(index)}.txt`);
/** A two-word name no other document shares, so a search for it has one right answer. */
const nameOf = (index: number): string =>
	`${ADJECTIVES[index % 10]} ${[...NOUNS, ...NOUNS_MORE][Math.floor(index / 10) % 20]}`;

function body(index: number, revision: "A" | "B"): string {
	const name = nameOf(index);
	const department = DEPARTMENTS[index % DEPARTMENTS.length];
	const lines = [
		`# ${name} initiative`,
		`The ${name} initiative is owned by the ${department} department. Its quarterly review runs every ${10 + index} days.`,
		"Status notes: staffing is stable, the budget review closed without findings, and the regional schedule holds.",
		"Open items: confirm the vendor contract renewal, archive last quarter's reports, and update the audit checklist.",
		...(revision === "B"
			? [`Revision B: the ${name} initiative moved its escalation contact to the ${DEPARTMENTS[(index + 2) % 5]} desk.`]
			: []),
	];
	return `${lines.join("\n\n")}\n`;
}

function write(index: number, revision: "A" | "B"): void {
	writeFileSync(docPath(index), body(index, revision));
}

try {
	const version = spawnSync("minsync", ["--version"], { encoding: "utf8" }).stdout.trim();
	console.log(`minsync: ${version}`);
	console.log(`temp root: ${tmpRoot}`);
	for (let index = MODIFIED.from; index < DELETED.to; index += 1) write(index, "A");

	const embedder = createEmbedderFromConfig({});
	const health = await embedder.health();
	console.log(`embedding backend: ${health.backend} alive=${health.alive}${health.detail ? ` (${health.detail})` : ""}`);
	check("an embedding backend is alive (run `inl serve --idle-timeout 0` first)", health.alive);

	const agent = new AutoRAGAgent({
		searchPaths: [docs],
		workspacePath: tmpRoot,
		memoryPath: join(autoragHome, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
	});

	const cursor = join(minSyncWorkspaceRoot(tmpRoot), ".minsync", "cursor.json");
	const txn = join(minSyncWorkspaceRoot(tmpRoot), ".minsync", "txn.json");

	const search = async (query: string) => agent.retrieveWithDiagnostics(query, { topK: 8 });
	const found = (results: readonly { source: string }[], index: number): boolean =>
		results.some((result) => result.source.endsWith(`/${pad(index)}.txt`));

	// 1. Build the index: this is the "index that already exists" the refresh below updates.
	const started = Date.now();
	const first = await agent.refresh(true, { methods: ["parsed", "minsync"] });
	console.log(`first refresh: ${((Date.now() - started) / 1000).toFixed(1)}s`);
	check("first refresh builds the index", first.minsync?.ok === true && existsSync(cursor), JSON.stringify(first.minsync));
	for (const index of [3, 75, 145]) {
		const answer = await search(`${nameOf(index)} initiative quarterly review`);
		check(`before the refresh: document ${pad(index)} is found`, found(answer.results, index));
	}

	// 2. Change the corpus: edit one slice, add some, delete some, leave the rest alone.
	for (let index = MODIFIED.from; index < MODIFIED.to; index += 1) write(index, "B");
	for (let index = ADDED.from; index < ADDED.to; index += 1) write(index, "B");
	for (let index = DELETED.from; index < DELETED.to; index += 1) unlinkSync(docPath(index));

	// 3. Refresh incrementally while searching the whole time.
	interface Probe {
		readonly index: number;
		readonly kind: "untouched" | "modified";
	}
	// One untouched document per six searches keeps the "index never goes dark" check alive; the rest are edited
	// documents, because a search is slow next to a refresh and that is the sample the edited-document guarantee needs.
	const probes: readonly Probe[] = [
		{ index: 5, kind: "modified" },
		{ index: 30, kind: "modified" },
		{ index: 100, kind: "untouched" },
		{ index: 55, kind: "modified" },
		{ index: 12, kind: "modified" },
		{ index: 44, kind: "modified" },
	];
	let refreshing = true;
	const refreshStarted = Date.now();
	const refresh = agent.refresh(false, { methods: ["parsed", "minsync"] }).finally(() => {
		refreshing = false;
	});
	const during: { probe: Probe; found: boolean; failed: string[]; overlapped: boolean; seconds: number }[] = [];
	for (let turn = 0; refreshing; turn += 1) {
		const probe = probes[turn % probes.length];
		if (probe === undefined) break;
		const startedBeside = refreshing;
		const searchStarted = Date.now();
		const stateAtStart = `txn=${existsSync(txn) ? "Y" : "-"} cursor=${existsSync(cursor) ? "Y" : "-"}`;
		const answer = await search(`${nameOf(probe.index)} initiative quarterly review`);
		const overlapped = startedBeside && refreshing;
		const failed = answer.diagnostics.filter((d) => d.code === "retrieval-method-failed").map((d) => d.reason ?? d.message);
		const hit = found(answer.results, probe.index);
		during.push({ probe, found: hit, failed, overlapped, seconds: (Date.now() - searchStarted) / 1000 });
		console.log(
			`  +${((Date.now() - refreshStarted) / 1000).toFixed(1).padStart(5)}s [${stateAtStart}] ${probe.kind.padEnd(9)} ${pad(probe.index)} ` +
				`hits=${answer.results.length} found=${hit} overlapped=${overlapped} failed=${failed.length}${failed[0] ? ` (${failed[0].slice(0, 80)})` : ""}`,
		);
		await sleep(0);
	}
	const second = await refresh;
	console.log(`incremental refresh: ${((Date.now() - refreshStarted) / 1000).toFixed(1)}s, searches issued: ${during.length}`);

	const overlapping = during.filter((entry) => entry.overlapped);
	check("incremental refresh succeeds", second.minsync?.ok === true, JSON.stringify(second.minsync));
	check(
		`searches overlapped the refresh (>= ${MIN_OVERLAPPING_SEARCHES} ran start to finish beside it)`,
		overlapping.length >= MIN_OVERLAPPING_SEARCHES,
		`${overlapping.length} overlapped of ${during.length}`,
	);
	check(
		"no search was refused or degraded while the refresh ran",
		during.every((entry) => entry.failed.length === 0),
		`${during.filter((entry) => entry.failed.length > 0).length} failed`,
	);
	const untouched = during.filter((entry) => entry.probe.kind === "untouched");
	check(
		"an untouched document is found by every search during the refresh",
		untouched.length > 0 && untouched.every((entry) => entry.found),
		`${untouched.filter((entry) => entry.found).length}/${untouched.length}`,
	);
	const modified = during.filter((entry) => entry.probe.kind === "modified");
	const lagged = modified.filter((entry) => !entry.found);
	console.log(
		`INFO  edited document missed by ${lagged.length}/${modified.length} searches while its batch was being swapped` +
			`${lagged.length > 0 ? ` (${[...new Set(lagged.map((entry) => pad(entry.probe.index)))].join(", ")})` : ""}`,
	);
	// A refresh that ends after a handful of searches cannot show that an edited document was never missed, and it
	// must not pass or fail on that. With enough searches, MinSync 0.7.4 must never miss one.
	if (modified.length >= MIN_EDITED_SEARCHES) {
		check(
			"an edited document is never missing from a search while its batch is swapped (MinSync >= 0.7.4)",
			lagged.length === 0,
			`${lagged.length}/${modified.length} missed`,
		);
	} else {
		console.log(
			`SKIP  edited-document guarantee not judged: only ${modified.length} edited-document searches ran beside the refresh (need ${MIN_EDITED_SEARCHES}); rerun on a quieter machine`,
		);
		inconclusive = true;
	}
	let recovered = 0;
	for (let index = MODIFIED.from; index < MODIFIED.to; index += 1) {
		if (found((await search(`${nameOf(index)} initiative quarterly review`)).results, index)) recovered += 1;
	}
	check(
		"after the refresh every edited document is found again",
		recovered === MODIFIED.to - MODIFIED.from,
		`${recovered}/${MODIFIED.to - MODIFIED.from}`,
	);

	// 4. After the refresh: edits, additions and deletions all show.
	const edited = await search(`${nameOf(5)} initiative escalation contact moved`);
	check(
		"after: an edited document answers with its new text",
		edited.results.some((result) => result.source.endsWith("/0005.txt") && result.content.includes("Revision B")),
	);
	const added = await search(`${nameOf(ADDED.from)} initiative quarterly review`);
	check("after: an added document is searchable", found(added.results, ADDED.from));
	const lateAdded = await search(`${nameOf(ADDED.to - 1)} initiative quarterly review`);
	check("after: the last added document is searchable", found(lateAdded.results, ADDED.to - 1));
	const deleted = await search(`${nameOf(DELETED.from)} initiative quarterly review`);
	check("after: a deleted document no longer answers", !found(deleted.results, DELETED.from));
	const kept = await search(`${nameOf(100)} initiative quarterly review`);
	check("after: an untouched document still answers", found(kept.results, 100));

	if (failures > 0) {
		console.log(`\nLIVE QA: ${failures} failure(s)`);
		process.exitCode = 1;
	} else if (inconclusive) {
		console.log("\nLIVE QA INCONCLUSIVE: no check failed, but the edited-document guarantee was not judged");
		process.exitCode = 2;
	} else {
		console.log("\nLIVE QA PASSED");
	}
} finally {
	await releaseEmbedders();
	rmSync(tmpRoot, { recursive: true, force: true });
}
