/**
 * Manual QA: a trashed duplicate leaves the version stack immediately.
 *
 * Proves, against the real app (production build, isolated userData):
 *  1. the head row carries a stack badge for a real dupey family
 *  2. opening the stack shows the duplicate members as child rows
 *  3. trashing a child removes that row from the stack right away — no waiting
 *     for the next scan — and decrements the badge
 *  4. the persisted snapshot is pruned, so a restart cannot resurrect it
 *  5. the file really moved into the OS Trash
 *
 * Fixture: ~/Downloads/ulw-dupey-trash-qa/ with three byte-identical documents,
 * named uniquely so process-table and Trash checks can never false-positive.
 * (Not ~/Desktop: a Desktop synced into iCloud Drive is not trashed into
 * ~/.Trash, so the Trash assertion would be meaningless there.)
 * Evidence: .omo/evidence/ai-finder-app/dupey-trash-prune/.
 * Run from the repo root after `cd app && bunx electron-vite build`:
 *   bun scripts/manual-qa/run-qa-finder-trash-prune.mjs
 */

import { _electron } from "playwright";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const EVIDENCE = join(REPO, ".omo", "evidence", "ai-finder-app", "dupey-trash-prune");
mkdirSync(EVIDENCE, { recursive: true });

const USER_DATA = join(tmpdir(), "pw-electron-qa-trash-prune");
rmSync(USER_DATA, { recursive: true, force: true });
const SNAPSHOT = join(USER_DATA, "dupey-cache", "version-families.json");

const FIXTURE_DIR = join(homedir(), "Downloads", "ulw-dupey-trash-qa");
const FOLDER = "ulw-dupey-trash-qa";
const MEMBERS = ["quarterly-report.txt", "quarterly-report-copy.txt", "quarterly-report (1).txt"];
/** The duplicate this run sends to the Trash (never the head, by construction). */
const VICTIM = "quarterly-report-copy.txt";
const VICTIM_PATH = join(FIXTURE_DIR, VICTIM);
/**
 * This Mac syncs Desktop & Documents into iCloud Drive, so a file under those
 * roots is trashed into the iCloud Drive Trash instead of ~/.Trash. Downloads
 * is local, but check both rather than assume the volume.
 */
const TRASH_ROOTS = [
	join(homedir(), "Library", "Mobile Documents", ".Trash"),
	join(homedir(), ".Trash"),
];
const trashedAt = () => TRASH_ROOTS.find((root) => existsSync(join(root, VICTIM))) ?? null;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fixture() {
	mkdirSync(FIXTURE_DIR, { recursive: true });
	const body = `${"Refund exceptions require director approval before payout. ".repeat(60)}\n`;
	for (const name of MEMBERS) writeFileSync(join(FIXTURE_DIR, name), body, "utf8");
}

function readJson(path) {
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch {
		return null;
	}
}

async function waitFor(predicate, { timeoutMs, label, stepMs = 250 }) {
	const deadline = Date.now() + timeoutMs;
	let last;
	while (Date.now() < deadline) {
		last = await predicate();
		if (last) return last;
		await sleep(stepMs);
	}
	throw new Error(`timed out waiting for ${label} (last: ${JSON.stringify(last)})`);
}

function check(label, value, expected) {
	const ok = value === expected;
	console.log(`${ok ? "PASS" : "FAIL"} ${label}: ${JSON.stringify(value)} (expected ${JSON.stringify(expected)})`);
	if (!ok) process.exitCode = 1;
}

/** The family the scan wrote for the fixture folder, if any. */
function fixtureFamily(snapshot) {
	const paths = new Set(MEMBERS.map((name) => join(FIXTURE_DIR, name)));
	return (snapshot?.families ?? []).find(
		(family) => paths.has(family.head) && family.members.some((member) => paths.has(member.path)),
	);
}

fixture();
rmSync(join(homedir(), ".Trash", VICTIM), { force: true });
rmSync(join(homedir(), "Library", "Mobile Documents", ".Trash", VICTIM), { force: true });

const app = await _electron.launch({
	args: [join(REPO, "app"), `--user-data-dir=${USER_DATA}`],
	cwd: REPO,
});
const page = await app.firstWindow();
page.setViewportSize({ width: 1440, height: 900 });
await page.waitForSelector('[role="row"]', { timeout: 60_000 });
await page.screenshot({ path: join(EVIDENCE, "0-launch.png") });

// 1. Wait for the cold scan to record the fixture family in the snapshot.
const family = await waitFor(async () => fixtureFamily(readJson(SNAPSHOT)), {
	timeoutMs: 420_000,
	label: "the startup scan to write the fixture family",
});
const head = family.head;
const headName = head.split("/").pop() ?? head;
console.log("SCAN family:", JSON.stringify({ head, members: family.members.map((m) => m.path) }));

// 2. Navigate into the fixture folder and open the stack. Scoping every
// locator to the head row keeps a neighbouring listing's badge out of the way.
await page.getByRole("button", { name: "Downloads" }).click();
await page.locator('[role="row"]', { hasText: FOLDER }).dblclick();
const headRow = page.locator('[role="row"]', { hasText: headName });
await headRow.waitFor({ timeout: 30_000 });

const badge = headRow.locator(".row__stackbadge");
const badgeText = async () => (await badge.textContent())?.trim() ?? null;
check("badge count before", await badgeText(), String(family.members.length));
await badge.click();
await page.locator(".row--child").first().waitFor({ timeout: 10_000 });

const childrenBefore = await page.locator(".row--child .row__label").allInnerTexts();
check("child rows before", childrenBefore.length, family.members.length);
check("victim is a child row", childrenBefore.includes(VICTIM), true);
await page.screenshot({ path: join(EVIDENCE, "1-stack-open.png") });

// 3. Trash the duplicate through the real context menu.
const victimRow = page.locator(".row--child", { hasText: VICTIM });
await victimRow.click({ button: "right" });
await page.getByRole("menuitem", { name: "휴지통으로 이동" }).click();

// The row must be gone the moment the trash lands — not after the next scan.
const vanished = await waitFor(async () => !(await page.locator(".row--child .row__label").allInnerTexts()).includes(VICTIM), {
	timeoutMs: 5000,
	label: "the trashed duplicate to leave the stack",
	stepMs: 100,
}).catch(() => false);
const childrenAfter = await page.locator(".row--child .row__label").allInnerTexts();
check("trashed row disappeared immediately", vanished, true);
check("child rows after", childrenAfter.length, family.members.length - 1);
check("badge count after", await badgeText(), String(family.members.length - 1));
await page.screenshot({ path: join(EVIDENCE, "2-after-trash.png") });

// 4. The file left its folder, landed in an OS Trash, and the snapshot no
// longer mentions it.
check("file left the fixture folder", existsSync(VICTIM_PATH), false);
const trashRoot = trashedAt();
console.log(`INFO trashed into: ${trashRoot ?? "NOWHERE"}`);
check("file is in an OS Trash", trashRoot !== null, true);
const after = readJson(SNAPSHOT);
const pruned = fixtureFamily(after);
check("snapshot still lists the family", pruned !== undefined, true);
check(
	"snapshot dropped the trashed member",
	(pruned?.members ?? []).some((member) => member.path === VICTIM_PATH),
	false,
);
check("snapshot kept the surviving members", (pruned?.members ?? []).length, family.members.length - 1);

// 5. Restart: the snapshot is the only source, so the deleted row cannot come back.
await app.close();
const relaunch = await _electron.launch({
	args: [join(REPO, "app"), `--user-data-dir=${USER_DATA}`],
	cwd: REPO,
});
const second = await relaunch.firstWindow();
second.setViewportSize({ width: 1440, height: 900 });
await second.waitForSelector('[role="row"]', { timeout: 60_000 });
await second.getByRole("button", { name: "Downloads" }).click();
await second.locator('[role="row"]', { hasText: FOLDER }).dblclick();
await second.locator('[role="row"]', { hasText: family.head.split("/").pop() ?? "" }).waitFor({ timeout: 30_000 });
await second.locator(".row__stackbadge").click();
await second.locator(".row--child").first().waitFor({ timeout: 10_000 });
const restarted = await second.locator(".row--child .row__label").allInnerTexts();
check("restart does not resurrect the trashed member", restarted.includes(VICTIM), false);
await second.screenshot({ path: join(EVIDENCE, "3-after-restart.png") });
await relaunch.close();

// Cleanup: only this run's own artifacts.
rmSync(FIXTURE_DIR, { recursive: true, force: true });
for (const root of TRASH_ROOTS) rmSync(join(root, VICTIM), { force: true });
console.log(`EVIDENCE ${EVIDENCE}`);
console.log(process.exitCode === 1 ? "RESULT FAIL" : "RESULT PASS");
