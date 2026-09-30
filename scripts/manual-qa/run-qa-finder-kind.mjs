/**
 * Manual QA — the Kind column shows the OS-detected kind (Finder's Kind).
 *
 * Drives the real production app (app/out) with Playwright's _electron API and
 * reads the Kind cell for fixtures created in ~/Documents. The expected value
 * is taken straight from `/usr/bin/mdls` (polled until Spotlight answers, since
 * a brand-new file can briefly report no kind), so the assertion is the OS's
 * own answer, not a copy of the app's table. Evidence goes to
 * .omo/evidence/ai-finder-app/qa-os-kind/.
 *
 * Run from the repo root: `bun scripts/manual-qa/run-qa-finder-kind.mjs`
 */

import { execSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { rm } from "node:fs/promises";
import { _electron } from "playwright";

const ROOT = new URL("../../", import.meta.url).pathname.replace(/\/$/, "");
const EVIDENCE = `${ROOT}/.omo/evidence/ai-finder-app/qa-os-kind/`;
mkdirSync(EVIDENCE, { recursive: true });

const stamp = Date.now();
const documents = `${process.env.HOME}/Documents`;
const folderName = `ulw-kind-qa-${stamp}-dir`;
const fixtures = [
	{ name: `ulw-kind-qa-${stamp}.mp4`, value: "QA fixture\n", expect: "MPEG-4 movie" },
	{ name: `ulw-kind-qa-${stamp}.pdf`, value: "%PDF-1.4\n", expect: "PDF document" },
];
const fallbackName = `ulw-kind-qa-${stamp}.weird`;

const mdlsKind = (path) => {
	try {
		return execSync(`/usr/bin/mdls -name kMDItemKind -raw ${JSON.stringify(path)}`, { encoding: "utf8" }).trim();
	} catch {
		return "";
	}
};

/** Spotlight reports no kind for a just-created file; wait for its answer. */
const osKindReady = async (path, timeoutMs = 10000) => {
	const start = Date.now();
	for (; ;) {
		const value = mdlsKind(path);
		if (value !== "" && value !== "(null)") return value;
		if (Date.now() - start > timeoutMs) return null;
		await Bun.sleep(250);
	}
};

const results = [];
const check = (label, actual, expected) => {
	const pass = actual === expected;
	results.push({ label, actual, expected, pass });
	console.log(`${pass ? "PASS" : "FAIL"}  ${label}: app=${JSON.stringify(actual)} os=${JSON.stringify(expected)}`);
};

mkdirSync(`${documents}/${folderName}`, { recursive: true });
for (const fixture of fixtures) await Bun.write(`${documents}/${fixture.name}`, fixture.value);
await Bun.write(`${documents}/${fallbackName}`, "unknown extension\n");

for (const fixture of fixtures) {
	const expected = await osKindReady(`${documents}/${fixture.name}`);
	if (expected === null) {
		console.log(`FAIL  ${fixture.name}: /usr/bin/mdls never reported a kind; cannot assert`);
		process.exitCode = 1;
	}
	fixture.expected = expected ?? fixture.expect;
}

let app;
try {
	app = await _electron.launch({ args: ["app"], cwd: ROOT });
	const page = await app.firstWindow();
	await page.waitForSelector('[role="row"]', { timeout: 30000 });
	await page.getByLabel("Places").getByRole("button", { name: "Documents", exact: true }).click();
	await page.waitForSelector(`[role="row"]:has-text("${fixtures[0].name}")`, { timeout: 30000 });

	const rowCellText = async (name) => {
		const row = page.locator('[role="row"]').filter({ hasText: name }).first();
		return row.locator('[role="gridcell"]').allInnerTexts();
	};

	const observed = [];
	for (const fixture of fixtures) {
		const cells = await rowCellText(fixture.name);
		observed.push({ name: fixture.name, cells });
		check(fixture.name, cells[3].trim(), fixture.expected);
	}
	const fallbackCells = await rowCellText(fallbackName);
	observed.push({ name: fallbackName, cells: fallbackCells });
	const fallbackKind = fallbackCells[3].trim();
	results.push({ label: fallbackName, actual: fallbackKind, expected: "non-empty fallback", pass: fallbackKind !== "" });
	console.log(`${fallbackKind !== "" ? "PASS" : "FAIL"}  ${fallbackName}: app=${JSON.stringify(fallbackKind)} (fallback row, non-empty)`);

	const folderCells = await rowCellText(folderName);
	observed.push({ name: folderName, cells: folderCells });
	check(folderName, folderCells[3].trim(), "Folder");

	check(`${fixtures[0].name} is not Markdown`, observed[0].cells[3].trim() === "Markdown" ? "Markdown" : "not-Markdown", "not-Markdown");

	const row = page.locator('[role="row"]').filter({ hasText: fixtures[0].name }).first();
	await page.screenshot({ path: `${EVIDENCE}/documents-list.png` });
	await row.screenshot({ path: `${EVIDENCE}/mp4-row.png` });
	await Bun.write(
		`${EVIDENCE}/result.json`,
		JSON.stringify({ platform: process.platform, checks: results, observedRows: observed }, null, 2),
	);
} finally {
	if (app !== undefined) await app.close();
	for (const fixture of fixtures) await rm(`${documents}/${fixture.name}`, { force: true });
	await rm(`${documents}/${fallbackName}`, { force: true });
	await rm(`${documents}/${folderName}`, { recursive: true, force: true });
}

const failed = results.filter((result) => !result.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) process.exitCode = 1;
