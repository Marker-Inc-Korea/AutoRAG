/**
 * Manual QA — a sidebar item with no backing location (e.g. Recents) must not
 * crash the fs ipc handler. It shows a toast instead and the current folder
 * view stays.
 *
 * Run from the repo root: `bun scripts/manual-qa/run-qa-finder-nav-guard.mjs`
 */

import { _electron } from "playwright";
import { mkdirSync } from "node:fs";

const ROOT = new URL("../../", import.meta.url).pathname.replace(/\/$/, "");
const EVIDENCE = `${ROOT}/.omo/evidence/ai-finder-app/qa-nav-guard/`;
mkdirSync(EVIDENCE, { recursive: true });

const app = await _electron.launch({ args: ["app"], cwd: ROOT });
const results = [];
const check = (label, pass) => {
	results.push({ label, pass });
	console.log(`${pass ? "PASS" : "FAIL"}  ${label}`);
};
try {
	const page = await app.firstWindow();
	const handlerErrors = [];
	page.on("console", (message) => {
		if (/Error occurred in handler/i.test(message.text())) handlerErrors.push(message.text());
	});
	await page.waitForSelector('[role="row"]', { timeout: 30000 });

	await page.getByLabel("Places").getByRole("button", { name: "Recents", exact: true }).click();
	await page.waitForSelector("text=아직 연결되지 않았습니다", { timeout: 10000 });

	check("Recents click shows the not-linked toast", true);
	check("no fs:listDir handler error after the click", handlerErrors.length === 0);
	const rows = await page.locator('[role="row"]').count();
	check(`current folder view still renders (${rows} rows)`, rows > 0);
	await page.screenshot({ path: `${EVIDENCE}/after-recents-click.png` });
	await Bun.write(`${EVIDENCE}/result.json`, JSON.stringify({ results, handlerErrors }, null, 2));
} catch (error) {
	results.push({ label: `script error: ${error instanceof Error ? error.message : String(error)}`, pass: false });
	console.log("FAIL  script-level error:", error instanceof Error ? error.message : String(error));
} finally {
	await app.close();
}
const failed = results.filter((result) => !result.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) process.exitCode = 1;
