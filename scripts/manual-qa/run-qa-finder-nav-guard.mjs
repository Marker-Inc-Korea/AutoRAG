/**
 * Manual QA — sidebar items must never crash fs:listDir.
 *
 * After the base landed a real Recents location, this script proves:
 *   1. "Recents" now navigates to the virtual Recents listing (no handler error,
 *      no "not linked" toast, rows render).
 *   2. A nav item with no backing location ("Slack") shows the not-linked toast
 *      instead of crashing (the Crash-to-toast regression guard from #1735).
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

	// 1. Recents is now a real location: it must navigate cleanly, with no toast.
	await page.getByLabel("Places").getByRole("button", { name: "Recents", exact: true }).click();
	await page.waitForTimeout(800);
	const recentsToasts = await page.locator("text=아직 연결되지 않았습니다").count();
	check("Recents navigates without a not-linked toast", recentsToasts === 0);
	check("no fs:listDir handler error after the Recents click", handlerErrors.length === 0);

	// 2. An item with no backing location toasts instead of crashing.
	await page.getByLabel("Places").getByRole("button", { name: "Slack", exact: true }).click();
	await page.waitForSelector("text=아직 연결되지 않았습니다", { timeout: 10000 });
	check("Slack click shows the not-linked toast", true);
	check("no fs:listDir handler error after the Slack click", handlerErrors.length === 0);
	const rows = await page.locator('[role="row"]').count();
	check(`view still renders rows (${rows})`, rows > 0);
	await page.screenshot({ path: `${EVIDENCE}/after-nav-clicks.png` });
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
