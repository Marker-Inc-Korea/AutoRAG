/**
 * Manual QA: dupey version-family persistence (SSOT) and the scan schedule.
 *
 * Proves, against the real app:
 *  1. the startup scan writes a snapshot to <userData>/dupey-cache/version-families.json
 *  2. the Finder renders stacks from that snapshot
 *  3. a restart serves stacks from the persisted snapshot and re-scans at startup
 *  4. the Settings scan interval persists into settings-state.json
 *
 * Evidence: .omo/evidence/ai-finder-app/dupey-cache/ (screenshots + logs).
 * Run from the repo root after `cd app && bunx electron-vite build`:
 *   bun scripts/manual-qa/run-qa-finder-dupey-cache.mjs
 */

import { _electron } from "playwright";
import { mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const EVIDENCE = join(REPO, ".omo", "evidence", "ai-finder-app", "dupey-cache");
mkdirSync(EVIDENCE, { recursive: true });

const USER_DATA = "/tmp/pw-electron-qa-dupey-cache";
rmSync(USER_DATA, { recursive: true, force: true });

const SNAPSHOT = join(USER_DATA, "dupey-cache", "version-families.json");
const SETTINGS = join(USER_DATA, "settings", "settings-state.json");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function readJson(path) {
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch {
		return null;
	}
}

async function waitFor(predicate, { timeoutMs, label }) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		const value = await predicate();
		if (value) return value;
		await sleep(1000);
	}
	throw new Error(`timed out waiting for ${label}`);
}

async function launch() {
	const app = await _electron.launch({
		args: [join(REPO, "app"), `--user-data-dir=${USER_DATA}`],
		cwd: REPO,
	});
	const page = await app.firstWindow();
	page.setViewportSize({ width: 1440, height: 900 });
	await page.waitForSelector('[role="row"]');
	return { app, page };
}

const report = {};

// ---- run 1: cold start scans, persists, and renders from the snapshot ----
{
	const { app, page } = await launch();
	const snapshot = await waitFor(async () => readJson(SNAPSHOT), {
		timeoutMs: 300_000,
		label: "startup scan to persist version-families.json",
	});
	report.coldStart = {
		version: snapshot.version,
		scannedAt: snapshot.scannedAt,
		locations: snapshot.locations?.length ?? 0,
		families: snapshot.families?.length ?? 0,
	};
	console.log("COLD-START snapshot:", JSON.stringify(report.coldStart));

	await page.getByRole("button", { name: "Downloads" }).click();
	await page.waitForSelector(".row__stackbadge", { timeout: 30_000 });
	report.coldStartBadges = await page.locator(".row__stackbadge").count();
	report.coldStartBanner = await page.locator(".list__error").count();
	console.log("COLD-START badges:", report.coldStartBadges, "banner:", report.coldStartBanner);
	await page.screenshot({ path: join(EVIDENCE, "1-cold-start.png") });
	await app.close();
}

// ---- run 2: restart serves the persisted snapshot, then re-scans ----
{
	const { app, page } = await launch();
	const started = Date.now();
	await page.getByRole("button", { name: "Downloads" }).click();
	await page.waitForSelector(".row__stackbadge", { timeout: 30_000 });
	report.warmBadgeMs = Date.now() - started;
	report.warmBadges = await page.locator(".row__stackbadge").count();
	console.log("WARM badges:", report.warmBadges, "after", report.warmBadgeMs, "ms");
	await page.screenshot({ path: join(EVIDENCE, "2-warm-start.png") });

	const rescanned = await waitFor(
		async () => {
			const snapshot = readJson(SNAPSHOT);
			return snapshot !== null && snapshot.scannedAt !== report.coldStart.scannedAt ? snapshot : null;
		},
		{ timeoutMs: 300_000, label: "the startup re-scan to advance scannedAt" },
	);
	report.rescannedAt = rescanned.scannedAt;
	console.log("WARM re-scanned:", report.rescannedAt);

	// Settings: change the scan interval and confirm it persists.
	await page.getByRole("button", { name: /Settings/ }).click();
	const interval = page.locator(".settings-panel select").nth(1);
	await interval.waitFor({ timeout: 15_000 });
	await interval.selectOption("15");
	await sleep(1500);
	report.interval = readJson(SETTINGS)?.settings?.dupeyScanIntervalMinutes ?? null;
	console.log("SETTINGS interval:", report.interval);
	await page.screenshot({ path: join(EVIDENCE, "3-settings-interval.png") });
	await app.close();
}

console.log("REPORT:", JSON.stringify(report, null, 1));
