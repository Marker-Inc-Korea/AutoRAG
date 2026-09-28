/**
 * capture-prototype-reference.mjs
 *
 * Launches Chrome via puppeteer-core, opens the AI Finder v6 prototype
 * as a file:// URL, and captures reference PNG screenshots in four states:
 *   1. main.png              – default main window
 *   2. settings.png           – Settings modal opened
 *   3. requests.png           – Requests popover opened
 *   4. evidence-collapsed.png – main window with Evidence panel collapsed
 *
 * The prototype uses zoom = min(1, vw/1360, vh/820) with 18px desk padding.
 * We capture at viewport 1396×856 (1360 + 36 padding, 820 + 36 padding)
 * so padding is visible and zoom lands at 1:1 for a natural look.
 *
 * Prints the absolute paths of every file it writes.
 * Kills the browser on exit.
 */

import puppeteer from "puppeteer-core";
import { existsSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, "../..");
const PROTOTYPE_DIR = resolve(
	REPO,
	"docs/design-reference/ai-finder/design_handoff_ai_finder_v6",
);
const EVIDENCE_DIR = resolve(REPO, ".omo/evidence/ai-finder-app/reference");

// Viewport: 1396×856 (1360+36 padding, 820+36 padding)
const VIEWPORT = { width: 1396, height: 856 };

// ── helpers ────────────────────────────────────────────────────────────────

function sleep(ms) {
	return new Promise((r) => setTimeout(r, ms));
}

// ── main ───────────────────────────────────────────────────────────────────

let browser;
try {
	if (!existsSync(EVIDENCE_DIR)) {
		mkdirSync(EVIDENCE_DIR, { recursive: true });
	}

	browser = await puppeteer.launch({
		executablePath:
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
		headless: true,
		args: ["--no-sandbox"],
	});

	const page = await browser.newPage();
	await page.setViewport(VIEWPORT);

	const protoUrl =
		"file://" + resolve(PROTOTYPE_DIR, "AI Finder v6.dc.html");
	console.log("  Opening", protoUrl);
	await page.goto(protoUrl, { waitUntil: "networkidle0", timeout: 30000 });

	// Let the dc-runtime hydrate and first render settle
	await sleep(800);

	// ── 1. main.png ────────────────────────────────────────────────────────
	{
		const png = resolve(EVIDENCE_DIR, "main.png");
		await page.screenshot({ path: png, fullPage: false });
		console.log("  → main.png", png);
	}

	// ── 2. settings.png ────────────────────────────────────────────────────
	// Use the dc-runtime component instance found via React fiber from the
	// root DOM node. The StandaloneRoot -> Root (StreamableComponent) renders
	// a div#dc-root -> div.sc-host chain. We walk the fiber to find the
	// StreamableComponent and call __setLogicState (exposed via logic.__host)
	// or the component's own setState method.
	{
		await page.evaluate(() => {
			// Walk the React 18 fiber tree from #dc-root to find the DC component
			const rootEl = document.getElementById("dc-root");
			if (!rootEl) throw new Error("#dc-root not found");

			// Get the first child element (the sc-host div rendered by StreamableComponent)
			const scHost = rootEl.firstElementChild;
			if (!scHost) throw new Error("sc-host not found");

			// React 18 stores fiber keys as __reactFiber$<hash> on DOM elements
			const fiberKey = Object.keys(scHost).find((k) =>
				k.startsWith("__reactFiber$"),
			);
			if (fiberKey) {
				let fiber = scHost[fiberKey];
				// Walk up to find the component instance (we want the class component,
				// which has stateNode)
				while (fiber) {
					if (
						fiber.stateNode &&
						fiber.stateNode.setState &&
						fiber.stateNode.__setLogicState
					) {
						// This is the StreamableComponent instance
						fiber.stateNode.__setLogicState({
							settings: true,
							inboxOpen: false,
						});
						return;
					}
					fiber = fiber.return;
				}
			}

			// Fallback: try clicking the Settings button in the sidebar
			const buttons = document.querySelectorAll("button");
			for (const btn of buttons) {
				if (btn.textContent.trim() === "Settings") {
					btn.click();
					return;
				}
			}
			throw new Error("Could not open Settings");
		});
		await sleep(500);

		const png = resolve(EVIDENCE_DIR, "settings.png");
		await page.screenshot({ path: png, fullPage: false });
		console.log("  → settings.png", png);
	}

	// ── 3. requests.png ────────────────────────────────────────────────────
	// Close settings and open Requests
	{
		await page.evaluate(() => {
			const rootEl = document.getElementById("dc-root");
			const scHost = rootEl?.firstElementChild;
			if (!scHost) throw new Error("sc-host not found");

			const fiberKey = Object.keys(scHost).find((k) =>
				k.startsWith("__reactFiber$"),
			);
			if (fiberKey) {
				let fiber = scHost[fiberKey];
				while (fiber) {
					if (
						fiber.stateNode &&
						fiber.stateNode.setState &&
						fiber.stateNode.__setLogicState
					) {
						fiber.stateNode.__setLogicState({
							settings: false,
							inboxOpen: true,
						});
						return;
					}
					fiber = fiber.return;
				}
			}

			// Fallback: find the Requests button via text content
			const buttons = document.querySelectorAll("button");
			for (const btn of buttons) {
				if (btn.textContent.trim().startsWith("Requests")) {
					btn.click();
					return;
				}
			}
			throw new Error("Could not open Requests");
		});
		await sleep(500);

		const png = resolve(EVIDENCE_DIR, "requests.png");
		await page.screenshot({ path: png, fullPage: false });
		console.log("  → requests.png", png);
	}

	// ── 4. evidence-collapsed.png ──────────────────────────────────────────
	{
		await page.evaluate(() => {
			const rootEl = document.getElementById("dc-root");
			const scHost = rootEl?.firstElementChild;
			if (!scHost) throw new Error("sc-host not found");

			const fiberKey = Object.keys(scHost).find((k) =>
				k.startsWith("__reactFiber$"),
			);
			if (fiberKey) {
				let fiber = scHost[fiberKey];
				while (fiber) {
					if (
						fiber.stateNode &&
						fiber.stateNode.setState &&
						fiber.stateNode.__setLogicState
					) {
						fiber.stateNode.__setLogicState({
							inboxOpen: false,
							evOpen: false,
						});
						return;
					}
					fiber = fiber.return;
				}
			}

			// Fallback: click the evidence toggle button (the one with chevron)
			const toggleBtns = document.querySelectorAll('button[title*="evidence" i]');
			for (const btn of toggleBtns) {
				btn.click();
				return;
			}
			throw new Error("Could not toggle evidence");
		});
		await sleep(500);

		const png = resolve(EVIDENCE_DIR, "evidence-collapsed.png");
		await page.screenshot({ path: png, fullPage: false });
		console.log("  → evidence-collapsed.png", png);
	}

	console.log("\nDone –", EVIDENCE_DIR);
} finally {
	if (browser) await browser.close();
}