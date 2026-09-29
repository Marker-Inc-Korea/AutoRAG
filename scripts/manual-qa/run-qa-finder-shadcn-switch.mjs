import { _electron } from "playwright";
import { mkdirSync } from "node:fs";

const EVIDENCE = new URL("../../.omo/evidence/ai-finder-app/shadcn-switch/", import.meta.url).pathname;
mkdirSync(EVIDENCE, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Production build first: cd app && bunx electron-vite build
const app = await _electron.launch({ args: ["app"] }); // run from the repo root
const page = await app.firstWindow();
await page.waitForSelector('[role="row"]');

await page.locator(".sidebar__item", { hasText: "Settings" }).click();
await page.waitForSelector(".settings-panel");

const sw = page.locator(".settings-row", { hasText: "Show hidden files" }).locator(".settings-switch");
const thumb = sw.locator(".settings-switch__thumb");

const geometry = async () => ({
	role: await sw.getAttribute("role"),
	state: await sw.getAttribute("data-state"),
	track: await sw.evaluate((el) => {
		const r = el.getBoundingClientRect();
		return { w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10, bg: getComputedStyle(el).backgroundColor, shadow: getComputedStyle(el).boxShadow };
	}),
	thumb: await thumb.evaluate((el) => {
		const r = el.getBoundingClientRect();
		const t = el.closest(".settings-switch").getBoundingClientRect();
		return { x: Math.round(r.left - t.left), size: Math.round(r.width) };
	}),
});

console.log("off:", JSON.stringify(await geometry()));
await page.screenshot({ path: `${EVIDENCE}/switch-off.png` });

await sw.click(); // trusted click — flips the real persisted setting
await sleep(600);
const onState = await geometry();
console.log("on:", JSON.stringify(onState));
await page.screenshot({ path: `${EVIDENCE}/switch-on.png` });

const fails = [];
if ((await sw.getAttribute("role")) !== "switch") fails.push("Radix role missing");
if (onState.track.w !== 32 || onState.track.h !== 18.4) fails.push(`track ${onState.track.w}x${onState.track.h}`);
if (onState.track.bg !== "rgb(255, 99, 99)") fails.push(`on bg ${onState.track.bg}`);
if (onState.thumb.size !== 16 || onState.thumb.x !== 15) fails.push(`thumb ${JSON.stringify(onState.thumb)}`);

await sw.click(); // restore off
await sleep(400);
console.log((await geometry()).state === "unchecked" ? "restored off ✓" : "restore FAIL");
if (fails.length) {
	console.log("FAIL:", fails);
	process.exitCode = 1;
} else {
	console.log("PASS: Radix switch geometry matches shadcn spec");
}
await app.close();
