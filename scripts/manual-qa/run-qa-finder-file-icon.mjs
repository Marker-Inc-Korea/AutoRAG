/**
 * Feasibility probe — can the OS give us the FileTile icon (Finder thumbnails)?
 *
 * Runs `app.getFileIcon` inside the real app's MAIN process via Playwright, for
 * a set of fixtures, and dumps every returned PNG plus its size/latency. Two
 * same-extension files with different content tell us whether the OS icon is a
 * content thumbnail (different bytes) or a generic type icon (identical bytes).
 *
 * Run from the repo root: `ICON_PROBE_DIR=/tmp/ulw-icon-probe bun scripts/manual-qa/run-qa-finder-file-icon.mjs`
 */

import { mkdirSync } from "node:fs";
import { _electron } from "playwright";

const ROOT = new URL("../../", import.meta.url).pathname.replace(/\/$/, "");
const EVIDENCE = `${ROOT}/.omo/evidence/ai-finder-app/qa-file-icon/`;
const DIR = process.env.ICON_PROBE_DIR;
mkdirSync(EVIDENCE, { recursive: true });

const paths = [];
for await (const name of new Bun.Glob("*").scan(DIR)) paths.push(`${DIR}/${name}`);
paths.sort();

const app = await _electron.launch({ args: ["app"], cwd: ROOT });
try {
	const page = await app.firstWindow();
	await page.waitForSelector('[role="row"]', { timeout: 30000 });
	const result = await app.evaluate(async ({ app }, targets) => {
		const out = [];
		for (const target of targets) {
			const t0 = Date.now();
			const small = await app.getFileIcon(target, { size: "small" });
			const smallMs = Date.now() - t0;
			const t1 = Date.now();
			const normal = await app.getFileIcon(target, { size: "normal" });
			const normalMs = Date.now() - t1;
			out.push({
				path: target,
				small: { size: small.getSize(), png: small.toPNG().toString("base64"), empty: small.isEmpty(), ms: smallMs },
				normal: {
					size: normal.getSize(),
					png: normal.toPNG().toString("base64"),
					empty: normal.isEmpty(),
					ms: normalMs,
				},
			});
		}
		return out;
	}, paths);

	const summary = [];
	for (const row of result) {
		const name = row.path.split("/").pop();
		await Bun.write(`${EVIDENCE}/normal-${name}`, Buffer.from(row.normal.png, "base64"));
		await Bun.write(`${EVIDENCE}/small-${name}`, Buffer.from(row.small.png, "base64"));
		summary.push({
			name,
			small: { w: row.small.size.width, h: row.small.size.height, bytes: row.small.png.length, ms: row.small.ms, empty: row.small.empty },
			normal: {
				w: row.normal.size.width,
				h: row.normal.size.height,
				bytes: row.normal.png.length,
				ms: row.normal.ms,
				empty: row.normal.empty,
			},
		});
		console.log(
			`${name.padEnd(14)} small=${row.small.size.width}x${row.small.size.height} ${String(row.small.png.length).padStart(6)}b ${String(row.small.ms).padStart(4)}ms || normal=${row.normal.size.width}x${row.normal.size.height} ${String(row.normal.png.length).padStart(6)}b ${String(row.normal.ms).padStart(4)}ms${row.normal.empty ? " EMPTY" : ""}`,
		);
	}
	await Bun.write(`${EVIDENCE}/probe.json`, JSON.stringify(summary, null, 2));
} finally {
	await app.close();
}
