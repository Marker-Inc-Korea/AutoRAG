/**
 * Manual QA — the row tile shows the OS thumbnail (Finder-style), not a letter.
 *
 * Drives the real production app (app/out) with Playwright and samples the
 * PIXELS of each rendered tile through a canvas. Two same-type fixtures with
 * different content (red vs blue) prove the tile is content-true rather than a
 * generic type icon, and the sample is taken from the live DOM, not from a file
 * the script generated itself. Evidence goes to
 * .omo/evidence/ai-finder-app/qa-thumbnails/.
 *
 * Run from the repo root: `bun scripts/manual-qa/run-qa-finder-thumbnails.mjs`
 */

import { execSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { rm } from "node:fs/promises";
import { deflateSync } from "node:zlib";
import { _electron } from "playwright";

const ROOT = new URL("../../", import.meta.url).pathname.replace(/\/$/, "");
const EVIDENCE = `${ROOT}/.omo/evidence/ai-finder-app/qa-thumbnails/`;
mkdirSync(EVIDENCE, { recursive: true });

const stamp = Date.now();
const documents = `${process.env.HOME}/Documents`;
const folderName = `ulw-thumb-${stamp}-dir`;
const probeDir = `${documents}/ulw-thumb-${stamp}-latency`;
const RED = [220, 40, 40];
const BLUE = [40, 60, 220];
const fixtures = [
	{ name: `ulw-thumb-${stamp}-red.png`, expect: "red" },
	{ name: `ulw-thumb-${stamp}-blue.png`, expect: "blue" },
	{ name: `ulw-thumb-${stamp}-red.mp4`, expect: "red" },
	{ name: `ulw-thumb-${stamp}-blue.mp4`, expect: "blue" },
];

function crc32(buf) {
	let crc = 0xffffffff;
	for (const byte of buf) {
		crc ^= byte;
		for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
	}
	return (crc ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
	const length = Buffer.alloc(4);
	length.writeUInt32BE(data.length);
	const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
	const crc = Buffer.alloc(4);
	crc.writeUInt32BE(crc32(body));
	return Buffer.concat([length, body, crc]);
}
function solidPng(width, height, [r, g, b]) {
	const ihdr = Buffer.alloc(13);
	ihdr.writeUInt32BE(width, 0);
	ihdr.writeUInt32BE(height, 4);
	ihdr[8] = 8;
	ihdr[9] = 2;
	const raw = Buffer.alloc(height * (1 + width * 3));
	for (let y = 0; y < height; y++) {
		const off = y * (1 + width * 3);
		for (let x = 0; x < width; x++) {
			raw[off + 1 + x * 3] = r;
			raw[off + 2 + x * 3] = g;
			raw[off + 3 + x * 3] = b;
		}
	}
	return Buffer.concat([
		Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
		chunk("IHDR", ihdr),
		chunk("IDAT", deflateSync(raw)),
		chunk("IEND", Buffer.alloc(0)),
	]);
}

mkdirSync(`${documents}/${folderName}`, { recursive: true });
await Bun.write(`${documents}/${fixtures[0].name}`, solidPng(64, 64, RED));
await Bun.write(`${documents}/${fixtures[1].name}`, solidPng(64, 64, BLUE));
const video = (color) =>
	execSync(
		`ffmpeg -y -loglevel error -f lavfi -i color=c=${color}:s=64x64:d=1:r=10 -pix_fmt yuv420p -c:v libx264 "${documents}/${color === "red" ? fixtures[2].name : fixtures[3].name}"`,
	);
video("red");
video("blue");

const results = [];
const check = (label, actual, expected) => {
	const pass = actual === expected;
	results.push({ label, actual, expected, pass });
	console.log(`${pass ? "PASS" : "FAIL"}  ${label}: ${JSON.stringify(actual)}`);
};

let app;
try {
	app = await _electron.launch({ args: ["app"], cwd: ROOT });
	const page = await app.firstWindow();
	await page.waitForSelector('[role="row"]', { timeout: 30000 });
	await page.getByLabel("Places").getByRole("button", { name: "Documents", exact: true }).click();
	await page.waitForSelector(`[role="row"]:has-text("${fixtures[0].name}")`, { timeout: 30000 });
	await page.waitForSelector("img.tile--image", { timeout: 15000 });

	const sampleTile = (rowName) =>
		page.evaluate(async (name) => {
			const row = [...document.querySelectorAll('[role="row"]')].find((candidate) =>
				(candidate.textContent ?? "").includes(name),
			);
			if (row === undefined) return { error: "row not found" };
			const image = row.querySelector("img.tile--image");
			if (image === null) {
				return { error: "no image tile", tileText: row.querySelector(".tile")?.textContent ?? null };
			}
			if (!image.complete) await image.decode();
			const canvas = document.createElement("canvas");
			canvas.width = image.naturalWidth || 1;
			canvas.height = image.naturalHeight || 1;
			const context = canvas.getContext("2d");
			context.drawImage(image, 0, 0);
			const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
			let r = 0;
			let g = 0;
			let b = 0;
			const pixels = data.length / 4;
			for (let i = 0; i < data.length; i += 4) {
				r += data[i];
				g += data[i + 1];
				b += data[i + 2];
			}
			return {
				src: image.getAttribute("src")?.slice(0, 30) ?? null,
				size: [canvas.width, canvas.height],
				avg: [Math.round(r / pixels), Math.round(g / pixels), Math.round(b / pixels)],
			};
		}, rowName);

	const dominant = (avg) => (avg[0] > avg[1] && avg[0] > avg[2] ? "red" : avg[2] > avg[1] ? "blue" : "other");
	const observed = [];
	for (const fixture of fixtures) {
		const sample = await sampleTile(fixture.name);
		observed.push({ name: fixture.name, sample });
		if (sample.error !== undefined) {
			check(fixture.name, sample.error, fixture.expect);
			continue;
		}
		check(
			`${fixture.name} tile is a PNG data URL`,
			sample.src.startsWith("data:image/png;base64,") ? "png-data-url" : sample.src,
			"png-data-url",
		);
		check(`${fixture.name} tile is ${fixture.expect}-true`, dominant(sample.avg), fixture.expect);
	}

	const folderSample = await sampleTile(folderName);
	observed.push({ name: folderName, sample: folderSample });
	check("folder renders no image tile", folderSample.error ?? "has image tile", "no image tile");

	// Informational: what the icon work costs a real listing (cold = icons generated now).
	mkdirSync(probeDir, { recursive: true });
	for (const fixture of fixtures) execSync(`cp "${documents}/${fixture.name}" "${probeDir}/"`);
	const timeList = (dir) =>
		page.evaluate(async (target) => {
			const start = performance.now();
			await window.autorag.fs.listDir(target);
			return Math.round(performance.now() - start);
		}, dir);
	const coldMs = await timeList(probeDir);
	const warmMs = await timeList(probeDir);
	console.log(`INFO  listDir(${probeDir}): cold=${coldMs}ms warm=${warmMs}ms`);

	await page.screenshot({ path: `${EVIDENCE}/documents-list.png` });
	const row = page.locator('[role="row"]').filter({ hasText: fixtures[0].name }).first();
	await row.screenshot({ path: `${EVIDENCE}/red-png-row.png` });
	await Bun.write(`${EVIDENCE}/result.json`, JSON.stringify({ checks: results, observed, latency: { coldMs, warmMs } }, null, 2));
} finally {
	if (app !== undefined) await app.close();
	for (const fixture of fixtures) await rm(`${documents}/${fixture.name}`, { force: true });
	await rm(`${documents}/${folderName}`, { recursive: true, force: true });
	await rm(probeDir, { recursive: true, force: true });
}

const failed = results.filter((result) => !result.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) process.exitCode = 1;
