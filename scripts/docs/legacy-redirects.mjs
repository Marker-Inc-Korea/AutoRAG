#!/usr/bin/env node
/**
 * Keeps the old legacy-docs URLs alive after the docs moved.
 *
 * The Python AutoRAG docs used to live at https://marker-inc-korea.github.io/AutoRAG/<page>.html and are linked from
 * the legacy README, PyPI, and many blog posts. They are now built into `<site>/legacy/`. GitHub Pages cannot serve
 * server-side redirects, so for every legacy HTML page this writes a small stub at the old path that sends the reader
 * (keeping `?query` and `#section`) to the same page under `/legacy/`.
 *
 * Usage: node scripts/docs/legacy-redirects.mjs <legacy-html-dir> <site-dir>
 *
 * A page the new site already serves (its own `index.html`, for example) is never replaced.
 */
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";

const SITE = "https://autorag-docs.nomadamas.org";

const [legacyDir, siteDir] = process.argv.slice(2);
if (!legacyDir || !siteDir) {
	console.error("usage: legacy-redirects.mjs <legacy-html-dir> <site-dir>");
	process.exit(2);
}
if (!existsSync(legacyDir) || !statSync(legacyDir).isDirectory()) {
	console.error(`legacy-redirects: ${legacyDir} does not exist; the legacy docs build produced nothing`);
	process.exit(1);
}

const escapeHtml = (value) =>
	value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

/** URL path for a legacy page: encoded, with `?` and `#` kept out of the path. */
const urlPath = (rel) => encodeURI(rel).replaceAll("?", "%3F").replaceAll("#", "%23");

function pages(dir, out = []) {
	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		// `_static`, `_sources`, `_modules` and friends are assets, not pages anyone links to by their old URL.
		if (entry.name.startsWith("_") || entry.name.startsWith(".")) continue;
		const full = join(dir, entry.name);
		if (entry.isDirectory()) pages(full, out);
		else if (entry.name.endsWith(".html")) out.push(full);
	}
	return out;
}

function stub(rel) {
	const path = urlPath(rel);
	const target = `/legacy/${path}`;
	const attr = escapeHtml(target);
	return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved: AutoRAG legacy docs</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="${escapeHtml(`${SITE}${target}`)}">
<meta http-equiv="refresh" content="0; url=${attr}">
<script>location.replace(${JSON.stringify(target)}.replace(/</g, "\\u003c") + location.search + location.hash);</script>
</head>
<body>
<p>The legacy AutoRAG docs moved to <a href="${attr}">${attr}</a>.</p>
</body>
</html>
`;
}

let written = 0;
let kept = 0;
for (const file of pages(legacyDir)) {
	// Forward slashes in URLs on every platform.
	const rel = relative(legacyDir, file).split(sep).join("/");
	const out = join(siteDir, ...rel.split("/"));
	if (existsSync(out)) {
		kept++;
		continue;
	}
	mkdirSync(dirname(out), { recursive: true });
	writeFileSync(out, stub(rel));
	written++;
}
console.log(`legacy-redirects: wrote ${written} redirect stub(s), kept ${kept} page(s) the new site already serves`);
