#!/usr/bin/env node
// Stand-in for the `minsync` CLI. It keeps the real command surface (`init`, `check`, `sync`, `query`) and
// answers `query` by substring-matching the files staged under `<cwd>/files`, so a test exercises the whole
// stage -> sync -> query -> hit-mapping path without an embedding model.
//
// Sibling files of this binary control it: `calls.jsonl` is the call log, and a `fail-sync` file makes `sync`
// exit 1 with "boom".
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, relative, sep } from "node:path";

const args = process.argv.slice(2);
const home = dirname(process.argv[1]);
const cwd = process.cwd();
const config = join(cwd, ".minsync", "config.toml");
const cursor = join(cwd, ".minsync", "cursor.json");
appendFileSync(join(home, "calls.jsonl"), `${JSON.stringify({ args, cwd })}\n`);

function stagedFiles(directory) {
	if (!existsSync(directory)) return [];
	return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
		const path = join(directory, entry.name);
		return entry.isDirectory() ? stagedFiles(path) : [path];
	});
}

if (args[0] === "init") {
	mkdirSync(dirname(config), { recursive: true });
	const languageIndex = args.indexOf("--language");
	const lexical = languageIndex === -1 ? "" : `\n[lexical]\nlanguage = "${args[languageIndex + 1]}"\n`;
	writeFileSync(config, `[embedder]\nid = "openai"\n${lexical}`);
	console.log(JSON.stringify({ initialized: true }));
	process.exit(0);
}

if (args[0] === "check") {
	console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true, all_passed: true }));
	process.exit(0);
}

if (args[0] === "sync") {
	if (existsSync(join(home, "fail-sync"))) {
		console.error("boom");
		process.exit(1);
	}
	mkdirSync(dirname(cursor), { recursive: true });
	writeFileSync(cursor, JSON.stringify({ ready: true }));
	console.log(JSON.stringify({ files_processed: stagedFiles(join(cwd, "files")).length }));
	process.exit(0);
}

if (args[0] === "query") {
	const mode = args[args.indexOf("--mode") + 1];
	if (mode === "hybrid" && existsSync(join(home, "fail-hybrid"))) {
		console.error("embedder unavailable");
		process.exit(1);
	}
	const k = Number(args[args.indexOf("-k") + 1]);
	const tokens = args[args.length - 1].toLowerCase().split(/\s+/u).filter(Boolean);
	const filesRoot = join(cwd, "files");
	const hits = stagedFiles(filesRoot)
		.map((file) => {
			const text = readFileSync(file, "utf8");
			const lower = text.toLowerCase();
			const score = tokens.reduce((sum, token) => sum + (lower.split(token).length - 1), 0);
			const path = `files/${relative(filesRoot, file).split(sep).join("/")}`;
			return { doc_id: createHash("sha256").update(path).digest("hex"), path, text, score };
		})
		.filter((hit) => hit.score > 0)
		.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path))
		.slice(0, k);
	console.log(JSON.stringify(hits));
	process.exit(0);
}

console.error(`unexpected fake minsync command: ${args.join(" ")}`);
process.exit(2);
