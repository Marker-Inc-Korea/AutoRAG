import { chmodSync, existsSync, readFileSync, writeFileSync } from "node:fs";

/** Deterministic minsync stand-in that stages nothing itself and queries `files/`. */
export function writeFakeMinSync(binaryPath: string, logPath?: string): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
${logPath ? `appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, cwd: process.cwd() }) + "\\n");` : ""}

if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}
if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
}
if (args[0] === "sync") {
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ files_processed: 1 }));
  process.exit(0);
}
if (args[0] === "query") {
  const filesRoot = join(process.cwd(), "files");
  const hits = [];
  const walk = (dir, rel) => {
    if (!existsSync(dir)) return;
    for (const ent of readdirSync(dir, { withFileTypes: true })) {
      const nextRel = rel ? rel + "/" + ent.name : ent.name;
      const nextPath = join(dir, ent.name);
      if (ent.isDirectory()) walk(nextPath, nextRel);
      else hits.push({ path: "files/" + nextRel, score: 0.9, text: readFileSync(nextPath, "utf8") });
    }
  };
  walk(filesRoot, "");
  console.log(JSON.stringify({ results: hits }));
  process.exit(0);
}
console.error("unexpected fake minsync command: " + args.join(" "));
process.exit(2);
`,
	);
	chmodSync(binaryPath, 0o755);
}

export function fakeMinSyncLoggedModes(logPath: string): string[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => JSON.parse(line) as { args: string[] })
		.filter((entry) => entry.args[0] === "query")
		.map((entry) => entry.args[entry.args.indexOf("--mode") + 1]);
}
