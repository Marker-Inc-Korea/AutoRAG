import { chmodSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type ModuleFormat = "esm" | "cjs";

/**
 * Platform name the MinSync resolver looks up on PATH and in the workspace
 * cache (`minsync` on POSIX, `minsync.exe` on Windows).
 */
export function fakeMinSyncExecutableName(platform: NodeJS.Platform = process.platform): string {
	return platform === "win32" ? "minsync.exe" : "minsync";
}

function fakeMinSyncScript(logPath: string | undefined, format: ModuleFormat): string {
	const loader =
		format === "esm"
			? `import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";`
			: `const { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } = require("node:fs");
const { dirname, join } = require("node:path");`;
	return `#!/usr/bin/env node
${loader}

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
`;
}

/** Deterministic minsync stand-in that stages nothing itself and queries `files/`. */
export function writeFakeMinSync(binaryPath: string, logPath?: string): void {
	writeFileSync(binaryPath, fakeMinSyncScript(logPath, "esm"));
	chmodSync(binaryPath, 0o755);
}

/**
 * Write the fake MinSync under the resolver's platform name into `directory`,
 * so a config-only stdio test resolves it from PATH or the workspace cache.
 * The resolver name has no `.mjs` extension, and Node loads a shebang script
 * without a module extension as CommonJS, so this entry uses `require` to work
 * on POSIX and on Windows (where the lookup name is `minsync.exe`).
 *
 * @returns the written binary path.
 */
export function writeFakeMinSyncExecutable(directory: string, logPath?: string): string {
	const binaryPath = join(directory, fakeMinSyncExecutableName());
	writeFileSync(binaryPath, fakeMinSyncScript(logPath, "cjs"));
	chmodSync(binaryPath, 0o755);
	return binaryPath;
}

/**
 * An inert `minsync` for the suite-wide PATH stand-in: every command succeeds
 * and a query never finds anything. It exists so an agent can resolve a binary
 * without a test's result depending on a real install or on whatever corpus a
 * `.autorag/` directory next to the test happens to hold.
 */
export function writeInertMinSyncExecutable(directory: string): string {
	const binaryPath = join(directory, fakeMinSyncExecutableName());
	const script = `#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "check") console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
else if (args[0] === "query") console.log(JSON.stringify({ results: [] }));
else if (args[0] === "sync") console.log(JSON.stringify({ files_processed: 0 }));
else console.log(JSON.stringify({ initialized: true }));
`;
	writeFileSync(binaryPath, script);
	chmodSync(binaryPath, 0o755);
	return binaryPath;
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
