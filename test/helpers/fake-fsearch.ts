import { chmodSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Write a minimal fsearch-cli stand-in for the MCP stdio smoke test. */
export function writeFakeFSearchExecutable(directory: string, resultPath: string): string {
	const binaryPath = join(directory, process.platform === "win32" ? "fsearch-cli.exe" : "fsearch-cli");
	const script = `#!/usr/bin/env node
const { mkdirSync, writeFileSync } = require("node:fs");
const { dirname } = require("node:path");
const args = process.argv.slice(2);
if (args[0] === "--version") {
  console.log("fsearch-cli 0.3");
  process.exit(0);
}
if (args[0] === "index") {
  // Like the real CLI, \`index --db <path>\` writes the database file.
  const db = args[args.indexOf("--db") + 1];
  mkdirSync(dirname(db), { recursive: true });
  writeFileSync(db, "db");
  console.log(JSON.stringify({ indexed: true }));
  process.exit(0);
}
if (args[0] === "stats") {
  // No watch daemon is ever launched; report live:false so a caller never
  // adopts a fictional daemon or waits on one.
  console.log(JSON.stringify({ live: false, files: 1, folders: 0 }));
  process.exit(0);
}
if (args[0] === "search") {
  console.log(JSON.stringify({ path: ${JSON.stringify(resultPath)}, type: "file", size: 1, mtime: 0 }));
  console.log(JSON.stringify({ done: true, num_results: 1 }));
  process.exit(0);
}
console.error("unexpected fake fsearch-cli command: " + args.join(" "));
process.exit(2);
`;
	writeFileSync(binaryPath, script);
	chmodSync(binaryPath, 0o755);
	return binaryPath;
}
