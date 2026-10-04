import { chmodSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Write a minimal fsearch-cli stand-in for the MCP stdio smoke test. */
export function writeFakeFSearchExecutable(directory: string, resultPath: string): string {
	const binaryPath = join(directory, process.platform === "win32" ? "fsearch-cli.exe" : "fsearch-cli");
	const script = `#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "--version") {
  console.log("fsearch-cli 0.3");
  process.exit(0);
}
if (args[0] === "index") {
  console.log(JSON.stringify({ indexed: true }));
  process.exit(0);
}
if (args[0] === "stats") {
  console.log(JSON.stringify({ live: true, files: 1, folders: 0 }));
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
