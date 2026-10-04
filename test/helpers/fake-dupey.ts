import { chmodSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Write a deterministic Dupey stand-in into `directory` under the filename the
 * scanner resolves on PATH when the MCP duplicates tool spawns the default
 * executable (`dupey` on POSIX, `dupey.exe` on Windows), and return its path.
 *
 * The script reports one exact duplicate group for the scanned directory.
 * Node loads a shebang script without a module extension as CommonJS, so
 * `require` works on POSIX and Windows alike (same trick as fake-minsync).
 */
export function writeFakeDupeyExecutable(directory: string): string {
	const binaryPath = join(directory, process.platform === "win32" ? "dupey.exe" : "dupey");
	const script = `#!/usr/bin/env node
const { join } = require("node:path");

const args = process.argv.slice(2);
if (args[0] !== "scan") {
  console.error("unexpected fake dupey command: " + args.join(" "));
  process.exit(2);
}
const dir = args[1] ?? ".";
console.log(JSON.stringify({
  dir,
  files: [
    { path: join(dir, "a.md"), content_hash: "dupey-fixture-hash" },
    { path: join(dir, "b.md"), content_hash: "dupey-fixture-hash" },
  ],
  families: [],
  errors: [],
}));
`;
	writeFileSync(binaryPath, script);
	chmodSync(binaryPath, 0o755);
	return binaryPath;
}
