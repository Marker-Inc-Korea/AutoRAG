import { readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const testRoot = join(root, "test");
const vitest = join(root, "node_modules", "vitest", "vitest.mjs");

function collectTestFiles(directory) {
	const files = [];
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) files.push(...collectTestFiles(path));
		else if (entry.isFile() && entry.name.endsWith(".test.ts")) files.push(path);
	}
	return files;
}

const testFiles = collectTestFiles(testRoot).sort();
const result = spawnSync(process.platform === "win32" ? "bun.exe" : "bun", [vitest, "run", ...testFiles], {
	cwd: root,
	stdio: "inherit",
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);

console.log(`\nAll ${testFiles.length} test files passed.`);
