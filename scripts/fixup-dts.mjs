// Rewrite relative `.ts` specifiers in emitted declaration files to `.js`.
// tsc emits d.ts with the original `.ts` extensions (allowImportingTsExtensions);
// shipped packages only contain .js/.d.ts, so the specifiers must point at .js.
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? "dist";
const specifier = /(["'])(\.{1,2}\/[^"']+)\.ts\1/g;

function walk(dir) {
	for (const entry of readdirSync(dir)) {
		const path = join(dir, entry);
		if (statSync(path).isDirectory()) {
			walk(path);
		} else if (path.endsWith(".d.ts")) {
			const source = readFileSync(path, "utf8");
			const fixed = source.replace(specifier, "$1$2.js$1");
			if (fixed !== source) writeFileSync(path, fixed);
		}
	}
}

walk(root);
