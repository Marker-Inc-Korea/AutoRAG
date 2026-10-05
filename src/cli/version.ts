import { readFileSync } from "node:fs";

/**
 * The installed package version. The manifest sits two levels above this module
 * in both layouts: `src/cli/version.ts` and the bundled `dist/cli/index.js`.
 */
export function readPackageVersion(): string {
	try {
		const manifest = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")) as {
			version?: unknown;
		};
		return typeof manifest.version === "string" && manifest.version.length > 0 ? manifest.version : "0.0.0";
	} catch {
		return "0.0.0";
	}
}
