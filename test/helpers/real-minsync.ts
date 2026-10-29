import { existsSync } from "node:fs";
import { basename, delimiter, join } from "node:path";

/**
 * The developer's own `minsync`, skipping the inert stand-in `test/setup/fake-minsync-on-path.ts`
 * prepends to PATH. Live tests that retrieve from a real index need this binary.
 */
export function realMinSync(): string {
	const name = process.platform === "win32" ? "minsync.exe" : "minsync";
	for (const directory of (process.env.PATH ?? "").split(delimiter)) {
		if (directory === "" || basename(directory).startsWith("autorag-fake-minsync-path-")) continue;
		const candidate = join(directory, name);
		if (existsSync(candidate)) return candidate;
	}
	throw new Error("a real `minsync` on PATH is required for this live test");
}
