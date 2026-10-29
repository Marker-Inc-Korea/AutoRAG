/**
 * Live E2E: every route in the conversion matrix the system prompt hands the
 * model is true for the real filee-cli (`filee-cli formats <format>`), so a
 * Filee upgrade that drops a route fails here instead of misleading the model.
 *
 *   AUTORAG_FILEE_LIVE=1 bun run vitest run test/live-e2e/filee-matrix.test.ts
 *
 * Uses `AUTORAG_FILEE_BIN` or a `filee-cli` on PATH.
 */
import { describe, expect, it } from "vitest";
import {
	type FileeCliOptions,
	listFileeFormats,
	ROUTES,
	reachableTargets,
	SOURCE_FAMILIES,
} from "../../src/convert/index.ts";

const LIVE = process.env.AUTORAG_FILEE_LIVE === "1";
const filee: FileeCliOptions = process.env.AUTORAG_FILEE_BIN ? { executable: process.env.AUTORAG_FILEE_BIN } : {};

/** Targets are the comma-separated ids on the indented lines of `formats <id>` (both engine sections). */
function parseTargets(output: string): Set<string> {
	const targets = new Set<string>();
	for (const line of output.split(/\r?\n/)) {
		if (!line.startsWith("  ")) continue;
		for (const item of line.split(",")) {
			const id = item.trim();
			if (id.length > 0 && !id.startsWith("(")) targets.add(id);
		}
	}
	return targets;
}

describe.skipIf(!LIVE)("live: the conversion matrix matches filee-cli", () => {
	it("every family member converts to every format its family route promises", async () => {
		const problems: string[] = [];
		for (const [family, members] of Object.entries(SOURCE_FAMILIES)) {
			const promised = reachableTargets(family);
			for (const member of members) {
				const targets = parseTargets(await listFileeFormats(member, filee));
				for (const target of promised) {
					if (target !== member && !targets.has(target)) problems.push(`${member} -> ${target}`);
				}
			}
		}
		expect(problems).toEqual([]);
		expect(Object.keys(ROUTES).length).toBeGreaterThan(10);
	}, 300_000);
});
