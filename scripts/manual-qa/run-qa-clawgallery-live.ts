/**
 * Live ClawGallery QA for issue #1477.
 *
 * Prerequisites:
 *   - `cargo install clawgallery`
 *   - a local folder containing at least one supported image
 *
 * Usage:
 *   bun scripts/manual-qa/run-qa-clawgallery-live.ts /path/to/images "login error"
 *
 * The script uses a temporary ClawGallery config directory and only performs
 * bootstrap plus read-only keyword search. Set CLAWGALLERY_VDR_BACKEND=vsplade
 * to also run the trusted V-SPLADE sync when its runtime is installed.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ClawGalleryClient } from "../../src/datasource/skills/clawgallery/client.ts";

const folder = process.argv[2];
const query = process.argv.slice(3).join(" ").trim() || "screenshot";
if (folder === undefined || folder.length === 0) {
	console.error('Usage: bun scripts/manual-qa/run-qa-clawgallery-live.ts /path/to/images "query"');
	process.exit(2);
}

const configDir = mkdtempSync(join(tmpdir(), "autorag-clawgallery-live-"));
try {
	const client = new ClawGalleryClient({
		configDir,
		path: folder,
		...(process.env.CLAWGALLERY_BINARY !== undefined ? { binaryPath: process.env.CLAWGALLERY_BINARY } : {}),
		...(process.env.CLAWGALLERY_PYTHON !== undefined ? { env: { CLAWGALLERY_PYTHON: process.env.CLAWGALLERY_PYTHON } } : {}),
	});
	const bootstrap = await client.bootstrap();
	if (!bootstrap.ok) throw new Error(`bootstrap failed: ${bootstrap.reason}`);
	if (process.env.CLAWGALLERY_VDR_BACKEND !== undefined) {
		const visual = await client.syncVisual();
		if (!visual.ok) console.warn(`visual sync warning: ${visual.reason}`);
	}
	const result = await client.search("hybrid", query, { topK: 5 });
	if (!result.ok) throw new Error(`search failed: ${result.reason}`);
	console.log(JSON.stringify({ configDir, query, indexed: bootstrap.data.indexed, hits: result.hits }, null, 2));
} finally {
	rmSync(configDir, { recursive: true, force: true });
}
