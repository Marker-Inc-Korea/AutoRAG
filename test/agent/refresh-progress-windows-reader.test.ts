import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readRefreshProgress } from "../../src/agent/refresh-progress.ts";
import { AutoRAGAgent } from "../../src/index.ts";
import { refreshProgressPath } from "../../src/mirror/paths.ts";

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-progress-reader-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let index = 0; index < 400; index += 1) writeFileSync(join(docs, `note-${index}.md`), `# Note ${index}\n`);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

// Windows refuses to replace a file another handle has open without delete-sharing (EPERM on rename);
// POSIX renames over open files, so this race only exists there.
describe.runIf(process.platform === "win32")("refresh() with a concurrent reader of refresh-progress.json", () => {
	it("completes while another reader keeps opening the marker", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
		});
		let polling = true;
		let reads = 0;
		const poller = (async () => {
			while (polling) {
				// Back-to-back opens keep a read handle on the marker for most of the run.
				await readFile(refreshProgressPath(root), "utf8").then(
					() => {
						reads += 1;
					},
					() => undefined,
				);
			}
		})();

		try {
			await agent.refresh(true, { methods: ["parsed"] });
		} finally {
			polling = false;
			await poller;
		}

		expect(reads).toBeGreaterThan(0);
		expect(readRefreshProgress(root)).toMatchObject({ state: "success", parsedCounts: { scanned: 400 } });
	}, 120_000);
});
