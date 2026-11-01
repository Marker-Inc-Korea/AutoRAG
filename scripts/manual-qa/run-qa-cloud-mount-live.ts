/**
 * Live manual QA for the cloud-mount release path (#2043).
 *
 * Walks the real user-visible path on a Mac: fixture documents are written into a throwaway folder inside a
 * cloud mount (iCloud Drive, Google Drive, OneDrive under `~/Library/CloudStorage` / `~/Library/Mobile
 * Documents`), released so they are data-less placeholders, then indexed through a real AutoRAG refresh and
 * found again through a real search — with an OS-absolute, readable source under the mount. It then proves the
 * refresh released them a second time and that an unchanged refresh writes nothing and hydrates nothing.
 *
 * Opt-in and destructive: it writes into the user's real cloud. It only runs when `AUTORAG_E2E_CLOUD_DIRS` is
 * a platform path-delimited list (`:` on POSIX, `;` on Windows) of existing writable parent folders inside
 * cloud mounts. In each it creates exactly one `autorag-e2e-<hex>/` folder and deletes only that folder.
 *
 * Run: AUTORAG_E2E_CLOUD_DIRS="$HOME/Library/CloudStorage/GoogleDrive-x/My Drive:..." \
 *        bun scripts/manual-qa/run-qa-cloud-mount-live.ts
 */

import {
	accessSync,
	constants,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, delimiter, join, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createEmbedderFromConfig, releaseEmbedders } from "../../src/embedding/index.ts";
import { DarwinCloudProbe } from "../../src/filesystem/cloud-placeholder.ts";

// Isolate before anything constructs the agent: never touch the maintainer's real ~/.autorag. No imported
// module reads these at import time, so setting them in the module body (after the imports) is enough.
const qaRoot = mkdtempSync(join(tmpdir(), "autorag-cloud-mount-live-"));
process.env.AUTORAG_HOME = join(qaRoot, ".autorag-home");
process.env.AUTORAG_CONFIG = join(process.env.AUTORAG_HOME, "config.json");
mkdirSync(process.env.AUTORAG_HOME, { recursive: true });

const TOKEN = `cloudmount${Date.now().toString(36)}`;

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

/** Poll the release helper until the provider has uploaded the just-written file and accepts the release. */
async function releaseWhenUploaded(probe: DarwinCloudProbe, file: string): Promise<string> {
	const deadline = Date.now() + 180_000;
	let result = await probe.evict(file);
	while (!result.ok && Date.now() < deadline) {
		await sleep(2_000);
		result = await probe.evict(file);
	}
	return result.ok ? "" : result.message;
}

async function runDarwin(dirs: readonly string[]): Promise<void> {
	const probe = new DarwinCloudProbe();
	const embedder = createEmbedderFromConfig({});
	const health = await embedder.health();
	console.log(`embedding backend: ${health.backend} alive=${health.alive}${health.detail ? ` (${health.detail})` : ""}`);
	check("an embedding backend is alive (run `minsync serve --idle-timeout 0` first)", health.alive);

	for (const dir of dirs) {
		const fixtureDir = join(dir, `autorag-e2e-${Math.random().toString(16).slice(2, 10)}`);
		const files = [join(fixtureDir, "notes-a.txt"), join(fixtureDir, "notes-b.txt")];
		const workspace = mkdtempSync(join(qaRoot, "ws-"));
		try {
			mkdirSync(fixtureDir, { recursive: true });
			writeFileSync(files[0]!, `Indexed cloud fixture alpha, project codename ${TOKEN}.\n`);
			writeFileSync(files[1]!, `Indexed cloud fixture beta, budget reviewed for ${TOKEN}.\n`);
			const before = new Map(files.map((file) => [file, statSync(file).size]));

			for (const file of files) {
				const refusal = await releaseWhenUploaded(probe, file);
				check(`${dir} (${basename(file)}): released to a placeholder`, refusal === "", refusal);
			}
			// Settle the placeholder the way a cloud file that has synced already is settled: the first hydrate of a
			// file written on this Mac rounds its mtime to the cloud's precision (measured ~100 ns), after which it is
			// stable. Without this, the next refresh re-downloads the file once (see docs/minsync-setup.md).
			for (const file of files) readFileSync(file, "utf8");
			for (const file of files) await releaseWhenUploaded(probe, file);
			const dataless = await probe.datalessAmong(files);
			check(`${dir}: both fixtures are data-less placeholders`, dataless.size === files.length);

			const agent = new AutoRAGAgent({
				searchPaths: [fixtureDir],
				workspacePath: workspace,
				minSync: { autoInstall: false },
				everything: false,
				fsearch: false,
				// Jikji's prepare and the dupey exact-duplicate scan open file contents themselves and re-download
				// released placeholders (measured; tracked separately, issue #2043 item 4). The release path is what
				// this lane verifies, so both placeholder-unaware readers are off, exactly as the #2044 measurement did.
				jikji: false,
				excludeExactDuplicates: false,
			});
			const first = await agent.refresh(true, { methods: ["parsed", "minsync"] });
			check(`${dir}: refresh indexed the placeholders`, first.written === files.length, `written=${first.written}`);

			const { results } = await agent.searchAllDocuments(TOKEN, { topK: 10 });
			// Compare canonically: the retrieval source and the fixture path can differ in symlink resolution and
			// Unicode normalization (the OneDrive mount name is Korean), so a raw prefix test is fragile.
			const fixtureRoot = `${realpathSync(fixtureDir)}${sep}`;
			const hit = results.find(
				(result) =>
					typeof result.source === "string" &&
					realpathSync(result.source).startsWith(fixtureRoot) &&
					result.content.includes(TOKEN),
			);
			if (hit !== undefined) console.log(`CLOUD_MOUNT_LIVE_QA source=${hit.source}`);
			let readable = false;
			if (hit?.source !== undefined) {
				try {
					accessSync(hit.source, constants.R_OK);
					readable = true;
				} catch {
					readable = false;
				}
			}
			// Checked with access, never a read: reading the source would hydrate the placeholder we just released.
			check(
				`${dir}: search returns the token with an absolute readable source under the mount`,
				hit !== undefined && readable,
				hit === undefined ? `${results.length} result(s): ${results.map((result) => result.source).join(", ")}` : undefined,
			);

			const releasedAfterRefresh = await probe.datalessAmong(files);
			check(`${dir}: refresh released the files again`, releasedAfterRefresh.size === files.length);

			const second = await agent.refresh(false, { methods: ["parsed", "minsync"] });
			const stillDataless = await probe.datalessAmong(files);
			const sizesUnchanged = files.every((file) => statSync(file).size === before.get(file));
			check(`${dir}: unchanged refresh wrote nothing`, second.written === 0, `written=${second.written}`);
			check(
				`${dir}: unchanged refresh hydrated nothing and left the files released`,
				stillDataless.size === files.length && sizesUnchanged,
			);
		} catch (error) {
			check(`${dir}: harness completed`, false, error instanceof Error ? error.message : String(error));
		} finally {
			// Delete only the folder this harness created; deleting inside a mount deletes it from the cloud.
			rmSync(fixtureDir, { recursive: true, force: true });
			rmSync(workspace, { recursive: true, force: true });
		}
	}
	await releaseEmbedders();
}

/**
 * Windows Cloud Files branch (unpin a placeholder with `attrib +u`). Seam for the stacked
 * `feat/2043-windows-cloud-release` sibling, which implements and verifies it on a signed-in account.
 */
async function runWindows(dirs: readonly string[]): Promise<void> {
	console.log(
		`SKIP  windows cloud-mount branch is not implemented in this checkout; ${dirs.length} dir(s) configured`,
	);
	process.exitCode = 1;
}

try {
	const dirs = (process.env.AUTORAG_E2E_CLOUD_DIRS ?? "")
		.split(delimiter)
		.map((dir) => dir.trim())
		.filter(Boolean);
	if (dirs.length === 0) {
		console.log("cloud-mount live QA needs AUTORAG_E2E_CLOUD_DIRS (platform path-delimited cloud folders)");
		process.exitCode = 1;
	} else if (process.platform === "darwin") {
		await runDarwin(dirs);
	} else if (process.platform === "win32") {
		await runWindows(dirs);
	} else {
		check("platform is macOS or Windows", false, process.platform);
	}
	console.log(failures === 0 ? "\nCLOUD_MOUNT LIVE QA PASSED" : `\nCLOUD_MOUNT LIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(qaRoot, { recursive: true, force: true });
}
