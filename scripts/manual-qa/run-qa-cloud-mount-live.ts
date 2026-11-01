/**
 * Live manual QA for the cloud-mount release path (#2043).
 *
 * macOS walks the real user-visible path: fixture documents are written into a throwaway folder inside a
 * cloud mount (iCloud Drive, Google Drive, OneDrive, Dropbox under `~/Library/CloudStorage` / `~/Library/Mobile
 * Documents`), released so they are data-less placeholders, then indexed through a real AutoRAG refresh and
 * found again through a real search — with an OS-absolute, readable source under the mount. It then proves the
 * refresh released them a second time and that an unchanged refresh writes nothing and hydrates nothing.
 * Opt-in and destructive on macOS: it writes into the user's real cloud, so it only runs when
 * `AUTORAG_E2E_CLOUD_DIRS` is a `:`-delimited list of existing writable parent folders inside cloud mounts. In
 * each it creates exactly one `autorag-e2e-<hex>/` folder and deletes only that folder.
 *
 * Windows runs the same checks against a synthetic Cloud Files provider
 * (`windows-cloud-files-provider.ps1`) that this harness registers in its own temp folder and always
 * unregisters; it never touches a real sync root. Opt in with `AUTORAG_E2E_CLOUD_SYNTHETIC=1`. Windows denies
 * downloads to third-party processes that cannot show its download prompt (error 395, e.g. over SSH in session
 * 0), so run it in an interactive desktop session.
 *
 * Run: AUTORAG_E2E_CLOUD_DIRS="$HOME/Library/CloudStorage/GoogleDrive-x/My Drive:..." \
 *        bun scripts/manual-qa/run-qa-cloud-mount-live.ts
 *      (Windows) set AUTORAG_E2E_CLOUD_SYNTHETIC=1 && bun scripts/manual-qa/run-qa-cloud-mount-live.ts
 */

import { type ChildProcess, execFileSync, spawn } from "node:child_process";
import {
	accessSync,
	appendFileSync,
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
import { basename, delimiter, isAbsolute, join, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createEmbedderFromConfig, releaseEmbedders } from "../../src/embedding/index.ts";
import {
	buildWindowsSyncRootsScript,
	DarwinCloudProbe,
	parseWindowsSyncRoots,
	WindowsCloudProbe,
} from "../../src/filesystem/cloud-placeholder.ts";

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

/** Sync roots currently registered for this user, read the way the probe reads them. */
function registeredSyncRoots(): string[] {
	const stdout = execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", buildWindowsSyncRootsScript()], {
		encoding: "utf8",
	});
	return parseWindowsSyncRoots(stdout).map((root) => root.path.toLowerCase().replace(/\\+$/u, ""));
}

/** Start the synthetic provider and wait until it has registered the root and created the placeholders. */
async function startSyntheticProvider(syncRoot: string, sourceDir: string): Promise<ChildProcess> {
	const providerScript = fileURLToPath(new URL("./windows-cloud-files-provider.ps1", import.meta.url));
	const child = spawn(
		"powershell.exe",
		["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", providerScript, "-SyncRoot", syncRoot, "-SourceRoot", sourceDir],
		{ stdio: ["pipe", "pipe", "inherit"] },
	);
	let output = "";
	child.stdout?.on("data", (chunk) => {
		output += String(chunk);
	});
	const deadline = Date.now() + 60_000;
	while (!output.includes("PROVIDER=READY")) {
		if (child.exitCode !== null || Date.now() > deadline) {
			throw new Error(`synthetic provider did not start: ${output.trim() || `exit ${child.exitCode}`}`);
		}
		await sleep(200);
	}
	return child;
}

async function stopSyntheticProvider(child: ChildProcess | undefined): Promise<void> {
	if (child === undefined || child.exitCode !== null) return;
	const { promise: exited, resolve } = Promise.withResolvers<void>();
	child.once("exit", () => resolve());
	child.stdin?.end("stop\n");
	await Promise.race([exited, sleep(15_000)]);
	if (child.exitCode === null) child.kill();
}

/**
 * Windows: the same product checks as macOS against a synthetic Cloud Files sync root registered in this run's
 * temp folder, plus the two refusals the platform enforces on release (unsynced changes, pinned file).
 */
async function runWindows(): Promise<void> {
	if (process.env.AUTORAG_E2E_CLOUD_SYNTHETIC !== "1") {
		check("AUTORAG_E2E_CLOUD_SYNTHETIC=1 is set for Windows Cloud Files QA", false);
		return;
	}
	const probe = new WindowsCloudProbe();
	const health = await createEmbedderFromConfig({}).health();
	console.log(`embedding backend: ${health.backend} alive=${health.alive}${health.detail ? ` (${health.detail})` : ""}`);
	check("an embedding backend is alive (run `minsync serve --idle-timeout 0` first)", health.alive);

	const sourceDir = mkdtempSync(join(qaRoot, "synthetic-source-"));
	const syncRoot = join(qaRoot, `autorag-e2e-${Math.random().toString(16).slice(2, 10)}`);
	const workspace = mkdtempSync(join(qaRoot, "ws-"));
	const names = ["notes-a.txt", "notes-b.txt", "notes-c.txt", "notes-d.txt"];
	const files = names.map((name) => join(syncRoot, name));
	const [, , unsynced, pinned] = files as [string, string, string, string];
	for (const name of names) writeFileSync(join(sourceDir, name), `Indexed cloud fixture ${name} for ${TOKEN}.\n`);
	check("no synthetic root is registered before the lane", !registeredSyncRoots().includes(syncRoot.toLowerCase()));
	let child: ChildProcess | undefined;
	try {
		child = await startSyntheticProvider(syncRoot, sourceDir);
		check("the synthetic files are placeholders", (await probe.datalessAmong(files)).size === files.length);

		const agent = new AutoRAGAgent({
			searchPaths: [syncRoot],
			workspacePath: workspace,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			// Same placeholder-unaware readers the macOS branch turns off (issue #2043 item 4).
			jikji: false,
			excludeExactDuplicates: false,
		});
		const first = await agent.refresh(true, { methods: ["parsed", "minsync"] });
		const firstCodes = first.diagnostics.map((diagnostic) => `${diagnostic.code}: ${diagnostic.message}`);
		check("refresh indexed the placeholders", first.written === files.length, `written=${first.written} ${firstCodes.join(" | ")}`);

		const { results } = await agent.searchAllDocuments(TOKEN, { topK: 10 });
		const hit = results.find(
			(result) =>
				typeof result.source === "string" &&
				isAbsolute(result.source) &&
				result.source.toLowerCase().startsWith(`${syncRoot.toLowerCase()}${sep}`) &&
				result.content.includes(TOKEN),
		);
		if (hit !== undefined) console.log(`CLOUD_MOUNT_LIVE_QA source=${hit.source}`);
		check(
			"search returns the token with an absolute source under the sync root",
			hit !== undefined,
			hit === undefined ? `${results.length} result(s): ${results.map((result) => result.source).join(", ")}` : undefined,
		);
		check("refresh released the files again", (await probe.datalessAmong(files)).size === files.length);

		const second = await agent.refresh(false, { methods: ["parsed", "minsync"] });
		check("unchanged refresh wrote nothing", second.written === 0, `written=${second.written}`);
		check("unchanged refresh hydrated nothing", (await probe.datalessAmong(files)).size === files.length);

		// Platform refusals, through the real release call. A local edit leaves the file out of sync with the cloud.
		appendFileSync(unsynced, "local edit the provider has not synced\n");
		const editedBytes = readFileSync(unsynced, "utf8");
		const unsyncedResult = await probe.evict(unsynced);
		check(
			"a file with unsynced changes is not released and keeps its bytes",
			!unsyncedResult.ok && readFileSync(unsynced, "utf8") === editedBytes,
			unsyncedResult.ok ? "released" : unsyncedResult.message,
		);
		readFileSync(pinned, "utf8");
		execFileSync("attrib.exe", ["+P", pinned]);
		const pinnedResult = await probe.evict(pinned);
		check(
			"a pinned file is not released",
			!pinnedResult.ok && !(await probe.datalessAmong([pinned])).has(pinned),
			pinnedResult.ok ? "released" : pinnedResult.message,
		);
	} catch (error) {
		check("Windows synthetic harness completed", false, error instanceof Error ? error.message : String(error));
	} finally {
		await stopSyntheticProvider(child);
		check("the synthetic root is unregistered after the lane", !registeredSyncRoots().includes(syncRoot.toLowerCase()));
		rmSync(sourceDir, { recursive: true, force: true });
		rmSync(syncRoot, { recursive: true, force: true });
		rmSync(workspace, { recursive: true, force: true });
		await releaseEmbedders();
	}
}

try {
	const dirs = (process.env.AUTORAG_E2E_CLOUD_DIRS ?? "")
		.split(delimiter)
		.map((dir) => dir.trim())
		.filter(Boolean);
	if (process.platform === "darwin") {
		if (dirs.length === 0) {
			console.log("cloud-mount live QA needs AUTORAG_E2E_CLOUD_DIRS (platform path-delimited cloud folders)");
			process.exitCode = 1;
		} else await runDarwin(dirs);
	} else if (process.platform === "win32") {
		await runWindows();
	} else {
		check("platform is macOS or Windows", false, process.platform);
	}
	console.log(failures === 0 ? "\nCLOUD_MOUNT LIVE QA PASSED" : `\nCLOUD_MOUNT LIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(qaRoot, { recursive: true, force: true });
}
