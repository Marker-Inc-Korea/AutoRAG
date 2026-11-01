/**
 * Live manual QA for the cloud-or-network root handling (#2043).
 *
 * macOS: rclone serves a temp source folder over WebDAV on a loopback port and `mount_webdav` mounts it in a
 * temp folder. A real AutoRAG refresh runs with Jikji and the dupey exact-duplicate exclusion ON in two
 * layouts — the search root is the mount, and the search root is a temp folder holding the mount as a nested
 * folder plus a local file. The rclone request log proves how often each mount file is read: Jikji and dupey
 * read none of them, the refresh reads each changed file once for the parsed mirror and once for its
 * fingerprint (macOS webdavfs serves the second read from its own content cache, so the network sees one read
 * per changed file — measured), an unchanged refresh reads nothing, and the duplicate is excluded on the next
 * refresh without a read. The mount is unmounted and remounted before every measured refresh, because
 * webdavfs keeps a file's contents cached for the life of a mount.
 *
 * Windows: a temp folder is shared over SMB and mapped to a free drive letter, and the same folder is also
 * addressed through its UNC path. The checks that are observable without a server log run against both
 * layouts.
 *
 * Opt-in: AUTORAG_E2E_NETWORK_MOUNT=1
 *   (Windows) an account allowed to create an SMB share and map a drive is required.
 *
 *   bun scripts/manual-qa/run-qa-network-mount-live.ts
 */

import { type ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import {
	accessSync,
	constants,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	readdirSync,
	realpathSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, isAbsolute, join, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createEmbedderFromConfig, releaseEmbedders } from "../../src/embedding/index.ts";
import type { CloudRoot } from "../../src/filesystem/cloud-placeholder.ts";
import {
	canonicalPath,
	DarwinCloudProbe,
	WindowsCloudProbe,
} from "../../src/filesystem/cloud-placeholder.ts";
import { loadCloudFingerprints } from "../../src/mirror/index.ts";

// Isolate before anything constructs the agent: never touch the maintainer's real ~/.autorag.
const qaRoot = mkdtempSync(join(tmpdir(), "autorag-network-mount-live-"));
process.env.AUTORAG_HOME = join(qaRoot, ".autorag-home");
process.env.AUTORAG_CONFIG = join(process.env.AUTORAG_HOME, "config.json");
mkdirSync(process.env.AUTORAG_HOME, { recursive: true });

const TOKEN = `netmount${Date.now().toString(36)}`;

/**
 * How often one changed mount file crosses the wire during a refresh: measured on macOS 26 with rclone 1.75.2.
 * The refresh reads the file twice (the parsed mirror, then its dupey fingerprint) and macOS webdavfs serves
 * the second read from the file content it cached for the mount, so the server logs one `GET` per file.
 */
const MOUNT_NETWORK_READS_PER_CHANGED_FILE = 1;

const DUPLICATE_BODY = `Indexed network fixture alpha, project codename ${TOKEN}.\n`;
const FIXTURE_FILES: readonly { readonly name: string; readonly body: string }[] = [
	{ name: "notes-a.txt", body: DUPLICATE_BODY },
	{ name: "notes-b.txt", body: `Indexed network fixture beta, budget reviewed for ${TOKEN}.\n` },
	{ name: "notes-dup.txt", body: DUPLICATE_BODY },
];

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

function writeFixtures(directory: string): void {
	mkdirSync(directory, { recursive: true });
	for (const fixture of FIXTURE_FILES) writeFileSync(join(directory, fixture.name), fixture.body);
}

/** Paths of every WebDAV `GET` rclone has logged so far — a content read; `PROPFIND` is metadata only. */
function webdavReads(logPath: string): readonly string[] {
	let log: string;
	try {
		log = readFileSync(logPath, "utf8");
	} catch {
		return [];
	}
	const paths: string[] = [];
	for (const line of log.split("\n")) {
		const marker = line.indexOf(": GET from ");
		if (marker < 0) continue;
		const head = line.slice(0, marker);
		const slash = head.indexOf("//");
		paths.push((slash < 0 ? head : head.slice(slash + 2)).trim());
	}
	return paths;
}

/** How many times each logged read names `name`; macOS also reads implicit files like `.DS_Store`. */
function readCount(paths: readonly string[], name: string): number {
	return paths.filter((path) => path === name || path.endsWith(`/${name}`)).length;
}

interface WebdavServer {
	readonly child: ChildProcess;
	readonly port: number;
	readonly logPath: string;
}

/** Start `rclone serve webdav` on a free loopback port and wait until it reports its address. */
async function startWebdav(sourceDir: string, logPath: string): Promise<WebdavServer> {
	for (let attempt = 0; attempt < 12; attempt += 1) {
		rmSync(logPath, { force: true });
		const port = 40000 + Math.floor(Math.random() * 20000);
		const child = spawn(
			"rclone",
			[
				"serve",
				"webdav",
				sourceDir,
				"--addr",
				`127.0.0.1:${port}`,
				"--log-file",
				logPath,
				"--log-level",
				"DEBUG",
				"--dir-cache-time",
				"0",
			],
			{ stdio: "ignore" },
		);
		const deadline = Date.now() + 15_000;
		while (Date.now() < deadline) {
			if (child.exitCode !== null) break;
			if (existsSync(logPath) && readFileSync(logPath, "utf8").includes("Server started on")) {
				return { child, port, logPath };
			}
			await sleep(200);
		}
		child.kill("SIGKILL");
	}
	throw new Error("rclone serve webdav did not start");
}

async function stopWebdav(server: WebdavServer | undefined): Promise<void> {
	if (server === undefined || server.child.exitCode !== null) return;
	const { promise, resolve } = Promise.withResolvers<void>();
	server.child.once("exit", () => resolve());
	server.child.kill("SIGTERM");
	await Promise.race([promise, sleep(10_000)]);
	if (server.child.exitCode === null) server.child.kill("SIGKILL");
}

function mountWebdav(server: WebdavServer, mountDir: string): void {
	mkdirSync(mountDir, { recursive: true });
	const result = spawnSync("/sbin/mount_webdav", ["-S", `http://127.0.0.1:${server.port}/`, mountDir], {
		encoding: "utf8",
	});
	if (result.status !== 0) throw new Error(`mount_webdav failed: ${result.stderr || `exit ${result.status}`}`);
}

function unmountWebdav(mountDir: string): void {
	spawnSync("/sbin/umount", [mountDir], { encoding: "utf8" });
}

/** Unmount and mount again, then wait for the listing: webdavfs caches file contents for a mount's life. */
async function remountWebdav(server: WebdavServer, mountDir: string): Promise<void> {
	unmountWebdav(mountDir);
	await sleep(300);
	mountWebdav(server, mountDir);
	await waitForMount(mountDir);
}

/**
 * Wait until the mount lists its entries. Checked with `readdir` (metadata) and never with a file read: macOS
 * webdavfs caches file contents for the life of a mount, so the first read of a file must be the one under test.
 */
async function waitForMount(mountDir: string): Promise<void> {
	const deadline = Date.now() + 15_000;
	let last = "no entries";
	while (Date.now() < deadline) {
		try {
			const entries = readdirSync(mountDir);
			if (entries.includes("notes-a.txt")) return;
			last = `entries: ${entries.join(", ")}`;
		} catch (error) {
			last = error instanceof Error ? error.message : String(error);
		}
		await sleep(200);
	}
	throw new Error(`the WebDAV mount at ${mountDir} never became readable (${last})`);
}

/** The retrieval hit under `root` whose content carries the token, or undefined. */
async function findHit(agent: AutoRAGAgent, root: string): Promise<{ readonly source: string } | undefined> {
	const { results } = await agent.searchAllDocuments(TOKEN, { topK: 10 });
	const canonicalRoot = `${realpathSync(root)}${sep}`;
	for (const result of results) {
		if (typeof result.source !== "string" || !result.content.includes(TOKEN)) continue;
		try {
			if (realpathSync(result.source).startsWith(canonicalRoot)) return { source: result.source };
		} catch {
			continue;
		}
	}
	return undefined;
}

/** Every product check that needs no server log, shared by the macOS and Windows layouts. */
async function runProductChecks(options: {
	readonly label: string;
	readonly searchPaths: readonly string[];
	readonly mountRoot: string;
	/** Snapshot of the server's content reads so far; absent on a host without a request log. */
	readonly reads?: () => readonly string[];
	/** The search root itself is inside a cloud or network root, so Jikji must be skipped whole. */
	readonly jikjiSkipped: boolean;
	/** The search root itself is inside such a root, so dupey must not scan it at all (no scan result). */
	readonly dupeySkipped: boolean;
	/** Drop the client's file-content cache before a measured call (macOS: unmount and mount again). */
	readonly remount?: () => Promise<void>;
}): Promise<void> {
	const { label, searchPaths, mountRoot, reads, jikjiSkipped, dupeySkipped, remount } = options;
	const fixtureNames = FIXTURE_FILES.map((fixture) => fixture.name);
	const workspace = mkdtempSync(join(qaRoot, "ws-"));
	const sizesBefore = fixtureNames.map((name) => statSync(join(mountRoot, name)).size);
	/** Run `action` after a cache-clearing remount and return how often each fixture was read over the wire. */
	const measure = async (action: () => Promise<void>): Promise<readonly number[]> => {
		await remount?.();
		const before = (reads?.() ?? []).length;
		await action();
		const during = (reads?.() ?? []).slice(before);
		return fixtureNames.map((name) => readCount(during, name));
	};
	try {
		const agent = new AutoRAGAgent({
			searchPaths: [...searchPaths],
			workspacePath: workspace,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
		});

		// Read every mount file once with no other reader, then isolate each such reader on its own: the counts
		// are the direct proof that neither opens a mount file (`PROPFIND` is metadata and never a `GET`).
		let jikjiResults: Awaited<ReturnType<AutoRAGAgent["prepareJikji"]>>;
		const jikjiReads = await measure(async () => {
			jikjiResults = await agent.prepareJikji();
		});
		if (reads !== undefined) {
			check(`${label}: Jikji read no mount file`, jikjiReads.every((count) => count === 0), jikjiReads.join(", "));
		}
		const results = jikjiResults;
		if (jikjiSkipped) {
			check(
				`${label}: Jikji skipped the network root whole`,
				results !== undefined && results.every((result) => result.ok === false && result.reason === "cloud-root-skipped"),
				JSON.stringify(results),
			);
		} else {
			check(
				`${label}: Jikji ran over the root and excluded the nested network folder`,
				results !== undefined && results.every((result) => result.ok),
				JSON.stringify(results),
			);
		}

		let scanCount = -1;
		const dupeyReads = await measure(async () => {
			const details = await agent.scanDuplicateDocuments();
			scanCount = details.scans.length;
		});
		if (dupeySkipped) {
			check(`${label}: dupey never scanned the network root`, scanCount === 0, `${scanCount} scan(s)`);
		}
		if (reads !== undefined) {
			check(`${label}: dupey read no mount file`, dupeyReads.every((count) => count === 0), dupeyReads.join(", "));
		}

		const firstReads = await measure(async () => {
			const first = await agent.refresh(true, { methods: ["parsed", "minsync", "jikji"] });
			const codes = first.diagnostics.map((diagnostic) => `${diagnostic.code}:${diagnostic.severity}`);
			check(`${label}: refresh indexed every fixture`, first.written >= fixtureNames.length, `written=${first.written} ${codes.join(" ")}`);
		});
		if (reads !== undefined) {
			console.log(`    network reads per mount file during the refresh: ${firstReads.join(", ")}`);
			check(
				`${label}: each changed mount file was read ${MOUNT_NETWORK_READS_PER_CHANGED_FILE} time(s) over the wire for the parsed mirror, its fingerprint served from the client cache`,
				firstReads.every((count) => count === MOUNT_NETWORK_READS_PER_CHANGED_FILE),
				firstReads.join(", "),
			);
		}

		const hit = await findHit(agent, mountRoot);
		let readable = false;
		if (hit !== undefined) {
			console.log(`NETWORK_MOUNT_LIVE_QA source=${hit.source}`);
			try {
				accessSync(hit.source, constants.R_OK);
				readable = isAbsolute(hit.source);
			} catch {
				readable = false;
			}
		}
		check(`${label}: search returns the token with an OS-absolute readable source under the mount`, hit !== undefined && readable);

		const fingerprints = new Set(Object.keys(loadCloudFingerprints(workspace).entries).map(canonicalPath));
		check(
			`${label}: a stored fingerprint was recorded for every fixture`,
			fixtureNames.every((name) => fingerprints.has(canonicalPath(join(mountRoot, name)))),
			`${fingerprints.size} entr(ies)`,
		);

		let secondOk = false;
		const secondReads = await measure(async () => {
			const second = await agent.refresh(false, { methods: ["parsed", "minsync", "jikji"] });
			secondOk =
				second.deleted === 1 && second.diagnostics.some((diagnostic) => diagnostic.code === "duplicate-excluded");
		});
		check(`${label}: the duplicate is excluded on the next refresh`, secondOk);
		if (reads !== undefined) {
			check(`${label}: excluding the duplicate read no mount file`, secondReads.every((count) => count === 0), secondReads.join(", "));
		}

		let thirdWritten = -1;
		const thirdReads = await measure(async () => {
			const third = await agent.refresh(false, { methods: ["parsed", "minsync", "jikji"] });
			thirdWritten = third.written;
		});
		check(`${label}: unchanged refresh wrote nothing`, thirdWritten === 0, `written=${thirdWritten}`);
		if (reads !== undefined) {
			check(`${label}: an unchanged refresh read no mount file`, thirdReads.every((count) => count === 0), thirdReads.join(", "));
		}
		check(
			`${label}: the mount files were never modified`,
			fixtureNames.every((name, index) => statSync(join(mountRoot, name)).size === sizesBefore[index]),
		);
	} catch (error) {
		check(`${label}: harness completed`, false, error instanceof Error ? error.message : String(error));
	} finally {
		rmSync(workspace, { recursive: true, force: true });
	}
}

async function runDarwin(): Promise<void> {
	const probe = new DarwinCloudProbe();
	const embedder = createEmbedderFromConfig({});
	const health = await embedder.health();
	console.log(`embedding backend: ${health.backend} alive=${health.alive}${health.detail ? ` (${health.detail})` : ""}`);
	check("an embedding backend is alive (run `minsync serve --idle-timeout 0` first)", health.alive);

	const sourceDir = join(qaRoot, "webdav-source");
	const holderDir = join(qaRoot, "holder");
	const mountDir = join(holderDir, "mnt");
	const logPath = join(qaRoot, "rclone.log");
	let server: WebdavServer | undefined;
	let mounted = false;
	try {
		writeFixtures(sourceDir);
		mkdirSync(holderDir, { recursive: true });
		writeFileSync(join(holderDir, "local.txt"), `A local file beside the mount for ${TOKEN}.\n`);
		server = await startWebdav(sourceDir, logPath);
		const webdav = server;
		mountWebdav(webdav, mountDir);
		mounted = true;
		await waitForMount(mountDir);

		// The mount is reported at its real path while the search root is the path we passed, so detection must
		// canonicalize both sides (`/private/var/...` for a `/var/...` mount).
		const networkRoots = await probe.cloudRoots([mountDir]);
		const mountRealPath = realpathSync(mountDir);
		check(
			"the WebDAV mount is classified as a network root under its real path",
			networkRoots.some((root: CloudRoot) => root.kind === "network" && root.path === mountRealPath),
			networkRoots.map((root) => `${root.kind}:${root.path}`).join(" "),
		);

		// Layout (a): the search root is the mount itself.
		await runProductChecks({
			label: "search root = the mount",
			searchPaths: [mountDir],
			mountRoot: mountDir,
			reads: () => webdavReads(logPath),
			jikjiSkipped: true,
			dupeySkipped: true,
			remount: () => remountWebdav(webdav, mountDir),
		});

		// Layout (b): the search root is a temp folder holding the mount as a nested folder plus a local file.
		await runProductChecks({
			label: "search root = a folder holding the mount",
			searchPaths: [holderDir],
			mountRoot: mountDir,
			reads: () => webdavReads(logPath),
			// The mount is nested under this root, so Jikji runs and must exclude it rather than skip the root.
			jikjiSkipped: false,
			dupeySkipped: false,
			remount: () => remountWebdav(webdav, mountDir),
		});
	} catch (error) {
		check("darwin network-mount harness completed", false, error instanceof Error ? error.message : String(error));
	} finally {
		if (mounted) unmountWebdav(mountDir);
		await stopWebdav(server);
		await releaseEmbedders();
	}
}

/** Whether two Windows paths name the same location: `realpath` may add a trailing separator to a share root. */
function sameWindowsPath(left: string, right: string): boolean {
	const normalize = (value: string): string => value.replaceAll("/", "\\").replace(/\\+$/u, "").toLowerCase();
	return normalize(left) === normalize(right);
}

/** A drive letter that is not currently used. */
function freeDriveLetter(): string {
	for (let index = 0; index < 8; index += 1) {
		const letter = String.fromCharCode("Y".charCodeAt(0) - index);
		if (!existsSync(`${letter}:\\`)) return letter;
	}
	throw new Error("no free drive letter");
}

async function runWindows(): Promise<void> {
	if (process.env.AUTORAG_E2E_NETWORK_MOUNT !== "1") {
		check("AUTORAG_E2E_NETWORK_MOUNT=1 is set for Windows network-mount QA", false);
		return;
	}
	const probe = new WindowsCloudProbe();
	const health = await createEmbedderFromConfig({}).health();
	console.log(`embedding backend: ${health.backend} alive=${health.alive}${health.detail ? ` (${health.detail})` : ""}`);
	check("an embedding backend is alive (run `minsync serve --idle-timeout 0` first)", health.alive);

	const shareDir = join(qaRoot, `autorag-e2e-${Math.random().toString(16).slice(2, 10)}`);
	const shareName = basename(shareDir);
	const drive = freeDriveLetter();
	const unc = `\\\\localhost\\${shareName}`;
	let shared = false;
	let mapped = false;
	try {
		writeFixtures(shareDir);
		execFileSync(
			"powershell.exe",
			[
				"-NoProfile",
				"-NonInteractive",
				"-Command",
				`New-SmbShare -Name '${shareName}' -Path '${shareDir}' -FullAccess "$env:USERNAME" | Out-Null`,
			],
			{ encoding: "utf8" },
		);
		shared = true;
		execFileSync("net", ["use", `${drive}:`, unc, "/persistent:no"], { encoding: "utf8" });
		mapped = true;

		const roots = await probe.cloudRoots([`${drive}:\\`, unc]);
		check(
			"the mapped drive and the UNC share are classified as network roots",
			roots.some((root) => root.kind === "network" && sameWindowsPath(root.path, `${drive}:\\`)) &&
				roots.some((root) => root.kind === "network" && sameWindowsPath(root.path, unc)),
			roots.map((root) => `${root.kind}:${root.path}`).join(" "),
		);

		await runProductChecks({
			label: "search root = the mapped drive",
			searchPaths: [`${drive}:\\`],
			mountRoot: `${drive}:\\`,
			jikjiSkipped: true,
			dupeySkipped: true,
		});
		await runProductChecks({
			label: "search root = the UNC path",
			searchPaths: [unc],
			mountRoot: unc,
			jikjiSkipped: true,
			dupeySkipped: true,
		});
	} catch (error) {
		check("windows network-mount harness completed", false, error instanceof Error ? error.message : String(error));
	} finally {
		if (mapped) spawnSync("net", ["use", `${drive}:`, "/delete", "/y"], { encoding: "utf8" });
		if (shared) {
			spawnSync(
				"powershell.exe",
				["-NoProfile", "-NonInteractive", "-Command", `Remove-SmbShare -Name '${shareName}' -Force`],
				{ encoding: "utf8" },
			);
		}
		rmSync(shareDir, { recursive: true, force: true });
		await releaseEmbedders();
	}
}

try {
	if (process.platform === "darwin") await runDarwin();
	else if (process.platform === "win32") await runWindows();
	else check("platform is macOS or Windows", false, process.platform);
	console.log(failures === 0 ? "\nNETWORK_MOUNT LIVE QA PASSED" : `\nNETWORK_MOUNT LIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(qaRoot, { recursive: true, force: true });
}
