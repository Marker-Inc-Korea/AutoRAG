import { accessSync, constants, existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = resolve(fileURLToPath(new URL("../..", import.meta.url)));

const NATIVE_LANES = Object.freeze([
	{ name: "lazykatok", binary: "lazykatok", command: "scripts/manual-qa/run-qa-lazykatok-live.ts", identityPattern: /\/kakao\/[^/]+\/chunks\/[^/\s]+/u },
	{ name: "discrawl", binary: "discrawl", command: "scripts/manual-qa/run-qa-discrawl-live.ts", identityPattern: /\/discord\/[^\s]+/u },
	{ name: "wacrawl", binary: "wacrawl" },
	{ name: "telecrawl", binary: "telecrawl" },
	{ name: "slacrawl", binary: "slacrawl" },
	{ name: "lark", binary: "lark-cli", command: "scripts/manual-qa/run-qa-lark-live.ts", identityPattern: /\/lark\/[^/]+\/(?:messages|docs)\/[^/\s]+/u },
	{ name: "notcrawl", binary: "notcrawl" },
	{ name: "obsidian", binary: "minsync", command: "scripts/manual-qa/run-qa-obsidian-live.ts", identityPattern: /\/obsidian\/[^/\s]+\/[^\s]+\.md/u },
	{ name: "mailcrawl", binary: "mailcrawl", command: "scripts/manual-qa/run-qa-mailcrawl-live.ts", identityPattern: /\/mailcrawl\/[^\s]+/u },
	{ name: "hister", binary: "hister", command: "scripts/manual-qa/run-qa-hister-live.ts", identityPattern: /\/browser\/[^\s]+\/pages\/[^\s]+/u },
	// Opt-in. macOS writes into the user's real cloud mounts (AUTORAG_E2E_CLOUD_DIRS names the parent folders);
	// Windows registers its own synthetic Cloud Files provider in a temp folder (AUTORAG_E2E_CLOUD_SYNTHETIC=1).
	// Its identity is an OS-absolute source under the throwaway `autorag-e2e-<hex>` folder the harness creates.
	{
		name: "cloud-mount",
		binary: process.platform === "win32" ? "powershell.exe" : "osascript",
		command: "scripts/manual-qa/run-qa-cloud-mount-live.ts",
		identityPattern: /[\\/]autorag-e2e-[0-9a-f]+[\\/]/u,
	},
	// Opt-in. macOS starts its own rclone WebDAV server on a loopback port and mounts it in a temp folder;
	// Windows shares a temp folder over SMB and maps it to a free drive letter plus its UNC path. Nothing the
	// user owns is touched, and every mount, share, mapping, server and temp folder is removed again.
	{
		name: "network-mount",
		binary: process.platform === "win32" ? "powershell.exe" : "rclone",
		command: "scripts/manual-qa/run-qa-network-mount-live.ts",
		identityPattern: /[\\/]autorag-(?:network-mount-live|e2e)-[A-Za-z0-9]+[\\/]/u,
	},
]);

// Native-store presence probes: a lane runs for real when its CLI's own
// default store exists on this host. An explicit
// E2E_DATASOURCE_<NAME>_CONFIGURED=1 still forces a lane on (the harness then
// fails loudly if the store is unusable); the probe only decides the default.
const NATIVE_STORE_PROBES = Object.freeze({
	lazykatok: () => existsSync(join(homedir(), "Library", "Application Support", "katok", "archive.sqlite3")),
	discrawl: () => existsSync(join(homedir(), "Library", "Application Support", "discrawl", "discrawl.db")) || existsSync(join(homedir(), ".discrawl", "discrawl.db")),
	wacrawl: () => existsSync(join(homedir(), ".wacrawl", "wacrawl.db")),
	telecrawl: () => existsSync(join(homedir(), ".telecrawl", "telecrawl.db")),
	slacrawl: () => existsSync(join(homedir(), ".slacrawl", "config.toml")),
	notcrawl: () => existsSync(join(homedir(), ".notcrawl", "notcrawl.db")),
	// Self-contained: the harness builds its own vault fixture, so the lane needs `minsync` and nothing else.
	obsidian: () => true,
	mailcrawl: () => existsSync(join(homedir(), ".mailcrawl")),
	// Opt-in: macOS writes to the user's real cloud, so it runs only when the operator names the parent folders;
	// Windows runs only when the synthetic provider is requested. Otherwise the lane is SKIP.
	"cloud-mount": () =>
		process.platform === "win32"
			? process.env.AUTORAG_E2E_CLOUD_SYNTHETIC === "1"
			: (process.env.AUTORAG_E2E_CLOUD_DIRS ?? "").trim().length > 0,
	// Opt-in: the harness builds and removes its own loopback WebDAV mount / temp SMB share, so it runs only
	// when the operator asks for it.
	"network-mount": () => process.env.AUTORAG_E2E_NETWORK_MOUNT === "1",
});

function defaultConfigured(name) {
	if (process.env[`E2E_DATASOURCE_${name.toUpperCase()}_CONFIGURED`] === "1") return true;
	const probe = NATIVE_STORE_PROBES[name];
	if (probe === undefined) return false;
	try {
		return probe();
	} catch {
		return false;
	}
}

export function buildDatasourceMatrix() {
	return [{ name: "local", kind: "local" }, ...NATIVE_LANES.map((lane) => ({ ...lane, kind: "native" }))];
}

// Default: run every registered lane (local + all native datasources). Native
// lanes without a binary or a configured native store still report SKIP with a
// reason; E2E_DATASOURCES only ever narrows this default.
export function parseDatasourceSelection(value = process.env.E2E_DATASOURCES || "local,configured") {
	return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean))];
}

export function validateNativeIdentity(source, laneName) {
	if (typeof source !== "string") return false;
	// Canonical lazykatok source: /kakao/<instance>/chunks/<chunk>. Opaque slash
	// hierarchy, not an OS path. Retired kakao:<chat>/<sender>/<chunk> is invalid.
	if (laneName === "lazykatok") return /^\/kakao\/[^/]+\/chunks\/[^/]+$/u.test(source);
	return !source.startsWith("/") && /^[a-z][a-z0-9-]*:.+/u.test(source);
}

export function sanitizeDiagnostic(value) {
	return String(value)
		.replace(/(token|password|secret|credential)\s*[=:]\s*[^\s,;]+/giu, "$1=[redacted]")
		.replace(/\/Users\/[^\s,;]+/gu, "[path-redacted]")
		.replace(/\/home\/[^\s,;]+/gu, "[path-redacted]");
}

function commandAvailable(binary, which) {
	return which(binary);
}

function defaultWhich(binary) {
	return spawnSync("which", [binary], { stdio: "ignore" }).status === 0;
}

function defaultRun(command, args, cwd) {
	const result = spawnSync(command, args, { cwd, encoding: "utf8", timeout: 15 * 60 * 1000 });
	return { ok: result.status === 0, stdout: result.stdout ?? "", stderr: result.stderr ?? "", code: result.status ?? 1 };
}

function localLane(root) {
	const source = resolve(join(root, "corpus", "sample.txt"));
	try {
		accessSync(source, constants.R_OK);
		const content = readFileSync(source, "utf8");
		if (content.length === 0) throw new Error("local-source-empty");
		return { name: "local", status: "PASS", reason: "absolute corpus source is readable", evidence: { source, identity: "local:corpus/sample.txt", sourceAbsolute: isAbsolute(source), sourceReadable: true, nativeIdentity: true } };
	} catch (error) {
		return { name: "local", status: "FAIL", reason: sanitizeDiagnostic(error instanceof Error ? error.message : error) };
	}
}

export async function runDatasourceMatrix(options = {}) {
	const root = resolve(options.root ?? process.cwd());
	const selection = options.selection ?? parseDatasourceSelection();
	const which = options.which ?? defaultWhich;
	const configured = options.configured ?? defaultConfigured;
	const run = options.run ?? ((command, args, cwd) => defaultRun(command, args, cwd));
	const matrix = buildDatasourceMatrix();
	const lanes = [];
	for (const selected of selection) {
		if (selected === "configured") {
			for (const lane of NATIVE_LANES) lanes.push(await runNativeLane(lane, root, which, configured, run));
			continue;
		}
		const lane = matrix.find((candidate) => candidate.name === selected);
		if (lane?.kind === "local") lanes.push(localLane(root));
		else if (lane?.kind === "native") lanes.push(await runNativeLane(lane, root, which, configured, run));
		else lanes.push({ name: selected, status: "SKIP", reason: "lane-not-supported" });
	}
	const summary = { pass: lanes.filter((lane) => lane.status === "PASS").length, skip: lanes.filter((lane) => lane.status === "SKIP").length, fail: lanes.filter((lane) => lane.status === "FAIL").length };
	return { selection, lanes, summary, exitCode: summary.fail > 0 ? 1 : 0 };
}

async function runNativeLane(lane, root, which, configured, run) {
	if (!commandAvailable(lane.binary, which)) return { name: lane.name, status: "SKIP", reason: `${lane.binary}-not-installed` };
	if (!configured(lane.name)) return { name: lane.name, status: "SKIP", reason: `${lane.name}-native-store-not-configured` };
	if (lane.command === undefined) return { name: lane.name, status: "SKIP", reason: `${lane.binary}-live-harness-not-registered` };
	const result = await run("bun", [resolve(REPO_ROOT, lane.command)], root);
	// The identity is matched against the raw output: redacting it first would erase an OS-absolute source
	// (a cloud-mount path under /Users), which is exactly the identity some lanes must verify. Only the
	// sanitized copy is stored in evidence.
	const output = sanitizeDiagnostic(result.stdout).slice(-2000);
	if (result.ok && lane.identityPattern?.test(result.stdout)) return { name: lane.name, status: "PASS", reason: "native-store harness passed with verified native identity", evidence: { binary: lane.binary, nativeStorePreserved: true, output, nativeIdentity: true } };
	if (result.ok) return { name: lane.name, status: "FAIL", reason: `${lane.name} native identity unverified`, evidence: { binary: lane.binary, nativeStorePreserved: true, output, nativeIdentity: false } };
	return { name: lane.name, status: "FAIL", reason: sanitizeDiagnostic(result.stderr || result.stdout || `${lane.binary}-failed`), evidence: { binary: lane.binary, nativeStorePreserved: true } };
}

export { NATIVE_LANES };
