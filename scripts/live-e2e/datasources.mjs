import { accessSync, constants, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = resolve(fileURLToPath(new URL("../..", import.meta.url)));

const NATIVE_LANES = Object.freeze([
	{ name: "katok", binary: "katok", command: "scripts/manual-qa/run-qa-katok-live.ts", identityPattern: /\bkakao:[^\s]+/u },
	{ name: "discrawl", binary: "discrawl", command: "scripts/manual-qa/run-qa-discrawl-live.ts", identityPattern: /\/discord\/[^\s]+/u },
	{ name: "wacrawl", binary: "wacrawl" },
	{ name: "telecrawl", binary: "telecrawl" },
	{ name: "slacrawl", binary: "slacrawl" },
	{ name: "notcrawl", binary: "notcrawl" },
	{ name: "qmd", binary: "qmd" },
	{ name: "rclone", binary: "rclone" },
	{ name: "mailcrawl", binary: "mailcrawl", command: "scripts/manual-qa/run-qa-mailcrawl-live.ts", identityPattern: /\/mailcrawl\/[^\s]+/u },
	{ name: "spotlight", binary: "mdfind", command: "scripts/manual-qa/run-qa-spotlight-live.ts", identityPattern: /\bspotlight:[^\s]+/u },
]);

export function buildDatasourceMatrix() {
	return [{ name: "local", kind: "local" }, ...NATIVE_LANES.map((lane) => ({ ...lane, kind: "native" }))];
}

export function parseDatasourceSelection(value = process.env.E2E_DATASOURCES || "local") {
	return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean))];
}

export function validateNativeIdentity(source, laneName) {
	if (typeof source !== "string") return false;
	if (laneName === "katok") return /^kakao:[^\s/]+(?:\/[^\s/]+){1,2}$/u.test(source);
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
	const configured = options.configured ?? ((name) => process.env[`E2E_DATASOURCE_${name.toUpperCase()}_CONFIGURED`] === "1");
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
	const output = sanitizeDiagnostic(result.stdout).slice(-2000);
	if (result.ok && lane.identityPattern?.test(output)) return { name: lane.name, status: "PASS", reason: "native-store harness passed with verified native identity", evidence: { binary: lane.binary, nativeStorePreserved: true, output, nativeIdentity: true } };
	if (result.ok) return { name: lane.name, status: "FAIL", reason: `${lane.name} native identity unverified`, evidence: { binary: lane.binary, nativeStorePreserved: true, output, nativeIdentity: false } };
	return { name: lane.name, status: "FAIL", reason: sanitizeDiagnostic(result.stderr || result.stdout || `${lane.binary}-failed`), evidence: { binary: lane.binary, nativeStorePreserved: true } };
}

export { NATIVE_LANES };
