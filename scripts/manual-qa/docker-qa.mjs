#!/usr/bin/env node

import { existsSync, mkdirSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DEFAULT_MODEL_ENV = [
	"OPENAI_API_KEY",
	"AUTORAG_OPENAI_API_KEY",
	"ANTHROPIC_API_KEY",
	"GEMINI_API_KEY",
	"GOOGLE_API_KEY",
	"OPENROUTER_API_KEY",
	"FIREWORKS_API_KEY",
	"XAI_API_KEY",
	"MISTRAL_API_KEY",
	"GROQ_API_KEY",
	"AZURE_OPENAI_API_KEY",
];

// Environment names fixed by the launcher itself; the credential allowlist
// must not be able to redefine them.
const RESERVED_ENV_NAMES = new Set([
	"HOME",
	"PATH",
	"AUTORAG_HOME",
	"AUTORAG_CONFIG",
	"E2E_DATASOURCES",
	"E2E_MODE",
	"E2E_EMBEDDER",
	"E2E_ARGS",
	"QA_MODEL_ENV",
	"AUTORAG_LIVE_E2E_EMBEDDER",
	"AUTORAG_LIVE_E2E_GIT_SHA",
	"AUTORAG_LIVE_E2E_GIT_DIRTY",
]);
const ENV_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/u;

export function configuredModelEnvNames() {
	// Unset selects the default allowlist; an explicitly empty value selects
	// none, so `make qa-shell QA_MODEL_ENV=` forwards no credentials.
	const raw = process.env.QA_MODEL_ENV;
	if (raw === undefined) return DEFAULT_MODEL_ENV;
	const names = raw.trim().length === 0 ? [] : raw.trim().split(/\s+/u);
	const invalid = names.filter((name) => !ENV_NAME_PATTERN.test(name));
	if (invalid.length > 0) {
		throw new Error(`QA_MODEL_ENV contains invalid names: ${invalid.join(" ")}`);
	}
	return names;
}

export function parseRunnerArgs(input) {
	// Split E2E_ARGS like a POSIX shell: single- and double-quoted words keep
	// their spaces, no expansions. Each word becomes one argv entry handed to
	// the runner through "$@", never through a shell string.
	const trimmed = input.trim();
	const words = [];
	let current = "";
	let hasWord = false;
	let quote = null;
	for (const char of trimmed) {
		if (quote !== null) {
			if (char === quote) quote = null;
			else current += char;
			continue;
		}
		if (char === '"' || char === "'") {
			quote = char;
			hasWord = true;
			continue;
		}
		if (char === " " || char === "\t") {
			if (hasWord) words.push(current);
			current = "";
			hasWord = false;
			continue;
		}
		current += char;
		hasWord = true;
	}
	if (quote !== null) throw new Error(`E2E_ARGS has an unterminated ${quote} quote`);
	if (hasWord) words.push(current);
	return words;
}

export function containerEnvironment() {
	const home = process.env.QA_CONTAINER_HOME?.trim() || "/tmp/autorag-home";
	const autoragHome = `${home}/.autorag`;
	const fixed = [
		["HOME", home],
		["AUTORAG_HOME", autoragHome],
		["AUTORAG_CONFIG", `${autoragHome}/config.json`],
	];
	const modelEnv = configuredModelEnvNames().filter((name) => {
		if (RESERVED_ENV_NAMES.has(name)) {
			throw new Error(`QA_MODEL_ENV must not include launcher-owned variable: ${name}`);
		}
		const value = process.env[name];
		return value !== undefined && value.length > 0;
	});
	return { fixed, modelEnv };
}

function commonDockerArgs(interactive) {
	const image = process.env.QA_IMAGE?.trim() || "autorag-qa-linux";
	const platform = process.env.QA_PLATFORM?.trim() || "linux/amd64";
	const args = ["run", "--rm", "--init"];
	if (interactive) args.push("-it");
	args.push("--platform", platform);
	const environment = containerEnvironment();
	for (const [name, value] of environment.fixed) args.push("--env", `${name}=${value}`);
	for (const name of environment.modelEnv) args.push("--env", name);
	args.push(
		"--mount",
		`type=bind,src=${REPO_ROOT},dst=/workspace`,
		"--mount",
		"type=volume,dst=/workspace/node_modules",
		"--workdir",
		"/workspace",
	);
	return args;
}

function dockerImageName() {
	return process.env.QA_IMAGE?.trim() || "autorag-qa-linux";
}

function runDocker(args) {
	const result = spawnSync("docker", args, { cwd: REPO_ROOT, stdio: "inherit" });
	if (result.error !== undefined) throw result.error;
	return result.status ?? 1;
}

function buildImage() {
	const image = dockerImageName();
	const platform = process.env.QA_PLATFORM?.trim() || "linux/amd64";
	const dockerfile = process.env.QA_DOCKERFILE?.trim() || "scripts/ci/qa.Dockerfile";
	return runDocker(["build", "--platform", platform, "-f", dockerfile, "-t", image, "scripts/ci"]);
}

function runShell() {
	const args = commonDockerArgs(true);
	args.push(
		dockerImageName(),
		"bash",
		"-lc",
		'mkdir -p "$AUTORAG_HOME" && bun install --frozen-lockfile && exec bash',
	);
	return runDocker(args);
}


function hostGitIdentity() {
	// A linked-worktree checkout keeps its git metadata outside the mounted
	// tree, so git cannot run inside the container. Resolve the identity on
	// the host — where git works — and hand it to the runner through env; a
	// regular clone still falls back to in-container git.
	const shaRun = spawnSync("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT, encoding: "utf8" });
	if (shaRun.error !== undefined || shaRun.status !== 0) return undefined;
	const sha = (shaRun.stdout ?? "").trim();
	if (sha.length === 0) return undefined;
	const porcelainRun = spawnSync("git", ["status", "--porcelain"], { cwd: REPO_ROOT, encoding: "utf8" });
	const dirty = porcelainRun.status === 0 ? (porcelainRun.stdout ?? "").trim().length > 0 : true;
	return { sha, dirty };
}
function runLive() {
	const root = resolve(process.env.E2E_ROOT ?? process.env.AUTORAG_LIVE_E2E_ROOT ?? REPO_ROOT);
	if (!existsSync(root)) {
		console.error(`E2E_ROOT does not exist: ${root}`);
		return 2;
	}
	const mode = process.env.E2E_MODE?.trim() || "cold";
	if (mode !== "cold" && mode !== "warm") {
		console.error(`E2E_MODE must be "cold" or "warm", got: ${mode}`);
		return 2;
	}
	const embedder = process.env.E2E_EMBEDDER?.trim() || "native";
	if (embedder !== "native" && embedder !== "gateway") {
		console.error(`E2E_EMBEDDER must be "native" or "gateway", got: ${embedder}`);
		return 2;
	}
	let runnerArgs;
	try {
		runnerArgs = parseRunnerArgs(process.env.E2E_ARGS ?? "");
	} catch (error) {
		console.error(error instanceof Error ? error.message : String(error));
		return 2;
	}
	const cloneState = resolve(REPO_ROOT, ".autorag-e2e");
	if (root === cloneState || root.startsWith(`${cloneState}/`)) {
		console.error(`E2E_ROOT must not live inside ${cloneState}`);
		return 2;
	}
	// A container cannot delete a bind mount's own mountpoint, so cold-mode
	// state removal happens on the host before the container starts. This
	// mirrors the runner's cold contract: delete .autorag-e2e, rebuild.
	if (mode === "cold" && existsSync(cloneState)) {
		rmSync(cloneState, { recursive: true, force: true });
	}
	mkdirSync(cloneState, { recursive: true });
	const args = commonDockerArgs(false);
	const gitIdentity = hostGitIdentity();
	args.push(
		"--env",
		`E2E_DATASOURCES=${process.env.E2E_DATASOURCES ?? ""}`,
		"--env",
		`AUTORAG_LIVE_E2E_EMBEDDER=${embedder}`,
		...(gitIdentity !== undefined
			? [
					"--env",
					`AUTORAG_LIVE_E2E_GIT_SHA=${gitIdentity.sha}`,
					"--env",
					`AUTORAG_LIVE_E2E_GIT_DIRTY=${gitIdentity.dirty ? "true" : "false"}`,
				]
			: []),
		"--mount",
		`type=bind,src=${cloneState},dst=/workspace/.autorag-e2e`,
		"--mount",
		`type=bind,src=${root},dst=/e2e-root,readonly`,
		dockerImageName(),
		"bash",
		"-lc",
		// Runner arguments travel as argv through "$@": they are data for the
		// runner, never shell source. The bind mount is owned by the host
		// user, so git needs safe.directory for the fingerprint's SHA.
		'git config --global --add safe.directory /workspace >/dev/null 2>&1 || true; ' +
			'mkdir -p "$AUTORAG_HOME" && bun install --frozen-lockfile && ' +
			'exec node scripts/live-e2e/runner.mjs live --root /e2e-root "$@"',
		"runner",
		"--mode",
		mode,
		...runnerArgs,
	);
	return runDocker(args);
}

function dispatch(command) {
	switch (command) {
		case "build":
			return buildImage();
		case "shell":
			return runShell();
		case "live":
			return runLive();
		default:
			console.error("Usage: docker-qa.mjs <build|shell|live>");
			return 2;
	}
}

const isDirectRun =
	process.argv[1] !== undefined && fileURLToPath(import.meta.url) === resolve(process.argv[1]);
if (isDirectRun) {
	try {
		process.exitCode = dispatch(process.argv[2]);
	} catch (error) {
		console.error(error instanceof Error ? error.message : String(error));
		process.exitCode = 2;
	}
}
