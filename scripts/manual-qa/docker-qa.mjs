#!/usr/bin/env node

import { existsSync, mkdirSync } from "node:fs";
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

function configuredModelEnvNames() {
	const configured = process.env.QA_MODEL_ENV?.trim();
	return configured === undefined || configured.length === 0 ? DEFAULT_MODEL_ENV : configured.split(/\s+/u);
}

function containerEnvironment() {
	const home = process.env.QA_CONTAINER_HOME?.trim() || "/tmp/autorag-home";
	const autoragHome = `${home}/.autorag`;
	const fixed = [
		["HOME", home],
		["AUTORAG_HOME", autoragHome],
		["AUTORAG_CONFIG", `${autoragHome}/config.json`],
	];
	const modelEnv = configuredModelEnvNames().filter((name) => {
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
		image,
	);
	return args;
}

function runDocker(args) {
	const result = spawnSync("docker", args, { cwd: REPO_ROOT, stdio: "inherit" });
	if (result.error !== undefined) throw result.error;
	return result.status ?? 1;
}

function buildImage() {
	const image = process.env.QA_IMAGE?.trim() || "autorag-qa-linux";
	const platform = process.env.QA_PLATFORM?.trim() || "linux/amd64";
	const dockerfile = process.env.QA_DOCKERFILE?.trim() || "scripts/ci/linux.Dockerfile";
	return runDocker(["build", "--platform", platform, "-f", dockerfile, "-t", image, "scripts/ci"]);
}

function runShell() {
	const args = commonDockerArgs(true);
	args.push("bash", "-lc", 'mkdir -p "$AUTORAG_HOME" && bun install --frozen-lockfile && exec bash');
	return runDocker(args);
}

function runLive() {
	const root = resolve(process.env.E2E_ROOT ?? process.env.AUTORAG_LIVE_E2E_ROOT ?? REPO_ROOT);
	if (!existsSync(root)) {
		console.error(`E2E_ROOT does not exist: ${root}`);
		return 2;
	}
	const mode = process.env.E2E_MODE?.trim() || "cold";
	const runnerArgs = process.env.E2E_ARGS?.trim() || "";
	const cloneState = resolve(REPO_ROOT, ".autorag-e2e");
	mkdirSync(cloneState, { recursive: true });
	const args = commonDockerArgs(false);
	const imageIndex = args.length - 1;
	args.splice(imageIndex, 0, "--env", `E2E_DATASOURCES=${process.env.E2E_DATASOURCES ?? ""}`);
	args.splice(args.length - 1, 0, "--env", `AUTORAG_LIVE_E2E_EMBEDDER=${process.env.E2E_EMBEDDER || "native"}`);
	args.splice(args.length - 1, 0, "--mount", `type=bind,src=${cloneState},dst=/workspace/.autorag-e2e`);
	args.splice(args.length - 1, 0, "--mount", `type=bind,src=${root},dst=/e2e-root,readonly`);
	const command =
		'mkdir -p "$AUTORAG_HOME" && bun install --frozen-lockfile && node scripts/live-e2e/runner.mjs live' +
		` --mode ${mode} --root /e2e-root ${runnerArgs}`;
	args.push("bash", "-lc", command);
	return runDocker(args);
}

const command = process.argv[2];
let exitCode;
switch (command) {
	case "build":
		exitCode = buildImage();
		break;
	case "shell":
		exitCode = runShell();
		break;
	case "live":
		exitCode = runLive();
		break;
	default:
		console.error("Usage: docker-qa.mjs <build|shell|live>");
		exitCode = 2;
}
process.exitCode = exitCode;
