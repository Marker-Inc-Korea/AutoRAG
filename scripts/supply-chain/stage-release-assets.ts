#!/usr/bin/env bun
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { SupplyChainError } from "./policy.ts";

const COPY_ALWAYS = ["LICENSE", "NOTICE"] as const;
const COPY_OPTIONAL = ["GOVERNANCE.md"] as const;

export function stageReleaseAssets(input: {
	readonly projectRoot: string;
	readonly outputDir: string;
	readonly sbomDir?: string;
}): { readonly files: readonly string[]; readonly checksumsPath: string } {
	if (!existsSync(join(input.projectRoot, "dist", "index.js"))) {
		throw new SupplyChainError("dist/index.js is missing; build before staging release assets");
	}
	mkdirSync(input.outputDir, { recursive: true });
	for (const name of COPY_ALWAYS) {
		const source = join(input.projectRoot, name);
		if (!existsSync(source)) throw new SupplyChainError(`${name} is missing`);
		copyFileSync(source, join(input.outputDir, name));
	}
	for (const name of COPY_OPTIONAL) {
		const source = join(input.projectRoot, name);
		if (existsSync(source)) copyFileSync(source, join(input.outputDir, name));
	}
	packNpmTarball(input.projectRoot, input.outputDir);
	copySbomJson(input.sbomDir, input.outputDir);
	const staged = readdirSync(input.outputDir)
		.filter((name) => name !== "SHA256SUMS.txt")
		.sort();
	if (!staged.some((name) => name.endsWith(".tgz"))) {
		throw new SupplyChainError("npm pack did not produce a tarball");
	}
	const checksumsPath = join(input.outputDir, "SHA256SUMS.txt");
	const lines = staged.map((name) => {
		const digest = createHash("sha256")
			.update(readFileSync(join(input.outputDir, name)))
			.digest("hex");
		return `${digest}  ${name}`;
	});
	writeFileSync(checksumsPath, `${lines.join("\n")}\n`);
	return { files: [...staged, "SHA256SUMS.txt"], checksumsPath };
}

function packNpmTarball(projectRoot: string, outputDir: string): void {
	const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
	const result = spawnSync(npmCommand, ["pack", "--pack-destination", outputDir], {
		cwd: projectRoot,
		encoding: "utf8",
		shell: process.platform === "win32",
	});
	if (result.status !== 0) {
		throw new SupplyChainError(`npm pack failed: ${result.error?.message || result.stderr || result.stdout}`);
	}
}

function copySbomJson(sbomDir: string | undefined, outputDir: string): void {
	if (sbomDir === undefined || !existsSync(sbomDir)) return;
	for (const name of readdirSync(sbomDir)) {
		const lower = name.toLowerCase();
		const isSbom =
			lower.endsWith(".cdx.json") ||
			lower.endsWith(".spdx.json") ||
			lower.includes("cyclonedx") ||
			lower.includes("spdx");
		if (!isSbom) continue;
		copyFileSync(join(sbomDir, name), join(outputDir, name));
	}
}

function parseArgs(argv: readonly string[]): { readonly out: string; readonly sbomDir?: string } {
	let out: string | undefined;
	let sbomDir: string | undefined;
	for (let index = 0; index < argv.length; index += 1) {
		const flag = argv[index];
		const value = argv[index + 1];
		if (flag === "--out" && value !== undefined) {
			out = value;
			index += 1;
			continue;
		}
		if (flag === "--sbom-dir" && value !== undefined) {
			sbomDir = value;
			index += 1;
			continue;
		}
		throw new SupplyChainError(`unknown argument ${flag}`);
	}
	if (out === undefined) throw new SupplyChainError("--out is required");
	return { out, sbomDir };
}

function isCli(): boolean {
	const entry = process.argv[1];
	if (entry === undefined) return false;
	return resolve(entry) === fileURLToPath(import.meta.url);
}

if (isCli()) {
	try {
		const flags = parseArgs(process.argv.slice(2));
		const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
		const staged = stageReleaseAssets({
			projectRoot,
			outputDir: resolve(flags.out),
			sbomDir: flags.sbomDir === undefined ? undefined : resolve(flags.sbomDir),
		});
		process.stdout.write(`${JSON.stringify(staged, null, 2)}\n`);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		process.stderr.write(`${message}\n`);
		process.exitCode = 1;
	}
}
