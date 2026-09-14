#!/usr/bin/env bun
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	copyFileSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { SupplyChainError } from "./policy.ts";

const COPY_ALWAYS = ["LICENSE", "NOTICE"] as const;
const COPY_OPTIONAL = ["GOVERNANCE.md"] as const;
const EMBEDDING_MANIFEST = join("licenses", "embedding-assets.json");

type PinnedAsset = {
	readonly kind: "model" | "runtime";
	readonly id: string;
	readonly filename: string;
	readonly url: string;
	readonly revision: string;
	readonly sha256: string;
	readonly licenseId: string;
	readonly noticeFile: string;
	readonly archiveMembers?: readonly string[];
};

type EmbeddingManifest = {
	readonly assets: readonly PinnedAsset[];
};

export function stageReleaseAssets(input: {
	readonly projectRoot: string;
	readonly outputDir: string;
	readonly sbomDir?: string;
	/** Download and validate pinned llama.cpp runtime archives for a release. */
	readonly stageEmbeddingAssets?: boolean;
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

	const shouldStageEmbeddingAssets =
		input.stageEmbeddingAssets ?? existsSync(join(input.projectRoot, EMBEDDING_MANIFEST));
	if (shouldStageEmbeddingAssets) stageEmbeddingReleaseAssets(input.projectRoot, input.outputDir);
	packNpmTarball(input.projectRoot, input.outputDir);
	copySbomJson(input.sbomDir, input.outputDir);

	const staged = listFiles(input.outputDir)
		.filter((name) => name !== "SHA256SUMS.txt")
		.sort();
	if (!staged.some((name) => name.endsWith(".tgz"))) {
		throw new SupplyChainError("npm pack did not produce a tarball");
	}
	const checksumsPath = join(input.outputDir, "SHA256SUMS.txt");
	const lines = staged.map((name) => {
		const digest = createHash("sha256")
			.update(readFileSync(join(input.outputDir, ...name.split("/"))))
			.digest("hex");
		return `${digest}  ${name}`;
	});
	writeFileSync(checksumsPath, `${lines.join("\n")}\n`);
	return { files: [...staged, "SHA256SUMS.txt"], checksumsPath };
}

function stageEmbeddingReleaseAssets(projectRoot: string, outputDir: string): void {
	const manifest = readEmbeddingManifest(join(projectRoot, EMBEDDING_MANIFEST));
	const noticeFiles = new Set(manifest.assets.map((asset) => asset.noticeFile));
	for (const noticeFile of noticeFiles) {
		if (!existsSync(join(projectRoot, noticeFile))) {
			throw new SupplyChainError(`embedding compliance notice is missing: ${noticeFile}`);
		}
	}
	copyDirectory(join(projectRoot, "licenses"), join(outputDir, "licenses"));
	const runtimeAssets = manifest.assets.filter((asset) => asset.kind === "runtime");
	for (const asset of runtimeAssets) {
		if (!asset.archiveMembers || asset.archiveMembers.length === 0) {
			throw new SupplyChainError(`runtime asset ${asset.id} has no required archive members`);
		}
		const destination = join(outputDir, asset.filename);
		mkdirSync(dirname(destination), { recursive: true });
		downloadPinnedAsset(asset, destination);
		verifyArchiveMembers(asset, destination);
	}
}

function readEmbeddingManifest(path: string): EmbeddingManifest {
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(path, "utf8"));
	} catch (error) {
		throw new SupplyChainError(`unable to read embedding compliance manifest: ${describe(error)}`);
	}
	if (!isRecord(parsed) || !Array.isArray(parsed.assets)) {
		throw new SupplyChainError("embedding compliance manifest must contain an assets array");
	}
	const assets = parsed.assets.map((value, index) => {
		if (!isRecord(value)) throw new SupplyChainError(`embedding manifest asset ${index} is not an object`);
		const fields = ["kind", "id", "filename", "url", "revision", "sha256", "licenseId", "noticeFile"] as const;
		for (const field of fields) {
			if (typeof value[field] !== "string" || value[field] === "") {
				throw new SupplyChainError(`embedding manifest asset ${index} is missing ${field}`);
			}
		}
		if (value.kind !== "model" && value.kind !== "runtime") {
			throw new SupplyChainError(`embedding manifest asset ${index} has an unknown kind`);
		}
		if (!/^[a-f0-9]{64}$/i.test(value.sha256 as string)) {
			throw new SupplyChainError(`embedding manifest asset ${index} has an invalid SHA-256`);
		}
		const archiveMembers = value.archiveMembers;
		if (
			archiveMembers !== undefined &&
			(!Array.isArray(archiveMembers) || archiveMembers.some((member) => typeof member !== "string"))
		) {
			throw new SupplyChainError(`embedding manifest asset ${index} has invalid archiveMembers`);
		}
		return {
			kind: value.kind as "model" | "runtime",
			id: value.id as string,
			filename: value.filename as string,
			url: value.url as string,
			revision: value.revision as string,
			sha256: value.sha256 as string,
			licenseId: value.licenseId as string,
			noticeFile: value.noticeFile as string,
			...(archiveMembers === undefined ? {} : { archiveMembers: archiveMembers as string[] }),
		};
	});
	return { assets };
}

function downloadPinnedAsset(asset: PinnedAsset, destination: string): void {
	const part = `${destination}.part`;
	rmSync(part, { force: true });
	const curl = process.platform === "win32" ? "curl.exe" : "curl";
	const result = spawnSync(
		curl,
		["--fail", "--silent", "--show-error", "--location", "--retry", "3", "--output", part, asset.url],
		{ encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
	);
	if (result.status !== 0) {
		rmSync(part, { force: true });
		throw new SupplyChainError(
			`download failed for ${asset.id} (${asset.url}): ${result.stderr || describe(result.error)}`,
		);
	}
	const actual = createHash("sha256").update(readFileSync(part)).digest("hex");
	if (actual !== asset.sha256.toLowerCase()) {
		rmSync(part, { force: true });
		throw new SupplyChainError(`SHA-256 mismatch for ${asset.id}: expected ${asset.sha256}, got ${actual}`);
	}
	try {
		// A verified part is the only file promoted to the release directory.
		rmSync(destination, { force: true });
		renameSync(part, destination);
	} finally {
		rmSync(part, { force: true });
	}
}

function verifyArchiveMembers(asset: PinnedAsset, archivePath: string): void {
	const isTarGz = asset.filename.endsWith(".tar.gz");
	const command = isTarGz ? "tar" : process.platform === "win32" ? "tar.exe" : "unzip";
	const args = isTarGz
		? ["-tzf", archivePath]
		: process.platform === "win32"
			? ["-tf", archivePath]
			: ["-Z1", archivePath];
	const result = spawnSync(command, args, { encoding: "utf8" });
	if (result.status !== 0) {
		throw new SupplyChainError(
			`unable to inspect archive members for ${asset.id}: ${result.stderr || describe(result.error)}`,
		);
	}
	const members = new Set(
		(result.stdout ?? "")
			.split(/\r?\n/)
			.map((member) => member.replace(/^\.\//, "").trim())
			.filter((member) => member.length > 0),
	);
	for (const expected of asset.archiveMembers ?? []) {
		const basename = expected.split("/").at(-1);
		const found =
			members.has(expected) ||
			(basename !== undefined && [...members].some((member) => member.endsWith(`/${basename}`)));
		if (!found) {
			throw new SupplyChainError(`runtime asset ${asset.id} is missing archive member ${expected}`);
		}
	}
}

function copyDirectory(source: string, destination: string): void {
	if (!existsSync(source)) throw new SupplyChainError(`${source} is missing`);
	for (const entry of readdirSync(source, { withFileTypes: true })) {
		const sourcePath = join(source, entry.name);
		const destinationPath = join(destination, entry.name);
		if (entry.isDirectory()) copyDirectory(sourcePath, destinationPath);
		else if (entry.isFile()) {
			mkdirSync(dirname(destinationPath), { recursive: true });
			copyFileSync(sourcePath, destinationPath);
		}
	}
}

function listFiles(root: string, current = root): string[] {
	const files: string[] = [];
	for (const entry of readdirSync(current, { withFileTypes: true })) {
		const path = join(current, entry.name);
		if (entry.isDirectory()) files.push(...listFiles(root, path));
		else if (entry.isFile()) files.push(relative(root, path).split("\\").join("/"));
	}
	return files;
}

function packNpmTarball(projectRoot: string, outputDir: string): void {
	const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
	const result = spawnSync(npmCommand, ["pack", "--pack-destination", outputDir], {
		cwd: projectRoot,
		encoding: "utf8",
		shell: process.platform === "win32",
	});
	if (result.status !== 0) {
		const detail = result.error?.message || result.stderr || result.stdout || "unknown process error";
		throw new SupplyChainError(`npm pack failed: ${detail}`);
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

function parseArgs(argv: readonly string[]): {
	readonly out: string;
	readonly sbomDir?: string;
	readonly stageEmbeddingAssets?: boolean;
} {
	let out: string | undefined;
	let sbomDir: string | undefined;
	let stageEmbeddingAssets: boolean | undefined;
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
		if (flag === "--embedding-assets") {
			stageEmbeddingAssets = true;
			continue;
		}
		throw new SupplyChainError(`unknown argument ${flag}`);
	}
	if (out === undefined) throw new SupplyChainError("--out is required");
	return { out, sbomDir, stageEmbeddingAssets };
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function describe(error: unknown): string {
	return error instanceof Error ? error.message : String(error ?? "unknown process error");
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
			stageEmbeddingAssets: flags.stageEmbeddingAssets,
		});
		process.stdout.write(`${JSON.stringify(staged, null, 2)}\n`);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		process.stderr.write(`${message}\n`);
		process.exitCode = 1;
	}
}
