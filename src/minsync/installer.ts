import { createHash } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, renameSync, rmSync } from "node:fs";
import { chmod, mkdtemp } from "node:fs/promises";
import { get } from "node:https";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { pipeline } from "node:stream/promises";
import { spawnProcess } from "./process.ts";

const LATEST_RELEASE_URL = "https://api.github.com/repos/NomaDamas/MinSync/releases/latest";
const SHA256_HEX_PATTERN = /^[a-f0-9]{64}$/;

export interface MinSyncReleaseAsset {
	readonly name: string;
	readonly downloadUrl: string;
	readonly sha256?: string;
}

export interface MinSyncRelease {
	readonly tagName: string;
	readonly assets: readonly MinSyncReleaseAsset[];
}

export interface InstalledMinSyncBinary {
	readonly binaryPath: string;
	readonly version: string;
}

export interface EnsureMinSyncBinaryOptions {
	readonly root: string;
	readonly platform?: NodeJS.Platform;
	readonly arch?: NodeJS.Architecture;
	readonly releaseProvider?: () => Promise<MinSyncRelease>;
	readonly assetInstaller?: (asset: MinSyncReleaseAsset, destination: string) => Promise<void>;
}

export async function ensureMinSyncBinary(options: EnsureMinSyncBinaryOptions): Promise<InstalledMinSyncBinary> {
	const binaryPath = join(options.root, ".autorag", "bin", executableName(options.platform ?? process.platform));
	if (existsSync(binaryPath)) return { binaryPath, version: "cached" };
	const releaseProvider = options.releaseProvider ?? fetchLatestMinSyncRelease;
	const release = await releaseProvider();
	const asset = selectReleaseAsset(release, options.platform ?? process.platform, options.arch ?? process.arch);
	requireSha256(asset);
	const assetInstaller = options.assetInstaller ?? installReleaseAsset;
	mkdirSync(dirname(binaryPath), { recursive: true });
	await assetInstaller(asset, binaryPath);
	await chmod(binaryPath, 0o755);
	return { binaryPath, version: release.tagName };
}

export async function fetchLatestMinSyncRelease(): Promise<MinSyncRelease> {
	const text = await readHttpsText(LATEST_RELEASE_URL);
	const parsed: unknown = JSON.parse(text);
	if (!isRecord(parsed) || typeof parsed.tag_name !== "string" || !Array.isArray(parsed.assets)) {
		throw new MinSyncReleaseError("GitHub latest release response did not match the expected shape");
	}
	return {
		tagName: parsed.tag_name,
		assets: parsed.assets.filter(isGitHubAsset).map((asset) => ({
			name: asset.name,
			downloadUrl: asset.browser_download_url,
			sha256: parseDigest(asset.digest),
		})),
	};
}

export function selectReleaseAsset(
	release: MinSyncRelease,
	platform: NodeJS.Platform,
	arch: NodeJS.Architecture,
): MinSyncReleaseAsset {
	const target = targetTriple(platform, arch);
	const asset = release.assets.find((candidate) => candidate.name.includes(target));
	if (!asset) throw new MinSyncReleaseError(`No MinSync ${release.tagName} asset found for ${target}`);
	return asset;
}

async function installReleaseAsset(asset: MinSyncReleaseAsset, destination: string): Promise<void> {
	const expectedSha256 = requireSha256(asset);
	const tempDir = await mkdtemp(join(tmpdir(), "autorag-minsync-install-"));
	try {
		const archive = join(tempDir, asset.name);
		await downloadFile(asset.downloadUrl, archive);
		await verifySha256(archive, expectedSha256);
		if (!asset.name.endsWith(".tar.gz") && !asset.name.endsWith(".zip")) {
			throw new MinSyncReleaseError(`Unsupported MinSync asset format: ${asset.name}`);
		}
		const extracted = await spawnProcess("tar", ["-xf", archive, "-C", tempDir], tempDir);
		if (!extracted.ok) throw new MinSyncReleaseError(extracted.stderr || `Could not extract ${asset.name}`);
		renameSync(join(tempDir, basename(destination)), destination);
	} finally {
		rmSync(tempDir, { recursive: true, force: true });
	}
}

async function downloadFile(url: string, destination: string): Promise<void> {
	await pipeline(await openHttps(url), createWriteStream(destination));
}

async function verifySha256(path: string, expected: string): Promise<void> {
	const hash = createHash("sha256");
	const stream = await import("node:fs").then((fs) => fs.createReadStream(path));
	await pipeline(stream, hash);
	const actual = hash.digest("hex");
	if (actual !== expected)
		throw new MinSyncReleaseError(`MinSync asset sha256 mismatch: expected ${expected}, got ${actual}`);
}

function readHttpsText(url: string, redirectCount = 0): Promise<string> {
	return new Promise((resolve, reject) => {
		get(url, { headers: { "User-Agent": "AutoRAG-MinSync" } }, (response) => {
			if (isRedirect(response.statusCode) && response.headers.location) {
				response.resume();
				readHttpsText(new URL(response.headers.location, url).toString(), redirectCount + 1).then(resolve, reject);
				return;
			}
			if (redirectCount > 5) {
				reject(new MinSyncReleaseError("Too many redirects while fetching MinSync release metadata"));
				response.resume();
				return;
			}
			if (response.statusCode !== 200) {
				reject(
					new MinSyncReleaseError(`GitHub release request failed with HTTP ${response.statusCode ?? "unknown"}`),
				);
				response.resume();
				return;
			}
			response.setEncoding("utf8");
			let body = "";
			response.on("data", (chunk: string) => {
				body += chunk;
			});
			response.on("end", () => {
				resolve(body);
			});
		}).on("error", reject);
	});
}

function openHttps(url: string, redirectCount = 0): Promise<NodeJS.ReadableStream> {
	return new Promise((resolve, reject) => {
		get(url, { headers: { "User-Agent": "AutoRAG-MinSync" } }, (response) => {
			if (isRedirect(response.statusCode) && response.headers.location) {
				response.resume();
				openHttps(new URL(response.headers.location, url).toString(), redirectCount + 1).then(resolve, reject);
				return;
			}
			if (redirectCount > 5) {
				reject(new MinSyncReleaseError("Too many redirects while downloading MinSync asset"));
				response.resume();
				return;
			}
			if (response.statusCode !== 200) {
				reject(
					new MinSyncReleaseError(`MinSync asset download failed with HTTP ${response.statusCode ?? "unknown"}`),
				);
				response.resume();
				return;
			}
			resolve(response);
		}).on("error", reject);
	});
}

function targetTriple(platform: NodeJS.Platform, arch: NodeJS.Architecture): string {
	if (platform === "darwin" && arch === "arm64") return "aarch64-apple-darwin";
	if (platform === "linux" && arch === "x64") return "x86_64-unknown-linux-gnu";
	if (platform === "win32" && arch === "x64") return "x86_64-pc-windows-msvc";
	throw new MinSyncReleaseError(`Unsupported MinSync platform: ${platform}/${arch}`);
}

export function executableName(platform: NodeJS.Platform): string {
	return platform === "win32" ? "minsync.exe" : "minsync";
}

function parseDigest(value: unknown): string | undefined {
	if (typeof value !== "string") return undefined;
	return value.startsWith("sha256:") ? value.slice("sha256:".length) : undefined;
}

function requireSha256(asset: MinSyncReleaseAsset): string {
	if (!asset.sha256) throw new MinSyncReleaseError(`MinSync asset ${asset.name} is missing sha256 digest`);
	if (!SHA256_HEX_PATTERN.test(asset.sha256)) {
		throw new MinSyncReleaseError(`MinSync asset ${asset.name} has malformed sha256 digest`);
	}
	return asset.sha256;
}

function isRedirect(statusCode: number | undefined): boolean {
	return statusCode === 301 || statusCode === 302 || statusCode === 303 || statusCode === 307 || statusCode === 308;
}

interface GitHubAsset {
	readonly name: string;
	readonly browser_download_url: string;
	readonly digest?: string;
}

function isGitHubAsset(value: unknown): value is GitHubAsset {
	return isRecord(value) && typeof value.name === "string" && typeof value.browser_download_url === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

export class MinSyncReleaseError extends Error {
	readonly name = "MinSyncReleaseError";
}
