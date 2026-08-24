/**
 * Provider-neutral incremental rclone mirror.
 *
 * `lsjson` inventories a remote, a workspace manifest computes its diff, and
 * only added/changed indexable files are copied into a managed mirror. The
 * manifest is published only after every changed file is available, so search
 * can continue using the previous completed snapshot after an interrupted run.
 */

import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, normalize } from "node:path";
import { createDefaultParserRegistry } from "../../../parser/defaults.ts";
import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, parseEpochMs } from "../../http.ts";

export interface RcloneRunResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export type RcloneRunner = (args: readonly string[], timeoutMs: number) => Promise<RcloneRunResult>;

export interface RcloneConnectorOptions {
	readonly binaryPath?: string;
	readonly remote?: string;
	readonly provider?: string;
	readonly workspaceRoot?: string;
	readonly instanceId?: string;
	readonly skillName?: string;
	readonly extensions?: readonly string[];
	readonly include?: readonly string[];
	readonly exclude?: readonly string[];
	readonly exportFormats?: string;
	readonly maxDocuments?: number;
	readonly maxBytesPerFile?: number;
	readonly concurrency?: number;
	readonly bandwidthLimit?: string;
	readonly dryRun?: boolean;
	readonly timeoutMs?: number;
	readonly runner?: RcloneRunner;
}

interface ManifestEntry {
	readonly path: string;
	readonly name: string;
	readonly size: number;
	readonly modTime?: string;
	readonly mimeType?: string;
	readonly hashes?: Readonly<Record<string, string>>;
	readonly remoteId?: string;
}

interface RcloneManifest {
	readonly version: 1;
	readonly remoteName: string;
	readonly entries: readonly ManifestEntry[];
}

const DEFAULT_BINARY = "rclone";
const DEFAULT_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".html", ".rst", ".org"] as const;
const DEFAULT_EXPORT_FORMATS = "txt,csv";
const DEFAULT_MAX_DOCUMENTS = 200;
const DEFAULT_MAX_BYTES_PER_FILE = 2 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_CONTENT_CHARS = 100_000;

export class RcloneConnector implements DatasourceConnector {
	private readonly options: RcloneConnectorOptions;
	private readonly runner: RcloneRunner;

	constructor(options: RcloneConnectorOptions = {}) {
		this.options = options;
		this.runner =
			options.runner ?? ((args, timeoutMs) => runBinary(options.binaryPath ?? DEFAULT_BINARY, args, timeoutMs));
	}

	async fetch(): Promise<ConnectorFetchResult> {
		const remote = this.options.remote;
		if (remote === undefined || remote.length === 0) {
			return { ok: false, reason: "not-configured", message: "rclone remote not configured" };
		}
		const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		const exportFormats = this.options.exportFormats ?? DEFAULT_EXPORT_FORMATS;
		let listResult: RcloneRunResult;
		try {
			listResult = await this.runner(
				["lsjson", remote, "--recursive", "--files-only", "--hash", "--drive-export-formats", exportFormats],
				timeoutMs,
			);
		} catch {
			return { ok: false, reason: "unavailable", message: "rclone binary not found or failed to spawn" };
		}
		if (!listResult.ok) {
			return { ok: false, reason: classifyFailure(listResult.stderr), message: shortFailure(listResult) };
		}

		let rawEntries: readonly unknown[];
		try {
			rawEntries = asArray(JSON.parse(listResult.stdout));
		} catch {
			return { ok: false, reason: "invalid-data", message: "rclone listing was not valid JSON" };
		}

		const extensions =
			this.options.extensions ??
			(this.options.workspaceRoot === undefined ? DEFAULT_EXTENSIONS : defaultParserExtensions());
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const maxBytes = this.options.maxBytesPerFile ?? DEFAULT_MAX_BYTES_PER_FILE;
		let skipped = 0;
		const inventory: ManifestEntry[] = [];
		for (const raw of rawEntries) {
			const entry = toManifestEntry(raw);
			if (entry === undefined) continue;
			if (!matchesRules(entry.path, this.options.include, this.options.exclude)) {
				skipped += 1;
				continue;
			}
			if (!isIndexable(entry.name, extensions) || entry.size > maxBytes) {
				skipped += 1;
				continue;
			}
			if (inventory.length < maxDocuments) inventory.push(entry);
		}

		if (this.options.workspaceRoot === undefined) {
			return this.fetchWithoutMirror(remote, inventory, exportFormats, maxBytes, timeoutMs, skipped);
		}
		return this.fetchIncremental(remote, inventory, exportFormats, timeoutMs, skipped);
	}

	private async fetchIncremental(
		remote: string,
		inventory: readonly ManifestEntry[],
		exportFormats: string,
		timeoutMs: number,
		skipped: number,
	): Promise<ConnectorFetchResult> {
		const instanceId = this.options.instanceId ?? "default";
		const skillName = this.options.skillName ?? "gdrive";
		const paths = mirrorPaths(this.options.workspaceRoot as string, skillName, instanceId);
		const previous = loadManifest(paths.manifestPath);
		const previousByPath = new Map((previous?.entries ?? []).map((entry) => [entry.path, entry]));
		const currentByPath = new Map(inventory.map((entry) => [entry.path, entry]));
		const changedEntries = inventory.filter((entry) => !sameEntry(previousByPath.get(entry.path), entry));
		const deletedEntries = (previous?.entries ?? []).filter((entry) => !currentByPath.has(entry.path));
		const changed = changedEntries.length > 0 || deletedEntries.length > 0;

		if (this.options.dryRun === true) {
			return {
				ok: true,
				changed: false,
				documents: await loadMirroredDocuments(
					paths.mirrorRoot,
					previous?.entries ?? [],
					skillName,
					instanceId,
					remote,
				),
				warnings: [`dry-run: ${changedEntries.length} download(s), ${deletedEntries.length} deletion(s) planned`],
			};
		}

		const staged: Array<{ readonly temp: string; readonly destination: string }> = [];
		let nextEntry = 0;
		let copyFailure:
			| {
					readonly reason: "not-configured" | "auth" | "permission" | "api-error" | "unavailable";
					readonly message: string;
			  }
			| undefined;
		const copyOne = async (): Promise<void> => {
			const entryIndex = nextEntry;
			nextEntry += 1;
			const entry = changedEntries[entryIndex];
			if (entry === undefined || copyFailure !== undefined) return;
			const destination = mirrorFilePath(paths.mirrorRoot, entry.path);
			const temp = `${destination}.autorag-tmp`;
			mkdirSync(dirname(temp), { recursive: true });
			let copied: RcloneRunResult;
			try {
				copied = await this.runner(
					copyArgs(remote, entry.path, temp, exportFormats, this.options.concurrency, this.options.bandwidthLimit),
					timeoutMs,
				);
			} catch {
				copyFailure = { reason: "unavailable", message: "rclone copy failed unexpectedly" };
				rmSync(temp, { force: true });
				return;
			}
			if (!copied.ok) {
				copyFailure = { reason: classifyFailure(copied.stderr), message: shortFailure(copied) };
				rmSync(temp, { force: true });
				return;
			}
			staged.push({ temp, destination });
			await copyOne();
		};
		const concurrency = Math.max(1, Math.min(this.options.concurrency ?? 4, changedEntries.length || 1));
		await Promise.all(Array.from({ length: concurrency }, () => copyOne()));
		if (copyFailure !== undefined) {
			for (const file of staged) rmSync(file.temp, { force: true });
			return { ok: false, reason: copyFailure.reason, message: copyFailure.message };
		}
		for (const file of staged) renameSync(file.temp, file.destination);
		for (const entry of deletedEntries) rmSync(mirrorFilePath(paths.mirrorRoot, entry.path), { force: true });
		saveManifest(paths.manifestPath, { version: 1, remoteName: remoteName(remote), entries: inventory });

		const warnings = skipped > 0 ? [`${skipped} file(s) skipped (filtered, non-text, or oversized)`] : undefined;
		return {
			ok: true,
			changed,
			documents: await loadMirroredDocuments(paths.mirrorRoot, changedEntries, skillName, instanceId, remote),
			deletedDocIds: deletedEntries.map((entry) => entry.path),
			...(warnings !== undefined ? { warnings } : {}),
		};
	}

	private async fetchWithoutMirror(
		remote: string,
		inventory: readonly ManifestEntry[],
		exportFormats: string,
		maxBytes: number,
		timeoutMs: number,
		skipped: number,
	): Promise<ConnectorFetchResult> {
		const documents: ConnectorDocument[] = [];
		let failures = 0;
		for (const entry of inventory) {
			const target = remoteTarget(remote, entry.path);
			let result: RcloneRunResult;
			try {
				result = await this.runner(
					["cat", target, "--drive-export-formats", exportFormats, "--count", String(maxBytes)],
					timeoutMs,
				);
			} catch {
				failures += 1;
				continue;
			}
			if (!result.ok) {
				failures += 1;
				continue;
			}
			const content = result.stdout.trim().slice(0, MAX_CONTENT_CHARS);
			if (content.length > 0) {
				documents.push(
					documentFromEntry(
						entry,
						content,
						this.options.skillName ?? "gdrive",
						this.options.instanceId ?? "default",
						remote,
					),
				);
			}
		}
		const warnings: string[] = [];
		if (failures > 0) warnings.push(`${failures} file(s) failed to read`);
		if (skipped > 0) warnings.push(`${skipped} file(s) skipped (filtered, non-text, or oversized)`);
		return { ok: true, changed: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}

function toManifestEntry(raw: unknown): ManifestEntry | undefined {
	const entry = asRecord(raw);
	const path = asString(entry?.Path);
	const name = asString(entry?.Name);
	if (entry === undefined || path === undefined || name === undefined) return undefined;
	const hashesRecord = asRecord(entry.Hashes);
	const hashes =
		hashesRecord === undefined
			? undefined
			: Object.fromEntries(
					Object.entries(hashesRecord).filter((pair): pair is [string, string] => typeof pair[1] === "string"),
				);
	return {
		path,
		name,
		size: typeof entry.Size === "number" ? entry.Size : 0,
		...(asString(entry.ModTime) !== undefined ? { modTime: asString(entry.ModTime) } : {}),
		...(asString(entry.MimeType) !== undefined ? { mimeType: asString(entry.MimeType) } : {}),
		...(hashes !== undefined ? { hashes } : {}),
		...(asString(entry.ID) !== undefined ? { remoteId: asString(entry.ID) } : {}),
	};
}

function isIndexable(name: string, extensions: readonly string[]): boolean {
	const dot = name.lastIndexOf(".");
	return dot >= 0 && extensions.includes(name.slice(dot).toLowerCase());
}

function globMatches(path: string, rule: string): boolean {
	const escaped = rule
		.replace(/[.+^${}()|[\]\\]/gu, "\\$&")
		.replace(/\*\*/gu, ".*")
		.replace(/\*/gu, "[^/]*")
		.replace(/\?/gu, ".");
	return new RegExp(`^${escaped}$`, "u").test(path);
}

function matchesRules(path: string, include?: readonly string[], exclude?: readonly string[]): boolean {
	const included = include === undefined || include.length === 0 || include.some((rule) => globMatches(path, rule));
	return included && !(exclude ?? []).some((rule) => globMatches(path, rule));
}

function sameEntry(previous: ManifestEntry | undefined, current: ManifestEntry): boolean {
	return (
		previous?.path === current.path &&
		previous.size === current.size &&
		previous.modTime === current.modTime &&
		previous.remoteId === current.remoteId &&
		JSON.stringify(previous.hashes) === JSON.stringify(current.hashes)
	);
}

function mirrorPaths(workspaceRoot: string, skillName: string, instanceId: string) {
	const base = join(workspaceRoot, ".autorag", "datasources", skillName, instanceId);
	return { mirrorRoot: join(base, "mirror"), manifestPath: join(base, "manifest.json") };
}

function mirrorFilePath(root: string, path: string): string {
	const safe = normalize(path).replace(/^(\.\.(?:[/\\]|$))+|^[/\\]+/gu, "");
	return join(root, safe);
}

function remoteTarget(remote: string, path: string): string {
	return remote.endsWith(":") || remote.endsWith("/") ? `${remote}${path}` : `${remote}/${path}`;
}

function remoteName(remote: string): string {
	const colon = remote.indexOf(":");
	return colon >= 0 ? remote.slice(0, colon) : remote;
}

function copyArgs(
	remote: string,
	path: string,
	destination: string,
	exportFormats: string,
	concurrency?: number,
	bandwidthLimit?: string,
): string[] {
	const args = ["copyto", remoteTarget(remote, path), destination, "--drive-export-formats", exportFormats];
	if (concurrency !== undefined) args.push("--transfers", String(concurrency));
	if (bandwidthLimit !== undefined) args.push("--bwlimit", bandwidthLimit);
	return args;
}

function loadManifest(path: string): RcloneManifest | undefined {
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as RcloneManifest;
		return parsed.version === 1 && Array.isArray(parsed.entries) ? parsed : undefined;
	} catch {
		return undefined;
	}
}

function saveManifest(path: string, manifest: RcloneManifest): void {
	mkdirSync(dirname(path), { recursive: true });
	const temp = `${path}.autorag-tmp`;
	writeFileSync(temp, `${JSON.stringify(manifest)}\n`, "utf8");
	renameSync(temp, path);
}

async function loadMirroredDocuments(
	root: string,
	entries: readonly ManifestEntry[],
	skillName: string,
	instanceId: string,
	remote: string,
): Promise<ConnectorDocument[]> {
	const documents: ConnectorDocument[] = [];
	const registry = createDefaultParserRegistry();
	for (const entry of entries) {
		try {
			const sourcePath = mirrorFilePath(root, entry.path);
			const parser = registry.getForVirtualPath(entry.path);
			if (parser === undefined) continue;
			const bytes = await readFile(sourcePath);
			const parsed = await parser.parse({ virtualPath: entry.path, sourcePath, bytes });
			const content = parsed.markdown.trim().slice(0, MAX_CONTENT_CHARS);
			if (content.length > 0) documents.push(documentFromEntry(entry, content, skillName, instanceId, remote));
		} catch {
			// Missing/corrupt completed mirror entries are omitted and repaired on the next changed inventory.
		}
	}
	return documents;
}

function defaultParserExtensions(): readonly string[] {
	const extensions = new Set<string>();
	for (const parser of createDefaultParserRegistry().list()) {
		for (const extension of parser.extensions) {
			extensions.add(extension.startsWith(".") ? extension.toLowerCase() : `.${extension.toLowerCase()}`);
		}
	}
	return [...extensions];
}

function documentFromEntry(
	entry: ManifestEntry,
	content: string,
	skillName: string,
	instanceId: string,
	remote: string,
): ConnectorDocument {
	const segments = entry.path.split("/");
	return {
		docId: entry.path,
		hierarchy: ["files", ...segments.slice(0, -1)],
		title: entry.name,
		content,
		publishedAt: parseEpochMs(entry.modTime),
		metadata: {
			virtualPath: `/${skillName}/${instanceId}/files/${entry.path}`,
			remote,
			...(entry.mimeType !== undefined ? { mimeType: entry.mimeType } : {}),
			...(entry.remoteId !== undefined ? { remoteId: entry.remoteId } : {}),
			...(entry.hashes !== undefined ? { hashes: entry.hashes } : {}),
			size: entry.size,
		},
	};
}

function classifyFailure(stderr: string): "not-configured" | "auth" | "permission" | "api-error" {
	const lower = stderr.toLowerCase();
	if (lower.includes("didn't find section") || lower.includes("config file") || lower.includes("no remotes")) {
		return "not-configured";
	}
	if (lower.includes("oauth") || lower.includes("token") || lower.includes("unauthorized") || lower.includes("401")) {
		return "auth";
	}
	if (lower.includes("permission") || lower.includes("403") || lower.includes("forbidden")) return "permission";
	return "api-error";
}

function shortFailure(result: RcloneRunResult): string {
	return `rclone exited with code ${result.code ?? "unknown"}`;
}

function runBinary(binary: string, args: readonly string[], timeoutMs: number): Promise<RcloneRunResult> {
	return new Promise((resolvePromise) => {
		const child = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
		child.stdout.on("data", (chunk: Buffer) => {
			stdout += chunk.toString("utf8");
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString("utf8");
		});
		child.on("error", () => {
			clearTimeout(timer);
			resolvePromise({ ok: false, stdout, stderr, code: null });
		});
		child.on("close", (code) => {
			clearTimeout(timer);
			resolvePromise({ ok: code === 0, stdout, stderr, code });
		});
	});
}
