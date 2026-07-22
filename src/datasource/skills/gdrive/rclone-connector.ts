/**
 * Rclone-backed drive connector (issue #1301, rclone path).
 *
 * Uses the external `rclone` CLI (https://rclone.org) as the trusted bridge
 * to Google Drive — or any of rclone's 70+ storage backends — so OAuth
 * setup and token refresh live entirely in `rclone config`, not here
 * (katok/himalaya pattern). Google Docs/Sheets are transparently exported
 * as text via rclone's `--drive-export-formats`. Never throws; failures map
 * onto the coarse connector failure union with path/PII-opaque messages.
 *
 * Listing: `rclone lsjson <remote> --recursive --files-only`
 * Content: `rclone cat <remote>/<path>` (bounded, text formats only)
 */

import { spawn } from "node:child_process";
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
	/** Path to the rclone binary. Default bare `rclone` PATH lookup. */
	readonly binaryPath?: string;
	/**
	 * Rclone remote path, e.g. `gdrive:` or `gdrive:Team/Docs`. Required
	 * trusted configuration (`rclone config` defines the remote).
	 */
	readonly remote?: string;
	/**
	 * Indexable file extensions (lowercase, with dot). Google Docs/Sheets
	 * surface as these after `--drive-export-formats txt,csv`.
	 */
	readonly extensions?: readonly string[];
	/** Drive export formats forwarded to rclone. Default `txt,csv`. */
	readonly exportFormats?: string;
	/** Max files whose contents are fetched per run. Default 200. */
	readonly maxDocuments?: number;
	/** Files larger than this are skipped. Default 2 MiB. */
	readonly maxBytesPerFile?: number;
	/** Per-spawn timeout. Default 60s (cloud listings can be slow). */
	readonly timeoutMs?: number;
	/** Injectable process runner for tests. */
	readonly runner?: RcloneRunner;
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
		const extensions = this.options.extensions ?? DEFAULT_EXTENSIONS;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const maxBytes = this.options.maxBytesPerFile ?? DEFAULT_MAX_BYTES_PER_FILE;

		// 1. Recursive listing as JSON.
		let listResult: RcloneRunResult;
		try {
			listResult = await this.runner(
				["lsjson", remote, "--recursive", "--files-only", "--drive-export-formats", exportFormats],
				timeoutMs,
			);
		} catch {
			return { ok: false, reason: "unavailable", message: "rclone binary not found or failed to spawn" };
		}
		if (!listResult.ok) {
			return { ok: false, reason: classifyFailure(listResult.stderr), message: shortFailure(listResult) };
		}
		let entries: readonly unknown[];
		try {
			entries = asArray(JSON.parse(listResult.stdout));
		} catch {
			return { ok: false, reason: "invalid-data", message: "rclone listing was not valid JSON" };
		}

		// 2. Cat text files (bounded); per-file failures degrade to warnings.
		const documents: ConnectorDocument[] = [];
		let readFailures = 0;
		let skipped = 0;
		for (const raw of entries) {
			if (documents.length >= maxDocuments) break;
			const entry = asRecord(raw);
			const path = asString(entry?.Path);
			const name = asString(entry?.Name);
			if (entry === undefined || path === undefined || name === undefined) continue;
			const extension = name.includes(".") ? name.slice(name.lastIndexOf(".")).toLowerCase() : "";
			if (!extensions.includes(extension)) {
				skipped += 1;
				continue;
			}
			const size = typeof entry.Size === "number" ? entry.Size : 0;
			if (size > maxBytes) {
				skipped += 1;
				continue;
			}
			const target = remote.endsWith(":") || remote.endsWith("/") ? `${remote}${path}` : `${remote}/${path}`;
			let content = "";
			try {
				const catResult = await this.runner(
					["cat", target, "--drive-export-formats", exportFormats, "--count", String(maxBytes)],
					timeoutMs,
				);
				if (!catResult.ok) {
					readFailures += 1;
					continue;
				}
				content = catResult.stdout.trim().slice(0, MAX_CONTENT_CHARS);
			} catch {
				readFailures += 1;
				continue;
			}
			if (content.length === 0) continue;
			const segments = path.split("/");
			documents.push({
				docId: path,
				hierarchy: ["files", ...segments.slice(0, -1)],
				title: name,
				content,
				publishedAt: parseEpochMs(asString(entry.ModTime)),
				metadata: {
					...(asString(entry.MimeType) !== undefined ? { mimeType: asString(entry.MimeType) } : {}),
					size,
				},
			});
		}

		const warnings: string[] = [];
		if (readFailures > 0) warnings.push(`${readFailures} file(s) failed to read`);
		if (skipped > 0) warnings.push(`${skipped} file(s) skipped (non-text or oversized)`);
		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
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

/** Short, path/PII-opaque failure summary (never raw stderr). */
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
