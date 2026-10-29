import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile as fsReadFile, stat } from "node:fs/promises";
import { basename, extname } from "node:path";
import type { ConnectorDocument, ConnectorFetchFail, ConnectorFetchOk } from "../../connector.ts";

export interface SpotlightConnectorFailure extends ConnectorFetchFail {
	readonly stderr?: string;
	readonly stdout?: string;
	readonly code?: number | null;
}

export type SpotlightConnectorResult = ConnectorFetchOk | SpotlightConnectorFailure;

export interface ProcessRunResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	/** Spawn error code, e.g. `"ENOENT"` when the binary is missing. */
	readonly error?: string;
}

/** Process execution seam; injectable for tests. Never throws. */
export type ProcessRunner = (command: string, args: readonly string[], timeoutMs: number) => Promise<ProcessRunResult>;

export type FileReader = (path: string) => Promise<string>;

export interface SpotlightConnectorOptions {
	/** Trusted Spotlight filters AND-ed with each user query. Required. */
	readonly queries?: readonly string[];
	/** Trusted directory restricting every query (`mdfind -onlyin`). */
	readonly onlyIn?: string;
	/** Hits kept per query after dedupe. Default 100. */
	readonly maxResultsPerQuery?: number;
	/** Maximum documents hydrated by one live search. Default 300. */
	readonly maxDocuments?: number;
	/** Characters kept per document body. Default 100000. */
	readonly maxContentChars?: number;
	/** Files larger than this are skipped. Default 256 KiB. */
	readonly maxBytesPerFile?: number;
	/** Per-process timeout in ms. Default 15s. */
	readonly timeoutMs?: number;
	/** Injectable platform; defaults to `process.platform`. */
	readonly platform?: NodeJS.Platform;
	/** Injectable process runner; defaults to a bounded spawn wrapper. */
	readonly run?: ProcessRunner;
	/** Injectable file reader; defaults to utf8 `fs.readFile`. */
	readonly readFile?: FileReader;
}

const DEFAULT_MAX_RESULTS_PER_QUERY = 100;
const DEFAULT_MAX_DOCUMENTS = 300;
const DEFAULT_MAX_BYTES_PER_FILE = 256 * 1024;
const DEFAULT_TIMEOUT_MS = 15_000;
const MAX_CONTENT_CHARS = 100_000;
const MAX_BUFFER_BYTES = 4 * 1024 * 1024;

const FDA_GUIDANCE =
	"grant Full Disk Access to the terminal or app running AutoRAG (System Settings -> Privacy & Security -> Full Disk Access), then restart it";

/** Executes live Spotlight queries. macOS owns indexing; no local snapshot is collected. */
export class SpotlightConnector {
	private readonly options: SpotlightConnectorOptions;
	private readonly run: ProcessRunner;
	private readonly readFile: FileReader;
	private readonly platform: NodeJS.Platform;

	constructor(options: SpotlightConnectorOptions = {}) {
		this.options = options;
		this.run = options.run ?? spawnRunner;
		this.readFile = options.readFile ?? ((path) => fsReadFile(path, "utf8"));
		this.platform = options.platform ?? process.platform;
	}

	/** Refresh-time health check only: never runs `mdfind` or hydrates files. */
	async probe(): Promise<SpotlightConnectorResult> {
		const queries = (this.options.queries ?? []).map((query) => query.trim()).filter((query) => query.length > 0);
		const failure = await this.readinessFailure(queries);
		return failure ?? { ok: true, documents: [] };
	}

	/** Search the current macOS Spotlight index and hydrate only the requested hits. */
	async search(
		query: string,
		limit = DEFAULT_MAX_DOCUMENTS,
		acceptDocument?: (docId: string) => boolean,
	): Promise<SpotlightConnectorResult> {
		const trimmed = query.trim();
		if (trimmed.length === 0 || limit <= 0) return { ok: true, documents: [] };
		const queries = (this.options.queries ?? []).map((query) => query.trim()).filter((query) => query.length > 0);
		const failure = await this.readinessFailure(queries);
		if (failure !== undefined) return failure;

		const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		const maxPerQuery = this.options.maxResultsPerQuery ?? DEFAULT_MAX_RESULTS_PER_QUERY;
		const maxDocuments = Math.min(this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS, limit);
		const seen = new Set<string>();
		const paths: string[] = [];
		const textQuery = trimmed
			.split(/\s+/u)
			.map((term) => `"${term.replace(/\\/gu, "\\\\").replace(/"/gu, '\\"')}"`)
			.join(" AND ");
		let mdfindMissing = false;
		let failedQueries = 0;
		const failures: ProcessRunResult[] = [];
		for (const filter of queries) {
			const query = `(${filter}) AND (${textQuery})`;
			const args = this.options.onlyIn !== undefined ? ["-onlyin", this.options.onlyIn, query] : [query];
			const result = await this.run("mdfind", args, timeoutMs);
			if (!result.ok) {
				if (result.error === "ENOENT") {
					mdfindMissing = true;
					break;
				}
				failedQueries += 1;
				failures.push(result);
				continue;
			}
			let kept = 0;
			for (const line of result.stdout.split("\n")) {
				const path = line.trim();
				if (path.length === 0 || seen.has(path)) continue;
				if (kept >= maxPerQuery) break;
				seen.add(path);
				paths.push(path);
				kept += 1;
			}
		}
		if (mdfindMissing) {
			return { ok: false, reason: "unavailable", message: "mdfind binary not found; Spotlight requires macOS" };
		}
		if (failedQueries === queries.length) {
			return {
				ok: false,
				reason: "unavailable",
				message: "every configured mdfind query failed",
				stderr: failures[0]?.stderr,
				stdout: failures[0]?.stdout,
				code: failures[0]?.code,
			};
		}

		const documents: ConnectorDocument[] = [];
		let permissionDenied = 0;
		let readFailures = 0;
		const maxBytes = this.options.maxBytesPerFile ?? DEFAULT_MAX_BYTES_PER_FILE;
		const maxContentChars = this.options.maxContentChars ?? MAX_CONTENT_CHARS;
		for (const path of paths) {
			if (documents.length >= maxDocuments) break;
			const docId = createHash("sha256").update(path).digest("hex").slice(0, 16);
			if (acceptDocument !== undefined && !acceptDocument(docId)) continue;
			try {
				const info = await stat(path);
				if (!info.isFile() || info.size > maxBytes) continue;
				const text = await this.readFile(path);
				const content = text.trim().slice(0, maxContentChars);
				if (content.length === 0) continue;
				documents.push({
					docId,
					hierarchy: ["files"],
					title: basename(path),
					content,
					publishedAt: info.mtimeMs,
					metadata: { path, extension: extname(path).toLowerCase() },
				});
			} catch (error) {
				const code = (error as NodeJS.ErrnoException).code;
				if (code === "EPERM" || code === "EACCES") permissionDenied += 1;
				else readFailures += 1;
			}
		}

		if (documents.length === 0 && permissionDenied > 0) {
			return {
				ok: false,
				reason: "permission",
				message: `macOS denied access to all Spotlight hits; ${FDA_GUIDANCE}`,
			};
		}
		if (documents.length === 0 && failedQueries > 0) {
			return {
				ok: false,
				reason: "unavailable",
				message: "some mdfind queries failed and no readable hits remained",
				stderr: failures[0]?.stderr,
				stdout: failures[0]?.stdout,
				code: failures[0]?.code,
			};
		}
		const warnings: string[] = [];
		if (failedQueries > 0) warnings.push(`${failedQueries} mdfind quer(y/ies) failed`);
		if (permissionDenied > 0) {
			warnings.push(
				`${permissionDenied} Spotlight hit(s) were denied by macOS privacy protections; ${FDA_GUIDANCE}`,
			);
		}
		if (readFailures > 0) warnings.push(`${readFailures} Spotlight hit(s) failed to read`);
		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}

	private async readinessFailure(queries: readonly string[]): Promise<ConnectorFetchFail | undefined> {
		if (this.platform !== "darwin") {
			return { ok: false, reason: "unavailable", message: "spotlight datasource is supported on macOS only" };
		}
		if (queries.length === 0) {
			return { ok: false, reason: "not-configured", message: "no spotlight queries configured" };
		}
		const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		const status = await this.run("mdutil", ["-s", "/"], timeoutMs);
		if (status.ok && /indexing disabled/iu.test(status.stdout)) {
			return {
				ok: false,
				reason: "unavailable",
				message: "Spotlight indexing is disabled; enable it with `sudo mdutil -i on /`",
			};
		}
		return undefined;
	}
}

/** Default runner: bounded buffered spawn that never rejects. */
const spawnRunner: ProcessRunner = (command, args, timeoutMs) =>
	new Promise((resolve) => {
		const child = spawn(command, [...args], { stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		let settled = false;
		const finish = (result: ProcessRunResult): void => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolve(result);
		};
		const timer = setTimeout(() => {
			child.kill("SIGKILL");
			finish({ ok: false, stdout, stderr: "process timed out", code: null });
		}, timeoutMs);
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			if (stdout.length < MAX_BUFFER_BYTES) stdout += chunk;
		});
		child.stderr.on("data", (chunk: string) => {
			if (stderr.length < MAX_BUFFER_BYTES) stderr += chunk;
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			finish({ ok: false, stdout, stderr: error.message, code: null, error: error.code });
		});
		child.on("close", (code) => {
			finish({ ok: code === 0, stdout, stderr, code });
		});
	});
