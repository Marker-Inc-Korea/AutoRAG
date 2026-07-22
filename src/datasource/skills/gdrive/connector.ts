/**
 * Google Drive connector (issue #1301).
 *
 * Lists files through the Drive v3 API with a trusted, server-configured
 * OAuth access token, exporting Google Docs/Sheets as plain text/CSV and
 * downloading plain-text files. Binary formats are skipped. Per-file export
 * failures degrade to warnings. Never throws; messages stay path/PII-opaque.
 */

import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, httpJson, httpText, parseEpochMs, resolveToken } from "../../http.ts";

export interface GDriveConnectorOptions {
	/** Drive v3 REST base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** OAuth access token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the access token. Default `GDRIVE_ACCESS_TOKEN`. */
	readonly tokenEnv?: string;
	/** Restrict to files whose parent is this folder id. */
	readonly folderId?: string;
	/** Include shared-drive items in listings. */
	readonly includeSharedDrives?: boolean;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxPages?: number;
	readonly maxDocuments?: number;
}

const DEFAULT_BASE_URL = "https://www.googleapis.com/drive/v3";
const DEFAULT_TOKEN_ENV = "GDRIVE_ACCESS_TOKEN";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_DOCUMENTS = 500;
const MAX_CONTENT_CHARS = 100_000;
const EXPORT_MIME: Readonly<Record<string, string>> = {
	"application/vnd.google-apps.document": "text/plain",
	"application/vnd.google-apps.spreadsheet": "text/csv",
};
const DOWNLOAD_MIME = new Set(["text/plain", "text/markdown", "text/csv", "application/json"]);

export class GDriveConnector implements DatasourceConnector {
	private readonly options: GDriveConnectorOptions;

	constructor(options: GDriveConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const token = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		if (token === undefined) return { ok: false, reason: "not-configured", message: "auth token not configured" };
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const request = {
			headers: { Authorization: `Bearer ${token}` },
			timeoutMs: this.options.timeoutMs,
			fetchImpl: this.options.fetchImpl,
			signal,
		};
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;

		// 1. List files (token-paginated).
		interface DriveFile {
			id: string;
			name: string;
			mimeType: string;
			modifiedTime?: string;
		}
		const files: DriveFile[] = [];
		let pageToken: string | undefined;
		for (let page = 0; page < maxPages; page += 1) {
			const url = new URL(`${baseUrl}/files`);
			url.searchParams.set("pageSize", "100");
			url.searchParams.set("fields", "nextPageToken,files(id,name,mimeType,modifiedTime,parents)");
			const query =
				this.options.folderId !== undefined
					? `trashed=false and '${this.options.folderId}' in parents`
					: "trashed=false";
			url.searchParams.set("q", query);
			if (this.options.includeSharedDrives === true) {
				url.searchParams.set("supportsAllDrives", "true");
				url.searchParams.set("includeItemsFromAllDrives", "true");
				url.searchParams.set("corpora", "allDrives");
			}
			if (pageToken !== undefined) url.searchParams.set("pageToken", pageToken);
			const result = await httpJson(url.toString(), request);
			if (!result.ok) return { ok: false, reason: result.reason, message: `file list failed: ${result.message}` };
			const envelope = asRecord(result.json);
			if (envelope === undefined) return { ok: false, reason: "invalid-data", message: "file list malformed" };
			for (const raw of asArray(envelope.files)) {
				const file = asRecord(raw);
				const id = asString(file?.id);
				const name = asString(file?.name);
				const mimeType = asString(file?.mimeType);
				if (id === undefined || name === undefined || mimeType === undefined) continue;
				files.push({ id, name, mimeType, modifiedTime: asString(file?.modifiedTime) });
			}
			pageToken = asString(envelope.nextPageToken);
			if (pageToken === undefined || pageToken.length === 0) break;
		}

		// 2. Export/download textual content per file.
		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		for (const file of files) {
			if (documents.length >= maxDocuments) break;
			const exportMime = EXPORT_MIME[file.mimeType];
			let contentUrl: string | undefined;
			if (exportMime !== undefined) {
				contentUrl = `${baseUrl}/files/${file.id}/export?mimeType=${encodeURIComponent(exportMime)}`;
			} else if (DOWNLOAD_MIME.has(file.mimeType)) {
				contentUrl = `${baseUrl}/files/${file.id}?alt=media`;
			} else {
				continue; // Binary formats and folders are skipped.
			}
			const contentResult = await httpText(contentUrl, request);
			if (!contentResult.ok) {
				warnings.push(`file export failed: ${contentResult.message}`);
				continue;
			}
			const content = contentResult.text.trim().slice(0, MAX_CONTENT_CHARS);
			if (content.length === 0) continue;
			documents.push({
				docId: file.id,
				hierarchy: ["files"],
				title: file.name,
				content,
				publishedAt: parseEpochMs(file.modifiedTime),
				metadata: {
					mimeType: file.mimeType,
					...(file.modifiedTime !== undefined ? { modifiedTime: file.modifiedTime } : {}),
				},
			});
		}

		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}
