/**
 * Local mail export connector (issue #1311).
 *
 * Parses `.eml` files and `.mbox` archives from trusted, server-configured
 * filesystem paths using the existing `mailparser` dependency. No network
 * access. Per-message parse failures aggregate into one count-only warning;
 * absolute paths never appear in warnings or failure messages.
 */

import { readdir, readFile, stat } from "node:fs/promises";
import { basename, dirname, extname, join } from "node:path";
import { simpleParser } from "mailparser";
import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";

export interface MailExportConnectorOptions {
	/** Files or directories containing `.eml` / `.mbox` archives. Required. */
	readonly paths?: readonly string[];
	readonly maxDocuments?: number;
	/** Files larger than this are skipped with a warning. Default 50 MiB. */
	readonly maxBytesPerFile?: number;
}

const DEFAULT_MAX_DOCUMENTS = 500;
const DEFAULT_MAX_BYTES_PER_FILE = 50 * 1024 * 1024;
const MAX_BODY_CHARS = 50_000;
/** Classic mbox `From ` separator at the start of a line, ending in a year. */
const MBOX_SEPARATOR = /^From .*\d{4}$/mu;

export class MailExportConnector implements DatasourceConnector {
	private readonly options: MailExportConnectorOptions;

	constructor(options: MailExportConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(): Promise<ConnectorFetchResult> {
		const paths = this.options.paths ?? [];
		if (paths.length === 0)
			return { ok: false, reason: "not-configured", message: "no mail export paths configured" };
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const maxBytes = this.options.maxBytesPerFile ?? DEFAULT_MAX_BYTES_PER_FILE;

		// 1. Expand configured paths into mail files.
		const files: string[] = [];
		let missingPaths = 0;
		for (const path of paths) {
			try {
				const info = await stat(path);
				if (info.isDirectory()) {
					const entries = await readdir(path, { recursive: true, withFileTypes: true });
					for (const entry of entries) {
						if (!entry.isFile()) continue;
						const extension = extname(entry.name).toLowerCase();
						if (extension === ".eml" || extension === ".mbox") {
							files.push(join(entry.parentPath, entry.name));
						}
					}
				} else if (info.isFile()) {
					files.push(path);
				}
			} catch {
				missingPaths += 1;
			}
		}
		if (missingPaths === paths.length) {
			return { ok: false, reason: "unavailable", message: "no configured mail export paths were readable" };
		}

		// 2. Parse messages; aggregate failures into count-only warnings.
		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		if (missingPaths > 0) warnings.push(`${missingPaths} configured path(s) were unreadable`);
		let parseFailures = 0;
		let skippedLarge = 0;
		for (const file of files.sort()) {
			if (documents.length >= maxDocuments) break;
			let bytes: Buffer;
			try {
				const info = await stat(file);
				if (info.size > maxBytes) {
					skippedLarge += 1;
					continue;
				}
				bytes = await readFile(file);
			} catch {
				parseFailures += 1;
				continue;
			}
			const stem = basename(file, extname(file));
			const mailboxDir = basename(dirname(file));
			const hierarchy = ["mailboxes", mailboxDir, stem];
			const extension = extname(file).toLowerCase();
			const rawMessages = extension === ".mbox" ? splitMbox(bytes.toString("utf8")) : [bytes.toString("utf8")];
			for (const [index, raw] of rawMessages.entries()) {
				if (documents.length >= maxDocuments) break;
				try {
					const parsed = await simpleParser(raw);
					const subject = parsed.subject ?? "(no subject)";
					const from = Array.isArray(parsed.from) ? parsed.from[0]?.text : parsed.from?.text;
					const to = Array.isArray(parsed.to) ? parsed.to[0]?.text : parsed.to?.text;
					const body = (parsed.text ?? "").trim().slice(0, MAX_BODY_CHARS);
					const headerBlock = [
						`Subject: ${subject}`,
						...(from !== undefined ? [`From: ${from}`] : []),
						...(to !== undefined ? [`To: ${to}`] : []),
						...(parsed.date !== undefined ? [`Date: ${parsed.date.toISOString()}`] : []),
					].join("\n");
					const content = `${headerBlock}\n\n${body}`.trim();
					if (content.length === 0) continue;
					documents.push({
						docId: extension === ".mbox" ? `${stem}-${index + 1}` : stem,
						hierarchy,
						title: subject,
						content,
						...(parsed.date !== undefined ? { publishedAt: parsed.date.getTime() } : {}),
						metadata: {
							subject,
							...(from !== undefined ? { from } : {}),
							messageIndex: index,
						},
					});
				} catch {
					parseFailures += 1;
				}
			}
		}

		if (skippedLarge > 0) warnings.push(`${skippedLarge} file(s) exceeded the size limit and were skipped`);
		if (parseFailures > 0) warnings.push(`${parseFailures} message(s) failed to parse`);
		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}

/** Split an mbox archive on classic `From ` separator lines. */
function splitMbox(text: string): readonly string[] {
	const lines = text.split(/\r?\n/u);
	const messages: string[] = [];
	let current: string[] = [];
	for (const line of lines) {
		if (MBOX_SEPARATOR.test(line) && line.startsWith("From ")) {
			if (current.length > 0) messages.push(current.join("\n"));
			current = [];
			continue; // The From_ separator line itself is not part of the message.
		}
		current.push(line);
	}
	if (current.length > 0) messages.push(current.join("\n"));
	return messages.filter((message) => message.trim().length > 0);
}
