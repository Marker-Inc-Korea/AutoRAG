/**
 * Himalaya-backed mail connector (issue #1304, IMAP path).
 *
 * Uses the external `himalaya` CLI (https://pimalaya.org) as the trusted
 * bridge to any IMAP/Maildir account it has configured — Gmail included —
 * so no OAuth token plumbing is needed here. Mirrors the katok pattern:
 * AutoRAG never opens the mailbox itself; it spawns the CLI, parses JSON
 * envelopes, and reads message bodies. Never throws; failures map onto the
 * coarse connector failure union with path/PII-opaque messages.
 */

import { spawn } from "node:child_process";
import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString } from "../../http.ts";

export interface HimalayaRunResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export type HimalayaRunner = (args: readonly string[], timeoutMs: number) => Promise<HimalayaRunResult>;

export interface HimalayaConnectorOptions {
	/** Path to the himalaya binary. Default bare `himalaya` PATH lookup. */
	readonly binaryPath?: string;
	/** Himalaya account name (`himalaya account list`). Default account when omitted. */
	readonly account?: string;
	/** Mail folder, e.g. `INBOX`. Himalaya's default folder when omitted. */
	readonly folder?: string;
	/** Envelopes per listing page. Default 100. */
	readonly pageSize?: number;
	/** Max messages whose bodies are fetched per run. Default 50. */
	readonly maxDocuments?: number;
	/** Per-spawn timeout. Default 30s (IMAP fetches can be slow). */
	readonly timeoutMs?: number;
	/** Injectable process runner for tests. */
	readonly runner?: HimalayaRunner;
}

const DEFAULT_BINARY = "himalaya";
const DEFAULT_PAGE_SIZE = 100;
const DEFAULT_MAX_DOCUMENTS = 50;
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_BODY_CHARS = 50_000;

export class HimalayaConnector implements DatasourceConnector {
	private readonly options: HimalayaConnectorOptions;
	private readonly runner: HimalayaRunner;

	constructor(options: HimalayaConnectorOptions = {}) {
		this.options = options;
		this.runner =
			options.runner ?? ((args, timeoutMs) => runBinary(options.binaryPath ?? DEFAULT_BINARY, args, timeoutMs));
	}

	async fetch(): Promise<ConnectorFetchResult> {
		const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const accountArgs = this.options.account !== undefined ? ["--account", this.options.account] : [];
		const folderArgs = this.options.folder !== undefined ? ["--folder", this.options.folder] : [];

		// 1. List envelopes as JSON.
		const listArgs = [
			"envelope",
			"list",
			...accountArgs,
			...folderArgs,
			"--page-size",
			String(this.options.pageSize ?? DEFAULT_PAGE_SIZE),
			"--output",
			"json",
		];
		let listResult: HimalayaRunResult;
		try {
			listResult = await this.runner(listArgs, timeoutMs);
		} catch {
			return { ok: false, reason: "unavailable", message: "himalaya binary not found or failed to spawn" };
		}
		if (!listResult.ok) {
			return { ok: false, reason: classifyFailure(listResult.stderr), message: shortFailure(listResult) };
		}
		let envelopes: readonly unknown[];
		try {
			envelopes = asArray(JSON.parse(listResult.stdout));
		} catch {
			return { ok: false, reason: "invalid-data", message: "envelope listing was not valid JSON" };
		}

		// 2. Read message bodies (bounded); per-message failures degrade.
		const account = this.options.account ?? "default";
		const folder = this.options.folder ?? "INBOX";
		const documents: ConnectorDocument[] = [];
		let readFailures = 0;
		for (const raw of envelopes.slice(0, maxDocuments)) {
			const envelope = asRecord(raw);
			const id = asString(envelope?.id);
			if (envelope === undefined || id === undefined) continue;
			const subject = asString(envelope.subject) ?? "(no subject)";
			const from = asRecord(envelope.from);
			const fromText = [asString(from?.name), asString(from?.addr)].filter(Boolean).join(" ").trim();
			let body = "";
			try {
				const readResult = await this.runner(["message", "read", ...accountArgs, ...folderArgs, id], timeoutMs);
				if (readResult.ok) body = readResult.stdout.trim().slice(0, MAX_BODY_CHARS);
				else readFailures += 1;
			} catch {
				readFailures += 1;
			}
			const headerBlock = [
				`Subject: ${subject}`,
				...(fromText.length > 0 ? [`From: ${fromText}`] : []),
				...(asString(envelope.date) !== undefined ? [`Date: ${asString(envelope.date)}`] : []),
			].join("\n");
			documents.push({
				docId: `${account}-${folder}-${id}`,
				hierarchy: ["accounts", account, folder],
				title: subject,
				content: body.length > 0 ? `${headerBlock}\n\n${body}` : headerBlock,
				publishedAt: parseHimalayaDate(asString(envelope.date)),
				metadata: {
					account,
					folder,
					messageId: id,
					...(asArray(envelope.flags).length > 0
						? {
								flags: asArray(envelope.flags)
									.map((flag) => asString(flag))
									.filter(Boolean),
							}
						: {}),
				},
			});
		}

		const warnings = readFailures > 0 ? [`${readFailures} message(s) failed to read`] : undefined;
		return { ok: true, documents, ...(warnings !== undefined ? { warnings } : {}) };
	}
}

/** Himalaya prints dates like `2026-07-20 23:27-07:00`. */
function parseHimalayaDate(value: string | undefined): number | undefined {
	if (value === undefined || value.length === 0) return undefined;
	const parsed = Date.parse(value.replace(" ", "T"));
	return Number.isNaN(parsed) ? undefined : parsed;
}

function classifyFailure(stderr: string): "not-configured" | "auth" | "api-error" {
	const lower = stderr.toLowerCase();
	if (lower.includes("cannot find") && lower.includes("account")) return "not-configured";
	if (lower.includes("auth") || lower.includes("login") || lower.includes("credential")) return "auth";
	return "api-error";
}

/** Short, path/PII-opaque failure summary (never raw stderr). */
function shortFailure(result: HimalayaRunResult): string {
	return `himalaya exited with code ${result.code ?? "unknown"}`;
}

function runBinary(binary: string, args: readonly string[], timeoutMs: number): Promise<HimalayaRunResult> {
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
