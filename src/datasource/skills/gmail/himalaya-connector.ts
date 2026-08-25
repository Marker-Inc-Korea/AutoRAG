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
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	boundDiagnosticText,
	type ConnectorDocument,
	type ConnectorFetchResult,
	type DatasourceConnector,
	sanitizeIdSegment,
} from "../../connector.ts";
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
	/**
	 * Workspace root for the connector's envelope fingerprint state. When set,
	 * subsequent refreshes fetch only new or changed envelopes and update the
	 * shared chunk store incrementally.
	 */
	readonly workspaceRoot?: string;
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
const STATE_VERSION = 1;

interface HimalayaState {
	readonly version: number;
	readonly fingerprints: Readonly<Record<string, string>>;
}

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
		// Himalaya v2 uses global --account and --json, and --mailbox on the
		// envelope subcommand. Older integration code used --folder/--output,
		// which makes current Himalaya reject the command before connecting.
		const mailboxArgs = this.options.folder !== undefined ? ["--mailbox", this.options.folder] : [];
		const state = this.loadState();

		// 1. List envelopes as JSON.
		const listArgs = [
			...accountArgs,
			"--json",
			"envelope",
			"list",
			...mailboxArgs,
			"--page-size",
			String(this.options.pageSize ?? DEFAULT_PAGE_SIZE),
		];
		let listResult: HimalayaRunResult;
		try {
			listResult = await this.runner(listArgs, timeoutMs);
		} catch {
			return { ok: false, reason: "unavailable", message: "himalaya binary not found or failed to spawn" };
		}
		if (!listResult.ok) {
			return {
				ok: false,
				reason: classifyFailure(`${listResult.stderr}\n${listResult.stdout}`),
				message: diagnosticFailure(listResult),
			};
		}
		let envelopes: readonly unknown[];
		try {
			envelopes = parseEnvelopes(listResult.stdout);
		} catch {
			return { ok: false, reason: "invalid-data", message: "envelope listing was not valid JSON" };
		}

		// 2. Read message bodies (bounded); per-message failures degrade.
		const account = this.options.account ?? "default";
		const folder = this.options.folder ?? "INBOX";
		const documents: ConnectorDocument[] = [];
		let readFailures = 0;
		const nextFingerprints = { ...state.fingerprints };
		const listedDocIds = new Set<string>();
		for (const raw of envelopes.slice(0, maxDocuments)) {
			const envelope = asRecord(raw);
			const id = asString(envelope?.id);
			if (envelope === undefined || id === undefined) continue;
			const docId = `${account}-${folder}-${id}`;
			listedDocIds.add(docId);
			const fingerprint = envelopeFingerprint(envelope);
			if (state.fingerprints[docId] === fingerprint) continue;
			const subject = asString(envelope.subject) ?? "(no subject)";
			const from = asRecord(envelope.from);
			const fromText = [asString(from?.name), asString(from?.addr)].filter(Boolean).join(" ").trim();
			let body = "";
			let bodyRead = false;
			try {
				const readResult = await this.runner(
					[...accountArgs, "--json", "message", "read", ...mailboxArgs, id],
					timeoutMs,
				);
				if (readResult.ok) {
					body = readResult.stdout.trim().slice(0, MAX_BODY_CHARS);
					bodyRead = true;
				}
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
				docId,
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
			if (bodyRead) nextFingerprints[docId] = fingerprint;
		}
		if (envelopes.length <= maxDocuments) {
			for (const docId of Object.keys(state.fingerprints)) {
				if (!listedDocIds.has(docId)) delete nextFingerprints[docId];
			}
		}

		const warnings = readFailures > 0 ? [`${readFailures} message(s) failed to read`] : undefined;
		this.saveState({ version: STATE_VERSION, fingerprints: nextFingerprints });
		const deletedDocIds =
			envelopes.length <= maxDocuments
				? Object.keys(state.fingerprints).filter((docId) => !listedDocIds.has(docId))
				: [];
		return {
			ok: true,
			documents,
			changed: Object.keys(nextFingerprints).length !== Object.keys(state.fingerprints).length || documents.length > 0,
			...(deletedDocIds.length > 0 ? { deletedDocIds } : {}),
			...(warnings !== undefined ? { warnings } : {}),
		};
	}

	private loadState(): HimalayaState {
		const path = this.statePath();
		if (path === undefined) return { version: STATE_VERSION, fingerprints: {} };
		try {
			const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<HimalayaState>;
			if (parsed.version !== STATE_VERSION || parsed.fingerprints === undefined) {
				return { version: STATE_VERSION, fingerprints: {} };
			}
			return { version: STATE_VERSION, fingerprints: parsed.fingerprints };
		} catch {
			return { version: STATE_VERSION, fingerprints: {} };
		}
	}

	private saveState(state: HimalayaState): void {
		const path = this.statePath();
		if (path === undefined) return;
		try {
			mkdirSync(dirname(path), { recursive: true });
			writeFileSync(path, `${JSON.stringify(state)}\n`, "utf8");
		} catch {
			// State persistence is best-effort; the current fetch remains usable.
		}
	}

	private statePath(): string | undefined {
		if (this.options.workspaceRoot === undefined) return undefined;
		return join(
			this.options.workspaceRoot,
			".autorag",
			"datasources",
			sanitizeIdSegment("gmail"),
			sanitizeIdSegment(this.options.account ?? "default"),
			sanitizeIdSegment(this.options.folder ?? "INBOX"),
			"himalaya-state.json",
		);
	}
}

/** Himalaya prints dates like `2026-07-20 23:27-07:00`. */
function parseHimalayaDate(value: string | undefined): number | undefined {
	if (value === undefined || value.length === 0) return undefined;
	const parsed = Date.parse(value.replace(" ", "T"));
	return Number.isNaN(parsed) ? undefined : parsed;
}

function classifyFailure(diagnostic: string): "not-configured" | "auth" | "api-error" {
	const lower = diagnostic.toLowerCase();
	if (lower.includes("cannot find") && lower.includes("account")) return "not-configured";
	if (lower.includes("auth") || lower.includes("login") || lower.includes("credential")) return "auth";
	return "api-error";
}

/** Bounded actionable diagnostic with common secret-bearing values redacted. */
function diagnosticFailure(result: HimalayaRunResult): string {
	const raw = result.stderr.trim() || result.stdout.trim();
	if (raw.length === 0) return `himalaya exited with code ${result.code ?? "unknown"}`;
	return boundDiagnosticText(
		raw
			.replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu, "<redacted-email>")
			.replace(/(?:\/Users|\/home|[A-Z]:\\Users)[^\s"']+/gu, "<redacted-path>")
			.replace(/(?:password|passwd|token|secret)\s*[:=]\s*\S+/giu, "$1=<redacted>"),
	);
}

function parseEnvelopes(stdout: string): readonly unknown[] {
	const parsed = JSON.parse(stdout) as unknown;
	if (Array.isArray(parsed)) return asArray(parsed);
	const record = asRecord(parsed);
	return asArray(record?.envelopes ?? record?.data);
}

function envelopeFingerprint(envelope: Record<string, unknown>): string {
	return JSON.stringify({
		subject: envelope.subject,
		date: envelope.date,
		from: envelope.from,
		to: envelope.to,
		flags: envelope.flags,
	});
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
