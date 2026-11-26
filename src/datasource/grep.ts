/**
 * Exact-match (`grep` / `rg`) engine over parsed datasource mirrors and local archives.
 *
 * Mirrors the local parsed-mirror grep (`src/agent/parsed-mirror-tools.ts`) for configured
 * datasources:
 * - File-backed mirrors (`wombat`, `obsidian`): invokes ripgrep (`rg`) over the parsed
 *   markdown mirror tree (skipping `.minsync/` internals) with a pure-JS fallback when `rg`
 *   is absent.
 * - Read-only SQLite sidecar archives (`telegram`/`telecrawl`, `whatsapp`/`wacrawl`,
 *   `slack`/`slacrawl`, `discord`/`discrawl`, `kakao`/`lazykatok`, `notion`/`notcrawl`,
 *   `mailcrawl` fallback): opens the crawler's local SQLite archive read-only and matches
 *   exact substrings or regexes line by line without invoking embedding or ranking models.
 * - Connector chunk stores (`github`, `github-gist`, `rss`): scans persisted parsed chunks.
 */

import { spawn } from "node:child_process";
import { type Dirent, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";
import {
	DEFAULT_MAX_BYTES,
	formatSize,
	getAgentDir,
	truncateHead,
	truncateLine,
} from "@earendil-works/pi-coding-agent";
import type { EvidenceLedger } from "../agent/evidence-ledger.ts";
import type { SearchDocumentRetrievalTraceResult } from "../agent/search-documents.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "./scope.ts";
import type { DatasourceGrepHit, DatasourceGrepOptions, DatasourceSkill } from "./types.ts";

const DEFAULT_GREP_LIMIT = 100;
const DEFAULT_MAX_PER_FILE = 20;
const MAX_CONTEXT_LINES = 20;
const GREP_MAX_LINE_LENGTH = 500;

export interface DirectoryGrepOptions {
	readonly rootDir: string;
	readonly pattern: string;
	readonly literal?: boolean;
	readonly ignoreCase?: boolean;
	readonly context?: number;
	readonly limit?: number;
	readonly maxPerFile?: number;
	/** File extension filter (default `.md`). */
	readonly extension?: string;
	/** Relative directory names to skip (default `[".minsync"]`). */
	readonly skipDirs?: readonly string[];
	readonly rgPath?: string;
	readonly signal?: AbortSignal;
}

export interface DirectoryGrepMatch {
	readonly absPath: string;
	/** Forward-slash path relative to `rootDir`. */
	readonly relPath: string;
	readonly lineNumber: number;
	readonly lineEnd: number;
	readonly line: string;
	readonly snippet: string;
}

/** Compile a user/model grep pattern (literal or regex) into a JS RegExp. */
export function compileGrepPattern(pattern: string, options?: { literal?: boolean; ignoreCase?: boolean }): RegExp {
	if (pattern.length === 0) throw new Error("grep pattern must not be empty");
	const source = options?.literal ? pattern.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&") : pattern;
	return new RegExp(source, options?.ignoreCase ? "i" : "");
}

/** Escape a literal string for SQLite `LIKE ? ESCAPE '\'`. */
export function escapeSqlLike(literal: string): string {
	return literal.replace(/[\\%_]/gu, "\\$&");
}

function isRegularFile(path: string): boolean {
	try {
		return statSync(path).isFile();
	} catch {
		return false;
	}
}

function isDirectoryPath(path: string): boolean {
	try {
		return statSync(path).isDirectory();
	} catch {
		return false;
	}
}

function openReadOnlySqlite(dbPath: string): DatabaseSync | undefined {
	// Node 24+ supports file://...?mode=ro URIs directly; Bun's node:sqlite shim expects a plain path with readOnly: true.
	const uri = `${pathToFileURL(dbPath).href}?mode=ro`;
	try {
		return new DatabaseSync(uri, { readOnly: true });
	} catch {
		try {
			return new DatabaseSync(dbPath, { readOnly: true });
		} catch {
			return undefined;
		}
	}
}

export function resolveRipgrepBinaryForDatasource(override?: string): string | undefined {
	if (override !== undefined) return existsSync(override) ? override : undefined;
	const names = process.platform === "win32" ? ["rg.exe", "rg"] : ["rg"];
	const pathEntries = (process.env.PATH ?? "").split(process.platform === "win32" ? ";" : ":").filter(Boolean);
	for (const dir of pathEntries) {
		for (const name of names) {
			const candidate = join(dir, name);
			if (isRegularFile(candidate)) return candidate;
		}
	}
	try {
		const piBinDir = join(getAgentDir(), "bin");
		for (const name of names) {
			const candidate = join(piBinDir, name);
			if (isRegularFile(candidate)) return candidate;
		}
	} catch {
		// Ignore when pi agent dir cannot be resolved.
	}
	return undefined;
}

interface RgTextField {
	readonly text?: string;
	readonly bytes?: string;
}

function decodeRgText(data: RgTextField | undefined): string {
	if (typeof data?.text === "string") return data.text;
	if (typeof data?.bytes === "string") return Buffer.from(data.bytes, "base64").toString("utf8");
	return "";
}

/** Extract surrounding lines around 1-based `lineNumber` from a pre-split lines array. */
export function sliceContextLines(
	lines: readonly string[],
	lineNumber: number,
	context: number,
): { readonly lineStart: number; readonly lineEnd: number; readonly snippet: string } {
	if (lines.length === 0 || lineNumber < 1) {
		return { lineStart: lineNumber, lineEnd: lineNumber, snippet: "" };
	}
	const idx = Math.min(Math.max(0, lineNumber - 1), lines.length - 1);
	const lo = Math.max(0, idx - context);
	const hi = Math.min(lines.length - 1, idx + context);
	return {
		lineStart: lo + 1,
		lineEnd: hi + 1,
		snippet: lines.slice(lo, hi + 1).join("\n"),
	};
}

/**
 * Search a directory of parsed markdown files with ripgrep (`rg`), falling back to a pure-JS
 * walker when `rg` is not installed. Always skips `.minsync` internal directories.
 */
export async function grepDirectoryWithRipgrep(options: DirectoryGrepOptions): Promise<DirectoryGrepMatch[]> {
	if (options.signal?.aborted) throw new Error("Operation aborted");
	if (options.pattern.length === 0 || !isDirectoryPath(options.rootDir)) return [];

	const extension = options.extension ?? ".md";
	const skipDirs = new Set(options.skipDirs ?? [".minsync"]);
	const context = Math.min(MAX_CONTEXT_LINES, Math.max(0, Math.floor(options.context ?? 0)));
	const limit = Math.max(1, Math.floor(options.limit ?? DEFAULT_GREP_LIMIT));
	const maxPerFile = Math.max(1, Math.floor(options.maxPerFile ?? DEFAULT_MAX_PER_FILE));
	const rootDir = resolve(options.rootDir);

	const rgPath = resolveRipgrepBinaryForDatasource(options.rgPath);
	if (rgPath === undefined) {
		return grepDirectoryJs(rootDir, options, extension, skipDirs, context, limit, maxPerFile);
	}

	const rawRows = await new Promise<
		Array<{ readonly relPath: string; readonly absPath: string; readonly lineNumber: number; readonly line: string }>
	>((resolvePromise, rejectPromise) => {
		const args: string[] = [
			"--no-config",
			"--json",
			"--line-number",
			"--color=never",
			"--no-ignore",
			"--max-count",
			String(maxPerFile),
			"--glob",
			`*${extension}`,
		];
		for (const skip of skipDirs) {
			args.push("--glob", `!${skip}/**`, "--glob", `!**/${skip}/**`);
		}
		if (options.ignoreCase) args.push("--ignore-case");
		if (options.literal) args.push("--fixed-strings");
		args.push("--", options.pattern, ".");

		const child = spawn(rgPath, args, { cwd: rootDir, stdio: ["ignore", "pipe", "pipe"] });
		const rl = createInterface({ input: child.stdout });
		let stderr = "";
		let aborted = false;
		let killedForLimit = false;
		const rows: Array<{
			readonly relPath: string;
			readonly absPath: string;
			readonly lineNumber: number;
			readonly line: string;
		}> = [];

		const onAbort = (): void => {
			aborted = true;
			if (!child.killed) child.kill();
		};
		options.signal?.addEventListener("abort", onAbort, { once: true });
		child.stderr?.on("data", (chunk: Buffer) => {
			stderr += chunk.toString("utf8");
		});
		const cleanup = (): void => {
			rl.close();
			options.signal?.removeEventListener("abort", onAbort);
		};

		rl.on("line", (line) => {
			if (killedForLimit || line.trim().length === 0) return;
			let event: { type?: string; data?: { path?: RgTextField; line_number?: number; lines?: RgTextField } };
			try {
				event = JSON.parse(line);
			} catch {
				return;
			}
			if (event.type !== "match") return;
			const rawPath = decodeRgText(event.data?.path);
			const lineNumber = event.data?.line_number;
			if (!rawPath || typeof lineNumber !== "number") return;
			const relPath = rawPath.replace(/\\/gu, "/").replace(/^\.\//u, "");
			const segments = relPath.split("/");
			if (segments.some((seg) => skipDirs.has(seg))) return;
			const absPath = resolve(rootDir, relPath);
			const lineText = decodeRgText(event.data?.lines).replace(/\r?\n?$/u, "");
			rows.push({ relPath, absPath, lineNumber, line: lineText });
			if (rows.length >= limit) {
				killedForLimit = true;
				if (!child.killed) child.kill();
			}
		});

		child.on("error", (error) => {
			cleanup();
			rejectPromise(new Error(`Failed to run ripgrep: ${error.message}`));
		});
		child.on("close", (code) => {
			cleanup();
			if (aborted) {
				rejectPromise(new Error("Operation aborted"));
				return;
			}
			if (!killedForLimit && code !== 0 && code !== 1) {
				rejectPromise(new Error(stderr.trim() || `ripgrep exited with code ${code}`));
				return;
			}
			resolvePromise(rows);
		});
	});

	// Deterministic file+line ordering + optional context slicing.
	rawRows.sort((a, b) => a.relPath.localeCompare(b.relPath) || a.lineNumber - b.lineNumber);
	const fileLinesCache = new Map<string, readonly string[]>();
	return rawRows.map((row) => {
		if (context === 0) {
			return {
				absPath: row.absPath,
				relPath: row.relPath,
				lineNumber: row.lineNumber,
				lineEnd: row.lineNumber,
				line: row.line,
				snippet: row.line,
			};
		}
		let lines = fileLinesCache.get(row.absPath);
		if (lines === undefined) {
			try {
				lines = readFileSync(row.absPath, "utf8").split(/\r?\n/u);
			} catch {
				lines = [row.line];
			}
			fileLinesCache.set(row.absPath, lines);
		}
		const sliced = sliceContextLines(lines, row.lineNumber, context);
		return {
			absPath: row.absPath,
			relPath: row.relPath,
			lineNumber: row.lineNumber,
			lineEnd: sliced.lineEnd,
			line: row.line,
			snippet: sliced.snippet,
		};
	});
}

function grepDirectoryJs(
	rootDir: string,
	options: DirectoryGrepOptions,
	extension: string,
	skipDirs: ReadonlySet<string>,
	context: number,
	limit: number,
	maxPerFile: number,
): DirectoryGrepMatch[] {
	const matcher = compileGrepPattern(options.pattern, {
		literal: options.literal,
		ignoreCase: options.ignoreCase,
	});
	const files: Array<{ readonly relPath: string; readonly absPath: string }> = [];
	const walk = (dir: string): void => {
		if (options.signal?.aborted) throw new Error("Operation aborted");
		let entries: Dirent[];
		try {
			entries = readdirSync(dir, { withFileTypes: true });
		} catch {
			return;
		}
		entries.sort((a, b) => a.name.localeCompare(b.name));
		for (const entry of entries) {
			if (entry.isDirectory()) {
				if (skipDirs.has(entry.name)) continue;
				walk(join(dir, entry.name));
			} else if (entry.isFile() && entry.name.endsWith(extension)) {
				const absPath = join(dir, entry.name);
				const relPath = relative(rootDir, absPath).split(sep).join("/");
				files.push({ relPath, absPath });
			}
		}
	};
	walk(rootDir);

	const matches: DirectoryGrepMatch[] = [];
	for (const file of files) {
		if (options.signal?.aborted) throw new Error("Operation aborted");
		let content: string;
		try {
			content = readFileSync(file.absPath, "utf8");
		} catch {
			continue;
		}
		const lines = content.split(/\r?\n/u);
		let perFile = 0;
		for (let idx = 0; idx < lines.length; idx += 1) {
			const line = lines[idx] ?? "";
			if (!matcher.test(line)) continue;
			const sliced =
				context > 0
					? sliceContextLines(lines, idx + 1, context)
					: { lineStart: idx + 1, lineEnd: idx + 1, snippet: line };
			matches.push({
				absPath: file.absPath,
				relPath: file.relPath,
				lineNumber: idx + 1,
				lineEnd: sliced.lineEnd,
				line,
				snippet: sliced.snippet,
			});
			perFile += 1;
			if (matches.length >= limit) return matches;
			if (perFile >= maxPerFile) break;
		}
	}
	return matches;
}

/**
 * Match lines inside a single text body (e.g. a chat message, email chunk, or Notion block)
 * and return up to `maxMatches` line-level snippets with optional surrounding context lines.
 */
export function matchLinesInText(
	text: string,
	matcher: RegExp,
	context: number,
	maxMatches: number,
): Array<{ readonly lineNumber: number; readonly lineEnd: number; readonly line: string; readonly snippet: string }> {
	const lines = text.split(/\r?\n/u);
	const out: Array<{
		readonly lineNumber: number;
		readonly lineEnd: number;
		readonly line: string;
		readonly snippet: string;
	}> = [];
	for (let idx = 0; idx < lines.length; idx += 1) {
		const line = lines[idx] ?? "";
		if (!matcher.test(line)) continue;
		const sliced =
			context > 0
				? sliceContextLines(lines, idx + 1, context)
				: { lineStart: idx + 1, lineEnd: idx + 1, snippet: line };
		out.push({
			lineNumber: idx + 1,
			lineEnd: sliced.lineEnd,
			line,
			snippet: sliced.snippet,
		});
		if (out.length >= maxMatches) break;
	}
	return out;
}

/**
 * Read-only SQLite sidecar grep for `notcrawl` (`~/.notcrawl/notcrawl.db`):
 * scans live pages (`pages.alive = 1`) and blocks (`blocks.alive = 1`) directly
 * without spawning `notcrawl` or relying on FTS5 tokenization.
 */
export function grepNotcrawlArchive(options: {
	readonly dbPath: string;
	readonly datasourceId: string;
	readonly instanceId: string;
	readonly pattern: string;
	readonly grepOptions?: DatasourceGrepOptions;
}): DatasourceGrepHit[] {
	if (!isRegularFile(options.dbPath)) return [];
	const grepOpts = options.grepOptions ?? {};
	const matcher = compileGrepPattern(options.pattern, {
		literal: grepOpts.literal,
		ignoreCase: grepOpts.ignoreCase,
	});
	const limit = Math.max(1, Math.floor(grepOpts.limit ?? DEFAULT_GREP_LIMIT));
	const maxPerSource = Math.max(1, Math.floor(grepOpts.maxPerSource ?? 10));
	const context = Math.min(MAX_CONTEXT_LINES, Math.max(0, Math.floor(grepOpts.context ?? 0)));

	const db = openReadOnlySqlite(options.dbPath);
	if (db === undefined) return [];

	try {
		const hits: DatasourceGrepHit[] = [];
		const perPage = new Map<string, number>();

		const pageClauses = ["alive = 1"];
		const pageParams: string[] = [];
		if (grepOpts.literal) {
			pageClauses.push("title LIKE ? ESCAPE '\\'");
			pageParams.push(`%${escapeSqlLike(options.pattern)}%`);
		}
		const pages = db
			.prepare(
				`SELECT id, title, url, last_edited_time FROM pages WHERE ${pageClauses.join(" AND ")} ORDER BY last_edited_time DESC`,
			)
			.all(...pageParams) as Array<{
			id: string;
			title: string;
			url: string | null;
			last_edited_time: number | null;
		}>;

		for (const page of pages) {
			if (!matcher.test(page.title)) continue;
			const source = datasourceSourcePath(options.datasourceId, options.instanceId, page.id);
			if (!matchesDatasourceScope(source, grepOpts.scope)) continue;
			perPage.set(page.id, (perPage.get(page.id) ?? 0) + 1);
			hits.push({
				source,
				content: `[${page.title}] (title) ${page.title}`,
				lineNumber: 1,
				lineEnd: 1,
				label: page.title,
				metadata: {
					datasourceId: options.datasourceId,
					instanceId: options.instanceId,
					pageId: page.id,
					title: page.title,
					...(page.url ? { url: page.url } : {}),
				},
			});
			if (hits.length >= limit) return hits;
		}

		const blockClauses = ["b.alive = 1", "p.alive = 1", "b.text IS NOT NULL", "length(b.text) > 0"];
		const blockParams: string[] = [];
		if (grepOpts.literal) {
			blockClauses.push("b.text LIKE ? ESCAPE '\\'");
			blockParams.push(`%${escapeSqlLike(options.pattern)}%`);
		}
		const blocks = db
			.prepare(
				`SELECT b.id, b.page_id, b.type, b.text, p.title, p.url
				FROM blocks b JOIN pages p ON p.id = b.page_id
				WHERE ${blockClauses.join(" AND ")}
				ORDER BY p.last_edited_time DESC, b.display_order ASC`,
			)
			.iterate(...blockParams) as Iterable<{
			id: string;
			page_id: string;
			type: string;
			text: string;
			title: string;
			url: string | null;
		}>;

		for (const block of blocks) {
			const used = perPage.get(block.page_id) ?? 0;
			if (used >= maxPerSource) continue;
			const source = datasourceSourcePath(options.datasourceId, options.instanceId, block.page_id);
			if (!matchesDatasourceScope(source, grepOpts.scope)) continue;
			const matched = matchLinesInText(
				block.text,
				matcher,
				context,
				Math.min(maxPerSource - used, limit - hits.length),
			);
			for (const m of matched) {
				perPage.set(block.page_id, (perPage.get(block.page_id) ?? 0) + 1);
				hits.push({
					source,
					content: `[${block.title}] ${m.snippet}`,
					lineNumber: m.lineNumber,
					lineEnd: m.lineEnd,
					label: block.title,
					metadata: {
						datasourceId: options.datasourceId,
						instanceId: options.instanceId,
						pageId: block.page_id,
						blockId: block.id,
						title: block.title,
						...(block.url ? { url: block.url } : {}),
					},
				});
				if (hits.length >= limit) return hits;
			}
		}
		return hits;
	} catch {
		return [];
	} finally {
		db.close();
	}
}

/**
 * Read-only SQLite sidecar grep for `mailcrawl`'s `archive.sqlite` when the installed
 * `mailcrawl` CLI predates the `mailcrawl grep` subcommand (or as a fast zero-spawn path).
 */
export function grepMailcrawlArchiveSqlite(options: {
	readonly dbPath: string;
	readonly datasourceId: string;
	readonly instanceId: string;
	readonly account?: string;
	readonly mailbox?: string;
	readonly pattern: string;
	readonly grepOptions?: DatasourceGrepOptions;
}): DatasourceGrepHit[] {
	if (!isRegularFile(options.dbPath)) return [];
	const grepOpts = options.grepOptions ?? {};
	const matcher = compileGrepPattern(options.pattern, {
		literal: grepOpts.literal,
		ignoreCase: grepOpts.ignoreCase,
	});
	const limit = Math.max(1, Math.floor(grepOpts.limit ?? DEFAULT_GREP_LIMIT));
	const maxPerSource = Math.max(1, Math.floor(grepOpts.maxPerSource ?? 10));
	const context = Math.min(MAX_CONTEXT_LINES, Math.max(0, Math.floor(grepOpts.context ?? 0)));

	const db = openReadOnlySqlite(options.dbPath);
	if (db === undefined) return [];

	try {
		const clauses: string[] = [];
		const params: string[] = [];
		if (options.account !== undefined && options.account.length > 0) {
			clauses.push("c.account_id = ?");
			params.push(options.account);
		}
		if (options.mailbox !== undefined && options.mailbox.length > 0) {
			clauses.push("c.mailbox = ?");
			params.push(options.mailbox);
		}
		if (grepOpts.literal) {
			const like = `%${escapeSqlLike(options.pattern)}%`;
			clauses.push(
				"(c.text LIKE ? ESCAPE '\\' OR m.subject LIKE ? ESCAPE '\\' OR m.from_address LIKE ? ESCAPE '\\')",
			);
			params.push(like, like, like);
		}
		const whereSql = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
		const rows = db
			.prepare(
				`SELECT c.chunk_id, c.message_id, c.thread_id, c.account_id, c.mailbox, c.section,
				        m.subject, m.from_address, m.date, c.text
				 FROM chunks c JOIN messages m ON m.message_id = c.message_id
				 ${whereSql}
				 ORDER BY m.date DESC, c.message_id ASC, c.ordinal ASC`,
			)
			.iterate(...params) as Iterable<{
			chunk_id: string;
			message_id: string;
			thread_id: string;
			account_id: string;
			mailbox: string;
			section: string;
			subject: string;
			from_address: string;
			date: string;
			text: string;
		}>;

		const hits: DatasourceGrepHit[] = [];
		const perMessage = new Map<string, number>();
		const matchedSubject = new Set<string>();

		for (const row of rows) {
			let used = perMessage.get(row.message_id) ?? 0;
			if (used >= maxPerSource) continue;
			const source = datasourceSourcePath(options.datasourceId, options.instanceId, row.chunk_id);
			if (!matchesDatasourceScope(source, grepOpts.scope)) continue;
			const label = `${row.subject} — ${row.from_address} (${row.date.slice(0, 10)})`;

			if (!matchedSubject.has(row.message_id) && matcher.test(row.subject)) {
				matchedSubject.add(row.message_id);
				used += 1;
				perMessage.set(row.message_id, used);
				hits.push({
					source,
					content: `[Subject] ${row.subject}`,
					lineNumber: 1,
					lineEnd: 1,
					label,
					metadata: {
						datasourceId: options.datasourceId,
						instanceId: options.instanceId,
						chunkId: row.chunk_id,
						messageId: row.message_id,
						threadId: row.thread_id,
						accountId: row.account_id,
						mailbox: row.mailbox,
						subject: row.subject,
						from: row.from_address,
						date: row.date,
					},
				});
				if (hits.length >= limit) break;
				if (used >= maxPerSource) continue;
			}

			const matched = matchLinesInText(
				row.text,
				matcher,
				context,
				Math.min(maxPerSource - used, limit - hits.length),
			);
			for (const m of matched) {
				used += 1;
				perMessage.set(row.message_id, used);
				hits.push({
					source,
					content: m.snippet,
					lineNumber: m.lineNumber,
					lineEnd: m.lineEnd,
					label,
					metadata: {
						datasourceId: options.datasourceId,
						instanceId: options.instanceId,
						chunkId: row.chunk_id,
						messageId: row.message_id,
						threadId: row.thread_id,
						accountId: row.account_id,
						mailbox: row.mailbox,
						subject: row.subject,
						from: row.from_address,
						date: row.date,
						section: row.section,
					},
				});
				if (hits.length >= limit) break;
			}
			if (hits.length >= limit) break;
		}
		return hits;
	} catch {
		return [];
	} finally {
		db.close();
	}
}

export type DatasourceGrepTarget =
	| { readonly kind: "all"; readonly skills: readonly DatasourceSkill[] }
	| { readonly kind: "single"; readonly skill: DatasourceSkill; readonly scope: string };

/**
 * Determine whether a `grep` tool `path` argument addresses a configured datasource virtual path:
 * - `/datasources` or `/datasources/` -> grep across every configured datasource supporting `grep()`
 * - `/<datasourceId>` or `/<datasourceId>/...` -> grep that connection with the full path as `scope`
 * Returns `undefined` for ordinary OS paths so local/mirror grep handles them unchanged.
 */
export function resolveDatasourceGrepTarget(
	requestedPath: string | undefined,
	skills: readonly DatasourceSkill[],
): DatasourceGrepTarget | undefined {
	if (requestedPath === undefined || skills.length === 0) return undefined;
	const trimmed = requestedPath.trim();
	if (!trimmed.startsWith("/")) return undefined;
	const normalized = trimmed.replace(/\/+$/u, "") || "/";
	if (normalized === "/datasources") {
		const greppable = skills.filter((skill) => typeof skill.grep === "function");
		return greppable.length > 0 ? { kind: "all", skills: greppable } : undefined;
	}
	const rest = normalized.slice(1);
	const slash = rest.indexOf("/");
	const head = slash === -1 ? rest : rest.slice(0, slash);
	if (head.length === 0) return undefined;
	const matched = skills.find((skill) => {
		const desc = skill.describe();
		return (desc.datasourceId ?? desc.name) === head;
	});
	if (matched === undefined || typeof matched.grep !== "function") return undefined;
	return { kind: "single", skill: matched, scope: normalized };
}

/** Run `grep` across one or more datasource skills, respecting the overall `limit`. */
export async function grepDatasourceSkills(
	skills: readonly DatasourceSkill[],
	pattern: string,
	options: DatasourceGrepOptions = {},
): Promise<{
	readonly hits: readonly DatasourceGrepHit[];
	readonly limitReached: boolean;
	readonly searchedDatasources: readonly string[];
}> {
	const limit = Math.max(1, Math.floor(options.limit ?? DEFAULT_GREP_LIMIT));
	const hits: DatasourceGrepHit[] = [];
	const searchedDatasources: string[] = [];
	let limitReached = false;

	for (const skill of skills) {
		if (options.signal?.aborted) throw new Error("Operation aborted");
		if (typeof skill.grep !== "function") continue;
		const id = skill.describe().datasourceId ?? skill.describe().name;
		searchedDatasources.push(id);
		const remaining = limit - hits.length;
		if (remaining <= 0) {
			limitReached = true;
			break;
		}
		const skillHits = await skill.grep(pattern, {
			...options,
			limit: remaining + 1,
		});
		if (skillHits.length > remaining) {
			hits.push(...skillHits.slice(0, remaining));
			limitReached = true;
			break;
		}
		hits.push(...skillHits);
	}
	return { hits, limitReached, searchedDatasources };
}

export interface FormattedDatasourceGrepResult {
	readonly output: string;
	readonly matchCount: number;
	readonly sources: readonly string[];
	readonly results: readonly SearchDocumentRetrievalTraceResult[];
	readonly truncated?: boolean;
}

/** Format datasource exact-grep hits and register each hit in the run's EvidenceLedger as `[eN]`. */
export function formatDatasourceGrepOutput(
	hits: readonly DatasourceGrepHit[],
	ledger: EvidenceLedger | undefined,
	meta: {
		readonly limit: number;
		readonly limitReached: boolean;
		readonly searchedDatasources?: readonly string[];
	},
): FormattedDatasourceGrepResult {
	if (hits.length === 0) {
		const scopeHint =
			meta.searchedDatasources && meta.searchedDatasources.length > 0
				? ` (${meta.searchedDatasources.join(", ")})`
				: "";
		return {
			output: `No exact matches found in datasource parsed mirrors/archives${scopeHint}.`,
			matchCount: 0,
			sources: [],
			results: [],
		};
	}

	const outputLines: string[] = [];
	const sources = new Set<string>();
	const traceResults: SearchDocumentRetrievalTraceResult[] = [];
	let linesTruncated = false;

	for (const [index, hit] of hits.entries()) {
		sources.add(hit.source);
		const evidenceId = ledger?.register({
			method: "grep",
			source: hit.source,
			content: hit.content,
			...(hit.lineNumber !== undefined ? { lineNumber: hit.lineNumber } : {}),
			...(hit.lineEnd !== undefined ? { lineEnd: hit.lineEnd } : {}),
			...(hit.label !== undefined ? { section: hit.label } : {}),
		});
		traceResults.push({
			source: hit.source,
			excerpt: hit.content.slice(0, 400),
			score: 1 / (index + 1),
		});
		const headerParts = [
			...(evidenceId !== undefined ? [`[${evidenceId}]`] : []),
			hit.source,
			...(hit.label !== undefined && hit.label.length > 0 ? [`(${hit.label})`] : []),
		];
		outputLines.push(headerParts.join(" "));
		const bodyLines = hit.content.split(/\r?\n/u);
		const baseLine = hit.lineNumber ?? 1;
		for (const [offset, rawLine] of bodyLines.entries()) {
			const { text, wasTruncated } = truncateLine(rawLine, GREP_MAX_LINE_LENGTH);
			if (wasTruncated) linesTruncated = true;
			const lineTag = hit.lineNumber !== undefined ? `L${baseLine + offset}: ` : "";
			outputLines.push(`  ${lineTag}${text}`);
		}
	}

	const rawOutput = outputLines.join("\n");
	const truncation = truncateHead(rawOutput, { maxLines: Number.MAX_SAFE_INTEGER });
	let output = truncation.content;
	const notices: string[] = [
		"Exact-match grep over parsed datasource mirrors/archives (no embedding/ranking models used); cite hits by [eN].",
	];
	if (meta.limitReached) {
		notices.push(`${meta.limit} matches limit reached. Use limit=${meta.limit * 2} or narrow pattern/path.`);
	}
	if (truncation.truncated) {
		notices.push(`${formatSize(DEFAULT_MAX_BYTES)} output limit reached.`);
	}
	if (linesTruncated) {
		notices.push(`Some lines truncated to ${GREP_MAX_LINE_LENGTH} chars.`);
	}
	output += `\n\n[${notices.join(" ")}]`;

	return {
		output,
		matchCount: hits.length,
		sources: [...sources],
		results: traceResults,
		...(truncation.truncated ? { truncated: true } : {}),
	};
}
