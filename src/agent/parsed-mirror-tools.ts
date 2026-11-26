import { spawn } from "node:child_process";
import { type Dirent, existsSync, readdirSync, readFileSync, realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import type {
	ExtensionAPI,
	ExtensionFactory,
	GrepToolDetails,
	ReadToolDetails,
	ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import {
	createGrepToolDefinition,
	createReadToolDefinition,
	DEFAULT_MAX_BYTES,
	formatSize,
	getAgentDir,
	truncateHead,
	truncateLine,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { formatDatasourceGrepOutput, grepDatasourceSkills, resolveDatasourceGrepTarget } from "../datasource/grep.ts";
import type { DatasourceSkill } from "../datasource/types.ts";
import { type CloudRoot, canonicalPath, isPathInside } from "../filesystem/cloud-placeholder.ts";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";
import {
	MINSYNC_VISUAL_SUFFIX,
	minSyncDocumentPath,
	minSyncVisualDocumentPath,
	minSyncWorkspaceRoot,
} from "../minsync/paths.ts";
import { readPagesSidecar } from "../mirror/pages-sidecar.ts";
import { parsedOutputPath, parsedVisualOutputPath } from "../mirror/paths.ts";
import { type ParsedPages, pageRange } from "../parser/pages.ts";
import { describeLocation } from "../retrieval/chunk-location.ts";
import type { EvidenceLedger } from "./evidence-ledger.ts";
import type { SearchDocumentRetrievalTraceResult } from "./search-documents.ts";

/**
 * Binary/compound document extensions that cannot be read or grepped directly as UTF-8 text.
 * Reading any of these through `read` transparently serves its parsed markdown mirror (with
 * line/page metadata) when one exists, and avoids downloading cloud placeholders or dumping
 * binary bytes into the model context.
 */
const PARSED_DOCUMENT_READ_EXTENSIONS = new Set([
	".pdf",
	".hwp",
	".hwpx",
	".hml",
	".hwpml",
	".docx",
	".doc",
	".xlsx",
	".xlsm",
	".xlsb",
	".xls",
	".ods",
	".pptx",
	".eml",
]);

/** Korean statute headings (`제5조(…)`, `제2장 …`) and markdown headings (matches `chunk-location.ts`). */
const ARTICLE_LINE = /^제\s*\d+(?:의\d+)?\s*[조장절관](?:\s|\(|$)/u;
const MARKDOWN_HEADING = /^#{1,6}\s+(.+?)\s*#*$/u;
const MAX_SECTION_CHARS = 80;
const DEFAULT_GREP_LIMIT = 100;
const DEFAULT_PER_FILE_MATCH_CAP = 20;
const GREP_MAX_LINE_LENGTH = 500;

export interface ParsedMirrorMapperOptions {
	readonly workspaceRoot: string;
	readonly searchPaths: readonly string[];
	/** Optional override for cloud roots; a file inside a cloud root prefers its mirror on `read` too. */
	readonly cloudRoots?: () => Promise<readonly CloudRoot[]>;
}

export interface MirrorFileMapping {
	readonly sourcePath: string;
	readonly virtualPath: string;
	readonly mirrorPath: string;
	readonly pagesMirrorPath: string;
	/** Staged visual-extraction mirror path when the file has visually-rich pages extracted. */
	readonly visualMirrorPath?: string;
	/** True when the original source file exists and has a strictly newer mtime than the staged mirror. */
	readonly stale: boolean;
}

export interface MirrorSourceResolution {
	readonly sourcePath: string;
	readonly virtualPath: string;
	readonly pagesMirrorPath: string;
	readonly mirrorKind?: "text" | "visual";
}

export type MirrorTarget =
	| {
			readonly kind: "file";
			readonly mapping: MirrorFileMapping;
			readonly mirrorPath: string;
	  }
	| {
			readonly kind: "directory";
			readonly mirrorPath: string;
			readonly sourceRootPath: string;
	  }
	| {
			readonly kind: "corpus";
			readonly mirrorPath: string;
	  };

export interface MirrorLineLocation {
	readonly pageStart?: number;
	readonly pageEnd?: number;
	readonly section?: string;
}

export interface ParsedMirrorMapper {
	readonly filesRoot: string;
	readonly sourceRoots: readonly SourceRoot[];
	/** Resolve an original local file path to its staged `.md` mirror, if it exists under searchPaths. */
	toMirrorFile(sourceFile: string): MirrorFileMapping | undefined;
	/** Resolve a file, directory, or omitted/workspace path to the matching mirror target. */
	toMirrorTarget(requestedPath: string | undefined, cwd?: string): MirrorTarget | undefined;
	/** Reverse-map a path inside `.autorag/minsync/files` back to its real OS source path. */
	toSourcePath(mirrorFilePath: string): MirrorSourceResolution | undefined;
	/** Measure the page (layout-only) and nearest section heading for a 1-based line in a mirror file. */
	locateLine(mirrorFilePath: string, lineNumber: number, lineEnd?: number): MirrorLineLocation | undefined;
	/** True when `sourcePath` lies inside a known cloud or network root. */
	isUnderCloudRoot(sourcePath: string): Promise<boolean>;
}

interface RootBinding {
	readonly root: SourceRoot;
	readonly resolvedRoot: string;
	readonly realRoot: string;
	readonly prefixSegment: string;
}

function safeRealpath(path: string): string {
	try {
		return realpathSync(path);
	} catch {
		return resolve(path);
	}
}

function isInsideDir(target: string, dir: string): boolean {
	const rel = relative(dir, target);
	return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

function expandUserPath(rawPath: string): string {
	const stripped = rawPath.trim().replace(/^@/, "");
	if (stripped === "~") return homedir();
	if (stripped.startsWith("~/") || stripped.startsWith("~\\")) {
		return join(homedir(), stripped.slice(2));
	}
	return stripped;
}

function resolveUserPath(rawPath: string, cwd: string): string {
	const expanded = expandUserPath(rawPath);
	return isAbsolute(expanded) ? resolve(expanded) : resolve(cwd, expanded);
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

function sectionLabel(line: string): string | undefined {
	const text = line.trim();
	const heading = MARKDOWN_HEADING.exec(text);
	if (heading?.[1] !== undefined) return heading[1].slice(0, MAX_SECTION_CHARS);
	return ARTICLE_LINE.test(text) ? text.slice(0, MAX_SECTION_CHARS) : undefined;
}

function isMirrorStale(sourcePath: string, mirrorPath: string): boolean {
	try {
		const sourceStat = statSync(sourcePath);
		const mirrorStat = statSync(mirrorPath);
		return sourceStat.mtimeMs > mirrorStat.mtimeMs + 1;
	} catch {
		return false;
	}
}

/**
 * Whether `read` should prefer the parsed markdown mirror over the original file.
 * Parsed document formats (PDF, HWP/HWPX, DOCX, XLSX, PPTX, EML) are binary or structured
 * containers that raw text reading cannot inspect meaningfully; plain text, code, and images
 * continue reading from the original file (unless the original is a cloud placeholder).
 */
export function shouldPreferMirrorForRead(filePath: string): boolean {
	return PARSED_DOCUMENT_READ_EXTENSIONS.has(extname(filePath).toLowerCase());
}

/**
 * Adapt a user/model glob written for original filenames (e.g. `*.pdf`, `contracts/**\/*.hwp`)
 * so it matches the mirror tree where every staged document is named `<original>.md`.
 */
export function rewriteMirrorGlob(glob: string): string {
	const cleaned = glob.replace(/^(!?)(?:\.\/)+/, "$1");
	if (cleaned.length === 0) return cleaned;
	const negated = cleaned.startsWith("!");
	const body = negated ? cleaned.slice(1) : cleaned;
	// Globs ending with a wildcard or directory slash already match `<anything>.md`.
	if (body.endsWith("*") || body.endsWith("/")) return cleaned;
	return `${negated ? "!" : ""}${body}.md`;
}

/** Companion glob matching supplementary visual-extraction mirrors (`<original>.autorag-visual.md`). */
export function rewriteVisualMirrorGlob(glob: string): string {
	const cleaned = glob.replace(/^(!?)(?:\.\/)+/, "$1");
	if (cleaned.length === 0) return cleaned;
	const negated = cleaned.startsWith("!");
	const body = negated ? cleaned.slice(1) : cleaned;
	if (body.endsWith("*") || body.endsWith("/")) return cleaned;
	return `${negated ? "!" : ""}${body}${MINSYNC_VISUAL_SUFFIX}`;
}

export function createParsedMirrorMapper(options: ParsedMirrorMapperOptions): ParsedMirrorMapper {
	const workspaceRoot = resolve(options.workspaceRoot);
	const filesRoot = join(minSyncWorkspaceRoot(workspaceRoot), "files");
	const sourceRoots = planSourceRoots(options.searchPaths);
	const bindings: RootBinding[] = sourceRoots.map((root) => ({
		root,
		resolvedRoot: resolve(root.rootPath),
		realRoot: safeRealpath(root.rootPath),
		prefixSegment: root.prefix.replace(/^\/+/, ""),
	}));
	// Longest containing root wins when one search root is nested under another.
	const bySpecificity = [...bindings].sort((a, b) => b.resolvedRoot.length - a.resolvedRoot.length);
	const byPrefix = new Map(bindings.map((binding) => [binding.prefixSegment, binding]));

	interface CachedMirrorDoc {
		readonly mtimeMs: number;
		readonly lines: readonly string[];
		readonly pages: ParsedPages | undefined;
	}
	const docCache = new Map<string, CachedMirrorDoc>();
	let cachedCloudRoots: Promise<readonly string[]> | undefined;

	const findOwner = (candidate: string): { readonly binding: RootBinding; readonly resolved: string } | undefined => {
		const abs = resolve(candidate);
		for (const binding of bySpecificity) {
			if (isInsideDir(abs, binding.resolvedRoot)) return { binding, resolved: abs };
		}
		const real = safeRealpath(abs);
		if (real !== abs) {
			for (const binding of bySpecificity) {
				if (isInsideDir(real, binding.realRoot)) {
					const rel = relative(binding.realRoot, real);
					return { binding, resolved: rel === "" ? binding.resolvedRoot : join(binding.resolvedRoot, rel) };
				}
			}
		}
		return undefined;
	};

	const toMirrorFile = (sourceFile: string): MirrorFileMapping | undefined => {
		const owner = findOwner(sourceFile);
		if (owner === undefined) return undefined;
		const virtualPath = sourceIdentifier(owner.binding.root, owner.resolved);
		const wsRoot = minSyncWorkspaceRoot(workspaceRoot);
		const mirrorPath = minSyncDocumentPath(wsRoot, virtualPath);
		if (!isRegularFile(mirrorPath)) return undefined;
		const candidateVisual = minSyncVisualDocumentPath(wsRoot, virtualPath);
		return {
			sourcePath: owner.resolved,
			virtualPath,
			mirrorPath,
			pagesMirrorPath: parsedOutputPath(workspaceRoot, virtualPath),
			...(isRegularFile(candidateVisual) ? { visualMirrorPath: candidateVisual } : {}),
			stale: isMirrorStale(owner.resolved, mirrorPath),
		};
	};

	const toMirrorTarget = (requestedPath: string | undefined, cwd?: string): MirrorTarget | undefined => {
		if (!isDirectoryPath(filesRoot)) return undefined;
		const trimmed = requestedPath?.trim();
		if (trimmed === undefined || trimmed === "" || trimmed === ".") {
			// When cwd is inside a search root, narrow to that directory; otherwise search the whole mirror corpus.
			if (cwd !== undefined) {
				const cwdOwner = findOwner(cwd);
				if (cwdOwner !== undefined) {
					const rel = relative(cwdOwner.binding.resolvedRoot, cwdOwner.resolved);
					const mirrorDir =
						rel === ""
							? join(filesRoot, cwdOwner.binding.prefixSegment)
							: join(filesRoot, cwdOwner.binding.prefixSegment, rel);
					if (isDirectoryPath(mirrorDir)) {
						return { kind: "directory", mirrorPath: mirrorDir, sourceRootPath: cwdOwner.resolved };
					}
				}
			}
			return { kind: "corpus", mirrorPath: filesRoot };
		}

		const abs = resolveUserPath(trimmed, cwd ?? workspaceRoot);
		// Explicit workspace root searches the entire parsed corpus across all searchPaths.
		if (abs === workspaceRoot || safeRealpath(abs) === safeRealpath(workspaceRoot)) {
			const wsOwner = findOwner(abs);
			if (wsOwner === undefined) {
				return { kind: "corpus", mirrorPath: filesRoot };
			}
		}

		const owner = findOwner(abs);
		if (owner === undefined) return undefined;

		// Check mirror directory first when the requested path is a directory (or corresponds to a mirror dir).
		const rel = relative(owner.binding.resolvedRoot, owner.resolved);
		const mirrorDir =
			rel === "" ? join(filesRoot, owner.binding.prefixSegment) : join(filesRoot, owner.binding.prefixSegment, rel);
		if (isDirectoryPath(owner.resolved) || (!isRegularFile(owner.resolved) && isDirectoryPath(mirrorDir))) {
			if (isDirectoryPath(mirrorDir)) {
				return { kind: "directory", mirrorPath: mirrorDir, sourceRootPath: owner.resolved };
			}
			return undefined;
		}

		const mapping = toMirrorFile(owner.resolved);
		if (mapping === undefined) return undefined;
		return { kind: "file", mapping, mirrorPath: mapping.mirrorPath };
	};

	const toSourcePath = (mirrorFilePath: string): MirrorSourceResolution | undefined => {
		const resolvedFilesRoot = safeRealpath(filesRoot);
		const candidate = safeRealpath(mirrorFilePath);
		const rel = relative(resolvedFilesRoot, candidate);
		if (rel === "" || rel.startsWith("..") || isAbsolute(rel)) return undefined;
		const posixRel = rel.split(sep).join("/");
		const isVisual = posixRel.endsWith(MINSYNC_VISUAL_SUFFIX);
		if (!isVisual && !posixRel.endsWith(".md")) return undefined;
		const withoutSuffix = posixRel.slice(0, -(isVisual ? MINSYNC_VISUAL_SUFFIX.length : ".md".length));
		const slash = withoutSuffix.indexOf("/");
		if (slash === -1) return undefined;
		const prefixSegment = withoutSuffix.slice(0, slash);
		const rest = withoutSuffix.slice(slash + 1);
		const binding = byPrefix.get(prefixSegment);
		if (binding === undefined || rest.length === 0) return undefined;
		const sourcePath = resolve(binding.resolvedRoot, ...rest.split("/"));
		const virtualPath = `/${prefixSegment}/${rest}`;
		return {
			sourcePath,
			virtualPath,
			pagesMirrorPath: isVisual
				? parsedVisualOutputPath(workspaceRoot, virtualPath)
				: parsedOutputPath(workspaceRoot, virtualPath),
			mirrorKind: isVisual ? "visual" : "text",
		};
	};

	const loadMirrorDoc = (mirrorFilePath: string): CachedMirrorDoc | undefined => {
		let mtimeMs: number;
		try {
			mtimeMs = statSync(mirrorFilePath).mtimeMs;
		} catch {
			return undefined;
		}
		const cached = docCache.get(mirrorFilePath);
		if (cached !== undefined && cached.mtimeMs === mtimeMs) return cached;
		let markdown: string;
		try {
			markdown = readFileSync(mirrorFilePath, "utf8");
		} catch {
			return undefined;
		}
		const resolved = toSourcePath(mirrorFilePath);
		const pages = resolved !== undefined ? readPagesSidecar(resolved.pagesMirrorPath, markdown) : undefined;
		const entry: CachedMirrorDoc = { mtimeMs, lines: markdown.split("\n"), pages };
		if (docCache.size > 256) docCache.clear();
		docCache.set(mirrorFilePath, entry);
		return entry;
	};

	const locateLine = (
		mirrorFilePath: string,
		lineNumber: number,
		lineEnd: number = lineNumber,
	): MirrorLineLocation | undefined => {
		const doc = loadMirrorDoc(mirrorFilePath);
		if (doc === undefined || lineNumber < 1 || lineNumber > doc.lines.length) return undefined;
		let section: string | undefined;
		for (let index = lineNumber - 1; index >= 0 && section === undefined; index -= 1) {
			section = sectionLabel(doc.lines[index] ?? "");
		}
		// Blank separator lines between pages are not in any page span; narrow the range to
		// non-blank content lines when computing layout pages so surrounding blank context lines
		// never erase a valid page number.
		let contentStart = lineNumber;
		let contentEnd = Math.min(lineEnd, doc.lines.length);
		while (contentStart < contentEnd && (doc.lines[contentStart - 1]?.trim() ?? "") === "") contentStart += 1;
		while (contentEnd > contentStart && (doc.lines[contentEnd - 1]?.trim() ?? "") === "") contentEnd -= 1;
		const range = doc.pages?.mode === "layout" ? pageRange(doc.pages, contentStart, contentEnd) : undefined;
		if (range === undefined && section === undefined) return undefined;
		return {
			...(range ?? {}),
			...(section !== undefined ? { section } : {}),
		};
	};

	const isUnderCloudRoot = async (sourcePath: string): Promise<boolean> => {
		if (options.cloudRoots === undefined) return false;
		cachedCloudRoots ??= options
			.cloudRoots()
			.then((roots) => roots.map((root) => canonicalPath(root.path)))
			.catch(() => []);
		const roots = await cachedCloudRoots;
		if (roots.length === 0) return false;
		const canonical = canonicalPath(sourcePath);
		return roots.some((root) => isPathInside(canonical, root));
	};

	return {
		filesRoot,
		sourceRoots,
		toMirrorFile,
		toMirrorTarget,
		toSourcePath,
		locateLine,
		isUnderCloudRoot,
	};
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

function resolveRipgrepBinary(override?: string): string | undefined {
	if (override !== undefined) return existsSync(override) ? override : undefined;
	const names = process.platform === "win32" ? ["rg.exe", "rg"] : ["rg"];
	const pathEntries = (process.env.PATH ?? "").split(process.platform === "win32" ? ";" : ":").filter(Boolean);
	for (const dir of pathEntries) {
		for (const name of names) {
			const candidate = join(dir, name);
			if (isRegularFile(candidate)) return candidate;
		}
	}
	// Check pi's managed tool bin directory (where pi downloads rg if not on PATH).
	try {
		const piBinDir = join(getAgentDir(), "bin");
		for (const name of names) {
			const candidate = join(piBinDir, name);
			if (isRegularFile(candidate)) return candidate;
		}
	} catch {
		// Ignore if agent dir cannot be resolved.
	}
	return undefined;
}

/**
 * Pure-JS fallback for searching mirror markdown files when `rg` is not installed.
 * Supports literal/regex, ignoreCase, context lines, per-file cap, and overall limit.
 */
function runJsMirrorGrep(options: {
	readonly mapper: ParsedMirrorMapper;
	readonly target: MirrorTarget;
	readonly pattern: string;
	readonly glob?: string;
	readonly ignoreCase?: boolean;
	readonly literal?: boolean;
	readonly contextValue: number;
	readonly effectiveLimit: number;
	readonly maxPerFile: number;
	readonly signal?: AbortSignal;
}): {
	readonly rows: MirrorGrepHit[];
	readonly matchCount: number;
	readonly matchLimitReached: boolean;
	readonly perFileCapped: boolean;
} {
	const flags = options.ignoreCase ? "i" : "";
	const matcher = options.literal
		? new RegExp(options.pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), flags)
		: new RegExp(options.pattern, flags);

	const globSpec = options.glob?.trim();
	const mirrorGlob = globSpec && globSpec.length > 0 ? rewriteMirrorGlob(globSpec) : undefined;
	const visualMirrorGlob = globSpec && globSpec.length > 0 ? rewriteVisualMirrorGlob(globSpec) : undefined;
	const globMatcher = mirrorGlob ? compileSimpleGlob(mirrorGlob) : undefined;
	const visualGlobMatcher =
		visualMirrorGlob && visualMirrorGlob !== mirrorGlob ? compileSimpleGlob(visualMirrorGlob) : undefined;

	const files: string[] = [];
	if (options.target.kind === "file") {
		files.push(options.target.mirrorPath);
		if (options.target.mapping.visualMirrorPath !== undefined) {
			files.push(options.target.mapping.visualMirrorPath);
		}
	} else {
		const rootDir = options.target.mirrorPath;
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
				const full = join(dir, entry.name);
				if (entry.isDirectory()) {
					walk(full);
				} else if (entry.isFile() && entry.name.endsWith(".md")) {
					if (globMatcher !== undefined) {
						const relPosix = relative(rootDir, full).split(sep).join("/");
						const matched =
							globMatcher(relPosix, entry.name) || visualGlobMatcher?.(relPosix, entry.name) === true;
						if (!matched) continue;
					}
					files.push(full);
				}
			}
		};
		walk(rootDir);
	}

	const rows: MirrorGrepHit[] = [];
	let matchCount = 0;
	let matchLimitReached = false;
	let perFileCapped = false;

	for (const mirrorPath of files) {
		if (options.signal?.aborted) throw new Error("Operation aborted");
		const mapped = options.target.kind === "file" ? options.target.mapping : options.mapper.toSourcePath(mirrorPath);
		if (mapped === undefined) continue;
		let text: string;
		try {
			text = readFileSync(mirrorPath, "utf8");
		} catch {
			continue;
		}
		const lines = text.split(/\r?\n/);
		const matchedIndices: number[] = [];
		for (let idx = 0; idx < lines.length; idx += 1) {
			if (!matcher.test(lines[idx] ?? "")) continue;
			if (matchedIndices.length >= options.maxPerFile) {
				perFileCapped = true;
				break;
			}
			if (matchCount + matchedIndices.length >= options.effectiveLimit) {
				matchLimitReached = true;
				break;
			}
			matchedIndices.push(idx);
		}
		if (matchedIndices.length === 0) {
			if (matchLimitReached) break;
			continue;
		}
		const matchSet = new Set(matchedIndices);
		const included = new Set<number>();
		for (const idx of matchedIndices) {
			const lo = Math.max(0, idx - options.contextValue);
			const hi = Math.min(lines.length - 1, idx + options.contextValue);
			for (let lineIdx = lo; lineIdx <= hi; lineIdx += 1) included.add(lineIdx);
		}
		const sortedIndices = [...included].sort((a, b) => a - b);
		for (const idx of sortedIndices) {
			rows.push({
				mirrorPath,
				sourcePath: mapped.sourcePath,
				lineNumber: idx + 1,
				text: lines[idx] ?? "",
				isMatch: matchSet.has(idx),
			});
		}
		matchCount += matchedIndices.length;
		if (matchLimitReached) break;
	}

	return { rows, matchCount, matchLimitReached, perFileCapped };
}

function compileSimpleGlob(glob: string): (relPosix: string, fileName: string) => boolean {
	const negated = glob.startsWith("!");
	const pattern = negated ? glob.slice(1) : glob;
	const hasSlash = pattern.includes("/");
	// Convert simple glob (*, **, ?, {a,b}) into a RegExp.
	let regexStr = "^";
	for (let i = 0; i < pattern.length; i += 1) {
		const ch = pattern[i] as string;
		if (ch === "*" && pattern[i + 1] === "*") {
			regexStr += ".*";
			i += 1;
			if (pattern[i + 1] === "/") i += 1;
		} else if (ch === "*") {
			regexStr += "[^/]*";
		} else if (ch === "?") {
			regexStr += "[^/]";
		} else if (ch === "{") {
			const end = pattern.indexOf("}", i);
			if (end !== -1) {
				const alts = pattern
					.slice(i + 1, end)
					.split(",")
					.map((alt) => alt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
				regexStr += `(?:${alts.join("|")})`;
				i = end;
			} else {
				regexStr += "\\{";
			}
		} else {
			regexStr += ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		}
	}
	regexStr += "$";
	const re = new RegExp(regexStr);
	return (relPosix: string, fileName: string) => {
		const matched = hasSlash ? re.test(relPosix) : re.test(fileName);
		return negated ? !matched : matched;
	};
}

interface MirrorGrepHit {
	readonly mirrorPath: string;
	readonly sourcePath: string;
	readonly lineNumber: number;
	readonly text: string;
	readonly isMatch: boolean;
}

export interface MirrorGrepExecutionResult {
	readonly output: string;
	readonly matchCount: number;
	readonly matchedSources: readonly string[];
	readonly details: GrepToolDetails & {
		readonly method: "grep";
		readonly source: "parsed-mirror";
		readonly resultCount: number;
		readonly matchCount: number;
		readonly sources: readonly string[];
		readonly results: readonly SearchDocumentRetrievalTraceResult[];
		readonly perFileCapped?: boolean;
	};
}

export interface RunMirrorGrepOptions {
	readonly mapper: ParsedMirrorMapper;
	readonly target: MirrorTarget;
	readonly pattern: string;
	readonly glob?: string;
	readonly ignoreCase?: boolean;
	readonly literal?: boolean;
	readonly context?: number;
	readonly limit?: number;
	readonly maxMatchesPerFile?: number;
	readonly ledger?: EvidenceLedger;
	readonly rgPath?: string;
	readonly signal?: AbortSignal;
}

export async function runMirrorGrep(options: RunMirrorGrepOptions): Promise<MirrorGrepExecutionResult> {
	if (options.signal?.aborted) throw new Error("Operation aborted");
	const rgPath = resolveRipgrepBinary(options.rgPath);

	const isDir = options.target.kind !== "file";
	const contextValue =
		options.context !== undefined && options.context > 0 ? Math.min(Math.floor(options.context), 20) : 0;
	const effectiveLimit = Math.max(1, Math.floor(options.limit ?? DEFAULT_GREP_LIMIT));
	const maxPerFile = Math.max(1, Math.floor(options.maxMatchesPerFile ?? DEFAULT_PER_FILE_MATCH_CAP));

	const hits =
		rgPath === undefined
			? runJsMirrorGrep({
					mapper: options.mapper,
					target: options.target,
					pattern: options.pattern,
					glob: options.glob,
					ignoreCase: options.ignoreCase,
					literal: options.literal,
					contextValue,
					effectiveLimit,
					maxPerFile,
					signal: options.signal,
				})
			: await new Promise<{
					readonly rows: MirrorGrepHit[];
					readonly matchCount: number;
					readonly matchLimitReached: boolean;
					readonly perFileCapped: boolean;
				}>((resolvePromise, rejectPromise) => {
					const args: string[] = [
						"--no-config",
						"--json",
						"--line-number",
						"--color=never",
						"--no-ignore",
						"--hidden",
						"--max-count",
						String(maxPerFile),
					];
					if (options.ignoreCase) args.push("--ignore-case");
					if (options.literal) args.push("--fixed-strings");
					if (options.glob !== undefined && options.glob.trim() !== "") {
						const trimmedGlob = options.glob.trim();
						const baseGlob = rewriteMirrorGlob(trimmedGlob);
						const visualGlob = rewriteVisualMirrorGlob(trimmedGlob);
						args.push("--glob", baseGlob);
						if (visualGlob !== baseGlob) args.push("--glob", visualGlob);
					}
					if (contextValue > 0) args.push("--context", String(contextValue));
					if (isDir) {
						args.push("--", options.pattern, ".");
					} else {
						args.push("--", options.pattern, options.target.mirrorPath);
						if (options.target.mapping.visualMirrorPath !== undefined) {
							args.push(options.target.mapping.visualMirrorPath);
						}
					}

					const child = spawn(rgPath, args, {
						cwd: isDir ? options.target.mirrorPath : undefined,
						stdio: ["ignore", "pipe", "pipe"],
					});
					const rl = createInterface({ input: child.stdout });
					let stderr = "";
					let matchCount = 0;
					let matchLimitReached = false;
					let perFileCapped = false;
					let aborted = false;
					let killedDueToLimit = false;
					let done = false;
					let limitHit = false;
					let lastFile: string | undefined;
					let trailingContextEnd = 0;
					const perFileCount = new Map<string, number>();
					const rows: MirrorGrepHit[] = [];

					const stopChild = (dueToLimit = false): void => {
						if (!child.killed) {
							killedDueToLimit = dueToLimit;
							child.kill();
						}
					};
					const onAbort = (): void => {
						aborted = true;
						stopChild();
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
						if (done || line.trim().length === 0) return;
						let event: {
							type?: string;
							data?: {
								path?: RgTextField;
								line_number?: number;
								lines?: RgTextField;
							};
						};
						try {
							event = JSON.parse(line);
						} catch {
							return;
						}
						const isMatch = event.type === "match";
						if (!isMatch && event.type !== "context") return;
						const rawPath = decodeRgText(event.data?.path);
						const lineNumber = event.data?.line_number;
						if (!rawPath || typeof lineNumber !== "number") return;
						const mirrorPath = isDir ? resolve(options.target.mirrorPath, rawPath) : resolve(rawPath);
						const mapped =
							options.target.kind === "file" ? options.target.mapping : options.mapper.toSourcePath(mirrorPath);
						if (mapped === undefined) return;
						const lineText = decodeRgText(event.data?.lines).replace(/\r?\n?$/, "");

						if (!limitHit) {
							lastFile = mirrorPath;
							rows.push({
								mirrorPath,
								sourcePath: mapped.sourcePath,
								lineNumber,
								text: lineText,
								isMatch,
							});
							if (!isMatch) return;
							matchCount += 1;
							const fileCount = (perFileCount.get(mirrorPath) ?? 0) + 1;
							perFileCount.set(mirrorPath, fileCount);
							if (fileCount >= maxPerFile) perFileCapped = true;
							if (matchCount >= effectiveLimit) {
								limitHit = true;
								trailingContextEnd = lineNumber + contextValue;
							}
							return;
						}

						const inWindow = mirrorPath === lastFile && lineNumber <= trailingContextEnd;
						if (inWindow) {
							rows.push({
								mirrorPath,
								sourcePath: mapped.sourcePath,
								lineNumber,
								text: lineText,
								isMatch: false,
							});
						}
						if (isMatch) matchLimitReached = true;
						if (matchLimitReached && !(inWindow && lineNumber < trailingContextEnd)) {
							done = true;
							stopChild(true);
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
						if (!killedDueToLimit && code !== 0 && code !== 1) {
							rejectPromise(new Error(stderr.trim() || `ripgrep exited with code ${code}`));
							return;
						}
						resolvePromise({ rows, matchCount, matchLimitReached, perFileCapped });
					});
				});

	if (hits.matchCount === 0) {
		return {
			output: "No matches found in parsed document mirrors.",
			matchCount: 0,
			matchedSources: [],
			details: {
				method: "grep",
				source: "parsed-mirror",
				resultCount: 0,
				matchCount: 0,
				sources: [],
				results: [],
			},
		};
	}

	// Group contiguous blocks per file so each match block registers one clean evidence id with context.
	const blocks: {
		readonly mirrorPath: string;
		readonly sourcePath: string;
		readonly rows: MirrorGrepHit[];
	}[] = [];
	for (const row of hits.rows) {
		const prev = blocks[blocks.length - 1];
		const lastRow = prev?.rows[prev.rows.length - 1];
		if (
			prev !== undefined &&
			lastRow !== undefined &&
			prev.mirrorPath === row.mirrorPath &&
			row.lineNumber <= lastRow.lineNumber + 1
		) {
			prev.rows.push(row);
		} else {
			blocks.push({ mirrorPath: row.mirrorPath, sourcePath: row.sourcePath, rows: [row] });
		}
	}

	const outputLines: string[] = [];
	const matchedSources = new Set<string>();
	const staleSources = new Set<string>();
	const traceResults: SearchDocumentRetrievalTraceResult[] = [];
	let linesTruncated = false;

	for (const [blockIndex, block] of blocks.entries()) {
		if (!matchedSources.has(block.sourcePath)) {
			matchedSources.add(block.sourcePath);
			if (isMirrorStale(block.sourcePath, block.mirrorPath)) staleSources.add(block.sourcePath);
		}
		const firstLine = block.rows[0]?.lineNumber ?? 1;
		const lastLine = block.rows[block.rows.length - 1]?.lineNumber ?? firstLine;
		const matchedRows = block.rows.filter((row) => row.isMatch);
		const matchStart = matchedRows[0]?.lineNumber ?? firstLine;
		const matchEnd = matchedRows[matchedRows.length - 1]?.lineNumber ?? lastLine;
		const loc = options.mapper.locateLine(block.mirrorPath, matchStart, matchEnd);
		const snippet = block.rows.map((row) => row.text).join("\n");
		const evidenceId = options.ledger?.register({
			method: "grep",
			source: block.sourcePath,
			content: snippet,
			lineNumber: firstLine,
			lineEnd: lastLine,
			...(loc?.pageStart !== undefined && loc?.pageEnd !== undefined
				? { pageStart: loc.pageStart, pageEnd: loc.pageEnd }
				: {}),
			...(loc?.section !== undefined ? { section: loc.section } : {}),
		});
		traceResults.push({
			source: block.sourcePath,
			excerpt: snippet.slice(0, 400),
			score: 1 / (blockIndex + 1),
		});
		const locLabel = describeLocation({
			...(loc?.pageStart !== undefined ? { pageStart: loc.pageStart, pageEnd: loc.pageEnd } : {}),
			...(loc?.section !== undefined ? { section: loc.section } : {}),
		});
		const headerParts = [
			...(evidenceId !== undefined ? [`[${evidenceId}]`] : []),
			block.sourcePath,
			...(locLabel !== undefined ? [`(${locLabel})`] : []),
		];
		outputLines.push(headerParts.join(" "));
		for (const row of block.rows) {
			const { text, wasTruncated } = truncateLine(row.text, GREP_MAX_LINE_LENGTH);
			if (wasTruncated) linesTruncated = true;
			const sepChar = row.isMatch ? ":" : "-";
			outputLines.push(`  L${row.lineNumber}${sepChar} ${text}`);
		}
	}

	const rawOutput = outputLines.join("\n");
	const truncation = truncateHead(rawOutput, { maxLines: Number.MAX_SAFE_INTEGER });
	let output = truncation.content;
	const notices: string[] = [
		"Searched parsed markdown mirrors (includes PDF/HWP/DOCX/XLSX/PPTX); line numbers (L#) are parsed-markdown lines — pass offset=L# to `read` on the original path to inspect surrounding context.",
	];
	if (staleSources.size > 0) {
		const listed = [...staleSources].slice(0, 5).join(", ");
		const extra = staleSources.size > 5 ? ` (+${staleSources.size - 5} more)` : "";
		notices.push(`Mirror may be stale (source changed since the last refresh): ${listed}${extra}.`);
	}
	if (hits.perFileCapped) {
		notices.push(`Capped at ${maxPerFile} matches per file.`);
	}
	if (hits.matchLimitReached) {
		notices.push(`${effectiveLimit} matches limit reached. Use limit=${effectiveLimit * 2} or narrow pattern/path.`);
	}
	if (truncation.truncated) {
		notices.push(`${formatSize(DEFAULT_MAX_BYTES)} output limit reached.`);
	}
	if (linesTruncated) {
		notices.push(`Some lines truncated to ${GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines.`);
	}
	output += `\n\n[${notices.join(" ")}]`;

	const details: MirrorGrepExecutionResult["details"] = {
		method: "grep",
		source: "parsed-mirror",
		resultCount: traceResults.length,
		matchCount: hits.matchCount,
		sources: [...matchedSources],
		results: traceResults,
		...(hits.matchLimitReached ? { matchLimitReached: effectiveLimit } : {}),
		...(truncation.truncated ? { truncation } : {}),
		...(linesTruncated ? { linesTruncated: true } : {}),
		...(hits.perFileCapped ? { perFileCapped: true } : {}),
	};

	return {
		output,
		matchCount: hits.matchCount,
		matchedSources: [...matchedSources],
		details,
	};
}

const grepSchema = Type.Object({
	pattern: Type.String({ description: "Search pattern (regex or literal string)" }),
	path: Type.Optional(
		Type.String({
			description:
				"Directory, file, or virtual datasource path to search. Omit to search across all indexed local document mirrors; pass a folder or file inside search folders to narrow; or pass '/<datasourceId>' (e.g. '/wombat', '/mailcrawl', '/telegram', '/slack', '/discord', '/kakao', '/obsidian', '/notion') or '/datasources' to run exact-match grep directly on parsed datasource mirrors/archives.",
		}),
	),
	glob: Type.Optional(
		Type.String({
			description:
				"Filter files by glob pattern relative to path, e.g. '*.pdf', '*.hwp', 'contracts/**/*.docx'. Automatically matches parsed markdown mirrors.",
		}),
	),
	ignoreCase: Type.Optional(Type.Boolean({ description: "Case-insensitive search (default: false)" })),
	literal: Type.Optional(
		Type.Boolean({ description: "Treat pattern as literal string instead of regex (default: false)" }),
	),
	context: Type.Optional(
		Type.Number({ description: "Number of lines to show before and after each match (default: 0)" }),
	),
	limit: Type.Optional(Type.Number({ description: "Maximum number of matches to return (default: 100)" })),
});

const readSchema = Type.Object({
	path: Type.String({ description: "Path to the file to read (relative or absolute)" }),
	offset: Type.Optional(Type.Number({ description: "Line number to start reading from (1-indexed)" })),
	limit: Type.Optional(Type.Number({ description: "Maximum number of lines to read" })),
});

export interface ParsedMirrorToolsExtensionOptions {
	readonly workspaceRoot: string;
	readonly searchPaths: readonly string[];
	readonly ledger?: EvidenceLedger;
	readonly cloudRoots?: () => Promise<readonly CloudRoot[]>;
	readonly rgPath?: string;
	/** Configured datasource skills whose parsed mirrors/archives can be grepped via `/<datasourceId>` or `/datasources`. */
	readonly datasourceSkills?: readonly DatasourceSkill[];
}

/**
 * pi extension that transparently routes `grep` and `read` through AutoRAG's parsed markdown mirror
 * (`.autorag/minsync/files`) when targeting configured search folders or parsed document formats:
 *
 * - `grep`: searches the local parsed markdown mirror with ripgrep — covering PDF, HWP/HWPX, DOCX,
 *   XLSX, PPTX, EML and plain text in milliseconds without touching cloud placeholders, maps results
 *   back to original absolute paths with page/section locations, and registers `[eN]` ids in the
 *   run's EvidenceLedger. Paths outside `searchPaths` or without mirrors fall back to pi's native `grep`.
 * - `read`: serves the parsed markdown mirror (with line/page/section header and `[eN]` id) for
 *   parsed document formats (PDF, HWP, DOCX, XLSX, PPTX, EML) and for cloud-hosted files that have a
 *   mirror, so `grep` → `read(offset=L#)` works seamlessly without dumping binary bytes or hydrating
 *   cloud placeholders. Plain text, code, and images fall back to pi's native `read`.
 */
export function createParsedMirrorToolsExtension(options: ParsedMirrorToolsExtensionOptions): ExtensionFactory {
	const mapper = createParsedMirrorMapper({
		workspaceRoot: options.workspaceRoot,
		searchPaths: options.searchPaths,
		cloudRoots: options.cloudRoots,
	});

	return (pi: ExtensionAPI): void => {
		if (typeof pi.registerTool !== "function") return;

		const baseGrep = createGrepToolDefinition(options.workspaceRoot);
		const baseRead = createReadToolDefinition(options.workspaceRoot);
		const configuredGrepPaths = [
			...new Set(
				(options.datasourceSkills ?? [])
					.filter((skill) => typeof skill.grep === "function")
					.map((skill) => `/${skill.describe().datasourceId ?? skill.describe().name}`),
			),
		];
		const configuredGrepHint =
			configuredGrepPaths.length > 0
				? ` Configured datasource virtual paths: ${configuredGrepPaths.map((p) => `'${p}'`).join(", ")}, or '/datasources' for all.`
				: " For configured datasource virtual paths ('/<datasourceId>' or '/datasources' for all), runs exact-match grep directly over each datasource's parsed mirror or read-only SQLite archive.";
		const dynamicGrepSchema =
			configuredGrepPaths.length > 0
				? Type.Object({
						...grepSchema.properties,
						path: Type.Optional(
							Type.String({
								description: `Directory, file, or virtual datasource path to search. Omit to search across all indexed local document mirrors; pass a folder or file inside search folders to narrow; or pass one of the configured datasource virtual paths (${configuredGrepPaths.map((p) => `'${p}'`).join(", ")}, or '/datasources' for all) to run exact-match grep directly on parsed datasource mirrors/archives.`,
							}),
						),
					})
				: grepSchema;

		const mirrorGrepTool: ToolDefinition<typeof grepSchema, GrepToolDetails | undefined> = {
			...baseGrep,
			name: "grep",
			label: "grep",
			description: `Search file or datasource contents for an exact string or regex pattern. For configured search folders (or when path is omitted), searches AutoRAG's local parsed markdown mirrors with ripgrep — covering PDF, HWP/HWPX, DOCX, XLSX, PPTX, EML, and text files rapidly without downloading cloud placeholders, and returns original file paths with [eN] citation ids, parsed line numbers (L#), and page/section locations. For configured datasource virtual paths, runs exact-match grep directly over each datasource's parsed mirror or read-only SQLite archive without invoking embedding/ranking models.${configuredGrepHint} Falls back to raw filesystem grep for paths outside search folders or unmirrored files. Output is truncated to ${DEFAULT_GREP_LIMIT} matches or ${DEFAULT_MAX_BYTES / 1024}KB.`,
			parameters: dynamicGrepSchema,
			async execute(toolCallId, params, signal, onUpdate, ctx) {
				const dsTarget = resolveDatasourceGrepTarget(params.path, options.datasourceSkills ?? []);
				if (dsTarget !== undefined) {
					const effectiveLimit = Math.max(1, Math.floor(params.limit ?? DEFAULT_GREP_LIMIT));
					const skills = dsTarget.kind === "all" ? dsTarget.skills : [dsTarget.skill];
					const { hits, limitReached, searchedDatasources } = await grepDatasourceSkills(skills, params.pattern, {
						literal: params.literal,
						ignoreCase: params.ignoreCase,
						context: params.context,
						limit: effectiveLimit,
						...(dsTarget.kind === "single" ? { scope: dsTarget.scope } : {}),
						signal,
					});
					const formatted = formatDatasourceGrepOutput(hits, options.ledger, {
						limit: effectiveLimit,
						limitReached,
						searchedDatasources,
					});
					return {
						content: [{ type: "text", text: formatted.output }],
						details: {
							method: "grep",
							source: "datasource-parsed-mirror",
							resultCount: formatted.results.length,
							matchCount: formatted.matchCount,
							sources: formatted.sources,
							results: formatted.results,
							...(limitReached ? { matchLimitReached: effectiveLimit } : {}),
						} as unknown as GrepToolDetails,
					};
				}
				const cwd = ctx?.cwd || options.workspaceRoot;
				const target = mapper.toMirrorTarget(params.path, cwd);
				if (target !== undefined) {
					// For a single non-document plain-text file whose original is newer than its mirror (and not in a cloud root),
					// prefer raw grep so newly edited text files are never missed.
					if (
						target.kind === "file" &&
						!shouldPreferMirrorForRead(target.mapping.sourcePath) &&
						target.mapping.stale &&
						!(await mapper.isUnderCloudRoot(target.mapping.sourcePath))
					) {
						return baseGrep.execute(toolCallId, params, signal, onUpdate, ctx);
					}
					const result = await runMirrorGrep({
						mapper,
						target,
						pattern: params.pattern,
						glob: params.glob,
						ignoreCase: params.ignoreCase,
						literal: params.literal,
						context: params.context,
						limit: params.limit,
						ledger: options.ledger,
						rgPath: options.rgPath,
						signal,
					});
					return {
						content: [{ type: "text", text: result.output }],
						details: result.details,
					};
				}
				return baseGrep.execute(toolCallId, params, signal, onUpdate, ctx);
			},
		};

		const mirrorReadTool: ToolDefinition<typeof readSchema, ReadToolDetails | undefined> = {
			...baseRead,
			name: "read",
			label: "read",
			description:
				"Read the contents of a file. For parsed documents (PDF, HWP/HWPX, DOCX, XLSX, PPTX, EML) and cloud-hosted indexed files, reads AutoRAG's parsed markdown mirror directly with page/section context and an [eN] citation id — matching line numbers returned by `grep`. Supports plain text, code, and images (jpg, png, gif, webp, bmp) directly from disk. Use offset/limit for large files.",
			parameters: readSchema,
			async execute(toolCallId, params, signal, onUpdate, ctx) {
				const dsReadTarget = resolveDatasourceGrepTarget(params.path, options.datasourceSkills ?? []);
				if (dsReadTarget !== undefined) {
					const targetPath = dsReadTarget.kind === "single" ? dsReadTarget.scope : "/datasources";
					return {
						content: [
							{
								type: "text",
								text: `${targetPath} is a virtual datasource identifier, not a local OS file path. Use \`grep(pattern, path="${targetPath}", context=3)\` for surrounding lines or the datasource's \`search_datasource_*\` tool.`,
							},
						],
						details: undefined,
					};
				}
				const cwd = ctx?.cwd || options.workspaceRoot;
				const abs = resolveUserPath(params.path, cwd);
				const mapping = mapper.toMirrorFile(abs);
				if (mapping !== undefined) {
					const preferMirror =
						shouldPreferMirrorForRead(mapping.sourcePath) || (await mapper.isUnderCloudRoot(mapping.sourcePath));
					if (preferMirror) {
						const mirrorResult = await baseRead.execute(
							toolCallId,
							{ ...params, path: mapping.mirrorPath },
							signal,
							onUpdate,
							ctx,
						);
						const textBlock = mirrorResult.content.find((item) => item.type === "text");
						if (textBlock !== undefined && textBlock.type === "text") {
							const startLine = params.offset ? Math.max(1, Math.floor(params.offset)) : 1;
							// Strip pi's trailing continuation/truncation footer before counting displayed content lines.
							const contentBody = textBlock.text.replace(
								/\n\n\[(?:Showing lines|(?:\d+) more lines)[^\]]*\]$/u,
								"",
							);
							const shownLines = contentBody.split("\n").length;
							const endLine = startLine + Math.max(0, shownLines - 1);
							const loc = mapper.locateLine(mapping.mirrorPath, startLine, endLine);
							const evidenceId = options.ledger?.register({
								method: "read",
								source: mapping.sourcePath,
								content: contentBody,
								lineNumber: startLine,
								lineEnd: endLine,
								...(loc?.pageStart !== undefined && loc?.pageEnd !== undefined
									? { pageStart: loc.pageStart, pageEnd: loc.pageEnd }
									: {}),
								...(loc?.section !== undefined ? { section: loc.section } : {}),
							});
							const locLabel = describeLocation({
								lineStart: startLine,
								lineEnd: endLine,
								...(loc?.pageStart !== undefined ? { pageStart: loc.pageStart, pageEnd: loc.pageEnd } : {}),
								...(loc?.section !== undefined ? { section: loc.section } : {}),
							});
							const staleNote = mapping.stale ? " — warning: source modified after last refresh" : "";
							const visualNote =
								mapping.visualMirrorPath !== undefined
									? ` · visual extraction mirror available: ${mapping.visualMirrorPath}`
									: "";
							const header = `[${evidenceId !== undefined ? `${evidenceId} | ` : ""}Parsed mirror of ${mapping.sourcePath}${locLabel !== undefined ? ` (${locLabel})` : ""}${staleNote}${visualNote}]`;
							const combinedText = `${header}\n${textBlock.text}`;
							return {
								...mirrorResult,
								content: [{ type: "text", text: combinedText }],
								structuredContent: combinedText,
							};
						}
						return mirrorResult;
					}
				}
				return baseRead.execute(toolCallId, params, signal, onUpdate, ctx);
			},
		};

		pi.registerTool(mirrorGrepTool);
		pi.registerTool(mirrorReadTool);
	};
}

/**
 * Helper for tests and diagnostics: returns the basename label of a mirror file's original path.
 */
export function displaySourceName(sourcePath: string): string {
	return basename(sourcePath);
}
