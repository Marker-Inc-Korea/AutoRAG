import { existsSync, statSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

/**
 * What a tool call does to the user's files, as far as the approval gate cares.
 *
 * `create` is the safe default (the agent writes new files and copies); every
 * other kind changes or removes something that may be the user's original.
 */
export type PermissionOperationKind = "create" | "delete" | "move" | "rename" | "overwrite" | "edit" | "script";

export interface PermissionOperation {
	readonly kind: PermissionOperationKind;
	/** Absolute paths the operation touches; empty when none could be resolved statically. */
	readonly paths: readonly string[];
	/** One line for the approval prompt. */
	readonly detail: string;
	/** True when part of the target is only known at run time (variables, globs, substitutions, pipes). */
	readonly dynamic?: true;
	/** Absolute destination of a move or rename, when known. */
	readonly dest?: string;
}

/** The slice of the filesystem a classification reads. */
export interface ClassifyFs {
	readonly cwd: string;
	/** Home directory `~` expands to; `~` stays literal (and dynamic) when absent. */
	readonly home?: string;
	readonly exists: (path: string) => boolean;
	readonly isDirectory: (path: string) => boolean;
}

/** {@link ClassifyFs} over the real filesystem. */
export function nodeClassifyFs(cwd: string, home?: string): ClassifyFs {
	return {
		cwd,
		...(home === undefined ? {} : { home }),
		exists: (path) => existsSync(path),
		isDirectory: (path) => {
			try {
				return statSync(path).isDirectory();
			} catch {
				return false;
			}
		},
	};
}

// ---------------------------------------------------------------------------
// Shell tokenizer
// ---------------------------------------------------------------------------

interface Word {
	readonly value: string;
	/** Expands at run time: `$VAR`, `$(...)`, a glob, a brace list. */
	readonly dynamic: boolean;
	/** Contained any quoting or escaping (decides whether a heredoc body expands). */
	readonly quoted: boolean;
}

interface Redirect {
	readonly op: string;
	readonly target: Word;
}

interface Segment {
	words: Word[];
	redirects: Redirect[];
	stdin?: string;
}

interface Parsed {
	readonly segments: Segment[];
	/** Source of every `$(...)`, backtick, and heredoc-embedded substitution. */
	readonly substitutions: string[];
}

const OUTPUT_REDIRECTS = new Set([">", ">>", ">|", "&>", "&>>", ">&"]);
const WORD_BREAK = new Set([";", "|", "&", "(", ")", "<", ">", "\n", " ", "\t", "\r"]);

/** Index of the `)` closing the `(` at `open`, or the text length when unbalanced. */
function matchParen(text: string, open: number): number {
	let depth = 0;
	for (let i = open; i < text.length; i++) {
		const c = text[i];
		if (c === "\\") {
			i++;
		} else if (c === "'") {
			const end = text.indexOf("'", i + 1);
			i = end === -1 ? text.length : end;
		} else if (c === '"') {
			for (i++; i < text.length && text[i] !== '"'; i++) if (text[i] === "\\") i++;
		} else if (c === "(") {
			depth++;
		} else if (c === ")") {
			depth--;
			if (depth === 0) return i;
		}
	}
	return text.length;
}

/** Index of the backtick closing the one at `open`, or the text length. */
function matchBacktick(text: string, open: number): number {
	for (let i = open + 1; i < text.length; i++) {
		if (text[i] === "\\") i++;
		else if (text[i] === "`") return i;
	}
	return text.length;
}

/** Every `$(...)` and backtick substitution in text a shell would expand (an unquoted heredoc body). */
function extractSubstitutions(text: string): string[] {
	const found: string[] = [];
	for (let i = 0; i < text.length; i++) {
		const c = text[i];
		if (c === "\\") {
			i++;
		} else if (c === "$" && text[i + 1] === "(") {
			const close = matchParen(text, i + 1);
			found.push(text.slice(i + 2, close));
			i = close;
		} else if (c === "`") {
			const close = matchBacktick(text, i);
			found.push(text.slice(i + 1, close));
			i = close;
		}
	}
	return found;
}

function parseShell(text: string, home: string | undefined): Parsed {
	const segments: Segment[] = [];
	const substitutions: string[] = [];
	const n = text.length;
	let seg: Segment = { words: [], redirects: [] };
	let pendingOp: string | undefined;
	let pendingHeredoc: { strip: boolean } | undefined;
	const heredocs: { delimiter: string; quoted: boolean; strip: boolean; seg: Segment }[] = [];
	let i = 0;

	const flush = (): void => {
		if (seg.words.length > 0 || seg.redirects.length > 0) segments.push(seg);
		seg = { words: [], redirects: [] };
		pendingOp = undefined;
	};

	const readWord = (): Word => {
		let value = "";
		let dynamic = false;
		let quoted = false;
		const start = i;
		while (i < n) {
			const c = text[i] as string;
			if (WORD_BREAK.has(c)) break;
			if (c === "\\") {
				quoted = true;
				if (text[i + 1] === "\n") {
					i += 2;
					continue;
				}
				value += text[i + 1] ?? "";
				i += 2;
				continue;
			}
			if (c === "'") {
				quoted = true;
				const end = text.indexOf("'", i + 1);
				const stop = end === -1 ? n : end;
				value += text.slice(i + 1, stop);
				i = stop + 1;
				continue;
			}
			if (c === '"') {
				quoted = true;
				i++;
				while (i < n && text[i] !== '"') {
					const d = text[i] as string;
					if (d === "\\") {
						const next = text[i + 1] ?? "";
						value += '$`"\\'.includes(next) ? next : `\\${next}`;
						i += 2;
					} else if (d === "$" && text[i + 1] === "(") {
						const close = matchParen(text, i + 1);
						substitutions.push(text.slice(i + 2, close));
						dynamic = true;
						value += text.slice(i, close + 1);
						i = close + 1;
					} else if (d === "`") {
						const close = matchBacktick(text, i);
						substitutions.push(text.slice(i + 1, close));
						dynamic = true;
						value += text.slice(i, close + 1);
						i = close + 1;
					} else {
						if (d === "$" && /[A-Za-z_{0-9@*#?!$-]/.test(text[i + 1] ?? "")) dynamic = true;
						value += d;
						i++;
					}
				}
				i++;
				continue;
			}
			if (c === "$") {
				if (text[i + 1] === "(") {
					const close = matchParen(text, i + 1);
					substitutions.push(text.slice(i + 2, close));
					dynamic = true;
					value += text.slice(i, close + 1);
					i = close + 1;
					continue;
				}
				if (/[A-Za-z_{0-9@*#?!$-]/.test(text[i + 1] ?? "")) dynamic = true;
				value += c;
				i++;
				continue;
			}
			if (c === "`") {
				const close = matchBacktick(text, i);
				substitutions.push(text.slice(i + 1, close));
				dynamic = true;
				value += text.slice(i, close + 1);
				i = close + 1;
				continue;
			}
			if (c === "*" || c === "?" || c === "[") dynamic = true;
			if (c === "{" && /^\{[^{}\s]*(?:,|\.\.)[^{}\s]*\}/.test(text.slice(i))) dynamic = true;
			if (c === "~" && i === start && value === "") {
				const next = text[i + 1];
				if (next === undefined || next === "/" || WORD_BREAK.has(next)) {
					if (home === undefined) dynamic = true;
					else {
						value += home;
						i++;
						continue;
					}
				} else {
					dynamic = true;
				}
			}
			value += c;
			i++;
		}
		return { value, dynamic, quoted };
	};

	const consumeHeredocBodies = (): void => {
		for (const heredoc of heredocs.splice(0)) {
			const lines: string[] = [];
			while (i < n) {
				const eol = text.indexOf("\n", i);
				const line = text.slice(i, eol === -1 ? n : eol);
				i = eol === -1 ? n : eol + 1;
				if ((heredoc.strip ? line.replace(/^\t+/, "") : line) === heredoc.delimiter) break;
				lines.push(line);
			}
			const body = lines.join("\n");
			heredoc.seg.stdin = body;
			if (!heredoc.quoted) substitutions.push(...extractSubstitutions(body));
		}
	};

	while (i < n) {
		const c = text[i] as string;
		if (c === "\n") {
			flush();
			i++;
			consumeHeredocBodies();
			continue;
		}
		if (c === " " || c === "\t" || c === "\r") {
			i++;
			continue;
		}
		if (c === "#") {
			const eol = text.indexOf("\n", i);
			i = eol === -1 ? n : eol;
			continue;
		}
		if (c === ";" || c === "(" || c === ")") {
			flush();
			i++;
			continue;
		}
		if (c === "|") {
			flush();
			i += text[i + 1] === "|" || text[i + 1] === "&" ? 2 : 1;
			continue;
		}
		if (c === "&" && text[i + 1] !== ">") {
			flush();
			i += text[i + 1] === "&" ? 2 : 1;
			continue;
		}
		if (c === ">" || c === "<" || c === "&") {
			let op: string;
			if (c === "&") {
				op = text.startsWith("&>>", i) ? "&>>" : "&>";
			} else if (c === ">") {
				op = text.startsWith(">>", i)
					? ">>"
					: text.startsWith(">|", i)
						? ">|"
						: text.startsWith(">&", i)
							? ">&"
							: ">";
			} else if (text.startsWith("<<<", i)) {
				op = "<<<";
			} else if (text.startsWith("<<-", i)) {
				op = "<<-";
			} else if (text.startsWith("<<", i)) {
				op = "<<";
			} else if (text.startsWith("<&", i)) {
				op = "<&";
			} else if (text.startsWith("<>", i)) {
				op = "<>";
			} else {
				op = "<";
			}
			i += op.length;
			if (op === "<<" || op === "<<-") pendingHeredoc = { strip: op === "<<-" };
			else pendingOp = op;
			continue;
		}
		const word = readWord();
		if (!word.quoted && /^\d+$/.test(word.value) && (text[i] === ">" || text[i] === "<")) continue; // fd prefix
		if (pendingHeredoc !== undefined) {
			heredocs.push({ delimiter: word.value, quoted: word.quoted, strip: pendingHeredoc.strip, seg });
			pendingHeredoc = undefined;
		} else if (pendingOp !== undefined) {
			const op = pendingOp;
			pendingOp = undefined;
			const fdDup = (op === ">&" || op === "<&") && /^(?:\d+|-)$/.test(word.value);
			if (OUTPUT_REDIRECTS.has(op) && !fdDup) seg.redirects.push({ op, target: word });
		} else {
			seg.words.push(word);
		}
	}
	flush();
	return { segments, substitutions };
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

interface Ctx {
	readonly fs: ClassifyFs;
	cwd: string;
	cwdKnown: boolean;
	readonly ops: PermissionOperation[];
	readonly depth: number;
	/** Paths earlier operations of this command create. */
	readonly created: Set<string>;
}

const MAX_DEPTH = 8;
const KEYWORDS = new Set(["{", "}", "!", "then", "do", "else", "elif", "if", "while", "until", "time"]);
const ENV_ASSIGNMENT = /^[A-Za-z_][A-Za-z0-9_]*=/;

const WRAPPERS: Readonly<Record<string, { readonly valueFlags?: readonly string[]; readonly positional?: number }>> = {
	sudo: { valueFlags: ["-u", "-g", "-h", "-p", "-C", "-T", "-r", "-t", "-U", "-D"] },
	doas: { valueFlags: ["-u", "-C"] },
	env: { valueFlags: ["-u", "-C", "-S"] },
	nohup: {},
	command: {},
	exec: { valueFlags: ["-a"] },
	builtin: {},
	nice: { valueFlags: ["-n"] },
	ionice: { valueFlags: ["-c", "-n", "-p"] },
	setsid: {},
	caffeinate: { valueFlags: ["-t", "-w"] },
	stdbuf: {},
	unbuffer: {},
	timeout: { valueFlags: ["-s", "-k"], positional: 1 },
};

const DELETE_COMMANDS: Readonly<Record<string, readonly string[]>> = {
	rm: [],
	rmdir: [],
	unlink: [],
	trash: [],
	"trash-put": [],
	shred: ["-n", "-s", "--iterations", "--size"],
	srm: [],
};

const SHELLS = new Set(["bash", "sh", "zsh", "dash", "ksh"]);
const INTERPRETERS = new Set([
	"python",
	"python3",
	"node",
	"nodejs",
	"bun",
	"deno",
	"ruby",
	"perl",
	"php",
	"lua",
	"osascript",
]);
const CODE_FLAGS = new Set(["-c", "-e", "-E", "-p", "-pe", "--eval", "--print"]);

/** Interpreter source text that removes, moves, or renames files. */
const SCRIPT_PATTERN =
	/\b(?:os\.(?:remove|unlink|rmdir|removedirs|rename|renames|replace)|shutil\.(?:rmtree|move)|send2trash|rmtree|rmSync|rmdirSync|unlinkSync|renameSync|fs(?:\.promises)?\.(?:rm|rmdir|unlink|rename)|fsp\.(?:rm|unlink|rename)|FileUtils\.(?:rm|rm_rf|rm_r|rmtree|mv)|File\.(?:delete|rename|unlink)|Remove-Item|Deno\.(?:remove|rename))\b|\.(?:unlink|rmdir)\s*\(|Path\([^)]*\)\.(?:rename|replace)\s*\(|(?<![.\w])(?:rename|unlink|rmdir)\s*\(|move to trash|tell application "Finder"[\s\S]*\bdelete\b/i;

interface ParsedArgs {
	readonly flags: string[];
	readonly positionals: Word[];
	readonly values: Map<string, Word>;
}

function parseArgs(args: readonly Word[], valueFlags: readonly string[] = []): ParsedArgs {
	const flags: string[] = [];
	const positionals: Word[] = [];
	const values = new Map<string, Word>();
	let literal = false;
	for (let k = 0; k < args.length; k++) {
		const word = args[k] as Word;
		if (literal || word.dynamic || !word.value.startsWith("-") || word.value === "-") {
			positionals.push(word);
			continue;
		}
		if (word.value === "--") {
			literal = true;
			continue;
		}
		flags.push(word.value);
		const eq = word.value.indexOf("=");
		if (word.value.startsWith("--") && eq > 0) {
			values.set(word.value.slice(0, eq), { value: word.value.slice(eq + 1), dynamic: false, quoted: false });
		} else if (valueFlags.includes(word.value) && k + 1 < args.length) {
			values.set(word.value, args[++k] as Word);
		}
	}
	return { flags, positionals, values };
}

function hasFlag(flags: readonly string[], short: string | undefined, long?: string): boolean {
	return flags.some(
		(flag) =>
			(long !== undefined && (flag === `--${long}` || flag.startsWith(`--${long}=`))) ||
			(short !== undefined && /^-[^-]/.test(flag) && flag.slice(1).includes(short)),
	);
}

function resolvePath(word: Word, ctx: Ctx): string | undefined {
	if (word.dynamic || word.value === "") return undefined;
	if (!isAbsolute(word.value) && !ctx.cwdKnown) return undefined;
	return resolve(ctx.cwd, word.value);
}

function describe(name: string, args: readonly Word[]): string {
	const text = [name, ...args.map((arg) => arg.value)].join(" ");
	return text.length > 200 ? `${text.slice(0, 197)}...` : text;
}

function push(ctx: Ctx, op: PermissionOperation): void {
	ctx.ops.push(op);
	if (op.kind === "create") for (const path of op.paths) ctx.created.add(path);
}

/** One operation per target word; a word known only at run time becomes one dynamic operation. */
function targetOps(kind: PermissionOperationKind, targets: readonly Word[], detail: string, ctx: Ctx): void {
	for (const target of targets) {
		const path = resolvePath(target, ctx);
		push(ctx, path === undefined ? { kind, paths: [], detail, dynamic: true } : { kind, paths: [path], detail });
	}
}

/** `create` for a path that does not exist yet, `overwrite` (or `edit`) for one that does. */
function writeOp(path: string, ctx: Ctx, existing: PermissionOperationKind = "overwrite"): void {
	const exists = ctx.fs.exists(path);
	push(ctx, {
		kind: exists ? existing : "create",
		paths: [path],
		detail: `${exists ? existing : "create"} ${path}`,
	});
}

function classifyMove(name: string, args: readonly Word[], ctx: Ctx): void {
	const { flags, positionals, values } = parseArgs(args, ["-t", "--target-directory", "-S", "--suffix"]);
	const detail = describe(name, args);
	const targetDir = values.get("-t") ?? values.get("--target-directory");
	let sources: Word[];
	let dest: Word | undefined;
	if (targetDir !== undefined) {
		sources = positionals;
		dest = targetDir;
	} else {
		if (positionals.length < 2) return;
		sources = positionals.slice(0, -1);
		dest = positionals.at(-1);
	}
	const destPath = dest === undefined ? undefined : resolvePath(dest, ctx);
	const intoDir =
		targetDir !== undefined ||
		sources.length > 1 ||
		(dest?.value.endsWith("/") ?? false) ||
		(destPath !== undefined && ctx.fs.isDirectory(destPath));
	void flags;
	for (const source of sources) {
		const from = resolvePath(source, ctx);
		if (from === undefined) {
			push(ctx, { kind: "move", paths: [], detail, dynamic: true });
			continue;
		}
		if (destPath === undefined) {
			push(ctx, { kind: "move", paths: [from], detail, dynamic: true });
			continue;
		}
		const to = intoDir ? join(destPath, basename(from)) : destPath;
		const kind = !intoDir && dirname(from) === dirname(destPath) ? "rename" : "move";
		push(ctx, { kind, paths: [from], detail, dest: to });
	}
}

function classifyCopy(args: readonly Word[], ctx: Ctx): void {
	const { flags, positionals, values } = parseArgs(args, ["-t", "--target-directory", "-S", "--suffix"]);
	const targetDir = values.get("-t") ?? values.get("--target-directory");
	const noClobber = hasFlag(flags, "n", "no-clobber");
	let sources: Word[];
	let dest: Word | undefined;
	if (targetDir !== undefined) {
		sources = positionals;
		dest = targetDir;
	} else {
		if (positionals.length < 2) return;
		sources = positionals.slice(0, -1);
		dest = positionals.at(-1);
	}
	const destPath = dest === undefined ? undefined : resolvePath(dest, ctx);
	if (destPath === undefined) return;
	const intoDir =
		targetDir !== undefined ||
		sources.length > 1 ||
		(dest?.value.endsWith("/") ?? false) ||
		ctx.fs.isDirectory(destPath);
	for (const source of sources) {
		const from = resolvePath(source, ctx);
		if (from === undefined) continue;
		const to = intoDir ? join(destPath, basename(from)) : destPath;
		if (noClobber && ctx.fs.exists(to)) continue;
		writeOp(to, ctx);
	}
}

function classifySed(args: readonly Word[], ctx: Ctx, detail: string): void {
	let inPlace = false;
	let suffix = "";
	let hasScript = false;
	const positionals: Word[] = [];
	for (let k = 0; k < args.length; k++) {
		const word = args[k] as Word;
		const v = word.value;
		if (word.dynamic || !v.startsWith("-") || v === "-") {
			positionals.push(word);
		} else if (v === "-e" || v === "-f") {
			hasScript = true;
			k++;
		} else if (v.startsWith("--expression") || v.startsWith("--file")) {
			hasScript = true;
			if (!v.includes("=")) k++;
		} else if (v === "--in-place" || v.startsWith("--in-place=")) {
			inPlace = true;
			suffix = v.startsWith("--in-place=") ? v.slice("--in-place=".length) : "";
		} else if (!v.startsWith("--") && /^-[A-Za-z]*e$/.test(v)) {
			hasScript = true;
			if (/i/.test(v)) inPlace = true;
			k++;
		} else if (!v.startsWith("--") && /^-[A-Za-z]*i/.test(v)) {
			inPlace = true;
			// `-i.bak` carries its backup suffix; a bare `-i` takes none (GNU) or an empty next word (BSD).
			suffix = v.slice(v.indexOf("i") + 1);
			if (v === "-i" && args[k + 1]?.value === "" && args[k + 1]?.quoted === true) k++;
		}
	}
	if (!inPlace) return;
	if (!hasScript) positionals.shift();
	editTargets(positionals, ctx, detail, suffix);
}

function classifyPerl(args: readonly Word[], ctx: Ctx, detail: string): void {
	let inPlace = false;
	const positionals: Word[] = [];
	let sawCode = false;
	let perlSuffix = "";
	for (let k = 0; k < args.length; k++) {
		const word = args[k] as Word;
		const v = word.value;
		if (word.dynamic || !v.startsWith("-") || v === "-") {
			positionals.push(word);
		} else if (!v.startsWith("--")) {
			if (/^-[A-Za-z]*i/.test(v)) {
				inPlace = true;
				perlSuffix = v
					.slice(v.indexOf("i") + 1)
					.replace(/[A-Za-z]+$/, (rest) => (/^[epwlnaFC]+$/.test(rest) ? "" : rest));
			}
			if (/^-[A-Za-z]*[eE]$/.test(v)) {
				sawCode = true;
				k++;
			}
		}
	}
	if (!inPlace) return;
	if (!sawCode) positionals.shift();
	editTargets(positionals, ctx, detail, perlSuffix);
}

/**
 * In-place editors: each existing target is an `edit`; one known only at run time is a dynamic `edit`.
 * A backup suffix (`-i.bak`) also writes a new file next to the target, reported as a creation so the
 * backup is as free to remove as any other file the agent made.
 */
function editTargets(targets: readonly Word[], ctx: Ctx, detail: string, suffix = ""): void {
	for (const target of targets) {
		const path = resolvePath(target, ctx);
		if (path === undefined) {
			push(ctx, { kind: "edit", paths: [], detail, dynamic: true });
		} else if (ctx.fs.exists(path)) {
			if (suffix !== "" && !suffix.includes("*")) {
				const backup = `${path}${suffix}`;
				if (!ctx.fs.exists(backup)) push(ctx, { kind: "create", paths: [backup], detail: `create ${backup}` });
			}
			push(ctx, { kind: "edit", paths: [path], detail });
		}
	}
}

function classifyFind(args: readonly Word[], ctx: Ctx): void {
	const starts: Word[] = [];
	let k = 0;
	for (; k < args.length; k++) {
		const word = args[k] as Word;
		if (!word.dynamic && (word.value.startsWith("-") || word.value === "!" || word.value === "(")) break;
		starts.push(word);
	}
	if (starts.length === 0) starts.push({ value: ".", dynamic: false, quoted: false });
	for (; k < args.length; k++) {
		const word = args[k] as Word;
		if (word.dynamic) continue;
		if (word.value === "-delete") {
			targetOps("delete", starts, describe("find", args), ctx);
		} else if (["-exec", "-execdir", "-ok", "-okdir"].includes(word.value)) {
			const command: Word[] = [];
			for (k++; k < args.length; k++) {
				const part = args[k] as Word;
				if (!part.dynamic && (part.value === ";" || part.value === "+")) break;
				command.push(part);
			}
			for (const start of starts) {
				classifyWords(
					command.map((part) =>
						part.value.includes("{}")
							? {
									value: part.value.replaceAll("{}", start.value),
									dynamic: part.dynamic || start.dynamic,
									quoted: part.quoted,
								}
							: part,
					),
					ctx,
				);
			}
		}
	}
}

function classifyGit(args: readonly Word[], ctx: Ctx): void {
	let k = 0;
	while (k < args.length) {
		const word = args[k] as Word;
		if (word.dynamic || !word.value.startsWith("-")) break;
		k += word.value === "-C" || word.value === "-c" ? 2 : 1;
	}
	const sub = args[k]?.value;
	const rest = args.slice(k + 1);
	const detail = describe("git", args);
	if (sub === "rm") {
		const { flags, positionals } = parseArgs(rest);
		if (!hasFlag(flags, undefined, "cached")) targetOps("delete", positionals, detail, ctx);
	} else if (sub === "mv") {
		classifyMove("git mv", rest, ctx);
	} else if (sub === "clean") {
		const { flags, positionals } = parseArgs(rest);
		if (hasFlag(flags, "n", "dry-run")) return;
		if (positionals.length > 0) targetOps("delete", positionals, detail, ctx);
		else push(ctx, { kind: "delete", paths: [ctx.cwd], detail });
	}
}

function classifyRsync(args: readonly Word[], ctx: Ctx): void {
	const { flags, positionals } = parseArgs(args, [
		"-e",
		"--rsh",
		"--exclude",
		"--include",
		"--filter",
		"-f",
		"--files-from",
	]);
	const detail = describe("rsync", args);
	const local = (words: readonly Word[]): Word[] => words.filter((word) => word.dynamic || !word.value.includes(":"));
	if (hasFlag(flags, undefined, "remove-source-files"))
		targetOps("delete", local(positionals.slice(0, -1)), detail, ctx);
	if (flags.some((flag) => /^--(?:del$|delete)/.test(flag))) {
		const dest = positionals.at(-1);
		if (dest !== undefined) targetOps("delete", local([dest]), detail, ctx);
	}
}

function classifyInterpreter(name: string, args: readonly Word[], seg: Segment, ctx: Ctx): void {
	const detail = describe(name, args);
	const sources: string[] = [];
	for (let k = 0; k < args.length; k++) {
		const word = args[k] as Word;
		if (!word.dynamic && (CODE_FLAGS.has(word.value) || (name === "perl" && /^-[A-Za-z]*[eE]$/.test(word.value)))) {
			const code = args[k + 1];
			if (code !== undefined) sources.push(code.value);
			k++;
		}
	}
	if (seg.stdin !== undefined) sources.push(seg.stdin);
	if (name === "perl") classifyPerl(args, ctx, detail);
	if (sources.some((source) => SCRIPT_PATTERN.test(source))) {
		push(ctx, { kind: "script", paths: [], detail, dynamic: true });
	}
}

function skipWrapper(name: string, words: readonly Word[], from: number): number | undefined {
	const spec = WRAPPERS[name] as NonNullable<(typeof WRAPPERS)[string]>;
	let k = from;
	if (name === "command" && words.slice(k).some((w) => w.value === "-v" || w.value === "-V")) return undefined;
	while (k < words.length) {
		const word = words[k] as Word;
		if (word.dynamic) break;
		if (name === "env" && ENV_ASSIGNMENT.test(word.value)) {
			k++;
			continue;
		}
		if (!word.value.startsWith("-") || word.value === "-") break;
		if (word.value === "--") {
			k++;
			break;
		}
		k += spec.valueFlags?.includes(word.value) ? 2 : 1;
	}
	return k + (spec.positional ?? 0);
}

/** Classify one command given as words (already tokenized). */
function classifyWords(words: readonly Word[], ctx: Ctx, seg: Segment = { words: [], redirects: [] }): void {
	let idx = 0;
	let name: string | undefined;
	for (;;) {
		const word = words[idx];
		if (word === undefined) return;
		if (!word.dynamic && (ENV_ASSIGNMENT.test(word.value) || KEYWORDS.has(word.value))) {
			idx++;
			continue;
		}
		if (word.dynamic) return; // command name only known at run time
		name = basename(word.value);
		if (Object.hasOwn(WRAPPERS, name)) {
			const next = skipWrapper(name, words, idx + 1);
			if (next === undefined) return;
			idx = next;
			continue;
		}
		break;
	}
	const args = words.slice(idx + 1);
	const detail = describe(name, args);
	if (Object.hasOwn(DELETE_COMMANDS, name)) {
		const { positionals } = parseArgs(args, DELETE_COMMANDS[name]);
		targetOps("delete", positionals, detail, ctx);
		return;
	}
	switch (name) {
		case "mv":
			classifyMove("mv", args, ctx);
			return;
		case "cp":
			classifyCopy(args, ctx);
			return;
		case "mkdir": {
			for (const target of parseArgs(args, ["-m", "--mode"]).positionals) {
				const path = resolvePath(target, ctx);
				if (path !== undefined && !ctx.fs.exists(path))
					push(ctx, { kind: "create", paths: [path], detail: `create ${path}` });
			}
			return;
		}
		case "touch": {
			for (const target of parseArgs(args, ["-t", "-d", "-r", "--date", "--reference"]).positionals) {
				const path = resolvePath(target, ctx);
				if (path !== undefined && !ctx.fs.exists(path))
					push(ctx, { kind: "create", paths: [path], detail: `create ${path}` });
			}
			return;
		}
		case "tee": {
			for (const target of parseArgs(args).positionals) {
				const path = resolvePath(target, ctx);
				if (path !== undefined) writeOp(path, ctx);
			}
			return;
		}
		case "dd": {
			for (const arg of args) {
				if (arg.value.startsWith("of=") && !arg.dynamic) writeOp(resolve(ctx.cwd, arg.value.slice(3)), ctx);
			}
			return;
		}
		case "truncate": {
			for (const target of parseArgs(args, ["-s", "--size", "-r", "--reference"]).positionals) {
				const path = resolvePath(target, ctx);
				if (path === undefined) push(ctx, { kind: "edit", paths: [], detail, dynamic: true });
				else writeOp(path, ctx, "edit");
			}
			return;
		}
		case "sed":
			classifySed(args, ctx, detail);
			return;
		case "find":
			classifyFind(args, ctx);
			return;
		case "git":
			classifyGit(args, ctx);
			return;
		case "gio": {
			// `gio trash` is the Linux trash command AutoRAG's own prompt tells the model to use for deleting.
			const { positionals } = parseArgs(args);
			if (positionals[0]?.value === "trash") targetOps("delete", positionals.slice(1), detail, ctx);
			return;
		}
		case "rsync":
			classifyRsync(args, ctx);
			return;
		case "cd":
		case "pushd": {
			const target = args.find((arg) => arg.dynamic || !arg.value.startsWith("-"));
			if (target === undefined) return;
			const path = resolvePath(target, ctx);
			if (path === undefined) ctx.cwdKnown = false;
			else ctx.cwd = path;
			return;
		}
		case "popd":
			ctx.cwdKnown = false;
			return;
		case "xargs": {
			let k = 0;
			while (k < args.length) {
				const word = args[k] as Word;
				if (word.dynamic || !word.value.startsWith("-")) break;
				k += ["-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a"].includes(word.value) ? 2 : 1;
			}
			classifyWords([...args.slice(k), { value: "?", dynamic: true, quoted: false }], ctx);
			return;
		}
		case "eval":
			classifyNested(args.map((arg) => arg.value).join(" "), ctx);
			return;
		default:
			break;
	}
	if (SHELLS.has(name)) {
		const flagIndex = args.findIndex((arg) => !arg.dynamic && /^-[A-Za-z]*c$/.test(arg.value));
		const code = flagIndex === -1 ? undefined : args[flagIndex + 1];
		if (code !== undefined) classifyNested(code.value, ctx);
		else if (seg.stdin !== undefined) classifyNested(seg.stdin, ctx);
		return;
	}
	if (INTERPRETERS.has(name.replace(/\.\d+$/, ""))) classifyInterpreter(name.replace(/\.\d+$/, ""), args, seg, ctx);
}

function classifyNested(text: string, ctx: Ctx): void {
	if (ctx.depth >= MAX_DEPTH) return;
	classifyText(text, { ...ctx, depth: ctx.depth + 1 });
}

function classifyRedirect(redirect: Redirect, ctx: Ctx): void {
	const path = resolvePath(redirect.target, ctx);
	if (path === undefined || path.startsWith("/dev/")) return;
	writeOp(path, ctx);
}

function classifyText(text: string, ctx: Ctx): void {
	const parsed = parseShell(text, ctx.fs.home);
	for (const substitution of parsed.substitutions) classifyNested(substitution, ctx);
	for (const seg of parsed.segments) {
		classifyWords(seg.words, ctx, seg);
		for (const redirect of seg.redirects) classifyRedirect(redirect, ctx);
	}
}

function newCtx(fs: ClassifyFs): Ctx {
	// What an earlier part of the same command creates exists for the later parts, though the command has not run yet.
	const created = new Set<string>();
	const overlay: ClassifyFs = {
		cwd: fs.cwd,
		...(fs.home === undefined ? {} : { home: fs.home }),
		exists: (path) => created.has(path) || fs.exists(path),
		isDirectory: (path) => fs.isDirectory(path),
	};
	return { fs: overlay, cwd: fs.cwd, cwdKnown: true, ops: [], depth: 0, created };
}

/** Operations a `bash` command line performs on files, in order of appearance. */
export function classifyBash(command: string, fs: ClassifyFs): PermissionOperation[] {
	const ctx = newCtx(fs);
	classifyText(command, ctx);
	return ctx.ops;
}

/** Operations of an automation rule's command argv (`appendFiles` adds run-time file arguments). */
export function classifyAutomationRun(
	run: readonly string[],
	fs: ClassifyFs,
	options: { readonly appendFiles?: boolean } = {},
): PermissionOperation[] {
	const ctx = newCtx(fs);
	const words: Word[] = run.map((value) => ({ value, dynamic: false, quoted: true }));
	if (options.appendFiles === true) words.push({ value: "?", dynamic: true, quoted: false });
	classifyWords(words, ctx);
	return ctx.ops;
}

/** Operations of pi's `write` and `edit` tools; other tools have none. */
export function classifyFileTool(
	tool: "write" | "edit",
	input: Readonly<Record<string, unknown>>,
	fs: ClassifyFs,
): PermissionOperation[] {
	const raw = input.path;
	if (typeof raw !== "string" || raw === "") return [];
	const expanded = fs.home !== undefined && (raw === "~" || raw.startsWith("~/")) ? join(fs.home, raw.slice(1)) : raw;
	const path = resolve(fs.cwd, expanded);
	const exists = fs.exists(path);
	if (tool === "write") {
		const kind = exists ? "overwrite" : "create";
		return [{ kind, paths: [path], detail: `${kind} ${path}` }];
	}
	return exists ? [{ kind: "edit", paths: [path], detail: `edit ${path}` }] : [];
}
