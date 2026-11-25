import { spawn, spawnSync } from "node:child_process";
import { constants, mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { access as fsAccess } from "node:fs/promises";
import { constants as osConstants, tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import type { BashOperations } from "@earendil-works/pi-coding-agent";

/**
 * OS-level read-only filesystem enforcement for `bash` commands in `readonly` permission mode.
 *
 * The goal is a real kernel boundary, not a command-name denylist: any program a shell spawns
 * (shell redirects, `rm`, `mv`, `sed -i`, `python3`, `perl`, `node`, `/usr/bin/trash`, compiled binaries)
 * is prevented from modifying or deleting files on disk, except inside a dedicated scratch temp directory
 * (plus standard virtual devices such as `/dev/null`).
 *
 * Platform backends:
 * - macOS (`darwin`): `/usr/bin/sandbox-exec` (Seatbelt) with `(deny file-write*)` and explicit
 *   `(allow file-write* (subpath <scratchDir>))` + `/dev/null`, `/dev/zero`, `/dev/tty`, `/dev/fd/*`, `/dev/ttys*`.
 * - Linux (`linux`): `bwrap` (bubblewrap) with `--ro-bind / /`, `--dev /dev`, `--proc /proc`, and
 *   `--bind <scratchDir> <scratchDir>` (recursively read-only root with writable scratch dir).
 * - Windows (`win32`) or hosts without an available OS read-only sandbox: fail closed — refuse
 *   unsandboxed `bash` execution in `readonly` mode with an explicit diagnostic explaining why and
 *   how to switch to `full` mode.
 */

export type ReadOnlySandboxBackend = "seatbelt" | "bwrap" | "unavailable";

export interface ReadOnlySandboxInfo {
	readonly backend: ReadOnlySandboxBackend;
	readonly scratchDir: string;
	readonly unavailableReason?: string;
}

export interface ReadOnlySandbox {
	readonly backend: ReadOnlySandboxBackend;
	/** Dedicated temp directory the sandbox allows writes into (and sets `TMPDIR` to). */
	readonly scratchDir: string;
	/** Present when {@link backend} is `"unavailable"`. */
	readonly unavailableReason?: string;
	/** Execute a shell command under the read-only filesystem boundary. */
	readonly operations: BashOperations;
	/** Remove the dedicated scratch directory. Safe to call multiple times. */
	dispose(): void;
}

export interface ReadOnlySandboxOptions {
	/** Override platform for tests. Default: `process.platform`. */
	readonly platform?: NodeJS.Platform;
	/** Override scratch directory (must exist or be creatable). Default: a fresh dir under `os.tmpdir()`. */
	readonly scratchDir?: string;
	/** Override shell binary path (passed through to the inner shell). */
	readonly shellPath?: string;
	/** Override binary lookup for tests (e.g. simulating missing `sandbox-exec` or `bwrap`). */
	readonly probeBinary?: (name: string) => string | undefined;
	/**
	 * Extra writable file/dir paths (for example trusted config files). Kept empty by default so
	 * `readonly` truly blocks every disk write outside the scratch dir.
	 */
	readonly extraWritablePaths?: readonly string[];
}

const SEATBELT_BIN = "/usr/bin/sandbox-exec";

/** Escape a path for a Seatbelt SBPL string literal (`"..."`). */
export function escapeSeatbeltString(value: string): string {
	return value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

/**
 * Build the macOS Seatbelt profile that denies all filesystem writes except into `writableDirs`
 * (real paths, plus their symlink aliases when different) and standard character devices.
 */
export function buildSeatbeltProfile(writableDirs: readonly string[], writableFiles: readonly string[] = []): string {
	const rules: string[] = [
		"(version 1)",
		"(allow default)",
		"(deny file-write*)",
		'(allow file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") (literal "/dev/dtracehelper"))',
		'(allow file-write* (regex #"^/dev/fd/") (regex #"^/dev/ttys"))',
	];
	const seenDirs = new Set<string>();
	for (const raw of writableDirs) {
		for (const candidate of canonicalVariants(raw)) {
			if (seenDirs.has(candidate)) continue;
			seenDirs.add(candidate);
			rules.push(`(allow file-write* (subpath "${escapeSeatbeltString(candidate)}"))`);
		}
	}
	const seenFiles = new Set<string>();
	for (const raw of writableFiles) {
		for (const candidate of canonicalVariants(raw)) {
			if (seenFiles.has(candidate)) continue;
			seenFiles.add(candidate);
			rules.push(`(allow file-write* (literal "${escapeSeatbeltString(candidate)}"))`);
		}
	}
	return rules.join("\n");
}

function canonicalVariants(path: string): string[] {
	const resolved = resolve(path);
	const out = [resolved];
	try {
		const real = realpathSync(resolved);
		if (real !== resolved) out.push(real);
	} catch {
		// Path may not exist yet (e.g. a file to be created); also check parent realpath.
		try {
			const parentReal = realpathSync(dirname(resolved));
			const joined = join(parentReal, resolved.slice(dirname(resolved).length + 1));
			if (joined !== resolved) out.push(joined);
		} catch {
			// Ignore if parent does not exist either.
		}
	}
	return out;
}

function defaultProbeBinary(name: string): string | undefined {
	if (isAbsolute(name)) {
		const probe = spawnSync(name, ["--help"], { stdio: "ignore", timeout: 3000 });
		return probe.error === undefined ? name : undefined;
	}
	const locator = process.platform === "win32" ? "where" : "which";
	const res = spawnSync(locator, [name], { encoding: "utf8", timeout: 3000 });
	if (res.status !== 0 || !res.stdout) return undefined;
	const first = res.stdout.trim().split(/\r?\n/)[0]?.trim();
	return first && first.length > 0 ? first : undefined;
}

function probeBwrapUsable(bwrapPath: string): { ok: true } | { ok: false; reason: string } {
	const probe = spawnSync(bwrapPath, ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--", "/bin/true"], {
		encoding: "utf8",
		timeout: 5000,
	});
	if (probe.status === 0 && probe.error === undefined) return { ok: true };
	const detail =
		probe.error?.message ??
		(probe.stderr?.trim() || probe.stdout?.trim() || `exit code ${probe.status ?? "unknown"}`);
	return { ok: false, reason: `bwrap (${bwrapPath}) cannot create a read-only mount namespace: ${detail}` };
}

function resolveShell(customShellPath?: string): { shell: string; args: readonly string[] } {
	if (customShellPath) return { shell: customShellPath, args: ["-c"] };
	for (const candidate of ["/bin/bash", "/usr/bin/bash", "/bin/sh"]) {
		const probe = spawnSync(candidate, ["-c", ":"], { stdio: "ignore", timeout: 2000 });
		if (probe.status === 0) return { shell: candidate, args: ["-c"] };
	}
	return { shell: "sh", args: ["-c"] };
}

function resolveSandboxInvocation(
	platform: NodeJS.Platform,
	scratchDir: string,
	extraWritablePaths: readonly string[],
	probeBinary: (name: string) => string | undefined,
): {
	readonly backend: ReadOnlySandboxBackend;
	readonly unavailableReason?: string;
	readonly wrap?: (
		shell: string,
		shellArgs: readonly string[],
		command: string,
	) => { readonly bin: string; readonly args: readonly string[] };
} {
	if (platform === "darwin") {
		const bin = probeBinary(SEATBELT_BIN);
		if (bin === undefined) {
			return {
				backend: "unavailable",
				unavailableReason: `macOS read-only sandbox binary not found at ${SEATBELT_BIN}`,
			};
		}
		const profile = buildSeatbeltProfile([scratchDir], extraWritablePaths);
		return {
			backend: "seatbelt",
			wrap: (shell, shellArgs, command) => ({
				bin,
				args: ["-p", profile, shell, ...shellArgs, command],
			}),
		};
	}

	if (platform === "linux") {
		const bin = probeBinary("bwrap");
		if (bin === undefined) {
			return {
				backend: "unavailable",
				unavailableReason:
					"Linux read-only sandbox requires bubblewrap (`bwrap`), which was not found on PATH. Install `bubblewrap` or switch to `full` permission mode.",
			};
		}
		const check = probeBwrapUsable(bin);
		if (!check.ok) {
			return {
				backend: "unavailable",
				unavailableReason: `${check.reason}. Enable unprivileged user namespaces or switch to \`full\` permission mode.`,
			};
		}
		const realScratch = canonicalVariants(scratchDir).at(-1) ?? scratchDir;
		return {
			backend: "bwrap",
			wrap: (shell, shellArgs, command) => {
				const args: string[] = [
					"--ro-bind",
					"/",
					"/",
					"--dev",
					"/dev",
					"--proc",
					"/proc",
					"--bind",
					realScratch,
					realScratch,
				];
				if (realScratch !== scratchDir) {
					args.push("--bind", realScratch, scratchDir);
				}
				for (const extra of extraWritablePaths) {
					const resolved = resolve(extra);
					args.push("--bind", resolved, resolved);
				}
				args.push("--", shell, ...shellArgs, command);
				return { bin, args };
			},
		};
	}

	return {
		backend: "unavailable",
		unavailableReason: `OS-level read-only filesystem sandboxing for bash is not available on platform "${platform}". Use read-only tools (read, grep, find, ls, search tools) or switch to \`full\` permission mode.`,
	};
}

function killProcessTree(pid: number): void {
	if (process.platform === "win32") {
		try {
			const child = spawn(join(process.env.SystemRoot || "C:\\Windows", "System32", "taskkill.exe"), [
				"/F",
				"/T",
				"/PID",
				String(pid),
			]);
			child.once("error", () => undefined);
		} catch {
			// Ignore.
		}
		return;
	}
	try {
		process.kill(-pid, "SIGKILL");
	} catch {
		try {
			process.kill(pid, "SIGKILL");
		} catch {
			// Already exited.
		}
	}
}

/**
 * Create a session-scoped OS read-only filesystem sandbox for `bash` execution.
 *
 * A fresh temp directory is allocated (unless `scratchDir` is provided) and exported as `TMPDIR`/`TEMP`/`TMP`
 * inside the sandboxed process so pipelines and tools that need temporary files keep working while every
 * other path on disk rejects writes with `EPERM` / `EROFS`.
 */
export function createReadOnlySandbox(options: ReadOnlySandboxOptions = {}): ReadOnlySandbox {
	const platform = options.platform ?? process.platform;
	const probeBinary = options.probeBinary ?? defaultProbeBinary;
	const ownedScratch = options.scratchDir === undefined;
	const rawScratch =
		options.scratchDir !== undefined ? resolve(options.scratchDir) : mkdtempSync(join(tmpdir(), "autorag-readonly-"));
	if (!ownedScratch) mkdirSync(rawScratch, { recursive: true });
	const scratchDir = realpathSync(rawScratch);
	const extraWritablePaths = options.extraWritablePaths ?? [];

	const resolved = resolveSandboxInvocation(platform, scratchDir, extraWritablePaths, probeBinary);

	const operations: BashOperations = {
		exec: async (command, cwd, { onData, signal, timeout, env }) => {
			if (signal?.aborted) throw new Error("aborted");
			if (resolved.backend === "unavailable" || resolved.wrap === undefined) {
				throw new Error(
					`Read-only mode blocked bash execution: ${resolved.unavailableReason ?? "no OS read-only sandbox available"}`,
				);
			}
			try {
				await fsAccess(cwd, constants.F_OK);
			} catch {
				throw new Error(`Working directory does not exist: ${cwd}\nCannot execute bash commands.`);
			}

			const { shell, args: shellArgs } = resolveShell(options.shellPath);
			const wrapped = resolved.wrap(shell, shellArgs, command);
			const childEnv: NodeJS.ProcessEnv = {
				...(env ?? process.env),
				TMPDIR: scratchDir,
				TEMP: scratchDir,
				TMP: scratchDir,
				AUTORAG_READONLY_SCRATCH: scratchDir,
			};

			const child = spawn(wrapped.bin, [...wrapped.args], {
				cwd,
				detached: process.platform !== "win32",
				env: childEnv,
				stdio: ["ignore", "pipe", "pipe"],
				windowsHide: true,
			});

			let timedOut = false;
			let timeoutHandle: NodeJS.Timeout | undefined;
			const onAbort = () => {
				if (child.pid !== undefined) killProcessTree(child.pid);
			};

			try {
				if (timeout !== undefined && Number.isFinite(timeout) && timeout > 0) {
					timeoutHandle = setTimeout(() => {
						timedOut = true;
						if (child.pid !== undefined) killProcessTree(child.pid);
					}, timeout * 1000);
				}

				child.stdout?.on("data", onData);
				child.stderr?.on("data", onData);

				if (signal !== undefined) {
					if (signal.aborted) onAbort();
					else signal.addEventListener("abort", onAbort, { once: true });
				}

				const exitCode = await new Promise<number | null>((resolveExit, rejectExit) => {
					child.once("error", rejectExit);
					child.once("close", (code) => resolveExit(code));
				});

				if (signal?.aborted) throw new Error("aborted");
				if (timedOut) throw new Error(`timeout:${timeout}`);

				const signalCode = child.signalCode;
				return {
					exitCode:
						exitCode ??
						(signalCode ? 128 + ((osConstants.signals as Record<string, number>)[signalCode] ?? 0) : 1),
				};
			} finally {
				if (timeoutHandle !== undefined) clearTimeout(timeoutHandle);
				if (signal !== undefined) signal.removeEventListener("abort", onAbort);
			}
		},
	};

	let disposed = false;
	return {
		backend: resolved.backend,
		scratchDir,
		...(resolved.unavailableReason === undefined ? {} : { unavailableReason: resolved.unavailableReason }),
		operations,
		dispose() {
			if (disposed) return;
			disposed = true;
			if (ownedScratch) {
				try {
					rmSync(scratchDir, { recursive: true, force: true });
				} catch {
					// Best-effort scratch cleanup.
				}
			}
		},
	};
}
