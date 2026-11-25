import { isAbsolute, relative, resolve, sep } from "node:path";
import {
	type BashOperations,
	createBashToolDefinition,
	createLocalBashOperations,
	type ExtensionAPI,
	type ExtensionContext,
	type ExtensionFactory,
} from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";
import { createReadOnlySandbox, type ReadOnlySandbox, type ReadOnlySandboxOptions } from "./readonly-sandbox.ts";

/**
 * Binary permission modes for the agent. No decision model (Jev) is consulted.
 *
 * - `readonly` (default): disk writes are prevented at the OS level for `bash` (except inside the session's
 *   dedicated scratch directory), and write-capable tools (`write`, `edit`, `convert_files`, `manage_automations`
 *   mutations, `manage_ltr` mutations, `powershell`) are blocked.
 * - `full`: unrestricted read/write access; the agent executes all tools and shell commands directly.
 */
export type PermissionMode = "readonly" | "full";

/** Order Shift+Tab cycles through. */
export const PERMISSION_MODES: readonly PermissionMode[] = ["readonly", "full"];

/** Mode a fresh interactive or CLI session starts in. */
export const DEFAULT_PERMISSION_MODE: PermissionMode = "readonly";

const STATUS_KEY = "autorag-permissions";

/**
 * Parse a permission mode string case-insensitively.
 * Accepts `"readonly"` (and `"read-only"` alias) and `"full"`. Rejects everything else.
 */
export function parsePermissionMode(value: unknown): PermissionMode | undefined {
	if (typeof value !== "string") return undefined;
	const normalized = value.trim().toLowerCase();
	if (normalized === "read-only") return "readonly";
	return PERMISSION_MODES.find((mode) => mode === normalized);
}

export function cyclePermissionMode(mode: PermissionMode): PermissionMode {
	return PERMISSION_MODES[(PERMISSION_MODES.indexOf(mode) + 1) % PERMISSION_MODES.length] as PermissionMode;
}

type ModeListener = (mode: PermissionMode) => void;

/**
 * Session-wide permission state: either `readonly` or `full`. One controller is shared by every pi
 * session of an agent so switching via Shift+Tab, `/permissions`, or the API takes effect immediately.
 */
export class PermissionController {
	private current: PermissionMode;
	private readonly listeners = new Set<ModeListener>();

	constructor(mode: PermissionMode = DEFAULT_PERMISSION_MODE) {
		this.current = mode;
	}

	get mode(): PermissionMode {
		return this.current;
	}

	get isReadOnly(): boolean {
		return this.current === "readonly";
	}

	set(mode: PermissionMode): void {
		if (mode === this.current) return;
		this.current = mode;
		for (const listener of [...this.listeners]) listener(mode);
	}

	cycle(): PermissionMode {
		this.set(cyclePermissionMode(this.current));
		return this.current;
	}

	subscribe(listener: ModeListener): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}
}

export interface PermissionGateOptions {
	readonly controller: PermissionController;
	/**
	 * Files the agent may write/edit even in `readonly` mode (for example the active config file during a
	 * self-configuration turn). Empty by default.
	 */
	readonly trustedPaths?: readonly string[];
	/** Options forwarded when constructing the OS-level read-only bash sandbox (used in tests). */
	readonly sandboxOptions?: ReadOnlySandboxOptions;
	/** Factory override for tests. Defaults to {@link createReadOnlySandbox}. */
	readonly createSandbox?: (options?: ReadOnlySandboxOptions) => ReadOnlySandbox;
}

/** Appended to the system prompt while in `readonly` mode so the model knows writes will fail. */
export const READONLY_PROMPT_ADDENDUM = [
	"Permission mode: READ-ONLY.",
	"This session has read-only access to the disk.",
	"Direct file modifications (`write`, `edit`, `convert_files`, saving/deleting automations, changing LTR settings) are disabled,",
	"and `bash` runs inside an OS-level read-only filesystem sandbox where creating, editing, moving, renaming, or deleting any file outside the session temp directory (`$TMPDIR`) is blocked by the kernel.",
	"Read, search, inspect, and answer freely. If the user asks you to create, modify, organize, convert, or delete files, explain that the session is in read-only mode and tell them they can switch to `full` mode with Shift+Tab, `/permissions full`, or `--permission-mode full`.",
].join(" ");

const MODE_SUMMARY: Readonly<Record<PermissionMode, string>> = {
	readonly: "read-only disk access (OS-sandboxed bash, write/edit/mutating tools blocked)",
	full: "full read and write access (no restrictions)",
};

const MODE_LABEL: Readonly<Record<PermissionMode, string>> = {
	readonly: "permissions: read-only",
	full: "permissions: full",
};

interface Decision {
	readonly block?: true;
	readonly reason?: string;
}

function resolveToolPath(raw: string, cwd: string): string {
	return resolve(cwd, raw);
}

function isWithinDir(candidate: string, dir: string): boolean {
	const rel = relative(dir, candidate);
	return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

/**
 * Whether a tool call mutates disk/state and must be blocked in `readonly` mode.
 * Returns a human-readable description of the blocked action, or `undefined` if read-only.
 */
export function describeReadOnlyBlock(
	toolName: string,
	input: Readonly<Record<string, unknown>>,
	cwd: string,
	trustedPaths: ReadonlySet<string>,
	scratchDir: string | undefined,
): string | undefined {
	if (toolName === "write" || toolName === "edit") {
		const rawPath = typeof input.path === "string" ? input.path : "";
		if (rawPath !== "") {
			const abs = resolveToolPath(rawPath, cwd);
			if (trustedPaths.has(abs)) return undefined;
			if (scratchDir !== undefined && isWithinDir(abs, scratchDir)) return undefined;
		}
		return `${toolName} (${rawPath || "file"})`;
	}
	if (toolName === "convert_files") {
		const outputDir = typeof input.outputDir === "string" ? resolveToolPath(input.outputDir, cwd) : undefined;
		if (outputDir !== undefined && scratchDir !== undefined && isWithinDir(outputDir, scratchDir)) {
			return undefined;
		}
		return "convert_files";
	}
	if (toolName === "manage_automations") {
		const action = typeof input.action === "string" ? input.action : "";
		if (action === "list") return undefined;
		return `manage_automations (${action || "mutate"})`;
	}
	if (toolName === "manage_ltr") {
		const action = typeof input.action === "string" ? input.action : "";
		if (action === "status") return undefined;
		return `manage_ltr (${action || "mutate"})`;
	}
	if (toolName === "powershell") {
		return "powershell";
	}
	return undefined;
}

function readOnlyBlockedReason(action: string): string {
	return (
		`Blocked: session is in read-only permission mode, so ${action} is not allowed. ` +
		"Do not retry mutating operations in read-only mode. Tell the user they can switch to full mode " +
		"(Shift+Tab or `/permissions full` in the TUI, or `--permission-mode full` on the CLI) if they want you to make changes."
	);
}

/**
 * pi extension enforcing binary `readonly` / `full` permissions with OS-level filesystem read-only
 * sandboxing for `bash`. No decision model (Jev) and no per-call approval dialogs.
 *
 * - Overrides the built-in `bash` tool with a mode-aware wrapper: in `readonly`, commands execute inside
 *   the OS read-only sandbox (`sandbox-exec` on macOS, `bwrap` on Linux, fail-closed otherwise) with a
 *   dedicated writable scratch dir (`$TMPDIR`); in `full`, commands execute via pi's standard local bash.
 * - Intercepts `user_bash` (`!` shell escapes in the TUI) with the same read-only operations while in `readonly`.
 * - Blocks write-capable tools (`write`, `edit`, `convert_files`, `manage_automations` mutations, `manage_ltr`
 *   mutations, `powershell`) in `tool_call` while in `readonly`.
 * - Adds `/permissions [readonly|full]` and Shift+Tab toggling in the TUI, with live footer status
 *   (`permissions: read-only` vs `permissions: full`).
 */
export function createPermissionGate(options: PermissionGateOptions): ExtensionFactory {
	const { controller } = options;
	const trustedPaths = new Set((options.trustedPaths ?? []).map((p) => resolve(p)));
	const sandboxFactory = options.createSandbox ?? createReadOnlySandbox;

	return (pi: ExtensionAPI) => {
		let boundUi: ExtensionContext["ui"] | undefined;
		let unsubscribeKeys: (() => void) | undefined;
		let unsubscribeMode: (() => void) | undefined;
		let sandbox: ReadOnlySandbox | undefined;
		let localOps: BashOperations | undefined;
		let localShellPath: string | undefined;

		const ensureSandbox = (): ReadOnlySandbox => {
			if (sandbox === undefined) {
				sandbox = sandboxFactory({
					...options.sandboxOptions,
					...(trustedPaths.size > 0
						? {
								extraWritablePaths: [...(options.sandboxOptions?.extraWritablePaths ?? []), ...trustedPaths],
							}
						: {}),
				});
			}
			return sandbox;
		};

		const getShellSettings = (): { readonly shellPath?: string; readonly shellCommandPrefix?: string } => {
			try {
				const settings = pi.getSettings?.();
				return {
					...(settings?.shellPath === undefined ? {} : { shellPath: settings.shellPath }),
					...(settings?.shellCommandPrefix === undefined
						? {}
						: { shellCommandPrefix: settings.shellCommandPrefix }),
				};
			} catch {
				return {};
			}
		};

		const getLocalOps = (): BashOperations => {
			const configuredShell = getShellSettings().shellPath;
			if (localOps === undefined || configuredShell !== localShellPath) {
				localShellPath = configuredShell;
				localOps = createLocalBashOperations(
					configuredShell === undefined ? undefined : { shellPath: configuredShell },
				);
			}
			return localOps;
		};

		// Mode-aware BashOperations shared by the overridden `bash` tool and `user_bash`.
		const modeAwareBashOps: BashOperations = {
			exec: (command, cwd, execOptions) => {
				const prefix = getShellSettings().shellCommandPrefix;
				const fullCommand = prefix ? `${prefix}\n${command}` : command;
				if (controller.isReadOnly) {
					return ensureSandbox().operations.exec(fullCommand, cwd, execOptions);
				}
				return getLocalOps().exec(fullCommand, cwd, execOptions);
			},
		};

		// Override pi's built-in `bash` tool so every model bash call routes through modeAwareBashOps
		// while preserving pi's schema, truncation, streaming output, and TUI renderers.
		// Note: `pi.getSettings()` is an action method unavailable during extension load, so shell settings
		// are resolved lazily inside `modeAwareBashOps.exec`.
		if (typeof pi.registerTool === "function") {
			const bashToolDef = createBashToolDefinition(process.cwd(), {
				operations: modeAwareBashOps,
			});
			pi.registerTool(bashToolDef);
		}

		// Also route TUI `!cmd` user_bash through the read-only sandbox when in `readonly` mode.
		pi.on("user_bash", () => {
			if (!controller.isReadOnly) return undefined;
			return { operations: ensureSandbox().operations };
		});

		const showStatus = (ui: ExtensionContext["ui"], mode: PermissionMode): void => {
			const text = MODE_LABEL[mode];
			const color = mode === "full" ? "warning" : "success";
			ui.setStatus(STATUS_KEY, ui.theme?.fg ? ui.theme.fg(color, text) : text);
		};

		const bindUi = (ui: ExtensionContext["ui"]): void => {
			showStatus(ui, controller.mode);
			if (boundUi === ui) return;
			unsubscribeKeys?.();
			unsubscribeMode?.();
			boundUi = ui;
			unsubscribeMode = controller.subscribe((mode) => showStatus(ui, mode));
			unsubscribeKeys = ui.onTerminalInput((data) => {
				if (!matchesKey(data, "shift+tab")) return undefined;
				const mode = controller.cycle();
				ui.notify(`Permission mode: ${mode} — ${MODE_SUMMARY[mode]}`, "info");
				return { consume: true };
			});
		};

		pi.on("session_start", (_event, ctx) => {
			if (ctx.hasUI) bindUi(ctx.ui);
		});

		pi.on("session_shutdown", () => {
			unsubscribeKeys?.();
			unsubscribeMode?.();
			unsubscribeKeys = undefined;
			unsubscribeMode = undefined;
			boundUi = undefined;
			sandbox?.dispose();
			sandbox = undefined;
		});

		pi.registerCommand("permissions", {
			description: "Show or set permission mode: readonly or full (Shift+Tab toggles)",
			getArgumentCompletions: (prefix) => {
				const items = PERMISSION_MODES.filter((mode) => mode.startsWith(prefix.trim().toLowerCase())).map(
					(mode) => ({
						value: mode,
						label: mode,
						description: MODE_SUMMARY[mode],
					}),
				);
				return items.length > 0 ? items : null;
			},
			handler: async (args, ctx) => {
				const wanted = args.trim();
				let mode = parsePermissionMode(wanted);
				if (wanted !== "" && mode === undefined) {
					ctx.ui.notify(
						`Unknown permission mode "${wanted}". Use one of: ${PERMISSION_MODES.join(", ")}.`,
						"warning",
					);
					return;
				}
				if (mode === undefined) {
					const choice = await ctx.ui.select(
						`Permission mode (now ${controller.mode})`,
						PERMISSION_MODES.map((candidate) => `${candidate} - ${MODE_SUMMARY[candidate]}`),
					);
					mode = parsePermissionMode(choice?.split(" ")[0]);
					if (mode === undefined) return;
				}
				controller.set(mode);
				ctx.ui.notify(`Permission mode: ${mode} — ${MODE_SUMMARY[mode]}`, "info");
			},
		});

		pi.on("before_agent_start", (event) => {
			if (!controller.isReadOnly) return undefined;
			return { systemPrompt: `${event.systemPrompt}\n\n${READONLY_PROMPT_ADDENDUM}` };
		});

		pi.on("tool_call", (event, ctx): Decision | undefined => {
			if (!controller.isReadOnly) return undefined;
			const input = event.input as Record<string, unknown>;
			const scratchDir = sandbox?.scratchDir;
			const blocked = describeReadOnlyBlock(event.toolName, input, ctx.cwd, trustedPaths, scratchDir);
			if (blocked === undefined) return undefined;
			return { block: true, reason: readOnlyBlockedReason(blocked) };
		});
	};
}
