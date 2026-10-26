import { arch as osArch, platform as osPlatform, release as osRelease } from "node:os";
import { getShellConfig } from "@earendil-works/pi-coding-agent";
import { currentDateAddendum } from "./current-date.ts";

const OS_NAMES: Readonly<Record<string, string>> = { darwin: "macOS", win32: "Windows", linux: "Linux" };

export interface EnvironmentInfo {
	readonly cwd: string;
	readonly platform: string;
	readonly arch: string;
	readonly release: string;
	/** The shell the `bash` tool runs commands in; undefined when pi found none. */
	readonly shell: string | undefined;
}

/** The shell pi's `bash` tool will use on this host, or undefined when none resolves. */
function resolveBashShell(): string | undefined {
	try {
		return getShellConfig().shell;
	} catch {
		return undefined;
	}
}

/**
 * Where the agent runs: OS, shell, and working directory. AutoRAG replaces pi's whole
 * system prompt, so without this the model writes POSIX commands for a Windows host
 * and relative paths against an unknown directory. Fixed for the life of a session.
 */
export function environmentAddendum(info: EnvironmentInfo): string {
	const os = OS_NAMES[info.platform] ?? info.platform;
	const shell =
		info.shell === undefined
			? "no shell could be found, so the `bash` tool will fail"
			: `the \`bash\` tool runs commands in ${info.shell}`;
	return `\n\n<environment>\nOS: ${os} (${info.platform} ${info.release}, ${info.arch}).\nShell: ${shell}.\nWorking directory: ${info.cwd} (relative paths resolve against it).\n</environment>`;
}

/**
 * Everything the host appends after AutoRAG's own prompt. Ordered most stable first:
 * the environment never changes in a session, the date changes every minute, so the
 * longest possible prefix of the prompt stays identical between turns for prompt caching.
 */
export function runtimeContextAddendum(cwd: string, now: Date = new Date()): string {
	return (
		environmentAddendum({
			cwd,
			platform: osPlatform(),
			arch: osArch(),
			release: osRelease(),
			shell: resolveBashShell(),
		}) + currentDateAddendum(now)
	);
}
