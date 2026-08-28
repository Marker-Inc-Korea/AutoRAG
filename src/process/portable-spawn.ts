import { existsSync, readFileSync } from "node:fs";
import { extname } from "node:path";

const NODE_SCRIPT_EXTENSIONS = new Set([".cjs", ".js", ".mjs"]);

export interface PortableSpawnCommand {
	readonly command: string;
	readonly args: readonly string[];
}

/**
 * Windows does not execute Unix shebang files or extensionless scripts through
 * child_process.spawn. Resolve test/dev script entrypoints through their
 * interpreter while leaving native executables untouched.
 */
export function portableSpawnCommand(
	command: string,
	args: readonly string[],
	platform: NodeJS.Platform = process.platform,
): PortableSpawnCommand {
	if (platform !== "win32" || !existsSync(command)) return { command, args };

	const extension = extname(command).toLowerCase();
	if (NODE_SCRIPT_EXTENSIONS.has(extension) || hasShebang(command, "node")) {
		if (hasShebang(command, "node") && extension.length === 0) {
			return {
				command: process.versions.bun ? "node" : process.execPath,
				args: [command, ...args],
			};
		}
		if (hasShebang(command, "node") && process.versions.bun) {
			return { command: "node", args: [command, ...args] };
		}
		return { command: process.execPath, args: [command, ...args] };
	}
	if (extension.length === 0 && (hasShebang(command, "/bin/sh") || hasShebang(command, "/usr/bin/env sh"))) {
		return { command: "bash", args: [command, ...args] };
	}
	return { command, args };
}

function hasShebang(path: string, interpreter: string): boolean {
	try {
		const prefix = readFileSync(path, "utf8").slice(0, 128);
		return prefix.startsWith("#!") && prefix.includes(interpreter);
	} catch {
		return false;
	}
}
