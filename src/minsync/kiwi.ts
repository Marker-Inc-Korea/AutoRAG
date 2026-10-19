import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { posix, win32 } from "node:path";

/** What the probe reads, injectable so every platform's layout can be checked from any host. */
export interface KiwiProbeEnvironment {
	readonly env: NodeJS.ProcessEnv;
	readonly platform: NodeJS.Platform;
	readonly home: string | undefined;
	readonly exists: (path: string) => boolean;
}

/**
 * MinSync's `ko` tokenizer loads the Kiwi native library and its base model at run time (through `kiwi-rs`); neither
 * ships with MinSync. Without them every sync that needs Korean tokenization fails, so AutoRAG checks first.
 *
 * This mirrors the filesystem part of `kiwi-rs`'s discovery (`KIWI_LIBRARY_PATH` / `KIWI_MODEL_PATH`, then the
 * per-user `~/.local/kiwi` install and the well-known system directories). Names left to the OS loader search path
 * are not probed, so a library found only that way reads as missing; point `KIWI_LIBRARY_PATH` at it to fix that.
 * kiwi-rs treats an explicit path as authoritative, and so does this probe.
 */
export function kiwiAvailable(environment: KiwiProbeEnvironment = defaultEnvironment()): boolean {
	const { env, exists } = environment;
	const library = env.KIWI_LIBRARY_PATH;
	const model = env.KIWI_MODEL_PATH;
	const libraryFound =
		library !== undefined && library !== "" ? exists(library) : libraryCandidates(environment).some(exists);
	const modelFound = model !== undefined && model !== "" ? exists(model) : modelCandidates(environment).some(exists);
	return libraryFound && modelFound;
}

function defaultEnvironment(): KiwiProbeEnvironment {
	let home: string | undefined;
	try {
		home = homedir();
	} catch {
		home = undefined;
	}
	return { env: process.env, platform: process.platform, home, exists: existsSync };
}

/** Per-user install roots that hold `lib/` and `models/`: `~/.local/kiwi` on POSIX, `%LOCALAPPDATA%\kiwi` on Windows. */
function userRoots({ env, platform, home }: KiwiProbeEnvironment): string[] {
	if (platform === "win32") {
		const roots: string[] = [];
		if (env.LOCALAPPDATA) roots.push(win32.join(env.LOCALAPPDATA, "kiwi"));
		const profile = env.USERPROFILE ?? home;
		if (profile) roots.push(win32.join(profile, "AppData", "Local", "kiwi"));
		roots.push("C:\\kiwi", "C:\\Program Files\\Kiwi");
		return roots;
	}
	return home ? [posix.join(home, ".local", "kiwi")] : [];
}

function libraryCandidates(environment: KiwiProbeEnvironment): string[] {
	const { platform } = environment;
	if (platform === "win32") return userRoots(environment).map((root) => win32.join(root, "lib", "kiwi.dll"));
	const file = platform === "darwin" ? "libkiwi.dylib" : "libkiwi.so";
	const system = platform === "darwin" ? ["/usr/local/lib", "/opt/homebrew/lib"] : ["/usr/local/lib", "/usr/lib"];
	return [
		...userRoots(environment).map((root) => posix.join(root, "lib", file)),
		...system.map((dir) => posix.join(dir, file)),
	];
}

function modelCandidates(environment: KiwiProbeEnvironment): string[] {
	const { platform } = environment;
	if (platform === "win32") return userRoots(environment).map((root) => win32.join(root, "models", "cong", "base"));
	const system =
		platform === "darwin"
			? ["/usr/local/models", "/opt/homebrew/models", "/usr/local/share/kiwi/models"]
			: ["/usr/local/models", "/usr/local/share/kiwi/models", "/usr/share/kiwi/models"];
	return [
		...userRoots(environment).map((root) => posix.join(root, "models", "cong", "base")),
		...system.map((dir) => posix.join(dir, "cong", "base")),
	];
}
