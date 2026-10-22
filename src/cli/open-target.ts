import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

type SpawnFn = typeof spawn;

/** The platform's default opener, the same one pi's own URL click uses. */
function systemOpener(url: string): readonly [string, readonly string[]] {
	if (process.platform === "darwin") return ["open", [url]];
	if (process.platform === "win32") return ["rundll32", ["url.dll,FileProtocolHandler", url]];
	return ["xdg-open", [url]];
}

/** Start `command` detached; `onFailure` runs once when it cannot start or exits non-zero. */
function launch(spawnFn: SpawnFn, command: string, args: readonly string[], onFailure?: () => void): void {
	let failed = false;
	const fail = (): void => {
		if (failed) return;
		failed = true;
		onFailure?.();
	};
	const child = spawnFn(command, [...args], { stdio: "ignore", detached: true });
	child.on("error", fail);
	child.on("close", (code) => {
		if (code !== null && code !== 0) fail();
	});
	child.unref();
}

/**
 * Open what a clicked TUI citation points at. A local file goes to Dazzi
 * (`dazzi <path>` shows it in its folder); when Dazzi is not installed or
 * refuses, and for web links, the platform's default opener takes over.
 */
export function openCitationTarget(url: string, spawnFn: SpawnFn = spawn): void {
	const [command, args] = systemOpener(url);
	if (!url.startsWith("file:")) {
		launch(spawnFn, command, args);
		return;
	}
	let path: string;
	try {
		path = fileURLToPath(url);
	} catch {
		launch(spawnFn, command, args);
		return;
	}
	launch(spawnFn, "dazzi", ["--", path], () => launch(spawnFn, command, args));
}
