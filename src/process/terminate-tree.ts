import { type ChildProcess, spawn } from "node:child_process";

/** Grace after the direct child exits for its stream-holding descendants to die. */
export const TREE_KILL_GRACE_MS = 500;

/**
 * Terminate a spawned external CLI together with the descendants it created.
 *
 * CLIs such as qmd 2.8.3 are launchers: the process AutoRAG spawns starts a
 * runtime descendant that inherits the launcher's stdout/stderr pipes. Node
 * emits `close` only once every holder of those pipes has exited, so signalling
 * just the direct child leaves the caller waiting on a descendant that may live
 * for minutes — the hang reported in #1578.
 *
 * Spawn the child with `detached: true` on POSIX so it leads its own process
 * group; the group survives the direct child and stays addressable as `-pid`.
 * Windows has no equivalent handle, so it uses `taskkill /t /f`, which walks the
 * live parent/child chain and therefore requires `detached: false` there.
 *
 * Returns whether the process group / tree was signalled.
 * Default SIGKILL: SIGTERM lets a Node descendant close inherited stdio during
 * shutdown while the pid is still kill(pid,0)-visible (zombie or exiting).
 */
export function terminateProcessTree(child: ChildProcess, signal: NodeJS.Signals = "SIGKILL"): boolean {
	const pid = child.pid;
	if (pid === undefined) return false;
	if (process.platform === "win32") {
		const killer = spawn("taskkill", ["/pid", String(pid), "/t", "/f"], { stdio: "ignore", windowsHide: true });
		killer.on("error", () => killDirectChild(child, signal));
		killer.on("close", (code) => {
			if (code !== 0) killDirectChild(child, signal);
		});
		return true;
	}
	try {
		process.kill(-pid, signal);
		return true;
	} catch {
		// The group is already gone; fall back to the direct child.
		return killDirectChild(child, signal);
	}
}

function killDirectChild(child: ChildProcess, signal: NodeJS.Signals): boolean {
	if (child.exitCode !== null || child.signalCode !== null) return false;
	try {
		return child.kill(signal);
	} catch {
		return false;
	}
}
