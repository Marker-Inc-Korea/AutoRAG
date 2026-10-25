import type { ChildProcess } from "node:child_process";
import { terminateProcessTree } from "./terminate-tree.ts";

const HOST_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

/**
 * Children still running. A POSIX child leads its own process group, so it no
 * longer receives the terminal's Ctrl-C or the host's group-wide signals; the
 * host takes the tree down with it instead of orphaning a lock-holding sync.
 */
const liveChildren = new Set<ChildProcess>();
const hostSignalHandlers = new Map<NodeJS.Signals, () => void>();

function killLiveChildren(): void {
	for (const child of liveChildren) terminateProcessTree(child, "SIGKILL");
}

function installHostCleanup(): void {
	if (hostSignalHandlers.size > 0) return;
	process.on("exit", killLiveChildren);
	for (const signal of HOST_SIGNALS) {
		const handler = (): void => {
			killLiveChildren();
			// Listening for a signal replaces its default "terminate" action. When
			// nothing else listens, restore it so the host still dies from the signal.
			if (process.listenerCount(signal) === 1) {
				removeHostCleanup();
				process.kill(process.pid, signal);
			}
		};
		hostSignalHandlers.set(signal, handler);
		process.on(signal, handler);
	}
}

function removeHostCleanup(): void {
	process.off("exit", killLiveChildren);
	for (const [signal, handler] of hostSignalHandlers) process.off(signal, handler);
	hostSignalHandlers.clear();
}

/**
 * Track a spawned child so it cannot outlive the host: a host exit, and
 * SIGINT/SIGTERM/SIGHUP, kill its process tree. Windows has no process group to
 * signal, so it keeps the parent/child chain `taskkill /t` walks and tracks
 * nothing.
 */
export function trackLiveChild(child: ChildProcess): void {
	if (process.platform === "win32") return;
	liveChildren.add(child);
	installHostCleanup();
}

/** Stop tracking a child that has exited; the host handlers go with the last one. */
export function untrackLiveChild(child: ChildProcess): void {
	liveChildren.delete(child);
	if (liveChildren.size === 0) removeHostCleanup();
}
