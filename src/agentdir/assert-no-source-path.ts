/**
 * Enforce agentdir path opacity: no source filesystem path may appear in any
 * agent-facing value (tool content, details, or error text).
 *
 * Throws if the serialized `value` contains any of the known source roots.
 * Used both as a test assertion and as a runtime guard before returning tool
 * output to the agent.
 */
export function assertNoSourcePath(value: unknown, knownSourceRoots: string[]): void {
	const serialized = typeof value === "string" ? value : JSON.stringify(value ?? "");
	for (const root of knownSourceRoots) {
		if (root && serialized.includes(root)) {
			throw new Error("agentdir path opacity violation: agent-facing output leaked a source filesystem path");
		}
	}
}

/** Non-throwing variant: returns true when `value` is free of every source root. */
export function isSourcePathFree(value: unknown, knownSourceRoots: string[]): boolean {
	try {
		assertNoSourcePath(value, knownSourceRoots);
		return true;
	} catch {
		return false;
	}
}
