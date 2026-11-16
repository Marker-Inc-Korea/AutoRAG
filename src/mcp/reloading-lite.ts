import { statSync } from "node:fs";
import { resolveConfigPath } from "../cli/config.ts";
import type { AutoRAGLite } from "../core.ts";

export interface ReloadingLiteOptions {
	/** Build a runtime from the config as it is now. May throw (missing or unparsable config). */
	readonly create: () => AutoRAGLite;
	/** A cheap fingerprint of whatever `create` reads; a different value means the runtime is out of date. */
	readonly signature: () => string;
	/** A config that could not be loaded. Called once per distinct broken config, never on every call. */
	readonly onError?: (error: Error) => void;
}

/**
 * A runtime that follows its config file. An MCP server is started once and then lives for as long as its host
 * keeps it (Hermes keeps one per gateway, across chat sessions), so a config written after it started, which is
 * exactly what first-time setup does, would otherwise never be seen: every search kept answering `index-not-ready`.
 *
 * Each access compares the config's signature (two `stat` calls) and rebuilds the runtime when it changed. The first
 * build happens eagerly, so a bad config still fails at start. A later config that cannot be loaded (half-written,
 * mistyped) keeps the last working runtime and is reported once, then retried when the file changes again.
 */
export function createReloadingLite(options: ReloadingLiteOptions): AutoRAGLite {
	// Read the signature before building, so a change made while building triggers one more rebuild.
	let attempted = options.signature();
	let current = options.create();

	const sync = (): AutoRAGLite => {
		const next = options.signature();
		if (next === attempted) return current;
		attempted = next;
		try {
			current = options.create();
		} catch (error) {
			options.onError?.(error instanceof Error ? error : new Error(String(error)));
		}
		return current;
	};

	return new Proxy({} as AutoRAGLite, {
		get(_target, property) {
			const runtime = sync();
			const value: unknown = Reflect.get(runtime, property, runtime);
			return typeof value === "function" ? value.bind(runtime) : value;
		},
		has(_target, property) {
			return property in sync();
		},
	});
}

/**
 * Fingerprint of the config file `createAutoRAGLite` would read: its inode, size and modification time, or `absent`.
 * An explicit `AUTORAG_CONFIG`, else `<AUTORAG_HOME>/config.json` (and the legacy `autorag.config.json` beside the
 * working directory) are covered, in the same order the resolver uses.
 */
export function configSignature(env: NodeJS.ProcessEnv = process.env): string {
	const { configPath, legacyPath } = resolveConfigPath({ flags: {}, env });
	return [configPath, legacyPath].map((path) => (path === undefined ? "-" : fingerprint(path))).join("|");
}

function fingerprint(path: string): string {
	try {
		const stats = statSync(path);
		return `${stats.ino}:${stats.size}:${stats.mtimeMs}`;
	} catch {
		return "absent";
	}
}
