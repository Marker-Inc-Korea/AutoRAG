import { statSync } from "node:fs";
import { join, toNamespacedPath } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { JikjiDoctorResult } from "./types.ts";

/** Mirrors jikji-cli 0.2.x's data_dir(), including its environment overrides. */
export function jikjiDatabasePath(env: NodeJS.ProcessEnv, platform: NodeJS.Platform = process.platform): string {
	const get = (name: string): string | undefined => {
		if (platform !== "win32") return env[name];
		// Match Node's child-process selection when Windows env contains aliases.
		const key = Object.keys(env)
			.sort()
			.find((key) => key.toUpperCase() === name && env[key] !== undefined);
		return key === undefined ? undefined : env[key];
	};
	let data = get("JIKJI_DATA_DIR");
	if (data === undefined || data.length === 0) {
		data =
			platform === "win32"
				? get("LOCALAPPDATA")
				: platform === "darwin"
					? env.HOME === undefined
						? undefined
						: join(env.HOME, "Library", "Application Support")
					: (env.XDG_DATA_HOME ?? (env.HOME === undefined ? undefined : join(env.HOME, ".local", "share")));
	}
	if (data === undefined) throw new Error("Cannot determine Jikji data directory from the process environment.");
	return join(data, "jikji", "index.sqlite");
}

/**
 * Native doctor opens/initializes its store and can register or migrate a root.
 * Only invoke it for a root with an existing manifest in the central store.
 * A root row alone is insufficient: prepare registers it before building anything.
 * Do not probe the legacy .jikji folder: doctor can migrate it as a side effect.
 */
export function hasPreparedJikjiRoot(root: string, env: NodeJS.ProcessEnv): boolean {
	const path = jikjiDatabasePath(env);
	try {
		statSync(path);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
		throw error;
	}
	const db = new DatabaseSync(path, { readOnly: true });
	try {
		const tables = db
			.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('roots', 'artifacts')")
			.all();
		if (tables.length !== 2) return false;
		// Rust canonicalize() uses Windows extended-length paths; Node realpath does not.
		return (
			db
				.prepare(
					"SELECT 1 FROM roots r JOIN artifacts a ON a.root_id = r.id WHERE r.canonical_root IN (?, ?) AND a.kind = 'manifest' LIMIT 1",
				)
				.get(root, toNamespacedPath(root)) !== undefined
		);
	} finally {
		db.close();
	}
}

/** doctor exits 1 for an unhealthy index but still emits its full JSON report. */
export function parseJikjiDoctorReport(stdout: string): JikjiDoctorResult {
	const value: unknown = JSON.parse(stdout);
	if (
		!isRecord(value) ||
		typeof value.ok !== "boolean" ||
		typeof value.prepared !== "boolean" ||
		!strings(value.errors) ||
		!strings(value.warnings)
	) {
		throw new Error("Jikji doctor report must contain ok, prepared, errors and warnings.");
	}
	const docsFailed = isRecord(value.manifest) ? value.manifest.docs_failed : undefined;
	if (
		docsFailed !== undefined &&
		(typeof docsFailed !== "number" || !Number.isSafeInteger(docsFailed) || docsFailed < 0)
	) {
		throw new Error("Jikji doctor manifest.docs_failed must be a non-negative safe integer.");
	}
	return {
		state: value.ok && value.prepared ? "prepared" : "needs-reprepare",
		...(typeof docsFailed === "number" ? { docsFailed } : {}),
		errors: value.errors,
		warnings: value.warnings,
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function strings(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((entry) => typeof entry === "string");
}
