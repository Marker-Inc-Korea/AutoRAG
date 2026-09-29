import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { hasErrorCode } from "./fs-entry";

/**
 * Recently opened / previewed files, owned by the app so the Recents view works
 * identically on macOS and Windows: the directory comes from Electron's
 * platform userData path and every entry is stored as the OS-native absolute
 * path, never split or rewritten here.
 */
export interface RecentsStore {
	/** Move a path to the front; opening it again refreshes its position. */
	record(path: string): Promise<void>;
	/** Recorded absolute paths, most recent first. */
	list(): Promise<readonly string[]>;
}

export interface RecentsStoreDeps {
	readonly directory: string;
	/** Deepest history kept; defaults to RECENTS_LIMIT. */
	readonly limit?: number;
}

export const RECENTS_FILENAME = "recents.json";
export const RECENTS_LIMIT = 50;

/** JSON persistence for the Recents list. Writes are serialized and atomic. */
export function createRecentsStore(deps: RecentsStoreDeps): RecentsStore {
	const filePath = join(deps.directory, RECENTS_FILENAME);
	const limit = Math.max(0, deps.limit ?? RECENTS_LIMIT);
	let paths: string[] | undefined;
	let queue: Promise<void> = Promise.resolve();

	async function loadPaths(): Promise<string[]> {
		if (paths !== undefined) return paths;
		try {
			const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
			paths = Array.isArray(parsed)
				? parsed.filter((value): value is string => typeof value === "string" && value.length > 0)
				: [];
		} catch (error) {
			if (error instanceof SyntaxError || hasErrorCode(error, "ENOENT") || hasErrorCode(error, "ENOTDIR")) {
				paths = [];
			} else {
				throw error;
			}
		}
		return paths;
	}

	async function writePaths(next: readonly string[]): Promise<void> {
		await mkdir(deps.directory, { recursive: true });
		const temporaryPath = join(deps.directory, `.${RECENTS_FILENAME}.${process.pid}.${randomUUID()}.tmp`);
		try {
			await writeFile(temporaryPath, `${JSON.stringify(next, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
			await rename(temporaryPath, filePath);
		} finally {
			await rm(temporaryPath, { force: true });
		}
	}

	function enqueue<T>(operation: () => Promise<T>): Promise<T> {
		const result = queue.then(operation);
		queue = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}

	return {
		record: (path) =>
			enqueue(async () => {
				if (path.length === 0) return;
				const loaded = await loadPaths();
				const next = [path, ...loaded.filter((existing) => existing !== path)].slice(0, limit);
				paths = next;
				await writePaths(next);
			}),
		list: () =>
			enqueue(async () => [...(await loadPaths())]),
	};
}
