import { type Dirent, readdirSync, type Stats, statSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { setTimeout as sleepFor } from "node:timers/promises";
import type { ExtensionAPI, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import {
	type CloudEvictResult,
	type CloudFileProbe,
	type CloudRoot,
	canonicalPath,
	isPathInside,
} from "../filesystem/cloud-placeholder.ts";
import { DEFAULT_CLOUD_EVICT_BACKOFF_MS, DEFAULT_CLOUD_EVICT_RETRIES } from "../mirror/cloud.ts";

/** Direct child files one directory candidate contributes; beyond this a directory is left unwatched. */
export const DEFAULT_CLOUD_RELEASE_CHILD_CAP = 500;

/** Words of a tool argument or a command, with a quoted run kept as one word so a path with spaces survives. */
const WORD = /"([^"]*)"|'([^']*)'|(\S+)/g;

/**
 * Every absolute path in a tool input value: a string that is itself absolute (a `path` argument), and the
 * absolute words of any string that is not (a `bash` command, where `cd "<dir>" && pdftotext "a.pdf" -` names
 * the directory `/…/dir` and no absolute file). Object and array values are walked, so a tool that nests its
 * arguments is covered too.
 */
export function absolutePathsIn(value: unknown): string[] {
	const found: string[] = [];
	const visit = (node: unknown): void => {
		if (typeof node === "string") {
			if (isAbsolute(node)) found.push(node);
			for (const match of node.matchAll(WORD)) {
				const word = match[1] ?? match[2] ?? match[3];
				if (word !== undefined && isAbsolute(word)) found.push(word);
			}
			return;
		}
		if (Array.isArray(node)) {
			for (const item of node) visit(item);
			return;
		}
		if (node !== null && typeof node === "object") {
			for (const item of Object.values(node)) visit(item);
		}
	};
	visit(value);
	return found;
}

export interface CloudReleaseExtensionOptions {
	/** Platform cloud probe; without one the extension registers nothing (a platform AutoRAG cannot release on). */
	readonly probe?: CloudFileProbe;
	/**
	 * The machine's non-physical roots, resolved lazily and cached by the caller. Without one, or when it
	 * resolves to no releasable root (Linux, or a machine with no cloud folders), the extension does nothing.
	 */
	readonly cloudRoots?: () => Promise<readonly CloudRoot[]>;
	/** Direct child files one directory candidate contributes. Default {@link DEFAULT_CLOUD_RELEASE_CHILD_CAP}. */
	readonly childCap?: number;
	/** Retries after the first failed release; `0` tries once. Default {@link DEFAULT_CLOUD_EVICT_RETRIES}. */
	readonly evictRetries?: number;
	/** First retry wait in ms, doubled each retry. Default {@link DEFAULT_CLOUD_EVICT_BACKOFF_MS}. */
	readonly evictBackoffMs?: number;
	/** Wait implementation; tests inject one that records instead of sleeping. */
	readonly sleep?: (ms: number) => Promise<void>;
	/** One line per file that could not be released; defaults to stderr. Never thrown, never shown in the UI. */
	readonly log?: (message: string) => void;
}

/**
 * pi extension that releases cloud files the agent itself opened during an answer.
 *
 * A refresh releases the placeholders it downloads, but the agent's own tools (`bash`, `read`, `grep`, …) open
 * originals too, and nothing released those. This extension watches the files a run named or was shown: before
 * each call it stats the candidates a tool input names (an absolute `path`, the absolute words of a `bash`
 * command, the direct children of a directory candidate) and remembers the ones that are placeholders now;
 * after each result it does the same for the absolute paths a retrieval result printed. At `agent_end` it
 * re-stats what it remembered and releases each file that is now on disk. A file that was already local when
 * first seen is never watched, so only what this run hydrated is released. Without a probe or without a
 * releasable cloud root the extension does nothing; a failed release is logged once and never thrown.
 */
export function createCloudReleaseExtension(options: CloudReleaseExtensionOptions): ExtensionFactory {
	return (pi: ExtensionAPI): void => {
		const probe = options.probe;
		if (probe === undefined) return;
		const childCap = Math.max(0, options.childCap ?? DEFAULT_CLOUD_RELEASE_CHILD_CAP);
		const evictRetries = Math.max(0, options.evictRetries ?? DEFAULT_CLOUD_EVICT_RETRIES);
		const evictBackoffMs = options.evictBackoffMs ?? DEFAULT_CLOUD_EVICT_BACKOFF_MS;
		const sleep = options.sleep ?? (async (ms) => void (await sleepFor(ms)));
		const log = options.log ?? ((message: string) => process.stderr.write(`[cloud-release] ${message}\n`));

		/** Releasable roots, canonicalized and resolved once; a probe failure reads as no roots, never an error. */
		let rootsPromise: Promise<readonly string[]> | undefined;
		const releasableRoots = async (): Promise<readonly string[]> => {
			if (options.cloudRoots === undefined) return [];
			// A `network` root needs no release (its files are read over the wire and nothing is downloaded), so
			// only the directories holding placeholder-backed files are watched.
			rootsPromise ??= options
				.cloudRoots()
				.then((roots) =>
					roots
						.filter((root) => root.kind === "file-provider" || root.kind === "cloud-files")
						.map((root) => canonicalPath(root.path)),
				)
				.catch(() => []);
			return rootsPromise;
		};

		/** Placeholders first seen this run: a file that was already local when named is never in here. */
		let watched = new Set<string>();

		const underRoot = (path: string, roots: readonly string[]): boolean =>
			roots.some((root) => isPathInside(path, root));

		/** Canonical candidates from a tool input, with a directory contributing its direct child files. */
		const candidatesFrom = async (input: unknown): Promise<string[]> => {
			const roots = await releasableRoots();
			if (roots.length === 0) return [];
			const files: string[] = [];
			let children = 0;
			for (const candidate of absolutePathsIn(input)) {
				const path = canonicalPath(candidate);
				if (!underRoot(path, roots)) continue;
				let stats: Stats;
				try {
					stats = statSync(path);
				} catch {
					continue;
				}
				if (!stats.isDirectory()) {
					files.push(path);
					continue;
				}
				let entries: Dirent[];
				try {
					entries = readdirSync(path, { withFileTypes: true });
				} catch {
					continue;
				}
				for (const entry of entries) {
					if (children >= childCap) break;
					if (entry.isDirectory()) continue;
					const child = canonicalPath(join(path, entry.name));
					if (!underRoot(child, roots)) continue;
					files.push(child);
					children += 1;
				}
			}
			return files;
		};

		/** Remember the candidates that are placeholders right now, in one batched stat. */
		const remember = async (files: readonly string[]): Promise<void> => {
			if (files.length === 0) return;
			try {
				const dataless = await probe.datalessAmong(files);
				for (const file of files) if (dataless.has(file)) watched.add(file);
			} catch {
				// A probe that cannot answer leaves nothing remembered; the run continues.
			}
		};

		/** Release one file, retrying a refusal with doubling waits like the refresh's gate. */
		const release = async (path: string): Promise<CloudEvictResult> => {
			let result = await probe.evict(path);
			for (let retry = 0; !result.ok && retry < evictRetries; retry += 1) {
				if (evictBackoffMs > 0) await sleep(evictBackoffMs * 2 ** retry);
				result = await probe.evict(path);
			}
			return result;
		};

		pi.on("agent_start", () => {
			watched = new Set();
		});

		pi.on("tool_call", async (event) => {
			try {
				await remember(await candidatesFrom(event.input));
			} catch {
				// Remembering is best-effort: a failure here must never block the tool.
			}
		});

		pi.on("tool_result", async (event) => {
			try {
				const roots = await releasableRoots();
				if (roots.length === 0) return;
				const text = event.content
					.filter((item) => item.type === "text")
					.map((item) => item.text)
					.join("\n");
				const files = absolutePathsIn(text)
					.map((path) => canonicalPath(path))
					.filter((path) => underRoot(path, roots));
				await remember(files);
			} catch {
				// A result that cannot be scanned leaves the run unchanged.
			}
		});

		pi.on("agent_end", async () => {
			const files = [...watched];
			watched = new Set();
			if (files.length === 0) return;
			let dataless: ReadonlySet<string>;
			try {
				dataless = await probe.datalessAmong(files);
			} catch {
				// Without a fresh answer, leave every file: an evict on a file that may still be a placeholder is worse.
				return;
			}
			for (const file of files) {
				// Still a placeholder: the run never hydrated it, so there is nothing to release.
				if (dataless.has(file)) continue;
				try {
					const provider = await probe.providerOf(file);
					if (!probe.canRelease(provider)) continue;
					const result = await release(file);
					if (!result.ok) log(`could not release ${file}: ${result.message}`);
				} catch (error) {
					log(`could not release ${file}: ${error instanceof Error ? error.message : String(error)}`);
				}
			}
		});
	};
}
