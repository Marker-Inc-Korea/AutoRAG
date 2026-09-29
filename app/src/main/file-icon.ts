/**
 * OS-provided file icons for the tile left of a file name.
 *
 * macOS produces real Finder-style thumbnails: one batched
 * `qlmanage -t -s 64 -o <tmpdir> <paths...>` covers a whole folder, the same
 * QuickLook family the Space preview already uses. Other platforms fall back to
 * Electron's `app.getFileIcon`, the OS's per-type icon. Generation failures
 * degrade to "no icon" (the caller keeps the letter tile) and are logged
 * verbatim.
 */

import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { errorMessage, hasErrorCode } from "./fs-entry";

/** Thumbnail edge in px; 64 covers an 18px tile at 3x. */
export const ICON_SIZE = 64;
/** One listing never generates more than this many icons. */
export const ICON_LIMIT = 240;

const ICON_TIMEOUT_MS = 20000;
const ICON_MAX_BUFFER = 1 << 20;
const THUMBNAIL_SUFFIX = ".png";

export interface IconTarget {
	/** Absolute OS path. */
	readonly path: string;
	/** Cache identity: a changed file must not reuse a stale thumbnail. */
	readonly modifiedAt: string;
}

export interface IconProvider {
	/** path -> PNG data URL, for the subset the OS produced an icon for. */
	icons(targets: readonly IconTarget[]): Promise<ReadonlyMap<string, string>>;
}

export interface IconProviderDeps {
	/** Defaults to process.platform. */
	readonly platform?: NodeJS.Platform;
	/** macOS thumbnail generation into outDir; defaults to qlmanage. */
	readonly generate?: (paths: readonly string[], outDir: string) => Promise<void>;
	/** OS type icon as a data URL, or null; defaults to Electron app.getFileIcon. */
	readonly systemIcon?: (path: string) => Promise<string | null>;
	/** Cap for a single call; defaults to ICON_LIMIT. */
	readonly limit?: number;
	/** Defaults to console.warn. */
	readonly warn?: (message: string) => void;
}

function cacheKey(target: IconTarget): string {
	return `${target.path}\u0000${target.modifiedAt}`;
}

function defaultGenerate(paths: readonly string[], outDir: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const args = ["-t", "-s", String(ICON_SIZE), "-o", outDir, ...paths];
		execFile("/usr/bin/qlmanage", args, { timeout: ICON_TIMEOUT_MS, maxBuffer: ICON_MAX_BUFFER }, (error) => {
			if (error) {
				reject(error);
				return;
			}
			resolve();
		});
	});
}

async function defaultSystemIcon(path: string): Promise<string | null> {
	const { app } = await import("electron");
	const image = await app.getFileIcon(path, { size: "normal" });
	return image.isEmpty() ? null : image.toDataURL();
}

function dataUrl(png: Buffer): string {
	return `data:image/png;base64,${png.toString("base64")}`;
}

export function createIconProvider(deps: IconProviderDeps = {}): IconProvider {
	const platform = deps.platform ?? process.platform;
	const generate = deps.generate ?? defaultGenerate;
	const systemIcon = deps.systemIcon ?? defaultSystemIcon;
	const limit = deps.limit ?? ICON_LIMIT;
	const warn = deps.warn ?? ((message: string) => console.warn(message));
	const cache = new Map<string, string>();

	async function thumbnails(targets: readonly IconTarget[]): Promise<ReadonlyMap<string, string>> {
		const outDir = await mkdtemp(join(tmpdir(), "autorag-icons-"));
		try {
			await generate(
				targets.map((target) => target.path),
				outDir,
			);
			const produced = new Map<string, string>();
			for (const target of targets) {
				const file = join(outDir, `${basename(target.path)}${THUMBNAIL_SUFFIX}`);
				try {
					produced.set(target.path, dataUrl(await readFile(file)));
				} catch (error) {
					// No thumbnail for this file: qlmanage simply writes nothing.
					if (!hasErrorCode(error, "ENOENT")) throw error;
				}
			}
			return produced;
		} finally {
			await rm(outDir, { recursive: true, force: true });
		}
	}

	async function systemIcons(targets: readonly IconTarget[]): Promise<ReadonlyMap<string, string>> {
		const produced = new Map<string, string>();
		for (const target of targets) {
			const url = await systemIcon(target.path);
			if (url !== null) produced.set(target.path, url);
		}
		return produced;
	}

	return {
		async icons(targets) {
			const found = new Map<string, string>();
			const missing: IconTarget[] = [];
			for (const target of targets) {
				const cached = cache.get(cacheKey(target));
				if (cached !== undefined) {
					found.set(target.path, cached);
					continue;
				}
				if (missing.length < limit) missing.push(target);
			}
			if (missing.length === 0) return found;

			let produced: ReadonlyMap<string, string>;
			try {
				produced = platform === "darwin" ? await thumbnails(missing) : await systemIcons(missing);
			} catch (error) {
				warn(`file-icon: icon generation failed for ${missing.length} file(s): ${errorMessage(error)}`);
				return found;
			}
			for (const target of missing) {
				const url = produced.get(target.path);
				if (url === undefined) continue;
				cache.set(cacheKey(target), url);
				found.set(target.path, url);
			}
			return found;
		},
	};
}
