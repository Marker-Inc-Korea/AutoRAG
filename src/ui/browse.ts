import { existsSync, readdirSync, type Stats, statSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { ConfigError } from "../cli/config.ts";

export interface BrowseEntry {
	readonly name: string;
	readonly path: string;
	readonly directory: boolean;
}

export interface BrowseResult {
	readonly path: string;
	readonly parent: string | null;
	readonly entries: readonly BrowseEntry[];
}

const MAX_ENTRIES = 400;

export function browseDirectory(inputPath: string | undefined): BrowseResult {
	const path = resolve(inputPath && inputPath.trim().length > 0 ? inputPath : homedir());
	if (!existsSync(path)) throw new ConfigError("Folder was not found.");
	let stat: Stats;
	try {
		stat = statSync(path);
	} catch {
		throw new ConfigError("Folder was not found.");
	}
	if (!stat.isDirectory()) throw new ConfigError("Path is not a folder.");

	let names: string[] = [];
	try {
		names = readdirSync(path);
	} catch {
		throw new ConfigError("Folder could not be read.");
	}

	const entries: BrowseEntry[] = [];
	for (const name of names) {
		if (name === "." || name === "..") continue;
		const child = join(path, name);
		let directory = false;
		try {
			directory = statSync(child).isDirectory();
		} catch {
			continue;
		}
		entries.push({ name, path: child, directory });
		if (entries.length >= MAX_ENTRIES) break;
	}
	entries.sort((a, b) => Number(b.directory) - Number(a.directory) || a.name.localeCompare(b.name));
	const parent = dirname(path);
	return {
		path,
		parent: parent === path ? null : parent,
		entries,
	};
}
