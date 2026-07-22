import { existsSync, readdirSync, readFileSync } from "node:fs";
import { extname, join } from "node:path";
import { parse as parseYaml } from "yaml";
import type { StoreManifest } from "./types.ts";

function isValidManifest(data: unknown): data is StoreManifest {
	if (typeof data !== "object" || data === null) return false;
	const obj = data as Record<string, unknown>;
	return (
		typeof obj.name === "string" &&
		obj.name.length > 0 &&
		typeof obj.description === "string" &&
		obj.description.length > 0 &&
		typeof obj.type === "string" &&
		["vector", "bm25", "hybrid", "visual"].includes(obj.type)
	);
}

export function loadManifest(filePath: string): StoreManifest {
	const content = readFileSync(filePath, "utf-8");
	const ext = extname(filePath).toLowerCase();
	let data: unknown;
	if (ext === ".yaml" || ext === ".yml") {
		data = parseYaml(content);
	} else if (ext === ".json") {
		data = JSON.parse(content);
	} else {
		throw new Error(`Unsupported manifest format: ${ext}`);
	}
	if (!isValidManifest(data)) {
		throw new Error(`Invalid manifest at ${filePath}: missing required fields (name, description, type)`);
	}
	return {
		name: data.name,
		description: data.description,
		type: data.type,
		dataDescription: typeof data.dataDescription === "string" ? data.dataDescription : "",
		contentTypes: Array.isArray(data.contentTypes) ? (data.contentTypes as string[]) : [],
		config: typeof data.config === "object" && data.config !== null ? (data.config as Record<string, unknown>) : {},
	};
}

export function loadManifests(dirPath: string): StoreManifest[] {
	if (!existsSync(dirPath)) {
		return [];
	}
	let entries: string[];
	try {
		entries = readdirSync(dirPath);
	} catch {
		return [];
	}
	const manifests: StoreManifest[] = [];
	for (const entry of entries) {
		const ext = extname(entry).toLowerCase();
		if (ext !== ".yaml" && ext !== ".yml" && ext !== ".json") continue;
		const filePath = join(dirPath, entry);
		try {
			const manifest = loadManifest(filePath);
			manifests.push(manifest);
		} catch (err) {
			console.warn(`[AutoRAG] Skipping invalid manifest ${filePath}: ${(err as Error).message}`);
		}
	}
	return manifests;
}
