import { readFileSync } from "node:fs";
import { builtinModules } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const manifest: unknown = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const PRODUCTION = new Set(
	typeof manifest === "object" &&
		manifest !== null &&
		"dependencies" in manifest &&
		typeof manifest.dependencies === "object"
		? Object.keys(manifest.dependencies ?? {})
		: [],
);
const BUILTIN = new Set(builtinModules.flatMap((name) => [name, `node:${name}`]));

/**
 * `zod` is imported by the MCP server but never declared; it reaches a published install only because
 * `@modelcontextprotocol/server` depends on it (checked below, so this cannot go stale unnoticed).
 */
const TRANSITIVE_THROUGH_SERVER = new Set(["zod"]);

/** The package a bare specifier resolves to: `@scope/name/sub` -> `@scope/name`, `name/sub` -> `name`. */
function packageOf(specifier: string): string {
	const parts = specifier.split("/");
	return specifier.startsWith("@") ? parts.slice(0, 2).join("/") : (parts[0] ?? specifier);
}

/** Static and dynamic import specifiers in one source file. Type-only imports are erased by the bundler. */
function importsOf(source: string): string[] {
	const found: string[] = [];
	// A specifier must look like a module name; this skips code that merely appears inside a template string.
	const isModule = (specifier: string): boolean =>
		/^(?:\.{1,2}\/|node:|@[a-z0-9][\w.-]*\/[\w.-]+|[a-z][\w.-]*)/.test(specifier) && !/\s/.test(specifier);
	for (const match of source.matchAll(/^import\s+(type\s+)?[^"'`;]*?from\s+["']([^"']+)["']/gm)) {
		if (match[1] === undefined && match[2] !== undefined) found.push(match[2]);
	}
	for (const match of source.matchAll(/^import\s+["']([^"']+)["']/gm))
		if (match[1] !== undefined) found.push(match[1]);
	for (const match of source.matchAll(/^[^\n"'`]*\bimport\(\s*["']([^"']+)["']\s*\)/gm))
		if (match[1] !== undefined) found.push(match[1]);
	return found.filter(isModule);
}

/** Every package the entry loads, following relative imports through the repo's own sources. */
function packagesReachableFrom(entry: string): Map<string, string> {
	const packages = new Map<string, string>();
	const seen = new Set<string>();
	const queue = [entry];
	for (let file = queue.pop(); file !== undefined; file = queue.pop()) {
		if (seen.has(file)) continue;
		seen.add(file);
		for (const specifier of importsOf(readFileSync(file, "utf8"))) {
			if (specifier.startsWith(".")) queue.push(resolve(dirname(file), specifier));
			else if (!BUILTIN.has(specifier) && !packages.has(packageOf(specifier)))
				packages.set(packageOf(specifier), file);
		}
	}
	return packages;
}

describe("autorag-mcp entry", () => {
	it("imports only packages a published install has (dependencies, never devDependencies)", () => {
		const reachable = packagesReachableFrom(join(ROOT, "src/mcp/index.ts"));
		expect(reachable.size).toBeGreaterThan(0);
		const missing = [...reachable]
			.filter(([name]) => !PRODUCTION.has(name) && !TRANSITIVE_THROUGH_SERVER.has(name))
			.map(([name, from]) => `${name} (from ${from.replace(`${ROOT}/`, "")})`);
		expect(missing).toEqual([]);
	});

	it("still gets zod from a declared dependency", () => {
		const server = JSON.parse(
			readFileSync(join(ROOT, "node_modules/@modelcontextprotocol/server/package.json"), "utf8"),
		);
		expect(Object.keys(server.dependencies ?? {})).toContain("zod");
	});
});
