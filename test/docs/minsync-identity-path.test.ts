import { readFileSync } from "node:fs";
import { basename, join, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { EMBEDDING_IDENTITY_FILE, MINSYNC_CONFIG_DIR, MINSYNC_CONFIG_FILE } from "../../src/minsync/embedder-config.ts";
import { minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));

/** Docs that describe the MinSync workspace layout for operators. */
const DOCS = ["docs/minsync-setup.md", "docs/embedding-runtime.md"] as const;

const WORKSPACE = "<workspace>";

function toDocumentedPath(path: string): string {
	return path.split(sep).join("/");
}

/**
 * The MinSync state directory the product uses when no explicit
 * `minSync.workspacePath` override is configured, expressed the way docs write it.
 */
const stateDir = toDocumentedPath(join(minSyncWorkspaceRoot(WORKSPACE), MINSYNC_CONFIG_DIR));

/** Sibling files MinSync keeps next to the identity record. */
const CURSOR_FILE = "cursor.json";

const stateFiles = [EMBEDDING_IDENTITY_FILE, CURSOR_FILE, MINSYNC_CONFIG_FILE] as const;

/** Every `<workspace>/...` path literal a doc tells an operator to inspect. */
function documentedWorkspacePaths(doc: string): string[] {
	return [...doc.matchAll(/<workspace>\/[^\s`)"'<>]+/g)].map((match) => match[0]);
}

function statePathsFor(doc: string): string[] {
	return documentedWorkspacePaths(doc).filter((path) => path.startsWith(`${stateDir}/`));
}

describe("documented MinSync state paths", () => {
	it("resolves the product-default state directory from the MinSync code contract", () => {
		expect(stateDir).toBe("<workspace>/.autorag/minsync/.minsync");
	});

	it.each(DOCS)("%s documents the embedding identity at the path the code writes", (docPath) => {
		const doc = readFileSync(join(repoRoot, docPath), "utf8");

		expect(documentedWorkspacePaths(doc).filter((path) => path.endsWith(EMBEDDING_IDENTITY_FILE))).toEqual([
			`${stateDir}/${EMBEDDING_IDENTITY_FILE}`,
		]);
	});

	it.each(DOCS)("%s documents cursor and config as siblings of the identity record", (docPath) => {
		const doc = readFileSync(join(repoRoot, docPath), "utf8");

		expect(new Set(statePathsFor(doc))).toEqual(new Set(stateFiles.map((file) => `${stateDir}/${file}`)));
	});

	it.each(DOCS)("%s never places MinSync state files directly under <workspace>/.minsync", (docPath) => {
		const doc = readFileSync(join(repoRoot, docPath), "utf8");
		const legacyStateDir = `${WORKSPACE}/.minsync`;

		expect(
			documentedWorkspacePaths(doc).filter(
				(path) => path.startsWith(`${legacyStateDir}/`) && stateFiles.includes(basename(path) as never),
			),
		).toEqual([]);
	});
});
