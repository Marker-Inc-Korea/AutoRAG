import {
	lstatSync,
	mkdirSync,
	realpathSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { saveMirrorIndex, type ParsedMirrorEntry } from "../../src/mirror/index-store.ts";
import { parsedOutputPath } from "../../src/mirror/paths.ts";
import type { CorpusDocument } from "./types.ts";

export interface BenchmarkWorkspace {
	readonly root: string;
	readonly mirrorFiles: readonly string[];
	readonly documentBySource: ReadonlyMap<string, string>;
}

interface MaterializedDocument {
	readonly document: CorpusDocument;
	readonly virtualPath: string;
	readonly markdown: string;
}

const BENCHMARK_PARSER_NAME = "miracl-benchmark";
const DETERMINISTIC_UPDATED_AT = "1970-01-01T00:00:00.000Z";

export function materializeBenchmarkWorkspace(
	root: string,
	corpus: readonly CorpusDocument[],
): BenchmarkWorkspace {
	const documents = validateCorpus(corpus);
	const canonicalRoot = validateNewWorkspaceRoot(root);
	mkdirSync(canonicalRoot, { mode: 0o700 });

	try {
		const mirrorFiles: string[] = [];
		const documentBySource = new Map<string, string>();
		const entries: Record<string, ParsedMirrorEntry> = {};

		for (const { document, virtualPath, markdown } of documents) {
			const outputPath = parsedOutputPath(canonicalRoot, virtualPath);
			mkdirSync(dirname(outputPath), { recursive: true, mode: 0o700 });
			writeFileSync(outputPath, markdown, { encoding: "utf8", flag: "wx", mode: 0o600 });
			mirrorFiles.push(outputPath);
			documentBySource.set(virtualPath, document.documentId);
			entries[virtualPath] = {
				virtualPath,
				sourcePath: virtualPath,
				outputPath,
				parserName: BENCHMARK_PARSER_NAME,
				sourceMtimeNs: 0,
				sourceSizeBytes: Buffer.byteLength(markdown, "utf8"),
				updatedAt: DETERMINISTIC_UPDATED_AT,
			};
		}

		saveMirrorIndex(canonicalRoot, { version: 1, entries });
		return { root: canonicalRoot, mirrorFiles, documentBySource };
	} catch (error) {
		rmSync(canonicalRoot, { recursive: true, force: true });
		throw error;
	}
}

function validateCorpus(corpus: readonly CorpusDocument[]): MaterializedDocument[] {
	const documentIds = new Set<string>();
	const virtualPaths = new Set<string>();
	return corpus.map((document) => {
		if (document.documentId.trim().length === 0) {
			throw new Error("MIRACL document id must not be blank");
		}
		if (documentIds.has(document.documentId)) {
			throw new Error(`duplicate MIRACL document id: ${document.documentId}`);
		}
		documentIds.add(document.documentId);

		const virtualPath = `/miracl/${encodeURIComponent(document.documentId)}.md`;
		if (virtualPaths.has(virtualPath)) {
			throw new Error("MIRACL document ids must encode to unique virtual paths");
		}
		virtualPaths.add(virtualPath);
		return {
			document,
			virtualPath,
			markdown: `# ${document.title}\n\n${document.text}\n`,
		};
	});
}

function validateNewWorkspaceRoot(root: string): string {
	if (root.trim().length === 0) {
		throw new Error("benchmark workspace root must not be blank");
	}
	const absoluteRoot = resolve(root);
	if (pathEntryExists(absoluteRoot)) {
		throw new Error("benchmark workspace root must not already exist");
	}

	const canonicalParent = realpathSync(dirname(absoluteRoot));
	const canonicalRoot = join(canonicalParent, basename(absoluteRoot));
	if (isInsideExistingAutorag(canonicalRoot)) {
		throw new Error("benchmark workspace root must not resolve inside an existing .autorag");
	}
	return canonicalRoot;
}

function pathEntryExists(path: string): boolean {
	try {
		lstatSync(path);
		return true;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
		throw error;
	}
}

function isInsideExistingAutorag(path: string): boolean {
	let current = dirname(path);
	while (true) {
		if (basename(current) === ".autorag" && pathEntryExists(current)) {
			return true;
		}
		const parent = dirname(current);
		if (parent === current) return false;
		current = parent;
	}
}
