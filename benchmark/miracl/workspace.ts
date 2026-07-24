import { lstatSync, mkdirSync, readlinkSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, parse, relative, resolve, sep } from "node:path";
import { type ParsedMirrorEntry, saveMirrorIndex } from "../../src/mirror/index-store.ts";
import { parsedMirrorIndexPath, parsedOutputPath } from "../../src/mirror/paths.ts";
import type { CorpusDocument } from "./types.ts";

export interface BenchmarkWorkspace {
	readonly root: string;
	readonly mirrorFiles: readonly string[];
	readonly documentBySource: ReadonlyMap<string, string>;
}

interface MaterializedDocument {
	readonly document: CorpusDocument;
	readonly virtualPath: string;
}

interface ValidatedCorpus {
	readonly documents: readonly MaterializedDocument[];
	readonly documentBySource: ReadonlyMap<string, string>;
}

export interface BenchmarkDirectoryIdentity {
	readonly device: number;
	readonly inode: number;
}

const BENCHMARK_PARSER_NAME = "miracl-benchmark";
const DETERMINISTIC_UPDATED_AT = "1970-01-01T00:00:00.000Z";

export function materializeBenchmarkWorkspace(root: string, corpus: readonly CorpusDocument[]): BenchmarkWorkspace {
	const validated = validateCorpus(corpus);
	const canonicalRoot = validateNewWorkspaceRoot(root);
	mkdirSync(canonicalRoot, { mode: 0o700 });
	const ownedRoot = snapshotBenchmarkDirectory(canonicalRoot);

	try {
		const mirrorFiles: string[] = [];
		const entries: Record<string, ParsedMirrorEntry> = {};

		for (const { document, virtualPath } of validated.documents) {
			const markdown = `# ${document.title}\n\n${document.text}\n`;
			const outputPath = parsedOutputPath(canonicalRoot, virtualPath);
			ensureTrustedDirectory(dirname(outputPath), canonicalRoot, ownedRoot);
			assertTrustedDirectory(dirname(outputPath), canonicalRoot, ownedRoot);
			writeFileSync(outputPath, markdown, { encoding: "utf8", flag: "wx", mode: 0o600 });
			assertTrustedDirectory(dirname(outputPath), canonicalRoot, ownedRoot);
			mirrorFiles.push(outputPath);
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

		ensureTrustedDirectory(dirname(parsedMirrorIndexPath(canonicalRoot)), canonicalRoot, ownedRoot);
		assertTrustedDirectory(dirname(parsedMirrorIndexPath(canonicalRoot)), canonicalRoot, ownedRoot);
		saveMirrorIndex(canonicalRoot, { version: 1, entries });
		assertTrustedDirectory(dirname(parsedMirrorIndexPath(canonicalRoot)), canonicalRoot, ownedRoot);
		return { root: canonicalRoot, mirrorFiles, documentBySource: validated.documentBySource };
	} catch (error) {
		cleanupOwnedRoot(canonicalRoot, ownedRoot);
		throw error;
	}
}

export function materializeEmptyBenchmarkWorkspace(root: string): string {
	const canonicalRoot = validateNewWorkspaceRoot(root);
	mkdirSync(canonicalRoot, { mode: 0o700 });
	return canonicalRoot;
}

function validateCorpus(corpus: readonly CorpusDocument[]): ValidatedCorpus {
	const documentIds = new Set<string>();
	const virtualPaths = new Set<string>();
	const documentBySource = new Map<string, string>();
	const documents = corpus.map((document) => {
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
		documentBySource.set(virtualPath, document.documentId);
		return { document, virtualPath };
	});
	return { documents, documentBySource };
}

function validateNewWorkspaceRoot(root: string): string {
	if (root.trim().length === 0) {
		throw new Error("benchmark workspace root must not be blank");
	}
	assertBenchmarkPathOutsideAutorag(root);
	const absoluteRoot = resolve(root);
	if (pathEntryExists(absoluteRoot)) {
		throw new Error("benchmark workspace root must not already exist");
	}

	const canonicalParent = realpathSync(dirname(absoluteRoot));
	assertNoAutoragComponent(canonicalParent);
	const canonicalRoot = join(canonicalParent, basename(absoluteRoot));
	assertNoAutoragComponent(canonicalRoot);
	return canonicalRoot;
}

export function assertBenchmarkPathOutsideAutorag(path: string): void {
	assertNoAutoragComponent(path);
	const absolutePath = resolve(path);
	assertNoAutoragComponent(absolutePath);
	inspectExistingComponents(absolutePath);
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

function assertNoAutoragComponent(path: string): void {
	const components = path.split(/[\\/]+/u);
	if (components.some((component) => component.toLowerCase() === ".autorag")) {
		throw new Error("benchmark workspace root must not resolve inside an existing .autorag");
	}
}

function inspectExistingComponents(path: string, inspectedSymlinks = new Set<string>()): void {
	assertNoAutoragComponent(path);
	const pathRoot = parse(path).root;
	let current = pathRoot;
	for (const component of path.slice(pathRoot.length).split(sep).filter(Boolean)) {
		current = join(current, component);
		let stats: ReturnType<typeof lstatSync>;
		try {
			stats = lstatSync(current);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
			throw error;
		}
		if (stats.isSymbolicLink()) {
			if (!inspectedSymlinks.has(current)) {
				inspectedSymlinks.add(current);
				const target = resolve(dirname(current), readlinkSync(current));
				assertNoAutoragComponent(target);
				inspectExistingComponents(target, inspectedSymlinks);
			}
		}
		assertNoAutoragComponent(realpathSync(current));
	}
}

export function snapshotBenchmarkDirectory(path: string): BenchmarkDirectoryIdentity {
	const stats = lstatSync(path);
	if (!stats.isDirectory() || stats.isSymbolicLink()) {
		throw new Error("benchmark workspace root changed");
	}
	return { device: stats.dev, inode: stats.ino };
}

export function assertBenchmarkDirectoryIdentity(root: string, identity: BenchmarkDirectoryIdentity): void {
	const current = snapshotBenchmarkDirectory(root);
	if (current.device !== identity.device || current.inode !== identity.inode) {
		throw new Error("benchmark workspace root changed");
	}
}

function ensureTrustedDirectory(path: string, root: string, identity: BenchmarkDirectoryIdentity): void {
	assertContainedPath(path, root);
	assertBenchmarkDirectoryIdentity(root, identity);
	mkdirSync(path, { recursive: true, mode: 0o700 });
	assertTrustedDirectory(path, root, identity);
}

function assertTrustedDirectory(path: string, root: string, identity: BenchmarkDirectoryIdentity): void {
	assertContainedPath(path, root);
	assertBenchmarkDirectoryIdentity(root, identity);
	const descendant = relative(root, path);
	let current = root;
	for (const component of descendant.split(sep).filter(Boolean)) {
		current = join(current, component);
		const stats = lstatSync(current);
		if (!stats.isDirectory() || stats.isSymbolicLink()) {
			throw new Error("benchmark workspace directory changed during materialization");
		}
	}
	if (realpathSync(path) !== path) {
		throw new Error("benchmark workspace directory changed during materialization");
	}
	assertBenchmarkDirectoryIdentity(root, identity);
}

function assertContainedPath(path: string, root: string): void {
	const descendant = relative(root, path);
	if (descendant === ".." || descendant.startsWith(`..${sep}`) || isAbsolute(descendant)) {
		throw new Error("benchmark workspace directory changed during materialization");
	}
}

function cleanupOwnedRoot(root: string, identity: BenchmarkDirectoryIdentity): void {
	try {
		const current = snapshotBenchmarkDirectory(root);
		if (current.device !== identity.device || current.inode !== identity.inode) return;
	} catch {
		return;
	}
	rmSync(root, { recursive: true, force: true });
}
