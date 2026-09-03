import { accessSync, constants, existsSync, realpathSync } from "node:fs";
import { basename, delimiter, join, normalize } from "node:path";
import { matchesVirtualPathScope } from "../retrieval/scope.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import type { MinSyncQueryMode } from "./client.ts";
import { MinSyncClient } from "./client.ts";
import { type EnsureMinSyncBinaryOptions, ensureMinSyncBinary, executableName } from "./installer.ts";
import { minSyncWorkspaceRoot } from "./paths.ts";
import type { MinSyncEmbedderConfig, MinSyncSyncResult } from "./types.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "./workspace.ts";

export interface MinSyncVectorMethodOptions {
	readonly root: string;
	/** Test/embedded override; CLI config intentionally does not persist this. */
	readonly binaryPath?: string;
	readonly workspacePath?: string;
	readonly installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
	readonly autoInstall?: boolean;
	readonly embedder?: MinSyncEmbedderConfig;
	readonly maxChunkSize?: number;
	readonly mode?: MinSyncQueryMode;
}

export type MinSyncHybridMethodOptions = Omit<MinSyncVectorMethodOptions, "mode">;

/** Degrade result returned when no binary can be resolved. */
function degrade(workspacePath: string, reason: string): MinSyncSyncResult {
	return { ok: false, synced: 0, workspacePath, reason };
}

/** Resolve the `minsync` executable from PATH directories. Returns the first match or undefined. */
function lookupInPath(env: NodeJS.ProcessEnv): string | undefined {
	const pathEnv = env.PATH;
	if (typeof pathEnv !== "string" || pathEnv.length === 0) return undefined;
	const execName = executableName(process.platform);
	for (const dir of pathEnv.split(delimiter)) {
		if (dir.length === 0) continue;
		const candidate = join(dir, execName);
		try {
			accessSync(candidate, constants.X_OK);
			return candidate;
		} catch {
			// not executable in this dir, continue
		}
	}
	return undefined;
}

export class MinSyncVectorMethod implements RetrievalMethod {
	private readonly root: string;
	private readonly binaryPath: string | undefined;
	private readonly workspacePath: string;
	private readonly installer: Omit<EnsureMinSyncBinaryOptions, "root"> | undefined;
	private readonly autoInstall: boolean;
	private readonly embedder: MinSyncEmbedderConfig | undefined;
	private readonly maxChunkSize: number | undefined;
	private readonly mode: MinSyncQueryMode;

	constructor(options: MinSyncVectorMethodOptions) {
		this.root = options.root;
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(options.root);
		this.installer = options.installer;
		this.autoInstall = options.autoInstall ?? true;
		this.embedder = options.embedder;
		this.maxChunkSize = options.maxChunkSize;
		this.mode = options.mode ?? "vector";
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "minsync",
			type: this.mode === "bm25" ? "bm25" : this.mode === "hybrid" ? "hybrid" : "vector",
			description:
				this.mode === "bm25"
					? "MinSync-backed BM25 lexical retrieval over parsed markdown mirrors"
					: this.mode === "hybrid"
						? "MinSync-backed hybrid retrieval over parsed markdown mirrors"
						: "MinSync-backed semantic vector retrieval over parsed markdown mirrors",
			status: "active",
			capabilities: [
				...(this.mode === "bm25" ? ["lexical"] : this.mode === "hybrid" ? ["semantic", "lexical"] : ["semantic"]),
				"incremental",
				"parsed-mirrors",
				"virtual-paths",
			],
		};
	}

	async sync(): Promise<MinSyncSyncResult> {
		syncMinSyncWorkspace(this.root, { workspacePath: this.workspacePath });
		const binaryResult = await this.resolveBinary();
		if (binaryResult === undefined) {
			return degrade(this.workspacePath, "missing-binary");
		}
		if (typeof binaryResult === "string") {
			const client = new MinSyncClient({
				binaryPath: binaryResult,
				workspacePath: this.workspacePath,
				embedder: this.embedder,
				maxChunkSize: this.maxChunkSize,
			});
			return client.sync();
		}
		// install-failed degrade result
		return binaryResult;
	}

	/** MinSync is resolved from PATH, then the workspace cache. */
	isBinaryMissing(): boolean {
		return this.binaryPath !== undefined && !existsSync(this.binaryPath);
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const topK = options.topK ?? 20;
		const queryK = options.scope ? Math.min(Math.max(topK * 5, topK + 20), 100) : topK;
		const byPath = buildMinSyncPathMap(this.root, this.workspacePath);
		const binaryResult = await this.resolveBinary();
		if (binaryResult === undefined || typeof binaryResult !== "string") return [];
		const client = new MinSyncClient({
			binaryPath: binaryResult,
			workspacePath: this.workspacePath,
			embedder: this.embedder,
			maxChunkSize: this.maxChunkSize,
		});
		const hits = await client.query(query, queryK, this.mode);
		const results: RetrievalResult[] = [];
		for (const hit of hits) {
			const entry = byPath.get(hit.path);
			if (!entry || !matchesVirtualPathScope(entry.virtualPath, options.scope)) continue;
			const chunkId = `minsync:${entry.virtualPath}:${basename(hit.path)}`;
			results.push({
				id: chunkId,
				content: hit.text,
				source: realpathSync(normalize(entry.sourcePath)),
				score: hit.score,
				metadata: {
					method: this.mode === "vector" ? "minsync" : `minsync-${this.mode}`,
					virtualPath: entry.virtualPath,
					minsyncChunkId: chunkId,
					minsyncPath: hit.path,
				},
			});
			if (results.length >= topK) break;
		}
		return results;
	}

	/**
	 * Resolve the MinSync CLI from the user's global PATH first, then the
	 * workspace cache, and finally the verified auto-install fallback.
	 *
	 * Returns: string path on success, undefined for missing-binary, or a MinSyncSyncResult
	 * for install-failed.
	 */
	protected async resolveBinary(): Promise<string | undefined | MinSyncSyncResult> {
		if (this.binaryPath && existsSync(this.binaryPath)) return this.binaryPath;
		if (this.binaryPath !== undefined) return undefined;
		const pathBinary = lookupInPath(process.env);
		if (pathBinary) return pathBinary;
		const cachedBinary = join(this.root, ".autorag", "bin", executableName(process.platform));
		if (existsSync(cachedBinary)) return cachedBinary;
		if (this.autoInstall) {
			try {
				const installed = await ensureMinSyncBinary({ ...this.installer, root: this.root });
				return installed.binaryPath;
			} catch {
				return degrade(this.workspacePath, "install-failed");
			}
		}
		return undefined;
	}
}

export class MinSyncHybridMethod extends MinSyncVectorMethod {
	constructor(options: MinSyncHybridMethodOptions) {
		super({ ...options, mode: "hybrid" });
	}

	override describe(): RetrievalMethodDescriptor {
		return {
			name: "hybrid",
			type: "hybrid",
			description: "MinSync-backed hybrid BM25+vector retrieval over shared CDC chunks",
			status: "active",
			capabilities: ["semantic", "lexical", "incremental", "parsed-mirrors", "virtual-paths"],
		};
	}
}
