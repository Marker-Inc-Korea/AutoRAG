import { accessSync, constants, existsSync } from "node:fs";
import { basename, delimiter, join } from "node:path";
import { matchesVirtualPathScope } from "../retrieval/scope.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import { MinSyncClient } from "./client.ts";
import { type EnsureMinSyncBinaryOptions, ensureMinSyncBinary, executableName } from "./installer.ts";
import { minSyncWorkspaceRoot } from "./paths.ts";
import type { MinSyncEmbedderConfig, MinSyncSyncResult } from "./types.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "./workspace.ts";

export interface MinSyncVectorMethodOptions {
	readonly root: string;
	readonly binaryPath?: string;
	readonly workspacePath?: string;
	readonly installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
	readonly autoInstall?: boolean;
	readonly embedder?: MinSyncEmbedderConfig;
}

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

	constructor(options: MinSyncVectorMethodOptions) {
		this.root = options.root;
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(options.root);
		this.installer = options.installer;
		this.autoInstall = options.autoInstall ?? true;
		this.embedder = options.embedder;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "minsync",
			type: "vector",
			description: "MinSync-backed semantic vector retrieval over parsed markdown mirrors",
			status: "active",
			capabilities: ["semantic", "incremental", "parsed-mirrors", "virtual-paths"],
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
			});
			return client.sync();
		}
		// install-failed degrade result
		return binaryResult;
	}

	/** True when a configured binary path is missing (an explicit degraded state). */
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
		});
		const hits = await client.query(query, queryK);
		const results: RetrievalResult[] = [];
		for (const hit of hits) {
			const entry = byPath.get(hit.path);
			if (!entry || !matchesVirtualPathScope(entry.virtualPath, options.scope)) continue;
			results.push({
				id: `minsync:${entry.virtualPath}:${basename(hit.path)}`,
				content: hit.text,
				source: entry.virtualPath,
				score: hit.score,
				metadata: { method: "minsync" },
			});
			if (results.length >= topK) break;
		}
		return results;
	}

	/**
	 * Resolve the minsync binary path following the priority chain:
	 * 1. explicit binaryPath if set and exists
	 * 2. PATH lookup for `minsync`
	 * 3. cached join(root, '.autorag', 'bin', execName) if exists
	 * 4. if autoInstall===true, try ensureMinSyncBinary in try/catch => install-failed degrade
	 * 5. else return undefined (missing-binary degrade)
	 *
	 * Returns: string path on success, undefined for missing-binary, or a MinSyncSyncResult
	 * for install-failed.
	 */
	private async resolveBinary(): Promise<string | undefined | MinSyncSyncResult> {
		// 1. explicit binaryPath
		if (this.binaryPath && existsSync(this.binaryPath)) {
			return this.binaryPath;
		}
		// 2. PATH lookup
		const pathBinary = lookupInPath(process.env);
		if (pathBinary) return pathBinary;
		// 3. cached bin
		const cachedBinary = join(this.root, ".autorag", "bin", executableName(process.platform));
		if (existsSync(cachedBinary)) return cachedBinary;
		// 4. autoInstall
		if (this.autoInstall) {
			try {
				const installed = await ensureMinSyncBinary({ ...this.installer, root: this.root });
				return installed.binaryPath;
			} catch {
				return degrade(this.workspacePath, "install-failed");
			}
		}
		// 5. missing-binary
		return undefined;
	}
}
