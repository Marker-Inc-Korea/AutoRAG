import { basename } from "node:path";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import { MinSyncClient } from "./client.ts";
import { type EnsureMinSyncBinaryOptions, ensureMinSyncBinary } from "./installer.ts";
import { minSyncWorkspaceRoot } from "./paths.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "./workspace.ts";

export interface MinSyncVectorMethodOptions {
	readonly root: string;
	readonly binaryPath?: string;
	readonly workspacePath?: string;
	readonly installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
}

export class MinSyncVectorMethod implements RetrievalMethod {
	private readonly root: string;
	private readonly binaryPath: string | undefined;
	private readonly workspacePath: string;
	private readonly installer: Omit<EnsureMinSyncBinaryOptions, "root"> | undefined;

	constructor(options: MinSyncVectorMethodOptions) {
		this.root = options.root;
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(options.root);
		this.installer = options.installer;
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

	async sync(): Promise<ReturnType<MinSyncClient["sync"]>> {
		syncMinSyncWorkspace(this.root, { workspacePath: this.workspacePath });
		const client = await this.client();
		return client.sync();
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const topK = options.topK ?? 20;
		const byPath = buildMinSyncPathMap(this.root, this.workspacePath);
		const client = await this.client();
		const hits = await client.query(query, topK);
		const results: RetrievalResult[] = [];
		for (const hit of hits) {
			const entry = byPath.get(hit.path);
			if (!entry) continue;
			results.push({
				id: `minsync:${entry.virtualPath}:${basename(hit.path)}`,
				content: hit.text,
				source: entry.virtualPath,
				score: hit.score,
				metadata: { method: "minsync" },
			});
		}
		return results;
	}

	private async client(): Promise<MinSyncClient> {
		const binaryPath =
			this.binaryPath ?? (await ensureMinSyncBinary({ ...this.installer, root: this.root })).binaryPath;
		return new MinSyncClient({ binaryPath, workspacePath: this.workspacePath });
	}
}
