import { type CloudFileProbe, type CloudProvider, defaultCloudProbe } from "../filesystem/cloud-placeholder.ts";

/** Placeholders one refresh may download before the rest wait for the next refresh. */
export const DEFAULT_CLOUD_MAX_FILES = 200;

/** Placeholder bytes one refresh may download. The first file is always allowed so a large one cannot starve. */
export const DEFAULT_CLOUD_MAX_BYTES = 512 * 1024 * 1024;

export interface CloudGateOptions {
	readonly probe?: CloudFileProbe;
	readonly maxFiles?: number;
	readonly maxBytes?: number;
	readonly evictAfterParse?: boolean;
}

/**
 * What a refresh does with one file that needs parsing:
 * - `local`: not a placeholder; read it as always.
 * - `download`: an iCloud placeholder inside the budget; read it, then release it again.
 * - `skip`: a placeholder AutoRAG cannot release again (Google Drive, OneDrive, ...); do not read it.
 * - `defer`: an iCloud placeholder over this refresh's budget; leave it for the next refresh.
 */
export type CloudDecision =
	| { readonly kind: "local" }
	| { readonly kind: "download" }
	| { readonly kind: "skip"; readonly provider: CloudProvider | undefined }
	| { readonly kind: "defer" };

const LOCAL: CloudDecision = { kind: "local" };

/**
 * One refresh's view of cloud placeholders. Reading a placeholder downloads it, so the refresh asks here
 * before every read; without a probe (non-macOS) every file is `local` and nothing changes.
 */
export class CloudGate {
	private readonly probe: CloudFileProbe | undefined;
	private readonly maxFiles: number;
	private readonly maxBytes: number;
	private readonly providerByDirectory = new Map<string, Promise<CloudProvider | undefined>>();
	private placeholders: ReadonlySet<string> = new Set();
	private downloadedFiles = 0;
	private downloadedBytes = 0;
	/** Whether a downloaded iCloud file is released again after it is parsed. */
	readonly evictAfterParse: boolean;

	constructor(options: CloudGateOptions = {}) {
		this.probe = options.probe ?? defaultCloudProbe();
		this.maxFiles = options.maxFiles ?? DEFAULT_CLOUD_MAX_FILES;
		this.maxBytes = options.maxBytes ?? DEFAULT_CLOUD_MAX_BYTES;
		this.evictAfterParse = options.evictAfterParse !== false;
	}

	/** Look up, in one batch, which of the files about to be read are placeholders. */
	async classify(paths: readonly string[]): Promise<void> {
		if (this.probe === undefined || paths.length === 0) return;
		this.placeholders = await this.probe.datalessAmong(paths);
	}

	isPlaceholder(path: string): boolean {
		return this.placeholders.has(path);
	}

	/** Decide, and when the answer is `download`, spend this refresh's budget on `sizeBytes`. */
	async decide(path: string, sizeBytes: number, directory: string): Promise<CloudDecision> {
		if (this.probe === undefined || !this.placeholders.has(path)) return LOCAL;
		const provider = await this.providerIn(directory, path);
		if (provider !== "icloud") return { kind: "skip", provider };
		const overFiles = this.downloadedFiles >= this.maxFiles;
		const overBytes = this.downloadedFiles > 0 && this.downloadedBytes + sizeBytes > this.maxBytes;
		if (overFiles || overBytes) return { kind: "defer" };
		this.downloadedFiles += 1;
		this.downloadedBytes += sizeBytes;
		return { kind: "download" };
	}

	/** Release the local copy of a file this refresh downloaded. */
	async release(path: string): Promise<{ readonly ok: true } | { readonly ok: false; readonly message: string }> {
		if (this.probe === undefined) return { ok: true };
		return this.probe.evict(path);
	}

	/** Providers are resolved once per directory: a File Provider domain covers the whole subtree. */
	private providerIn(directory: string, path: string): Promise<CloudProvider | undefined> {
		const probe = this.probe;
		if (probe === undefined) return Promise.resolve(undefined);
		let provider = this.providerByDirectory.get(directory);
		if (provider === undefined) {
			provider = probe.providerOf(path);
			this.providerByDirectory.set(directory, provider);
		}
		return provider;
	}
}
