import { setTimeout as sleepFor } from "node:timers/promises";
import { type CloudFileProbe, type CloudProvider, defaultCloudProbe } from "../filesystem/cloud-placeholder.ts";

/** Placeholders one refresh may download before the rest wait for the next refresh. */
export const DEFAULT_CLOUD_MAX_FILES = 200;

/** Placeholder bytes one refresh may download. The first file is always allowed so a large one cannot starve. */
export const DEFAULT_CLOUD_MAX_BYTES = 512 * 1024 * 1024;

/**
 * Retries after the first failed release. A just-hydrated file is often briefly held by the provider, so the
 * release is refused once and succeeds on a retry; a short doubling wait usually clears it.
 */
export const DEFAULT_CLOUD_EVICT_RETRIES = 3;

/** First wait before retrying an evict; doubles each retry (500, 1000, 2000 ms by default). */
export const DEFAULT_CLOUD_EVICT_BACKOFF_MS = 500;

export interface CloudGateOptions {
	readonly probe?: CloudFileProbe;
	readonly maxFiles?: number;
	readonly maxBytes?: number;
	readonly evictAfterParse?: boolean;
	/** Retries after the first failed evict; `0` tries once. Default {@link DEFAULT_CLOUD_EVICT_RETRIES}. */
	readonly evictRetries?: number;
	/** First retry wait in ms, doubled each retry. Default {@link DEFAULT_CLOUD_EVICT_BACKOFF_MS}. */
	readonly evictBackoffMs?: number;
	/** Wait implementation; tests inject one that records instead of sleeping. */
	readonly sleep?: (ms: number) => Promise<void>;
}

export type CloudReleaseResult = { readonly ok: true } | { readonly ok: false; readonly message: string };

/**
 * What a refresh does with one file that needs parsing:
 * - `local`: not a placeholder; read it as always.
 * - `download`: a placeholder the probe can release, inside the budget; read it, then release it again.
 * - `skip`: a placeholder whose provider the probe cannot release; do not read it.
 * - `defer`: a releasable placeholder over this refresh's budget; leave it for the next refresh.
 */
export type CloudDecision =
	| { readonly kind: "local" }
	| { readonly kind: "download" }
	| { readonly kind: "skip"; readonly provider: CloudProvider | undefined }
	| { readonly kind: "defer" };

const LOCAL: CloudDecision = { kind: "local" };

/**
 * One refresh's view of cloud placeholders. Reading a placeholder downloads it, so the refresh asks here
 * before every read; without a probe (a platform AutoRAG has no cloud handling for) every file is `local`.
 */
export class CloudGate {
	private readonly probe: CloudFileProbe | undefined;
	private readonly maxFiles: number;
	private readonly maxBytes: number;
	private readonly evictRetries: number;
	private readonly evictBackoffMs: number;
	private readonly sleep: (ms: number) => Promise<void>;
	private readonly providerByDirectory = new Map<string, Promise<CloudProvider | undefined>>();
	private placeholders: ReadonlySet<string> = new Set();
	private downloadedFiles = 0;
	private downloadedBytes = 0;
	/** Whether a downloaded cloud file is released again after it is parsed. */
	readonly evictAfterParse: boolean;

	constructor(options: CloudGateOptions = {}) {
		this.probe = options.probe ?? defaultCloudProbe();
		this.maxFiles = options.maxFiles ?? DEFAULT_CLOUD_MAX_FILES;
		this.maxBytes = options.maxBytes ?? DEFAULT_CLOUD_MAX_BYTES;
		this.evictAfterParse = options.evictAfterParse !== false;
		this.evictRetries = Math.max(0, options.evictRetries ?? DEFAULT_CLOUD_EVICT_RETRIES);
		this.evictBackoffMs = options.evictBackoffMs ?? DEFAULT_CLOUD_EVICT_BACKOFF_MS;
		this.sleep = options.sleep ?? (async (ms) => void (await sleepFor(ms)));
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
		if (!this.probe.canRelease(provider)) return { kind: "skip", provider };
		const overFiles = this.downloadedFiles >= this.maxFiles;
		const overBytes = this.downloadedFiles > 0 && this.downloadedBytes + sizeBytes > this.maxBytes;
		if (overFiles || overBytes) return { kind: "defer" };
		this.downloadedFiles += 1;
		this.downloadedBytes += sizeBytes;
		return { kind: "download" };
	}

	/**
	 * Release the local copy of a file this refresh downloaded. A failed attempt is retried with doubling waits;
	 * once the retries are spent the last error is returned and the file is left on disk. Giving up is final for
	 * this refresh: the file stays indexed, and an unchanged file is not processed again, so nothing retries it later.
	 */
	async release(path: string): Promise<CloudReleaseResult> {
		const probe = this.probe;
		if (probe === undefined) return { ok: true };
		let result = await probe.evict(path);
		for (let retry = 0; !result.ok && retry < this.evictRetries; retry += 1) {
			if (this.evictBackoffMs > 0) await this.sleep(this.evictBackoffMs * 2 ** retry);
			result = await probe.evict(path);
		}
		if (result.ok) return result;
		const attempts = this.evictRetries + 1;
		return {
			ok: false,
			message: `${result.message} (gave up after ${attempts} attempt${attempts === 1 ? "" : "s"})`,
		};
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
