import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	type CloudEvictResult,
	type CloudFileProbe,
	type CloudProvider,
	loadMirrorIndex,
	type ParsedMirrorSyncOptions,
	type ParsedMirrorSyncResult,
	syncParsedMirrors,
} from "../../src/mirror/index.ts";
import {
	createDefaultParserRegistry,
	type ParseInput,
	type ParseOutput,
	ParserRegistry,
	PlainTextParser,
} from "../../src/parser/index.ts";

let root: string;
let source: string;

beforeEach(() => {
	waits = [];
	root = mkdtempSync(join(tmpdir(), "autorag-cloud-hydration-"));
	source = join(root, "docs");
	mkdirSync(source, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** A probe over ordinary temp files: the test decides which ones count as data-less placeholders. */
class FakeCloud implements CloudFileProbe {
	readonly dataless = new Set<string>();
	readonly providers = new Map<string, CloudProvider>();
	/** Providers this probe claims it can release; the gate must not download any other provider's placeholder. */
	readonly releasable = new Set<CloudProvider>(["icloud"]);
	readonly evicted: string[] = [];
	evictResult: CloudEvictResult = { ok: true };
	/** Results handed out one per evict call before falling back to `evictResult`. */
	evictScript: CloudEvictResult[] = [];

	markDataless(path: string, provider: CloudProvider): void {
		this.dataless.add(path);
		this.providers.set(path, provider);
	}
	async datalessAmong(paths: readonly string[]): Promise<ReadonlySet<string>> {
		return new Set(paths.filter((path) => this.dataless.has(path)));
	}
	async providerOf(path: string): Promise<CloudProvider | undefined> {
		return this.providers.get(path);
	}
	canRelease(provider: CloudProvider | undefined): boolean {
		return provider !== undefined && this.releasable.has(provider);
	}
	async evict(path: string): Promise<CloudEvictResult> {
		this.evicted.push(path);
		return this.evictScript.shift() ?? this.evictResult;
	}
}

/**
 * Reading a placeholder makes the provider hydrate it. Measured on macOS 26 with iCloud Drive: the first
 * hydrate of a file written on this Mac moves ctime and shifts mtime by tens of nanoseconds (size and inode
 * stay put), because the provider rounds the timestamp to the precision the cloud stores. The fake applies a
 * larger shift (1 microsecond) so it survives double-precision `utimes`.
 */
class HydratingParser extends PlainTextParser {
	readonly parsed: string[] = [];
	readonly placeholders: ReadonlySet<string>;

	constructor(placeholders: ReadonlySet<string>) {
		super();
		this.placeholders = placeholders;
	}
	async parse(input: ParseInput): Promise<ParseOutput> {
		this.parsed.push(input.virtualPath);
		const path = input.sourcePath;
		if (path !== undefined && this.placeholders.has(path)) {
			const mtime = statSync(path).mtimeMs / 1000 + 0.000001;
			utimesSync(path, mtime, mtime);
		}
		return super.parse(input);
	}
}

/** Rewrites the file mid-parse, the way a user editing a document would. */
class EditingParser extends PlainTextParser {
	async parse(input: ParseInput): Promise<ParseOutput> {
		const parsed = await super.parse(input);
		if (input.sourcePath !== undefined)
			writeFileSync(input.sourcePath, "edited while parsing, and longer than before\n");
		return parsed;
	}
}

function write(name: string, text = `${name}\n`): string {
	const path = join(source, name);
	writeFileSync(path, text);
	return path;
}

/** Backoff waits the refresh asked for, in order. The tests never really sleep. */
let waits: number[];

function sync(cloud: FakeCloud, extra: Partial<ParsedMirrorSyncOptions> = {}): Promise<ParsedMirrorSyncResult> {
	return syncParsedMirrors({
		root,
		searchPaths: [source],
		registry: createDefaultParserRegistry(),
		transientRetryBackoffMs: 0,
		cloudProbe: cloud,
		cloudSleep: async (ms) => {
			waits.push(ms);
		},
		...extra,
	});
}

function codes(result: ParsedMirrorSyncResult): string[] {
	return result.diagnostics.map((diagnostic) => diagnostic.code);
}

describe("iCloud placeholders are indexed, then released", () => {
	it("parses a placeholder, keeps it indexed, and evicts exactly that file afterwards", async () => {
		const cloud = new FakeCloud();
		const placeholder = write("remote.txt", "remote only\n");
		write("local.txt");
		cloud.markDataless(placeholder, "icloud");
		const parser = new HydratingParser(cloud.dataless);

		const result = await sync(cloud, { registry: new ParserRegistry([parser]) });

		expect(result.written).toBe(2);
		expect(codes(result)).not.toContain("source-changed");
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual(["/docs/local.txt", "/docs/remote.txt"]);
		expect(cloud.evicted).toEqual([placeholder]);
	});

	it("never evicts a file that was already on disk", async () => {
		const cloud = new FakeCloud();
		write("local.txt");

		await sync(cloud, { registry: new ParserRegistry([new HydratingParser(cloud.dataless)]) });

		expect(cloud.evicted).toEqual([]);
	});

	it("leaves the file materialized when eviction is switched off", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");

		const result = await sync(cloud, { cloudEvictAfterParse: false });

		expect(result.written).toBe(1);
		expect(cloud.evicted).toEqual([]);
	});

	it("gives up after the retries, keeps the document indexed, and reports the last error once", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");
		cloud.evictScript = [
			{ ok: false, message: "brctl: busy 1" },
			{ ok: false, message: "brctl: busy 2" },
			{ ok: false, message: "brctl: busy 3" },
			{ ok: false, message: "brctl: busy 4" },
		];

		const result = await sync(cloud, { cloudEvictRetries: 3 });

		expect(result.written).toBe(1);
		expect(cloud.evicted).toHaveLength(4); // first attempt + 3 retries
		const failures = result.diagnostics.filter((diagnostic) => diagnostic.code === "cloud-evict-failed");
		expect(failures).toHaveLength(1);
		expect(failures[0]?.source).toBe("/docs/remote.txt");
		expect(failures[0]?.message).toContain("brctl: busy 4");
		expect(failures[0]?.message).toContain("4 attempts");
		expect(loadMirrorIndex(root).entries["/docs/remote.txt"]).toBeDefined();
	});

	it("succeeds on a later attempt without a diagnostic", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");
		cloud.evictScript = [{ ok: false, message: "busy" }, { ok: false, message: "busy" }, { ok: true }];

		const result = await sync(cloud);

		expect(cloud.evicted).toHaveLength(3);
		expect(codes(result)).not.toContain("cloud-evict-failed");
	});

	it("doubles the wait between attempts", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");
		cloud.evictResult = { ok: false, message: "busy" };

		await sync(cloud, { cloudEvictRetries: 4, cloudEvictBackoffMs: 100 });

		expect(waits).toEqual([100, 200, 400, 800]);
	});

	it("does not wait or retry when the first attempt works", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");

		await sync(cloud);

		expect(cloud.evicted).toHaveLength(1);
		expect(waits).toEqual([]);
	});

	it("tries once and reports when retries are set to 0", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");
		cloud.evictResult = { ok: false, message: "busy" };

		const result = await sync(cloud, { cloudEvictRetries: 0 });

		expect(cloud.evicted).toHaveLength(1);
		expect(waits).toEqual([]);
		expect(codes(result)).toContain("cloud-evict-failed");
	});

	it("gives each file its own retries and keeps releasing the rest after one gives up", async () => {
		const cloud = new FakeCloud();
		for (const name of ["a.txt", "b.txt", "c.txt"]) cloud.markDataless(write(name), "icloud");
		// a.txt exhausts its 2 retries (3 calls); b.txt and c.txt then succeed first time.
		cloud.evictScript = [
			{ ok: false, message: "busy" },
			{ ok: false, message: "busy" },
			{ ok: false, message: "busy" },
		];

		const result = await sync(cloud, { cloudEvictRetries: 2 });

		expect(cloud.evicted.map((path) => basename(path))).toEqual(["a.txt", "a.txt", "a.txt", "b.txt", "c.txt"]);
		expect(result.diagnostics.filter((diagnostic) => diagnostic.code === "cloud-evict-failed")).toHaveLength(1);
		expect(result.written).toBe(3);
	});

	it("does not evict a file the user changed while it was being parsed", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt", "short\n"), "icloud");

		const result = await sync(cloud, { registry: new ParserRegistry([new EditingParser()]) });

		expect(codes(result)).toContain("source-changed");
		expect(cloud.evicted).toEqual([]);
	});

	it("still catches an edit that lands in the same second when the size changes", async () => {
		const cloud = new FakeCloud();
		const path = write("remote.txt", "short\n");
		cloud.markDataless(path, "icloud");
		const second = Math.floor(statSync(path).mtimeMs / 1000);
		class SameSecondEdit extends PlainTextParser {
			async parse(input: ParseInput): Promise<ParseOutput> {
				const parsed = await super.parse(input);
				if (input.sourcePath === undefined) return parsed;
				writeFileSync(input.sourcePath, "edited in the very same second, so mtime seconds match\n");
				utimesSync(input.sourcePath, second, second);
				return parsed;
			}
		}
		utimesSync(path, second, second);

		const result = await sync(cloud, { registry: new ParserRegistry([new SameSecondEdit()]) });

		expect(codes(result)).toContain("source-changed");
		expect(cloud.evicted).toEqual([]);
	});

	it("does not touch an unchanged placeholder on the next refresh", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("remote.txt"), "icloud");
		await sync(cloud);
		cloud.evicted.length = 0;
		const parser = new HydratingParser(cloud.dataless);

		const result = await sync(cloud, { registry: new ParserRegistry([parser]) });

		expect(parser.parsed).toEqual([]);
		expect(result.written).toBe(0);
		expect(cloud.evicted).toEqual([]);
	});
});

describe("download budget", () => {
	it("downloads at most cloudMaxFiles placeholders per refresh and finishes the rest on the next one", async () => {
		const cloud = new FakeCloud();
		for (const name of ["a.txt", "b.txt", "c.txt"]) cloud.markDataless(write(name), "icloud");

		const first = await sync(cloud, { cloudMaxFiles: 2 });

		expect(first.written).toBe(2);
		expect(first.diagnostics.filter((diagnostic) => diagnostic.code === "cloud-download-deferred")).toHaveLength(1);
		expect(Object.keys(loadMirrorIndex(root).entries)).toHaveLength(2);

		const second = await sync(cloud, { cloudMaxFiles: 2 });

		expect(second.written).toBe(1);
		expect(codes(second)).not.toContain("cloud-download-deferred");
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual(["/docs/a.txt", "/docs/b.txt", "/docs/c.txt"]);
	});

	it("stops downloading once cloudMaxBytes would be exceeded", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("a.txt", "x".repeat(600)), "icloud");
		cloud.markDataless(write("b.txt", "y".repeat(600)), "icloud");

		const result = await sync(cloud, { cloudMaxBytes: 1000 });

		expect(result.written).toBe(1);
		expect(codes(result)).toContain("cloud-download-deferred");
	});

	it("always downloads one file per refresh even when it alone exceeds cloudMaxBytes", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("big.txt", "z".repeat(600)), "icloud");

		const result = await sync(cloud, { cloudMaxBytes: 100 });

		expect(result.written).toBe(1);
		expect(codes(result)).not.toContain("cloud-download-deferred");
	});

	it("does not spend the budget on files that are already local", async () => {
		const cloud = new FakeCloud();
		for (const name of ["l1.txt", "l2.txt", "l3.txt"]) write(name);
		cloud.markDataless(write("remote.txt"), "icloud");

		const result = await sync(cloud, { cloudMaxFiles: 1 });

		expect(result.written).toBe(4);
		expect(codes(result)).not.toContain("cloud-download-deferred");
	});

	it("keeps the last good mirror of a changed file whose download was deferred", async () => {
		const cloud = new FakeCloud();
		const changed = write("z.txt", "version one\n");
		await sync(cloud);
		const outputPath = loadMirrorIndex(root).entries["/docs/z.txt"]?.outputPath;
		if (outputPath === undefined) throw new Error("z.txt was not indexed");
		write("z.txt", "version two, which is longer\n");
		cloud.markDataless(changed, "icloud");
		cloud.markDataless(write("a.txt"), "icloud");

		const result = await sync(cloud, { cloudMaxFiles: 1 });

		expect(result.diagnostics.find((diagnostic) => diagnostic.code === "cloud-download-deferred")?.source).toBe(
			"/docs/z.txt",
		);
		expect(readFileSync(outputPath, "utf8")).toContain("version one");
		expect(loadMirrorIndex(root).entries["/docs/z.txt"]).toBeDefined();
	});
});

describe("providers the probe cannot release", () => {
	it("skips a Google Drive placeholder without reading it and keeps its previous mirror", async () => {
		const cloud = new FakeCloud();
		const path = write("drive.txt", "indexed while it was local\n");
		await sync(cloud);
		write("drive.txt", "changed upstream and now a placeholder\n");
		cloud.markDataless(path, "google-drive");
		const parser = new HydratingParser(cloud.dataless);

		const result = await sync(cloud, { registry: new ParserRegistry([parser]) });

		expect(parser.parsed).toEqual([]);
		expect(result.written).toBe(0);
		const skipped = result.diagnostics.find((diagnostic) => diagnostic.code === "cloud-placeholder-skipped");
		expect(skipped?.source).toBe("/docs/drive.txt");
		expect(skipped?.reason).toBe("google-drive");
		expect(loadMirrorIndex(root).entries["/docs/drive.txt"]).toBeDefined();
		expect(cloud.evicted).toEqual([]);
	});

	it("skips a placeholder whose provider is unknown", async () => {
		const cloud = new FakeCloud();
		cloud.markDataless(write("unknown.txt"), "file-provider");

		const result = await sync(cloud);

		expect(result.written).toBe(0);
		expect(codes(result)).toContain("cloud-placeholder-skipped");
	});

	it("downloads, indexes and releases a placeholder of any provider the probe can release", async () => {
		const cloud = new FakeCloud();
		cloud.releasable.add("onedrive");
		const path = write("onedrive.txt", "only in the cloud\n");
		cloud.markDataless(path, "onedrive");

		const result = await sync(cloud);

		expect(result.written).toBe(1);
		expect(codes(result)).not.toContain("cloud-placeholder-skipped");
		expect(cloud.evicted).toEqual([path]);
	});
});
