import { createHash } from "node:crypto";
import type * as Fs from "node:fs";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import type * as FsPromises from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	loadMirrorIndex,
	type ParsedMirrorSyncOptions,
	type ParsedMirrorSyncResult,
	parsedMirrorRoot,
	syncParsedMirrors,
} from "../../src/mirror/index.ts";
import {
	createDefaultParserRegistry,
	type ParseInput,
	type ParseOutput,
	ParserRegistry,
	PlainTextParser,
} from "../../src/parser/index.ts";

const fsMock = vi.hoisted(() => ({
	readFileHook: undefined as ((path: unknown) => void) | undefined,
	writeFileSyncHook: undefined as ((path: unknown) => void) | undefined,
	realWriteFileSync: undefined as typeof Fs.writeFileSync | undefined,
}));

vi.mock("node:fs/promises", async () => {
	const actual = await vi.importActual<typeof FsPromises>("node:fs/promises");
	return {
		...actual,
		readFile: ((...args: Parameters<typeof FsPromises.readFile>) => {
			fsMock.readFileHook?.(args[0]);
			return actual.readFile(...args);
		}) as typeof FsPromises.readFile,
	};
});

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof Fs>("node:fs");
	fsMock.realWriteFileSync = actual.writeFileSync;
	return {
		...actual,
		writeFileSync: (...args: Parameters<typeof actual.writeFileSync>) => {
			fsMock.writeFileSyncHook?.(args[0]);
			return actual.writeFileSync(...args);
		},
	};
});

let root: string;
let source: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-mirror-failure-"));
	source = join(root, "docs");
	mkdirSync(source, { recursive: true });
});

afterEach(() => {
	fsMock.readFileHook = undefined;
	fsMock.writeFileSyncHook = undefined;
	rmSync(root, { recursive: true, force: true });
});

function errno(code: string): NodeJS.ErrnoException {
	const error: NodeJS.ErrnoException = new Error(`${code}: injected failure`);
	error.code = code;
	return error;
}

function writeCorpus(): void {
	writeFileSync(join(source, "a.txt"), "Alpha\n");
	writeFileSync(join(source, "b.txt"), "Bravo\n");
	writeFileSync(join(source, "c.txt"), "Charlie\n");
}

function leftoverTmpFiles(): string[] {
	const dir = parsedMirrorRoot(root);
	const found: string[] = [];
	const walk = (directory: string): void => {
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			const path = join(directory, entry.name);
			if (entry.isDirectory()) walk(path);
			else if (entry.name.endsWith(".tmp")) found.push(path);
		}
	};
	walk(dir);
	return found;
}

class CrashOnParser extends PlainTextParser {
	readonly crashPath: string;
	readonly thrown: () => unknown;

	constructor(crashPath: string, thrown: () => unknown) {
		super();
		this.crashPath = crashPath;
		this.thrown = thrown;
	}
	async parse(input: ParseInput): Promise<ParseOutput> {
		if (input.virtualPath === this.crashPath) throw this.thrown();
		return super.parse(input);
	}
}

function sync(extra: Partial<ParsedMirrorSyncOptions> = {}): Promise<ParsedMirrorSyncResult> {
	return syncParsedMirrors({
		root,
		searchPaths: [source],
		registry: createDefaultParserRegistry(),
		transientRetryBackoffMs: 0,
		...extra,
	});
}

describe("per-file parser crash isolation", () => {
	it("records exactly one parser-crashed skip for a plain Error and indexes the rest", async () => {
		writeCorpus();
		const result = await sync({
			registry: new ParserRegistry([new CrashOnParser("/docs/b.txt", () => new TypeError("third-party bug"))]),
		});

		const crashed = result.diagnostics.filter((d) => d.code === "parser-crashed");
		expect(crashed).toHaveLength(1);
		expect(crashed[0]?.source).toBe("/docs/b.txt");
		expect(crashed[0]?.severity).toBe("warning");
		expect(crashed[0]?.reason).toBe("TypeError");
		expect(JSON.stringify(result.diagnostics)).not.toContain(root);
		expect(result.written).toBe(2);
		expect(result.skipped).toBe(1);
		const index = loadMirrorIndex(root);
		expect(Object.keys(index.entries).sort()).toEqual(["/docs/a.txt", "/docs/c.txt"]);
	});

	it("isolates a non-Error throw from a parser", async () => {
		writeCorpus();
		const result = await sync({
			registry: new ParserRegistry([new CrashOnParser("/docs/a.txt", () => "plain string")]),
		});

		expect(result.diagnostics.filter((d) => d.code === "parser-crashed")).toHaveLength(1);
		expect(result.written).toBe(2);
	});

	it("drops a stale mirror for a file whose parser now crashes", async () => {
		writeCorpus();
		await sync();
		writeFileSync(join(source, "b.txt"), "Bravo changed\n");

		const result = await sync({
			registry: new ParserRegistry([new CrashOnParser("/docs/b.txt", () => new Error("boom"))]),
		});

		expect(result.diagnostics.filter((d) => d.code === "parser-crashed")).toHaveLength(1);
		expect(loadMirrorIndex(root).entries["/docs/b.txt"]).toBeUndefined();
	});
});

describe("source read failures", () => {
	it("retries a transient EMFILE read and indexes the file with no diagnostic", async () => {
		writeCorpus();
		let failures = 0;
		fsMock.readFileHook = (path) => {
			if (String(path).endsWith("b.txt") && failures < 2) {
				failures += 1;
				throw errno("EMFILE");
			}
		};

		const result = await sync();

		expect(failures).toBe(2);
		expect(result.diagnostics).toEqual([]);
		expect(result.written).toBe(3);
	});

	it("records source-read-failed after the retries are exhausted and keeps indexing", async () => {
		writeCorpus();
		let attempts = 0;
		fsMock.readFileHook = (path) => {
			if (String(path).endsWith("b.txt")) {
				attempts += 1;
				throw errno("EMFILE");
			}
		};

		const result = await sync();

		expect(attempts).toBe(4); // first attempt + default 3 retries
		const failed = result.diagnostics.filter((d) => d.code === "source-read-failed");
		expect(failed).toHaveLength(1);
		expect(failed[0]?.source).toBe("/docs/b.txt");
		expect(failed[0]?.severity).toBe("warning");
		expect(failed[0]?.reason).toBe("EMFILE");
		expect(result.written).toBe(2);
		expect(result.skipped).toBe(1);
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual(["/docs/a.txt", "/docs/c.txt"]);
	});

	it("honours transientRetries and does not retry a non-transient error", async () => {
		writeCorpus();
		let attempts = 0;
		fsMock.readFileHook = (path) => {
			if (String(path).endsWith("b.txt")) {
				attempts += 1;
				throw errno("EIO");
			}
		};
		const noRetry = await sync({ transientRetries: 0 });
		expect(attempts).toBe(1);
		expect(noRetry.diagnostics.filter((d) => d.code === "source-read-failed")).toHaveLength(1);

		attempts = 0;
		fsMock.readFileHook = (path) => {
			if (String(path).endsWith("b.txt")) {
				attempts += 1;
				throw errno("EINVAL");
			}
		};
		const nonTransient = await sync({ force: true });
		expect(attempts).toBe(1);
		expect(nonTransient.diagnostics.filter((d) => d.code === "source-read-failed")).toHaveLength(1);
	});

	it("does not persist a source-read-failed skip, so the next refresh retries the file", async () => {
		writeCorpus();
		fsMock.readFileHook = (path) => {
			if (String(path).endsWith("b.txt")) throw errno("EIO");
		};
		await sync();
		expect(loadMirrorIndex(root).skipped?.["/docs/b.txt"]).toBeUndefined();

		fsMock.readFileHook = undefined;
		const result = await sync();
		expect(result.diagnostics).toEqual([]);
		expect(loadMirrorIndex(root).entries["/docs/b.txt"]).toBeDefined();
	});
});

describe("mirror write failures", () => {
	it("aborts on ENOSPC with a workspace-level error and leaves no tmp file", async () => {
		writeCorpus();
		fsMock.writeFileSyncHook = (path) => {
			if (String(path).includes(parsedMirrorRoot(root)) && String(path).endsWith(".tmp")) {
				fsMock.realWriteFileSync?.(String(path), "partial");
				throw errno("ENOSPC");
			}
		};

		const error = await sync().then(
			() => undefined,
			(thrown: unknown) => thrown,
		);

		expect(error).toBeInstanceOf(Error);
		expect((error as Error).name).toBe("MirrorWorkspaceError");
		expect((error as Error).message).toContain(root);
		expect((error as Error).message).toContain("ENOSPC");
		fsMock.writeFileSyncHook = undefined;
		expect(leftoverTmpFiles()).toEqual([]);
	});

	it("aborts on EACCES against the workspace", async () => {
		writeCorpus();
		fsMock.writeFileSyncHook = (path) => {
			if (String(path).endsWith(".tmp")) throw errno("EACCES");
		};

		const error = await sync().then(
			() => undefined,
			(thrown: unknown) => thrown,
		);

		expect((error as Error).name).toBe("MirrorWorkspaceError");
		expect((error as Error).message).toContain(root);
		fsMock.writeFileSyncHook = undefined;
		expect(leftoverTmpFiles()).toEqual([]);
	});

	it("keeps already mirrored files in the index when it aborts", async () => {
		writeCorpus();
		let writes = 0;
		fsMock.writeFileSyncHook = (path) => {
			if (!/\.md\.[^/]+\.tmp$/.test(String(path))) return;
			writes += 1;
			if (writes === 2) throw errno("ENOSPC");
		};

		await sync().catch(() => undefined);
		fsMock.writeFileSyncHook = undefined;

		expect(Object.keys(loadMirrorIndex(root).entries)).toEqual(["/docs/a.txt"]);
	});

	it("records mirror-write-failed for a per-file write error, cleans the tmp file, and continues", async () => {
		writeCorpus();
		fsMock.writeFileSyncHook = (path) => {
			// Fail every write of the second mirror file (all retries), regardless of tmp uuid.
			if (/\.md\.[^/]+\.tmp$/.test(String(path)) && String(path).includes(sha("/docs/b.txt"))) {
				fsMock.realWriteFileSync?.(String(path), "partial");
				throw errno("EIO");
			}
		};

		const result = await sync();
		fsMock.writeFileSyncHook = undefined;

		const failed = result.diagnostics.filter((d) => d.code === "mirror-write-failed");
		expect(failed).toHaveLength(1);
		expect(failed[0]?.source).toBe("/docs/b.txt");
		expect(failed[0]?.reason).toBe("EIO");
		expect(result.written).toBe(2);
		expect(result.skipped).toBe(1);
		expect(leftoverTmpFiles()).toEqual([]);
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual(["/docs/a.txt", "/docs/c.txt"]);
	});

	it("retries a transient write error and succeeds", async () => {
		writeCorpus();
		let failures = 0;
		fsMock.writeFileSyncHook = (path) => {
			if (/\.md\.[^/]+\.tmp$/.test(String(path)) && failures < 1) {
				failures += 1;
				throw errno("EBUSY");
			}
		};

		const result = await sync();
		fsMock.writeFileSyncHook = undefined;

		expect(failures).toBe(1);
		expect(result.diagnostics).toEqual([]);
		expect(result.written).toBe(3);
		expect(leftoverTmpFiles()).toEqual([]);
	});
});

function sha(virtualPath: string): string {
	// Mirror output file name is the sha256 of the virtual path.
	return createHash("sha256").update(virtualPath).digest("hex");
}
