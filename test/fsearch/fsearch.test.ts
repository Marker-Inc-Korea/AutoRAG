import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildFSearchIndexArgs,
	buildFSearchSearchArgs,
	FSearchClient,
	type FSearchRunResult,
	fsearchSocketPath,
	parseFSearchJsonLines,
	parseFSearchStats,
	walkFileSearch,
} from "../../src/fsearch/index.ts";

// FSearch is macOS/Linux-only: the client, daemon socket (/tmp), and path-prefix
// filtering assume POSIX paths, and on Windows the agent never constructs the
// client (isSupported() === false), so the suite is skipped there.
const describeFSearch = process.platform === "win32" ? describe.skip : describe;

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-fsearch-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

describeFSearch("buildFSearchSearchArgs", () => {
	const SOCKET = "/tmp/autorag-fsearch-501-a1b2c3d4e5f6.sock";

	it("builds a plain name search with the query after --", () => {
		expect(buildFSearchSearchArgs("/ws/.autorag/fsearch/fsearch.db", SOCKET, { query: "invoice 2026" })).toEqual([
			"search",
			"--db",
			"/ws/.autorag/fsearch/fsearch.db",
			"--socket",
			SOCKET,
			"--json",
			"--limit",
			"100",
			"--",
			"invoice 2026",
		]);
	});

	it("maps typed filters to fsearch-cli flags", () => {
		const args = buildFSearchSearchArgs("/db", SOCKET, {
			query: "보고서",
			regex: true,
			matchCase: true,
			matchPath: true,
			kind: "folders",
			sort: "date-modified-descending",
		});
		expect(args).toEqual([
			"search",
			"--db",
			"/db",
			"--socket",
			SOCKET,
			"--json",
			"-r",
			"-c",
			"-p",
			"-F",
			"--sort",
			"mtime",
			"--desc",
			"--limit",
			"100",
			"--",
			"보고서",
		]);
	});

	it("maps files kind and ascending sorts", () => {
		const args = buildFSearchSearchArgs("/db", SOCKET, { query: "a", kind: "files", sort: "size-ascending" });
		expect(args).toContain("-f");
		expect(args).not.toContain("-F");
		const sortIndex = args.indexOf("--sort");
		expect(args[sortIndex + 1]).toBe("size");
		expect(args).not.toContain("--desc");
	});

	it("fetches offset+maxResults so the client can page locally, capped at 1000", () => {
		const paged = buildFSearchSearchArgs("/db", SOCKET, { query: "a", offset: 50, maxResults: 25 });
		expect(paged[paged.indexOf("--limit") + 1]).toBe("75");
		const capped = buildFSearchSearchArgs("/db", SOCKET, { query: "a", maxResults: 5000 });
		expect(capped[capped.indexOf("--limit") + 1]).toBe("1000");
	});

	it("over-fetches up to the cap when a path filter needs local prefix filtering", () => {
		const args = buildFSearchSearchArgs("/db", SOCKET, { query: "a", path: "/docs", maxResults: 10 });
		expect(args[args.indexOf("--limit") + 1]).toBe("1000");
	});
});

describeFSearch("fsearchSocketPath", () => {
	it("derives a short per-database socket path that fits the unix sun_path limit", () => {
		const dbPath = join(root, ".autorag", "fsearch", "fsearch.db");
		const socket = fsearchSocketPath(dbPath);
		expect(socket).toMatch(/^\/tmp\/autorag-fsearch-\d+-[0-9a-f]{12}\.sock$/);
		expect(socket.length).toBeLessThan(100);
		expect(fsearchSocketPath(dbPath)).toBe(socket);
		expect(fsearchSocketPath(join(root, "other.db"))).not.toBe(socket);
	});

	it("resolves symlinked database spellings to the same socket", () => {
		// realpath needs the file to exist; without it the helper falls back to resolve().
		writeFileSync(join(root, "fsearch.db"), "db");
		const linked = join(root, "link");
		symlinkSync(root, linked);
		const direct = fsearchSocketPath(join(root, "fsearch.db"));
		expect(fsearchSocketPath(join(linked, "fsearch.db"))).toBe(direct);
	});
});

describeFSearch("buildFSearchIndexArgs", () => {
	it("includes every folder and excludes workspace state directories", () => {
		expect(buildFSearchIndexArgs("/db", ["/docs", "/media"], ["/ws/.autorag", "/docs/.autorag"])).toEqual([
			"index",
			"--db",
			"/db",
			"--include",
			"/docs",
			"--include",
			"/media",
			"--exclude-path",
			"/ws/.autorag",
			"--exclude-path",
			"/docs/.autorag",
		]);
	});
});

describeFSearch("parseFSearchJsonLines", () => {
	it("parses entries, normalizes names, converts mtime, and reads the done line total", () => {
		const stdout = [
			'{"path":"/docs","name":"/docs","type":"folder","size":6,"mtime":1790956870}',
			'{"path":"/docs/환불 정책.pdf","name":"환불 정책.pdf","type":"file","size":5,"mtime":1790956800}',
			'{"done":true,"num_results":7,"num_returned":2}',
			"",
		].join("\n");
		const parsed = parseFSearchJsonLines(stdout);
		expect(parsed.total).toBe(7);
		expect(parsed.entries).toEqual([
			{
				path: "/docs",
				name: "docs",
				type: "folder",
				size: 6,
				dateModified: new Date(1790956870 * 1000).toISOString(),
			},
			{
				path: "/docs/환불 정책.pdf",
				name: "환불 정책.pdf",
				type: "file",
				size: 5,
				dateModified: new Date(1790956800 * 1000).toISOString(),
			},
		]);
	});

	it("treats empty output as zero results", () => {
		expect(parseFSearchJsonLines("")).toEqual({ entries: [], total: undefined });
		expect(parseFSearchJsonLines("\n")).toEqual({ entries: [], total: undefined });
	});

	it("throws the raw line when fsearch-cli does not return JSON", () => {
		expect(() => parseFSearchJsonLines("fsearch-cli: database not found")).toThrow(/database not found/);
	});
});

describeFSearch("parseFSearchStats", () => {
	it("reads counts and the live flag", () => {
		expect(parseFSearchStats('{"db":"/db","live":true,"files":2,"folders":3,"includes":[]}')).toEqual({
			files: 2,
			folders: 3,
			live: true,
		});
	});

	it("throws the raw output when stats is not JSON", () => {
		expect(() => parseFSearchStats("no database")).toThrow(/no database/);
	});
});

interface FakeBackend {
	client: FSearchClient;
	calls: Array<{ command: string; args: readonly string[] }>;
	launched: Array<{ command: string; args: readonly string[] }>;
	killed: number[];
	dbPath: string;
	pidPath: string;
}

function fakeClient(
	responses: Array<Partial<FSearchRunResult> & { code: number | null }>,
	options: { binaryPath?: string; alivePids?: number[] } = {},
): FakeBackend {
	const calls: FakeBackend["calls"] = [];
	const launched: FakeBackend["launched"] = [];
	const killed: number[] = [];
	let nextPid = 4242;
	const client = new FSearchClient({
		root,
		folders: [join(root, "docs")],
		platform: "darwin",
		binaryPath: options.binaryPath ?? "fsearch-cli",
		run: async (command, args) => {
			calls.push({ command, args });
			const next = responses.shift() ?? { code: 0, stdout: "" };
			return { code: next.code, stdout: next.stdout ?? "", stderr: next.stderr ?? "" };
		},
		launch: (command, args) => {
			launched.push({ command, args });
			return nextPid++;
		},
		isProcessAlive: (pid) => (options.alivePids ?? []).includes(pid),
		killProcess: (pid) => killed.push(pid),
		startupTimeoutMs: 50,
		pollIntervalMs: 1,
	});
	return {
		client,
		calls,
		launched,
		killed,
		dbPath: join(root, ".autorag", "fsearch", "fsearch.db"),
		pidPath: join(root, ".autorag", "fsearch", "watch.pid"),
	};
}

const VERSION_OK = { code: 0, stdout: "fsearch-cli 0.3\n" };

describeFSearch("FSearchClient", () => {
	it("is supported on macOS and Linux, not on Windows", () => {
		expect(new FSearchClient({ root, folders: [root], platform: "darwin" }).isSupported()).toBe(true);
		expect(new FSearchClient({ root, folders: [root], platform: "linux" }).isSupported()).toBe(true);
		expect(new FSearchClient({ root, folders: [root], platform: "win32" }).isSupported()).toBe(false);
	});

	it("returns unsupported-platform on Windows instead of searching", async () => {
		const client = new FSearchClient({ root, folders: [root], platform: "win32" });
		expect(await client.search({ query: "a" })).toMatchObject({ ok: false, reason: "unsupported-platform" });
		expect(await client.index()).toMatchObject({ ok: false, reason: "unsupported-platform" });
	});

	it("indexes the configured folders, starts the watch daemon, and reports the counted items", async () => {
		const { client, calls, launched, dbPath, pidPath } = fakeClient([
			VERSION_OK,
			{ code: 0, stdout: "indexed 2 files, 1 folders\n" },
			{ code: 0, stdout: '{"db":"/db","live":false,"files":2,"folders":1,"includes":[]}' },
			{ code: 0, stdout: '{"db":"/db","live":false,"files":2,"folders":1,"includes":[]}' },
			{ code: 0, stdout: '{"db":"/db","live":true,"files":2,"folders":1,"includes":[]}' },
		]);
		const result = await client.index();
		expect(result).toEqual({
			ok: true,
			indexedFolders: [join(root, "docs")],
			indexedItems: 3,
			watchLive: true,
		});
		const indexArgs = calls.find((call) => call.args[0] === "index");
		expect(indexArgs?.args).toEqual(
			expect.arrayContaining([
				"--db",
				dbPath,
				"--include",
				join(root, "docs"),
				"--exclude-path",
				join(root, ".autorag"),
			]),
		);
		expect(launched).toHaveLength(1);
		expect(launched[0]!.args).toEqual(["watch", "--db", dbPath, "--socket", fsearchSocketPath(dbPath), "--quiet"]);
		expect(readFileSync(pidPath, "utf8")).toBe("4242");
	});

	it("restarts the tracked watch daemon on reindex so the fresh index is served", async () => {
		const { client, launched, killed, pidPath } = fakeClient(
			[
				VERSION_OK,
				{ code: 0 },
				{ code: 0, stdout: '{"live":false,"files":1,"folders":1}' },
				{ code: 0, stdout: '{"live":false,"files":1,"folders":1}' },
				{ code: 0, stdout: '{"live":false,"files":1,"folders":1}' },
				{ code: 0, stdout: '{"live":true,"files":1,"folders":1}' },
			],
			{ alivePids: [31337, 4242] },
		);
		mkdirSync(join(root, ".autorag", "fsearch"), { recursive: true });
		writeFileSync(pidPath, "31337");
		const result = await client.index();
		expect(result).toMatchObject({ ok: true, watchLive: true });
		expect(killed).toEqual([31337]);
		expect(launched).toHaveLength(1);
		expect(readFileSync(pidPath, "utf8")).toBe("4242");
	});

	it("adopts an already-serving untracked daemon instead of launching a contender", async () => {
		const { client, launched, pidPath } = fakeClient([
			VERSION_OK,
			{ code: 0 },
			{ code: 0, stdout: '{"live":false,"files":1,"folders":1}' },
			{ code: 0, stdout: '{"live":true,"files":1,"folders":1}' },
		]);
		const result = await client.index();
		expect(result).toMatchObject({ ok: true, watchLive: true });
		expect(launched).toHaveLength(0);
		expect(existsSync(pidPath)).toBe(false);
	});

	it("removes the pid file when the launched daemon never serves", async () => {
		const { client, launched, pidPath } = fakeClient(
			[VERSION_OK, { code: 0 }, { code: 0, stdout: '{"live":false,"files":1,"folders":1}' }],
			{ alivePids: [] },
		);
		const result = await client.index();
		expect(result).toMatchObject({ ok: true, watchLive: false });
		expect(launched).toHaveLength(1);
		expect(existsSync(pidPath)).toBe(false);
	});

	it("never builds the database on a search; it walks the folders until refresh indexes them", async () => {
		// Index builds belong to `autorag refresh`, never to a query turn.
		mkdirSync(join(root, "docs"), { recursive: true });
		writeFileSync(join(root, "docs", "a.txt"), "a");
		const { client, calls } = fakeClient([VERSION_OK]);
		const result = await client.search({ query: "a" });
		expect(result).toMatchObject({ ok: true, backend: "walk" });
		if (!result.ok) throw new Error("expected ok");
		expect(result.results.map((entry) => entry.name)).toContain("a.txt");
		expect(calls.map((call) => call.args[0])).toEqual(["--version"]);
	});

	it("searches an existing database without reindexing", async () => {
		const { client, calls, dbPath } = fakeClient([
			VERSION_OK,
			{ code: 0, stdout: '{"done":true,"num_results":0,"num_returned":0}\n' },
		]);
		mkdirSync(join(root, ".autorag", "fsearch"), { recursive: true });
		writeFileSync(dbPath, "db");
		const result = await client.search({ query: "nothing" });
		expect(result).toMatchObject({ ok: true, backend: "fsearch-cli" });
		expect(calls.map((call) => call.args[0])).toEqual(["--version", "search"]);
	});

	it("applies offset, maxResults, and the path prefix filter locally", async () => {
		const lines = Array.from({ length: 10 }, (_, index) => {
			const path = index < 5 ? `/docs/a${index}.txt` : `/other/a${index}.txt`;
			return `{"path":"${path}","name":"a${index}.txt","type":"file","size":1,"mtime":1790956800}`;
		});
		const { client, dbPath } = fakeClient([
			VERSION_OK,
			{ code: 0, stdout: `${lines.join("\n")}\n{"done":true,"num_results":10,"num_returned":10}\n` },
		]);
		mkdirSync(join(root, ".autorag", "fsearch"), { recursive: true });
		writeFileSync(dbPath, "db");
		const result = await client.search({ query: "a", path: "/docs", offset: 1, maxResults: 2 });
		if (!result.ok) throw new Error("expected ok");
		expect(result.results.map((entry) => entry.path)).toEqual(["/docs/a1.txt", "/docs/a2.txt"]);
		expect(result.total).toBe(10);
	});

	it("surfaces fsearch-cli exit status and stderr verbatim when a search fails", async () => {
		const { client, dbPath } = fakeClient([VERSION_OK, { code: 2, stderr: "fsearch-cli: database is corrupt" }]);
		mkdirSync(join(root, ".autorag", "fsearch"), { recursive: true });
		writeFileSync(dbPath, "db");
		const result = await client.search({ query: "x" });
		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected failure");
		expect(result.reason).toBe("search-failed");
		expect(result.message).toContain("exit 2");
		expect(result.message).toContain("fsearch-cli: database is corrupt");
	});

	it("surfaces an index failure verbatim", async () => {
		const { client } = fakeClient([VERSION_OK, { code: 1, stderr: "fsearch-cli: cannot read /docs" }]);
		const result = await client.index();
		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected failure");
		expect(result.reason).toBe("index-failed");
		expect(result.message).toContain("fsearch-cli: cannot read /docs");
	});

	it("falls back to a slow filesystem walk when fsearch-cli is not installed", async () => {
		mkdirSync(join(root, "docs"), { recursive: true });
		writeFileSync(join(root, "docs", "walk-target.txt"), "hi");
		const { client } = fakeClient([{ code: null, stderr: "fsearch-cli: spawn fsearch-cli ENOENT" }]);
		const result = await client.search({ query: "walk-target" });
		if (!result.ok) throw new Error("expected ok");
		expect(result.backend).toBe("walk");
		expect(result.results.map((entry) => entry.name)).toEqual(["walk-target.txt"]);
		expect(result.note).toContain("spawn fsearch-cli ENOENT");
	});

	it("reports binary-missing when indexing without fsearch-cli installed", async () => {
		const { client } = fakeClient([{ code: null, stderr: "fsearch-cli: spawn fsearch-cli ENOENT" }]);
		const result = await client.index();
		expect(result).toMatchObject({ ok: false, reason: "binary-missing" });
		if (result.ok) throw new Error("expected failure");
		expect(result.message).toContain("spawn fsearch-cli ENOENT");
	});

	it("stop terminates the recorded watch daemon and removes the pid file", async () => {
		const { client, killed, pidPath } = fakeClient([], { alivePids: [4242] });
		mkdirSync(join(root, ".autorag", "fsearch"), { recursive: true });
		writeFileSync(pidPath, "4242");
		await client.stop();
		expect(killed).toEqual([4242]);
		expect(existsSync(pidPath)).toBe(false);
	});

	it("stop is a no-op without a live daemon", async () => {
		const { client, killed } = fakeClient([]);
		await client.stop();
		expect(killed).toEqual([]);
	});
});

describeFSearch("walkFileSearch", () => {
	let docs: string;

	beforeEach(() => {
		docs = join(root, "docs");
		mkdirSync(join(docs, "sub"), { recursive: true });
		writeFileSync(join(docs, "invoice-2026.txt"), "a");
		writeFileSync(join(docs, "sub", "환불 정책.hwp"), "b");
		writeFileSync(join(docs, "notes.md"), "c");
		mkdirSync(join(docs, ".autorag"), { recursive: true });
		writeFileSync(join(docs, ".autorag", "state.json"), "{}");
	});

	it("matches names case-insensitively and returns size, mtime, and type", async () => {
		const result = await walkFileSearch([docs], { query: "INVOICE" });
		expect(result.truncated).toBe(false);
		expect(result.entries).toHaveLength(1);
		expect(result.entries[0]).toMatchObject({
			path: join(docs, "invoice-2026.txt"),
			name: "invoice-2026.txt",
			type: "file",
			size: 1,
		});
		expect(result.entries[0]!.dateModified).toBeDefined();
	});

	it("matches case-sensitively and against the full path on request", async () => {
		expect((await walkFileSearch([docs], { query: "INVOICE", matchCase: true })).entries).toHaveLength(0);
		const byPath = await walkFileSearch([docs], { query: join("sub", "환불"), matchPath: true });
		expect(byPath.entries.map((entry) => entry.name)).toEqual(["환불 정책.hwp"]);
	});

	it("supports regex queries", async () => {
		const result = await walkFileSearch([docs], { query: "\\.hwp$", regex: true });
		expect(result.entries.map((entry) => entry.name)).toEqual(["환불 정책.hwp"]);
	});

	it("filters by kind and skips .autorag state directories", async () => {
		const folders = await walkFileSearch([docs], { query: "", kind: "folders" });
		expect(folders.entries.map((entry) => entry.name).sort()).toEqual(["docs", "sub"]);
		const everything = await walkFileSearch([docs], { query: "state" });
		expect(everything.entries).toHaveLength(0);
	});

	it("sorts by name, size, and modification time", async () => {
		writeFileSync(join(docs, "aa-big.txt"), "0123456789");
		const byName = await walkFileSearch([docs], { query: ".txt", sort: "name-descending" });
		expect(byName.entries.map((entry) => entry.name)).toEqual(["invoice-2026.txt", "aa-big.txt"]);
		const bySize = await walkFileSearch([docs], { query: ".txt", sort: "size-descending" });
		expect(bySize.entries[0]!.name).toBe("aa-big.txt");
	});

	it("applies offset and maxResults after sorting", async () => {
		const result = await walkFileSearch([docs], { query: ".", sort: "name-ascending", offset: 1, maxResults: 2 });
		expect(result.entries).toHaveLength(2);
	});

	it("restricts results to the requested path prefix", async () => {
		const result = await walkFileSearch([docs], { query: "", path: join(docs, "sub") });
		expect(result.entries.map((entry) => entry.name).sort()).toEqual(["sub", "환불 정책.hwp"]);
	});

	it("marks the walk truncated when the visit cap is hit", async () => {
		const result = await walkFileSearch([docs], { query: "" }, { maxVisited: 2 });
		expect(result.truncated).toBe(true);
		expect(result.visited).toBe(2);
	});
});
