import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, toNamespacedPath } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JikjiClient } from "../../src/jikji/client.ts";
import { jikjiDoctorDiagnostics } from "../../src/jikji/diagnostics.ts";
import { jikjiDatabasePath } from "../../src/jikji/doctor.ts";

let root: string;
let corpus: string;
let data: string;
let database: string;
let binary: string;
let calls: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-doctor-"));
	corpus = join(root, "corpus");
	data = join(root, "data");
	database = join(data, "jikji", "index.sqlite");
	binary = join(root, "jikji.cjs");
	calls = join(root, "calls.jsonl");
	mkdirSync(corpus);
});

afterEach(() => {
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
});

function registerRoot(path = realpathSync(corpus), manifest = true): void {
	mkdirSync(dirname(database), { recursive: true });
	const db = new DatabaseSync(database);
	try {
		db.exec(`CREATE TABLE IF NOT EXISTS roots (id INTEGER PRIMARY KEY, canonical_root TEXT NOT NULL UNIQUE, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (root_id INTEGER NOT NULL, ordinal INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, row_json TEXT NOT NULL);`);
		db.prepare("INSERT OR IGNORE INTO roots VALUES (1, ?, 0)").run(path);
		if (manifest) db.prepare("INSERT INTO artifacts (root_id, kind, row_json) VALUES (1, 'manifest', '{}')").run();
	} finally {
		db.close();
	}
}

function nativeReport(overrides: Record<string, unknown> = {}): string {
	return JSON.stringify({
		ok: true,
		prepared: true,
		errors: [],
		warnings: [],
		manifest: { docs_failed: 0 },
		...overrides,
	});
}

function fakeDoctor(stdout = nativeReport(), code = 0, stderr = ""): JikjiClient {
	writeFileSync(
		binary,
		`#!/usr/bin/env node\nconst fs = require('node:fs');
fs.appendFileSync(${JSON.stringify(calls)}, JSON.stringify(process.argv.slice(2)) + '\\n');
process.stdout.write(${JSON.stringify(stdout)});
process.stderr.write(${JSON.stringify(stderr)});
process.exit(${code});\n`,
	);
	chmodSync(binary, 0o755);
	return new JikjiClient({ binaryPath: binary, root, env: { JIKJI_DATA_DIR: data } });
}

describe("Jikji doctor", () => {
	it("does not create a store or invoke native doctor for an unprepared or legacy-only root", async () => {
		mkdirSync(join(corpus, ".jikji"));
		const result = await fakeDoctor().doctor(corpus);
		expect(result).toEqual({ state: "not-prepared", errors: [], warnings: [] });
		expect(existsSync(data)).toBe(false);
		expect(existsSync(calls)).toBe(false);
		expect(existsSync(join(root, ".autorag", "bin"))).toBe(false);
	});

	it.each(["other-root", "registered-without-manifest"])(
		"skips doctor for %s without changing the existing store",
		async (condition) => {
			registerRoot(
				condition === "other-root" ? `${realpathSync(corpus)}-other` : realpathSync(corpus),
				condition === "other-root",
			);
			const before = readFileSync(database);
			const result = await fakeDoctor().doctor(corpus);
			expect(result.state).toBe("not-prepared");
			expect(existsSync(calls)).toBe(false);
			expect(readFileSync(database)).toEqual(before);
		},
	);

	it("uses the registered canonical root and obtains failed-document counts", async () => {
		registerRoot(toNamespacedPath(realpathSync(corpus)));
		const result = await fakeDoctor(nativeReport({ manifest: { docs_failed: 3 } })).doctor(corpus);
		expect(result).toEqual({ state: "prepared", docsFailed: 3, errors: [], warnings: [] });
		expect(JSON.parse(readFileSync(calls, "utf8").trim())).toEqual(["doctor", realpathSync(corpus), "--json"]);
		expect(jikjiDoctorDiagnostics(result)).toContainEqual(
			expect.objectContaining({ code: "jikji-documents-failed", message: "Jikji manifest reports docs_failed=3." }),
		);
	});

	it("reads an unhealthy exit-1 report and preserves native errors and warnings verbatim", async () => {
		registerRoot();
		const error = "missing required artifact row: wiki_index";
		const warning = `original warning at ${join(corpus, "한글 file.md")}\nsecond line`;
		const result = await fakeDoctor(
			nativeReport({ ok: false, prepared: false, errors: [error], warnings: [warning] }),
			1,
		).doctor(corpus);
		expect(result).toEqual({ state: "needs-reprepare", docsFailed: 0, errors: [error], warnings: [warning] });
		expect(jikjiDoctorDiagnostics(result).map((entry) => entry.message)).toEqual([error, warning]);
	});

	it("does not invent a zero failure count when the native manifest omits it", async () => {
		registerRoot();
		const result = await fakeDoctor(nativeReport({ manifest: {} })).doctor(corpus);
		expect(result.state).toBe("prepared");
		expect(result.docsFailed).toBeUndefined();
	});

	it.each(["not JSON", nativeReport({ errors: "not an array" }), nativeReport({ manifest: { docs_failed: -1 } })])(
		"reports an invalid native report as unavailable: %s",
		async (stdout) => {
			registerRoot();
			const result = await fakeDoctor(stdout, 1, "native stderr\n").doctor(corpus);
			expect(result.state).toBe("unavailable");
			expect(result.errors.join("\n")).toContain(stdout);
			expect(result.errors.join("\n")).toContain("native stderr\n");
			expect(result.errors.join("\n")).toContain("exit status 1");
		},
	);

	it("retains native process failure details", async () => {
		registerRoot();
		const stderr = `cannot open ${database}: disk I/O error\n`;
		const result = await fakeDoctor("native output\n", 7, stderr).doctor(corpus);
		expect(result.state).toBe("unavailable");
		expect(result.errors[0]).toContain("exit status 7");
		expect(result.errors[0]).toContain(stderr);
		expect(result.errors[0]).toContain("native output\n");
	});

	it("reports a damaged database instead of treating it as an unprepared root", async () => {
		mkdirSync(dirname(database), { recursive: true });
		writeFileSync(database, "not a sqlite database");
		const result = await fakeDoctor().doctor(corpus);
		expect(result.state).toBe("unavailable");
		expect(result.errors.join("\n")).toContain("not a database");
		expect(existsSync(calls)).toBe(false);
	});

	it("rechecks disk state on the next call without reconstructing the client", async () => {
		const client = fakeDoctor();
		expect((await client.doctor(corpus)).state).toBe("not-prepared");
		registerRoot();
		expect((await client.doctor(corpus)).state).toBe("prepared");
	});

	it("does not spawn for an already aborted inspection", async () => {
		registerRoot();
		const result = await fakeDoctor().doctor(corpus, { signal: AbortSignal.abort() });
		expect(result.state).toBe("unavailable");
		expect(result.errors.join("\n")).toContain("aborted");
		expect(existsSync(calls)).toBe(false);
	});

	it("does not inherit native background-worker markers into the doctor process", async () => {
		registerRoot();
		vi.stubEnv("JIKJI_BACKGROUND_REFRESH_ROOT", join(root, "unrelated-root"));
		vi.stubEnv("JIKJI_BACKGROUND_REFRESH_LOCK", join(root, "worker.lock"));
		writeFileSync(
			binary,
			`#!/usr/bin/env node\nconst report = ${nativeReport()};
report.warnings = ['JIKJI_BACKGROUND_REFRESH_ROOT', 'JIKJI_BACKGROUND_REFRESH_LOCK'].filter(key => process.env[key] !== undefined);
process.stdout.write(JSON.stringify(report));\n`,
		);
		chmodSync(binary, 0o755);
		const client = new JikjiClient({ binaryPath: binary, env: { JIKJI_DATA_DIR: data } });
		expect(await client.doctor(corpus)).toEqual({ state: "prepared", docsFailed: 0, errors: [], warnings: [] });
	});
});

describe("Jikji data directory", () => {
	it("matches Windows environment names regardless of casing", () => {
		expect(jikjiDatabasePath({ Jikji_Data_Dir: data, LocalAppData: "unused" }, "win32")).toBe(database);
		expect(jikjiDatabasePath({ LocalAppData: data }, "win32")).toBe(database);
		expect(jikjiDatabasePath({ JIKJI_DATA_DIR: data, jikji_data_dir: "unused" }, "win32")).toBe(database);
	});
	it("honors an explicit data directory before platform defaults", () => {
		expect(jikjiDatabasePath({ JIKJI_DATA_DIR: data, LOCALAPPDATA: "unused" }, "win32")).toBe(database);
	});
	it("matches the native platform defaults and empty override behavior", () => {
		expect(jikjiDatabasePath({ LOCALAPPDATA: data, JIKJI_DATA_DIR: "" }, "win32")).toBe(database);
		expect(jikjiDatabasePath({ HOME: root }, "darwin")).toBe(
			join(root, "Library", "Application Support", "jikji", "index.sqlite"),
		);
		expect(jikjiDatabasePath({ HOME: root }, "linux")).toBe(join(root, ".local", "share", "jikji", "index.sqlite"));
		expect(jikjiDatabasePath({ HOME: root, XDG_DATA_HOME: data }, "linux")).toBe(database);
	});
});
