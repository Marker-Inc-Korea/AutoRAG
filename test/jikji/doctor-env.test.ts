import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, toNamespacedPath } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JikjiClient } from "../../src/jikji/client.ts";

const workerKeys = ["jikji_background_refresh_root", "JiKjI_BACKGROUND_REFRESH_LOCK"];
let root: string;
let corpus: string;
let data: string;
let binaryDirectory: string;
let calls: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-doctor-env-"));
	corpus = join(root, "corpus");
	data = process.platform === "darwin" ? join(root, "Library", "Application Support") : join(root, "data");
	binaryDirectory = join(root, "bin");
	calls = join(root, "calls.jsonl");
	mkdirSync(corpus);
	mkdirSync(binaryDirectory);
	const database = join(data, "jikji", "index.sqlite");
	mkdirSync(dirname(database), { recursive: true });
	const db = new DatabaseSync(database);
	try {
		db.exec(`CREATE TABLE roots (id INTEGER PRIMARY KEY, canonical_root TEXT NOT NULL UNIQUE);
CREATE TABLE artifacts (root_id INTEGER NOT NULL, kind TEXT NOT NULL);`);
		db.prepare("INSERT INTO roots VALUES (1, ?)").run(toNamespacedPath(realpathSync(corpus)));
		db.exec("INSERT INTO artifacts VALUES (1, 'manifest')");
	} finally {
		db.close();
	}
	const binary = join(binaryDirectory, "jikji");
	writeFileSync(
		binary,
		`#!/usr/bin/env node
const fs = require('node:fs');
const args = process.argv.slice(2);
const markers = Object.keys(process.env).filter(key => /^JIKJI_BACKGROUND_REFRESH_(ROOT|LOCK)$/i.test(key)).sort();
const dataDirectory = process.env.JIKJI_DATA_DIR ?? null;
fs.appendFileSync(${JSON.stringify(calls)}, JSON.stringify({ args, markers, dataDirectory }) + '\\n');
if (args[0] === '--version') console.log('jikji 0.2.1');
else console.log(JSON.stringify({ ok: true, prepared: true, errors: [], warnings: [], manifest: {} }));
`,
	);
	chmodSync(binary, 0o755);
	vi.stubEnv("JIKJI_BACKGROUND_REFRESH_ROOT", undefined);
	vi.stubEnv("JIKJI_BACKGROUND_REFRESH_LOCK", undefined);
});

afterEach(() => {
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
});

interface RecordedCall {
	args: string[];
	markers: string[];
	dataDirectory: string | null;
}

async function inspect(configured: Record<string, string | undefined> = {}): Promise<RecordedCall[]> {
	const client = new JikjiClient({
		root,
		env: {
			PATH: `${binaryDirectory}${delimiter}${process.env.PATH ?? ""}`,
			JIKJI_DATA_DIR: data,
			...configured,
		},
	});
	expect((await client.doctor(corpus)).state).toBe("prepared");
	const recorded: RecordedCall[] = readFileSync(calls, "utf8")
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line));
	expect(recorded.map((call) => call.args[0])).toEqual(["--version", "doctor"]);
	return recorded;
}

describe("Jikji doctor worker environment", () => {
	it.each(["inherited", "configured"])(
		"clears Windows worker-key variants from %s variables before version and doctor",
		async (source) => {
			const configured: Record<string, string | undefined> = {};
			for (const key of workerKeys) {
				if (source === "inherited") vi.stubEnv(key, join(root, key));
				else configured[key] = join(root, key);
			}
			const expected = process.platform === "win32" ? [] : [...workerKeys].sort();
			expect((await inspect(configured)).map((call) => call.markers)).toEqual([expected, expected]);
		},
	);

	it.skipIf(process.platform !== "win32")(
		"does not restore inherited variants removed by an explicit override",
		async () => {
			const configured: Record<string, undefined> = {};
			for (const key of workerKeys) {
				vi.stubEnv(key, join(root, key));
				configured[key] = undefined;
			}
			expect((await inspect(configured)).map((call) => call.markers)).toEqual([[], []]);
		},
	);

	it("keeps an explicit data-directory removal consistent between preflight and native calls", async () => {
		vi.stubEnv("JIKJI_DATA_DIR", join(root, "unrelated-data"));
		const recorded = await inspect({
			JIKJI_DATA_DIR: undefined,
			LOCALAPPDATA: data,
			XDG_DATA_HOME: data,
			HOME: root,
		});
		expect(recorded.map((call) => call.dataDirectory)).toEqual([null, null]);
	});
});
