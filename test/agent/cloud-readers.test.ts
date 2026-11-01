import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import type { CloudEvictResult, CloudFileProbe, CloudProvider } from "../../src/filesystem/cloud-placeholder.ts";
import { loadCloudFingerprints, loadMirrorIndex } from "../../src/mirror/index.ts";

let root: string;
let docs: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-cloud-readers-")));
	docs = join(root, "cloud");
	mkdirSync(docs, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** A probe over ordinary temp files: it reports the given roots and treats every file as already on disk. */
class FakeProbe implements CloudFileProbe {
	private readonly roots: readonly string[];
	constructor(roots: readonly string[]) {
		this.roots = roots;
	}
	async cloudRoots(): Promise<readonly string[]> {
		return this.roots;
	}
	async datalessAmong(): Promise<ReadonlySet<string>> {
		return new Set();
	}
	async providerOf(): Promise<CloudProvider | undefined> {
		return undefined;
	}
	canRelease(): boolean {
		return false;
	}
	async evict(): Promise<CloudEvictResult> {
		return { ok: true };
	}
}

/** fingerprint output for `path`, echoing the file's trimmed text as its exact hash (like dupey's tab format). */
function fingerprintOutput(path: string): string {
	const text = readFileSync(path, "utf8");
	return `path\t${path}\nformat\tTxt\nchars\t${text.length}\nexact\t${text.trim()}\nminhash_width\t128\nmodified\t-\nrevision\t-\n`;
}

/** A dupey double that records every command and answers scans with an empty result. */
function fakeDupey(): { readonly calls: string[][]; readonly run: (args: readonly string[]) => Promise<string> } {
	const calls: string[][] = [];
	return {
		calls,
		run: async (args) => {
			calls.push([...args]);
			if (args[0] === "scan") return JSON.stringify({ dir: args[1], files: [], families: [], errors: [] });
			if (args[0] === "fingerprint") return fingerprintOutput(args[1] ?? "");
			throw new Error(`unexpected dupey command: ${args[0]}`);
		},
	};
}

function makeAgent(extra: Omit<AutoRAGAgentOptions, "searchPaths">): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
		...extra,
	});
}

describe("dupey and cloud folders", () => {
	it("never runs dupey scan over a search root inside a cloud root", async () => {
		writeFileSync(join(docs, "note.txt"), "hello\n");
		const dupey = fakeDupey();
		const agent = makeAgent({ cloudProbe: new FakeProbe([docs]), dupey: { run: dupey.run } });

		await agent.refresh(true);

		expect(dupey.calls.some((args) => args[0] === "scan")).toBe(false);
	});

	it("excludes the cloud folders nested under a local search root by name", async () => {
		const local = join(root, "local");
		mkdirSync(local, { recursive: true });
		writeFileSync(join(local, "note.txt"), "hello\n");
		const nested = join(local, "Library", "CloudStorage", "GoogleDrive-user");
		mkdirSync(nested, { recursive: true });
		const dupey = fakeDupey();
		const agent = new AutoRAGAgent({
			searchPaths: [local],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			cloudProbe: new FakeProbe([nested]),
			dupey: { run: dupey.run },
		});

		await agent.scanDuplicateDocuments();

		const scanCalls = dupey.calls.filter((args) => args[0] === "scan");
		expect(scanCalls).toHaveLength(1);
		expect(scanCalls[0]).toEqual(["scan", local, "--json", "--exclude-dir", "GoogleDrive-user"]);
	});

	it("returns no scans for a search root inside a cloud root", async () => {
		writeFileSync(join(docs, "note.txt"), "hello\n");
		const dupey = fakeDupey();
		const agent = makeAgent({ cloudProbe: new FakeProbe([docs]), dupey: { run: dupey.run } });

		const details = await agent.scanDuplicateDocuments();

		expect(details.scans).toEqual([]);
		expect(dupey.calls).toEqual([]);
	});
});

describe("jikji and cloud folders", () => {
	it("skips Jikji for a search root inside a cloud root and reports it", async () => {
		writeFileSync(join(docs, "note.txt"), "hello\n");
		const agent = makeAgent({ cloudProbe: new FakeProbe([docs]), jikji: {} });

		const results = await agent.prepareJikji();

		expect(results?.[0]).toMatchObject({ ok: false, reason: "cloud-root-skipped" });
	});
});

describe("cloud fingerprints across refreshes", () => {
	it("excludes an exact duplicate among cloud files on the following refresh without reading them", async () => {
		const older = join(docs, "older.txt");
		const newer = join(docs, "newer.txt");
		writeFileSync(older, "identical cloud content\n");
		writeFileSync(newer, "identical cloud content\n");
		utimesSync(older, 1, 1);
		utimesSync(newer, 2, 2);
		const dupey = fakeDupey();
		const agent = makeAgent({ cloudProbe: new FakeProbe([docs]), dupey: { run: dupey.run } });

		const first = await agent.refresh(true);

		expect(first.written).toBe(2);
		expect(Object.keys(loadCloudFingerprints(root).entries).sort()).toEqual([older, newer].sort());

		const second = await agent.refresh(false);
		const index = loadMirrorIndex(root);

		expect(second.deleted).toBe(1);
		expect(second.diagnostics.some((diagnostic) => diagnostic.code === "duplicate-excluded")).toBe(true);
		expect(index.entries["/cloud/newer.txt"]).toBeDefined();
		expect(index.entries["/cloud/older.txt"]).toBeUndefined();
		// The dedupe used the stored fingerprints: dupey never scanned the cloud root.
		expect(dupey.calls.some((args) => args[0] === "scan")).toBe(false);
	});
});
