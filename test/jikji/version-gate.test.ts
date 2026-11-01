import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { JikjiClient } from "../../src/jikji/index.ts";
import { cachedJikjiBinaryPath, cargoExecutableName } from "../../src/jikji/installer.ts";
import { JIKJI_INSTALL_VERSION, JIKJI_MIN_VERSION } from "../../src/jikji/version.ts";

const ANSWER_PACK = {
	answer_paths: ["/repo/src/a.ts"],
	paths: ["/repo/src/a.ts"],
	candidates: [{ path: "/repo/src/a.ts", next_read: "original" }],
	evidence_pack: [{ path: "/repo/src/a.ts", next_read: "original" }],
	handoff_action: "direct_use",
	tool_call_policy: { stop_after_find: true, forbidden_tools: [], allowed_followups: [] },
	agent_should_not_rerank: true,
};

let root: string;
let corpus: string;
let binDir: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-version-gate-"));
	corpus = join(root, "corpus");
	binDir = join(root, "bin");
	mkdirSync(corpus, { recursive: true });
	mkdirSync(binDir, { recursive: true });
	// The stand-ins are `#!/usr/bin/env node` scripts. PATH is only binDir so the ambient
	// machine's real jikji/cargo can never be picked up; node is exposed through binDir itself.
	symlinkSync(process.execPath, join(binDir, "node"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

/** A jikji stand-in that answers `--version` without logging, then logs every other call. */
function jikjiScript(versionLine: string, logPath: string): string {
	return [
		"#!/usr/bin/env node",
		'const { appendFileSync } = require("node:fs");',
		"const args = process.argv.slice(2);",
		`if (args[0] === "--version") { console.log(${JSON.stringify(versionLine)}); process.exit(0); }`,
		`appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");`,
		`if (args[0] === "find") console.log(${JSON.stringify(JSON.stringify(ANSWER_PACK))});`,
		"else console.log(JSON.stringify({ prepared: true }));",
		"",
	].join("\n");
}

function writeJikji(path: string, versionLine: string, logPath: string): void {
	mkdirSync(join(path, ".."), { recursive: true });
	writeFileSync(path, jikjiScript(versionLine, logPath));
	chmodSync(path, 0o755);
}

/** cargo stand-in: records argv, then installs a jikji reporting `installedVersion` under `--root`. */
function writeFakeCargo(cargoLog: string, installedVersion: string, installedLog: string): void {
	// The client's PATH is only binDir (so no real jikji/cargo is ever found); this script needs the
	// system utilities, so it sets its own PATH.
	const script = `#!/bin/sh
PATH=/usr/bin:/bin
printf '%s\\n' "$*" >> '${cargoLog}'
root=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--root" ]; then shift; root="$1"; fi
  shift
done
mkdir -p "$root/bin"
cat > "$root/bin/jikji" <<'JIKJI'
${jikjiScript(`jikji ${installedVersion}`, installedLog)}
JIKJI
chmod +x "$root/bin/jikji"
exit 0
`;
	writeFileSync(join(binDir, cargoExecutableName()), script);
	chmodSync(join(binDir, cargoExecutableName()), 0o755);
}

function lines(path: string): string[] {
	if (!existsSync(path)) return [];
	return readFileSync(path, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0);
}

function pathEnv(): { readonly PATH: string } {
	return { PATH: binDir };
}

describe.skipIf(process.platform === "win32")("JikjiClient version floor", () => {
	it("skips a PATH jikji below the floor and uses the cached pinned binary", async () => {
		const staleLog = join(root, "stale.jsonl");
		const goodLog = join(root, "good.jsonl");
		writeJikji(join(binDir, "jikji"), "jikji 0.1.1", staleLog);
		writeJikji(cachedJikjiBinaryPath(root), `jikji ${JIKJI_INSTALL_VERSION}`, goodLog);
		const client = new JikjiClient({ root, env: pathEnv(), autoInstall: false });

		const result = await client.find(corpus, "q");

		expect(result).toMatchObject({ ok: true });
		expect(lines(goodLog)).toHaveLength(1);
		expect(lines(staleLog)).toHaveLength(0);
	});

	it("keeps preferring a PATH jikji that meets the floor", async () => {
		const pathLog = join(root, "path.jsonl");
		const cachedLog = join(root, "cached.jsonl");
		writeJikji(join(binDir, "jikji"), `jikji ${JIKJI_MIN_VERSION}`, pathLog);
		writeJikji(cachedJikjiBinaryPath(root), `jikji ${JIKJI_INSTALL_VERSION}`, cachedLog);
		const client = new JikjiClient({ root, env: pathEnv(), autoInstall: false });

		await client.find(corpus, "q");

		expect(lines(pathLog)).toHaveLength(1);
		expect(lines(cachedLog)).toHaveLength(0);
	});

	it("names the detected version, the path and the floor instead of spawning a stale jikji", async () => {
		const staleLog = join(root, "stale.jsonl");
		const stale = join(binDir, "jikji");
		writeJikji(stale, "jikji 0.1.1", staleLog);
		const client = new JikjiClient({ root, env: pathEnv(), autoInstall: false });

		const found = await client.find(corpus, "q");
		const prepared = await client.prepare(corpus);

		for (const result of [found, prepared]) {
			expect(result.ok).toBe(false);
			if (result.ok) continue;
			expect(result.reason).toBe("unsupported-version");
			expect(result.stderr).toContain("0.1.1");
			expect(result.stderr).toContain(stale);
			expect(result.stderr).toContain(JIKJI_MIN_VERSION);
		}
		expect(lines(staleLog)).toHaveLength(0);
	});

	it("never installs from find, even when a stale PATH jikji is found", async () => {
		const cargoLog = join(root, "cargo.jsonl");
		writeJikji(join(binDir, "jikji"), "jikji 0.1.1", join(root, "stale.jsonl"));
		writeFakeCargo(cargoLog, JIKJI_INSTALL_VERSION, join(root, "installed.jsonl"));
		const client = new JikjiClient({ root, env: pathEnv() });

		const result = await client.find(corpus, "q");

		expect(result).toMatchObject({ ok: false, reason: "unsupported-version" });
		expect(lines(cargoLog)).toHaveLength(0);
	});

	it("replaces a stale PATH jikji on prepare by installing the pinned version, then keeps using it", async () => {
		const staleLog = join(root, "stale.jsonl");
		const cargoLog = join(root, "cargo.jsonl");
		const installedLog = join(root, "installed.jsonl");
		writeJikji(join(binDir, "jikji"), "jikji 0.1.1", staleLog);
		writeFakeCargo(cargoLog, JIKJI_INSTALL_VERSION, installedLog);
		const client = new JikjiClient({ root, env: pathEnv(), timeoutMs: 10_000 });

		const first = await client.prepare(corpus);
		const second = await client.prepare(corpus);

		expect(first).toMatchObject({ ok: true });
		expect(second).toMatchObject({ ok: true });
		expect(lines(cargoLog)).toHaveLength(1);
		expect(lines(cargoLog)[0]).toContain(`--version ${JIKJI_INSTALL_VERSION}`);
		expect(lines(cargoLog)[0]).not.toContain("--force");
		expect(lines(installedLog)).toHaveLength(2);
		expect(lines(staleLog)).toHaveLength(0);
	});

	it("reinstalls over a stale cached binary with --force", async () => {
		const cargoLog = join(root, "cargo.jsonl");
		writeJikji(cachedJikjiBinaryPath(root), "jikji 0.1.1", join(root, "stale.jsonl"));
		writeFakeCargo(cargoLog, JIKJI_INSTALL_VERSION, join(root, "installed.jsonl"));
		const client = new JikjiClient({ root, env: pathEnv(), timeoutMs: 10_000 });

		const result = await client.prepare(corpus);

		expect(result).toMatchObject({ ok: true });
		expect(lines(cargoLog)).toHaveLength(1);
		expect(lines(cargoLog)[0]).toContain("--force");
		expect(lines(join(root, "installed.jsonl"))).toHaveLength(1);
	});

	it("reports why the pinned install could not happen when cargo is missing", async () => {
		const stale = join(binDir, "jikji");
		writeJikji(stale, "jikji 0.1.1", join(root, "stale.jsonl"));
		const client = new JikjiClient({ root, env: pathEnv() });

		const result = await client.prepare(corpus);

		expect(result.ok).toBe(false);
		if (result.ok) return;
		expect(result.reason).toBe("unsupported-version");
		expect(result.stderr).toContain("0.1.1");
		expect(result.stderr).toContain("cargo");
	});

	it("still uses a jikji whose --version output it does not recognise", async () => {
		const log = join(root, "custom.jsonl");
		writeJikji(join(binDir, "jikji"), "custom build without a version", log);
		const client = new JikjiClient({ root, env: pathEnv(), autoInstall: false });

		const result = await client.find(corpus, "q");

		expect(result).toMatchObject({ ok: true });
		expect(lines(log)).toHaveLength(1);
	});

	it("does not probe a jikji it was told to use by explicit path", async () => {
		const log = join(root, "explicit.jsonl");
		const explicit = join(root, "explicit-jikji");
		writeJikji(explicit, "jikji 0.1.1", log);
		const client = new JikjiClient({ binaryPath: explicit, root, autoInstall: false });

		const result = await client.find(corpus, "q");

		expect(result).toMatchObject({ ok: true });
		expect(lines(log)).toHaveLength(1);
	});
});

describe.skipIf(process.platform === "win32")("AutoRAGAgent refresh with a stale jikji", () => {
	it("surfaces the unsupported version in refresh diagnostics instead of a generic failure", async () => {
		const stale = join(binDir, "jikji");
		writeJikji(stale, "jikji 0.1.1", join(root, "stale.jsonl"));
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "a.txt"), "hello\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { env: pathEnv(), autoInstall: false },
			everything: false,
			fsearch: false,
		});

		await agent.refresh(true, { methods: ["jikji"] });

		const status = await agent.getRefreshStatus();
		const diag = status.diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-unsupported-version");
		expect(diag?.message).toContain("0.1.1");
		expect(diag?.message).toContain(JIKJI_MIN_VERSION);
	});
});
