import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JikjiClient } from "../../src/jikji/client.ts";
import {
	cachedJikjiBinaryPath,
	cargoExecutableName,
	ensureJikjiBinary,
	jikjiExecutableName,
	lookupExecutableInPath,
} from "../../src/jikji/installer.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-installer-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("lookupExecutableInPath", () => {
	it("finds an executable in a PATH directory", () => {
		const binDir = join(root, "bin");
		mkdirSync(binDir, { recursive: true });
		writeFileSync(join(binDir, "jikji"), "#!/bin/sh\n");
		expect(lookupExecutableInPath("jikji", { PATH: binDir })).toBe(join(binDir, "jikji"));
	});

	it("returns undefined when PATH is empty or missing", () => {
		expect(lookupExecutableInPath("jikji", {})).toBeUndefined();
		expect(lookupExecutableInPath("jikji", { PATH: "" })).toBeUndefined();
	});
});

describe("ensureJikjiBinary", () => {
	it("returns the cached binary without invoking cargo", async () => {
		const cached = cachedJikjiBinaryPath(root);
		mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
		writeFileSync(cached, "binary");
		const result = await ensureJikjiBinary({
			root,
			runner: () => {
				throw new Error("runner must not be called for a cached binary");
			},
		});
		expect(result).toEqual({ ok: true, binaryPath: cached, source: "cached" });
	});

	it("degrades with missing-cargo when no Rust toolchain is on PATH", async () => {
		const result = await ensureJikjiBinary({
			root,
			env: { PATH: join(root, "empty") },
			runner: () => {
				throw new Error("runner must not be called without cargo");
			},
		});
		expect(result.ok).toBe(false);
		if (!result.ok) expect(result.reason).toBe("missing-cargo");
	});

	it("installs through cargo and reports the cached binary path", async () => {
		const calls: string[][] = [];
		const result = await ensureJikjiBinary({
			root,
			env: { PATH: join(root, "cargo-bin") },
			cargoLocator: () => "/usr/bin/cargo",
			runner: async (args) => {
				calls.push([...args]);
				mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
				writeFileSync(cachedJikjiBinaryPath(root), "binary");
				return { code: 0, stderr: "" };
			},
		});
		expect(result).toEqual({ ok: true, binaryPath: cachedJikjiBinaryPath(root), source: "installed" });
		expect(calls).toEqual([["install", "jikji-cli", "--locked", "--root", join(root, ".autorag")]]);
	});

	it("pins the version when requested", async () => {
		const calls: string[][] = [];
		await ensureJikjiBinary({
			root,
			version: "0.1.1",
			cargoLocator: () => "/usr/bin/cargo",
			runner: async (args) => {
				calls.push([...args]);
				mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
				writeFileSync(cachedJikjiBinaryPath(root), "binary");
				return { code: 0, stderr: "" };
			},
		});
		expect(calls[0]).toContain("--version");
		expect(calls[0]).toContain("0.1.1");
	});

	it("degrades with install-failed when cargo exits nonzero", async () => {
		const result = await ensureJikjiBinary({
			root,
			cargoLocator: () => "/usr/bin/cargo",
			runner: async () => ({ code: 101, stderr: "error: could not compile" }),
		});
		expect(result.ok).toBe(false);
		if (!result.ok) {
			expect(result.reason).toBe("install-failed");
			expect(result.message).toContain("could not compile");
		}
	});
});

describe("JikjiClient binary resolution", () => {
	it("uses the cached .autorag/bin binary when PATH has no jikji", async () => {
		const cached = cachedJikjiBinaryPath(root);
		mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
		writeFileSync(
			cached,
			process.platform === "win32"
				? '#!/usr/bin/env node\nconsole.log(JSON.stringify({ not: "a pack" }));\n'
				: '#!/bin/sh\nprintf \'{"not":"a pack"}\\n\'\n',
		);
		chmodSync(cached, 0o755);
		const client = new JikjiClient({ root, autoInstall: false, timeoutMs: 5_000, env: { PATH: "" } });
		const result = await client.find(root, "query");
		// The fixture emits invalid JSON, proving the cached binary ran.
		expect(result.ok).toBe(false);
		if (!result.ok) expect(result.reason).toBe("bad-answer-pack");
	});

	it("falls back to the bare jikji command when autoInstall is disabled and nothing is installed", async () => {
		const client = new JikjiClient({ root, autoInstall: false, timeoutMs: 5_000 });
		// No binary anywhere: resolution falls back to `jikji`, which spawns or
		// fails exactly as before this change (no throw either way).
		const result = await client.find(root, "query");
		expect(typeof result.ok).toBe("boolean");
	});

	it("names the executable jikji.exe on win32", () => {
		expect(jikjiExecutableName("win32")).toBe("jikji.exe");
		expect(jikjiExecutableName("darwin")).toBe("jikji");
	});
});

// ---------------------------------------------------------------------------
// Auto-install coordination (POSIX sh fixtures; Windows cannot exec them).
// ---------------------------------------------------------------------------

const JIKJI_ANSWER_PACK_JSON = JSON.stringify({
	answer_paths: ["/repo/src/a.ts"],
	paths: ["/repo/src/a.ts"],
	candidates: [{ path: "/repo/src/a.ts", next_read: "cache" }],
	evidence_pack: [{ path: "/repo/src/a.ts", next_read: "cache" }],
	handoff_action: "direct_use",
	tool_call_policy: { stop_after_find: true, forbidden_tools: [], allowed_followups: [] },
	agent_should_not_rerank: true,
});

const FAKE_LOG_TIMEOUT_MS = 10_000;

/**
 * Write a `cargo` stand-in into `directory`. It records its argv, optionally
 * waits until `releasePath` appears (to hold one install open across concurrent
 * prepares), then writes a `jikji` stand-in into `<--root>/bin/jikji` exactly
 * where {@link cachedJikjiBinaryPath} expects it. Uses `#!/bin/sh` so no PATH
 * lookup of an interpreter is needed; the embedded utility calls (`mkdir`,
 * `chmod`, `cat`) resolve from the ambient PATH.
 */
function writeFakeCargo(directory: string, cargoLogPath: string, jikjiLogPath: string, releasePath?: string): void {
	const wait = releasePath === undefined ? "" : `while [ ! -f '${releasePath}' ]; do sleep 0.05; done`;
	const script = `#!/bin/sh
printf '%s\\n' "$*" >> '${cargoLogPath}'
${wait}
root=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--root" ]; then shift; root="$1"; fi
  shift
done
mkdir -p "$root/bin"
cat > "$root/bin/jikji" <<'JIKJI'
#!/bin/sh
printf '%s\\n' "$*" >> '${jikjiLogPath}'
printf '%s\\n' '${JIKJI_ANSWER_PACK_JSON}'
JIKJI
chmod +x "$root/bin/jikji"
exit 0
`;
	writeFileSync(join(directory, cargoExecutableName()), script);
	chmodSync(join(directory, cargoExecutableName()), 0o755);
}

function logLines(path: string): readonly string[] {
	if (!existsSync(path)) return [];
	return readFileSync(path, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0);
}

async function waitForLog(path: string): Promise<void> {
	// The writer is a child process, so there is no in-process signal to await:
	// poll the real filesystem until the fake cargo records its invocation.
	await vi.waitUntil(() => existsSync(path) && statSync(path).size > 0, {
		timeout: FAKE_LOG_TIMEOUT_MS,
		interval: 25,
	});
}

/** Drop PATH entries that would expose a real jikji/cargo binary. */
function stripExecutableDirs(pathValue: string, names: readonly string[]): string {
	return pathValue
		.split(delimiter)
		.filter((directory) => directory.length > 0 && names.every((name) => !existsSync(join(directory, name))))
		.join(delimiter);
}

describe.skipIf(process.platform === "win32")("JikjiClient auto-install coordination", () => {
	let savedPath: string | undefined;
	let binDir: string;
	let cargoLog: string;
	let jikjiLog: string;

	beforeEach(() => {
		savedPath = process.env.PATH;
		// Hide any real jikji/cargo from the ambient PATH: forwarding the
		// configured env to the installer is part of the behavior under test, and
		// a regression must fail fast (missing-cargo) instead of compiling the crate.
		process.env.PATH = stripExecutableDirs(savedPath ?? "", [jikjiExecutableName(), cargoExecutableName()]);
		binDir = join(root, "fake-bin");
		cargoLog = join(root, "cargo-calls.jsonl");
		jikjiLog = join(root, "jikji-calls.jsonl");
		mkdirSync(binDir, { recursive: true });
	});

	afterEach(() => {
		if (savedPath === undefined) delete process.env.PATH;
		else process.env.PATH = savedPath;
	});

	/** cargo is discoverable only through the client's configured env. */
	function clientEnv(): { readonly PATH: string } {
		return { PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}` };
	}

	it("shares a single cargo install across concurrent prepares for multiple roots", async () => {
		const releasePath = join(root, "cargo-release");
		writeFakeCargo(binDir, cargoLog, jikjiLog, releasePath);
		const client = new JikjiClient({ root, env: clientEnv(), timeoutMs: 10_000 });

		// All three prepares are initiated before the installer is allowed to
		// finish: one cargo install must serve every root.
		const preparing = Promise.all([
			client.prepare(join(root, "corpus-a")),
			client.prepare(join(root, "corpus-b")),
			client.prepare(join(root, "corpus-a")),
		]);
		await waitForLog(cargoLog); // install started and is blocked on the release
		writeFileSync(releasePath, "");

		const results = await preparing;
		expect(results.every((result) => result.ok)).toBe(true);
		const cargoCalls = logLines(cargoLog);
		expect(cargoCalls).toHaveLength(1);
		expect(cargoCalls[0]).toContain(`--root ${join(root, ".autorag")}`);
		expect(logLines(jikjiLog)).toHaveLength(3);
	});

	it("never lets a concurrent find trigger or wait for the in-flight install", async () => {
		const releasePath = join(root, "cargo-release");
		writeFakeCargo(binDir, cargoLog, jikjiLog, releasePath);
		const client = new JikjiClient({ root, env: clientEnv(), timeoutMs: 10_000 });

		const preparing = client.prepare(join(root, "corpus-a"));
		await waitForLog(cargoLog); // the install is now blocked on the release file

		const findResult = await client.find(join(root, "corpus-a"), "query");

		// The install was still blocked when find settled: it neither waited for it
		// nor started a second installer, and degraded to the bare fallback.
		expect(findResult).toMatchObject({ ok: false, reason: "spawn-error" });
		expect(logLines(cargoLog)).toHaveLength(1);
		expect(logLines(jikjiLog)).toHaveLength(0);

		writeFileSync(releasePath, "");
		expect(await preparing).toMatchObject({ ok: true });
		expect(logLines(jikjiLog)).toHaveLength(1);
	});

	it("still installs on a later refresh after a query-time miss", async () => {
		writeFakeCargo(binDir, cargoLog, jikjiLog);
		const client = new JikjiClient({ root, env: clientEnv(), timeoutMs: 10_000 });

		const miss = await client.find(join(root, "corpus-a"), "query");
		expect(miss).toMatchObject({ ok: false, reason: "spawn-error" });
		expect(logLines(cargoLog)).toHaveLength(0); // find never installs

		const prepared = await client.prepare(join(root, "corpus-a"));
		expect(prepared).toMatchObject({ ok: true });
		expect(logLines(cargoLog)).toHaveLength(1);
	});
});
