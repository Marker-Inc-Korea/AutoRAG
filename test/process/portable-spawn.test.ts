import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { portableSpawnCommand } from "../../src/process/portable-spawn.ts";

const tempDirs: string[] = [];

afterEach(() => {
	for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("portableSpawnCommand", () => {
	it("leaves commands unchanged on the current non-Windows platform", () => {
		expect(portableSpawnCommand("native-command", ["arg"])).toEqual({
			command: "native-command",
			args: ["arg"],
		});
	});

	it("keeps existing native executables unchanged", () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const binary = join(root, "tool.exe");
		writeFileSync(binary, "native-placeholder", "utf8");

		expect(portableSpawnCommand(binary, ["arg"])).toEqual({ command: binary, args: ["arg"] });
	});

	it("executes JavaScript fixtures with Node when running on Windows", () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const script = join(root, "tool.mjs");
		writeFileSync(script, "process.exit(0);", "utf8");
		chmodSync(script, 0o755);

		expect(portableSpawnCommand(script, ["arg"], "win32")).toEqual({
			command: process.execPath,
			args: process.versions.bun ? ["run", script, "--", "arg"] : [script, "arg"],
		});
	});

	it("executes extensionless Node shebang fixtures with Node on Windows", () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const script = join(root, "fake-jikji");
		writeFileSync(script, "#!/usr/bin/env node\nprocess.exit(0);\n", "utf8");
		chmodSync(script, 0o755);

		expect(portableSpawnCommand(script, ["find"], "win32")).toEqual({
			command: process.execPath,
			args: process.versions.bun ? ["run", script, "--", "find"] : [script, "find"],
		});
	});

	it("executes Node shebang fixtures with an executable extension on Windows", () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const script = join(root, "fake-minsync.exe");
		writeFileSync(script, "#!/usr/bin/env node\nprocess.exit(0);\n", "utf8");
		chmodSync(script, 0o755);

		expect(portableSpawnCommand(script, ["sync"], "win32")).toEqual({
			command: process.execPath,
			args: process.versions.bun ? ["run", script, "--", "sync"] : [script, "sync"],
		});
	});

	it("runs extensionless Node shebang fixtures with Node module semantics", async () => {
		if (process.platform !== "win32") return;
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const script = join(root, "tool");
		writeFileSync(script, "#!/usr/bin/env node\nprocess.stdout.write('ok');\n", "utf8");
		chmodSync(script, 0o755);

		const portable = portableSpawnCommand(script, [], "win32");
		const { spawnSync } = await import("node:child_process");
		const result = spawnSync(portable.command, [...portable.args], { encoding: "utf8" });

		expect(result.status).toBe(0);
		expect(result.stdout).toBe("ok");
	});
});
