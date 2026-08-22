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
			args: [script, "arg"],
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
			args: [script, "find"],
		});
	});
});
