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
			command: process.versions.bun ? "node.exe" : process.execPath,
			args: [script, "find"],
		});
	});

	it.each([
		["/bin/sh with LF", "#!/bin/sh\n"],
		["/bin/sh with CRLF", "#!/bin/sh\r\n"],
		["env sh with LF", "#!/usr/bin/env sh\n"],
		["env sh with CRLF", "#!/usr/bin/env sh\r\n"],
	])("uses the %s shebang even when the body mentions node", (_name, shebang) => {
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const script = join(root, "shell-wrapper");
		writeFileSync(script, `${shebang}printf '%s\\n' 'node'\n`, "utf8");
		const args = ["argument with spaces", "--flag"];

		expect(portableSpawnCommand(script, args, "win32")).toEqual({
			command: "bash",
			args: [script, ...args],
		});
	});

	it.each(["node", "/bin/sh", "/usr/bin/env sh"])(
		"does not infer an interpreter from %s in the body of an unsupported script",
		(interpreter) => {
			const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
			tempDirs.push(root);
			const script = join(root, "python-wrapper");
			writeFileSync(script, `#!/usr/bin/env python3\nprint("${interpreter}")\n`, "utf8");

			expect(portableSpawnCommand(script, ["arg"], "win32")).toEqual({
				command: script,
				args: ["arg"],
			});
		},
	);

	it.each(["linux", "darwin"] as const)("leaves existing shebang scripts unchanged on %s", (platform) => {
		const root = mkdtempSync(join(tmpdir(), "autorag-portable-spawn-"));
		tempDirs.push(root);
		const script = join(root, "shell-wrapper");
		writeFileSync(script, "#!/bin/sh\nprintf '%s\\n' 'node'\n", "utf8");

		expect(portableSpawnCommand(script, ["arg"], platform)).toEqual({
			command: script,
			args: ["arg"],
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
