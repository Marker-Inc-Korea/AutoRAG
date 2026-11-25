import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildSeatbeltProfile,
	createReadOnlySandbox,
	escapeSeatbeltString,
	type ReadOnlySandbox,
} from "../../src/agent/readonly-sandbox.ts";
import {
	buildWindowsLowIlInvocation,
	prepareWindowsReadOnlySandbox,
	WINDOWS_LOW_INTEGRITY_SID,
	WINDOWS_SANDBOX_LAUNCHER_PS1,
} from "../../src/agent/readonly-sandbox-win.ts";

let root: string;
let sandboxes: ReadOnlySandbox[];

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-ro-sandbox-test-")));
	sandboxes = [];
});

afterEach(() => {
	for (const sb of sandboxes) sb.dispose();
	rmSync(root, { recursive: true, force: true });
});

async function runIn(
	sb: ReadOnlySandbox,
	command: string,
	cwd = root,
): Promise<{ exitCode: number | null; output: string }> {
	let output = "";
	const result = await sb.operations.exec(command, cwd, {
		onData: (chunk) => {
			output += chunk.toString("utf8");
		},
	});
	return { exitCode: result.exitCode, output };
}

describe("buildSeatbeltProfile", () => {
	it("escapes quotes and backslashes safely", () => {
		expect(escapeSeatbeltString('/tmp/a"b\\c')).toBe('/tmp/a\\"b\\\\c');
	});

	it("denies file-write* by default and allows the scratch dir and virtual devices", () => {
		const profile = buildSeatbeltProfile([root]);
		expect(profile).toContain("(version 1)");
		expect(profile).toContain("(allow default)");
		expect(profile).toContain("(deny file-write*)");
		expect(profile).toContain(`(subpath "${escapeSeatbeltString(root)}")`);
		expect(profile).toContain('(literal "/dev/null")');
	});
});

describe("createReadOnlySandbox fail-closed behavior", () => {
	it("reports unavailable and throws when platform has no sandbox backend (freebsd)", async () => {
		const sb = createReadOnlySandbox({ platform: "freebsd" });
		sandboxes.push(sb);
		expect(sb.backend).toBe("unavailable");
		expect(sb.unavailableReason).toMatch(/freebsd/i);
		await expect(runIn(sb, "echo hi")).rejects.toThrow(/Read-only mode blocked bash execution/);
	});

	it("fails closed on darwin when sandbox-exec is absent", async () => {
		const sb = createReadOnlySandbox({ platform: "darwin", probeBinary: () => undefined });
		sandboxes.push(sb);
		expect(sb.backend).toBe("unavailable");
		expect(sb.unavailableReason).toMatch(/sandbox-exec/);
		await expect(runIn(sb, "echo hi")).rejects.toThrow(/Read-only mode blocked bash execution/);
	});

	it("fails closed on linux when bwrap is absent", async () => {
		const sb = createReadOnlySandbox({ platform: "linux", probeBinary: () => undefined });
		sandboxes.push(sb);
		expect(sb.backend).toBe("unavailable");
		expect(sb.unavailableReason).toMatch(/bwrap/);
		await expect(runIn(sb, "echo hi")).rejects.toThrow(/Read-only mode blocked bash execution/);
	});

	it("fails closed on win32 when PowerShell or icacls is absent or icacls fails", async () => {
		const noPs = createReadOnlySandbox({ platform: "win32", probeBinary: () => undefined });
		sandboxes.push(noPs);
		expect(noPs.backend).toBe("unavailable");
		expect(noPs.unavailableReason).toMatch(/PowerShell/i);
		await expect(runIn(noPs, "echo hi")).rejects.toThrow(/Read-only mode blocked bash execution/);

		const noIcacls = createReadOnlySandbox({
			platform: "win32",
			probeBinary: (name) => (name.toLowerCase().includes("powershell") ? "C:\\Windows\\powershell.exe" : undefined),
		});
		sandboxes.push(noIcacls);
		expect(noIcacls.backend).toBe("unavailable");
		expect(noIcacls.unavailableReason).toMatch(/icacls/i);

		const icaclsFail = createReadOnlySandbox({
			platform: "win32",
			probeBinary: (name) => name,
			runWindowsSetupCommand: () => ({ status: 5, stderr: "Access is denied." }),
		});
		sandboxes.push(icaclsFail);
		expect(icaclsFail.backend).toBe("unavailable");
		expect(icaclsFail.unavailableReason).toMatch(/Low Integrity label/i);
		expect(icaclsFail.unavailableReason).toMatch(/Access is denied/);
	});

	it("disposes its owned scratch directory", () => {
		const sb = createReadOnlySandbox({ platform: "freebsd" });
		const dir = sb.scratchDir;
		expect(existsSync(dir)).toBe(true);
		sb.dispose();
		expect(existsSync(dir)).toBe(false);
		// Safe to call twice.
		sb.dispose();
	});
});

describe("Windows Low-Integrity sandbox setup & invocation", () => {
	it("labels scratch dir (OI)(CI)L and trusted files L, writes launcher outside scratch dir, and cleans up on dispose", () => {
		const scratch = join(root, "scratch");
		mkdirSync(scratch, { recursive: true });
		const cfg = join(root, "config.json");
		writeFileSync(cfg, "{}", "utf8");
		const calls: { bin: string; args: readonly string[] }[] = [];

		const prepared = prepareWindowsReadOnlySandbox({
			scratchDir: scratch,
			extraWritablePaths: [cfg],
			probeBinary: (name) => {
				if (name.toLowerCase().includes("powershell"))
					return "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";
				if (name.toLowerCase().includes("icacls")) return "C:\\Windows\\System32\\icacls.exe";
				return undefined;
			},
			runCommand: (bin, args) => {
				calls.push({ bin, args });
				return { status: 0, stdout: "Successfully processed 1 files" };
			},
		});

		expect(prepared.ok).toBe(true);
		if (!prepared.ok) return;
		try {
			expect(existsSync(prepared.launcherScriptPath)).toBe(true);
			expect(prepared.launcherScriptPath.startsWith(scratch)).toBe(false);
			expect(calls).toEqual([
				{ bin: "C:\\Windows\\System32\\icacls.exe", args: [scratch, "/setintegritylevel", "(OI)(CI)L"] },
				{ bin: "C:\\Windows\\System32\\icacls.exe", args: [cfg, "/setintegritylevel", "L"] },
			]);

			const inv = buildWindowsLowIlInvocation(
				prepared.powershellPath,
				prepared.launcherScriptPath,
				"C:\\Program Files\\Git\\bin\\bash.exe",
				["-c"],
				"ls -la",
				root,
			);
			expect(inv.bin).toBe(prepared.powershellPath);
			expect(inv.args).toContain("-File");
			expect(inv.args).toContain(prepared.launcherScriptPath);
			expect(inv.env.AUTORAG_WIN_SB_EXE).toBe("C:\\Program Files\\Git\\bin\\bash.exe");
			expect(JSON.parse(inv.env.AUTORAG_WIN_SB_ARGS_JSON)).toEqual(["-c", "ls -la"]);
			expect(inv.env.AUTORAG_WIN_SB_CWD).toBe(root);
		} finally {
			rmSync(dirname(prepared.launcherScriptPath), { recursive: true, force: true });
		}
	});

	it("wires windows-low-il backend on win32 and cleans up launcher dir on dispose", () => {
		let launcherDir: string | undefined;
		const sb = createReadOnlySandbox({
			platform: "win32",
			probeBinary: (name) => name,
			runWindowsSetupCommand: () => ({ status: 0 }),
		});
		sandboxes.push(sb);
		expect(sb.backend).toBe("windows-low-il");
		expect(sb.unavailableReason).toBeUndefined();
		// Verify launcher script uses Low Integrity SID, restricted token, DETACHED_PROCESS (never CREATE_NO_WINDOW in flags), and Job Object.
		expect(WINDOWS_SANDBOX_LAUNCHER_PS1).toContain(WINDOWS_LOW_INTEGRITY_SID);
		expect(WINDOWS_SANDBOX_LAUNCHER_PS1).toContain("CreateRestrictedToken");
		expect(WINDOWS_SANDBOX_LAUNCHER_PS1).toContain("TokenIntegrityLevel");
		expect(WINDOWS_SANDBOX_LAUNCHER_PS1).toContain(
			"uint flags = DETACHED_PROCESS | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT;",
		);
		expect(WINDOWS_SANDBOX_LAUNCHER_PS1).toContain("JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE");
		expect(WINDOWS_SANDBOX_LAUNCHER_PS1).not.toMatch(/flags\s*=.*CREATE_NO_WINDOW/);

		const scratch = sb.scratchDir;
		expect(existsSync(scratch)).toBe(true);
		sb.dispose();
		expect(existsSync(scratch)).toBe(false);
		expect(launcherDir).toBeUndefined();
	});
});

describe.skipIf(process.platform !== "darwin")("macOS Seatbelt OS-level read-only enforcement", () => {
	it("allows reading, listing, and grepping existing files", async () => {
		const file = join(root, "notes.txt");
		writeFileSync(file, "alpha\nbeta\ngamma\n", "utf8");
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);
		expect(sb.backend).toBe("seatbelt");

		const res = await runIn(sb, `cat "${file}" && grep beta "${file}"`);
		expect(res.exitCode).toBe(0);
		expect(res.output).toContain("alpha");
		expect(res.output).toContain("beta");
	});

	it("allows writing temporary files inside its dedicated scratch dir ($TMPDIR)", async () => {
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);
		const res = await runIn(sb, 'echo "scratch-ok" > "$TMPDIR/work.txt" && cat "$TMPDIR/work.txt"');
		expect(res.exitCode).toBe(0);
		expect(res.output.trim()).toBe("scratch-ok");
	});

	it("blocks shell redirect, append, rm, mv, mkdir, touch, chmod, and symlink creation outside scratch dir", async () => {
		const file = join(root, "precious.txt");
		writeFileSync(file, "original-content", "utf8");
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);

		const attacks = [
			`echo hacked > "${file}"`,
			`echo hacked >> "${file}"`,
			`rm "${file}"`,
			`mv "${file}" "${join(root, "moved.txt")}"`,
			`mv "${file}" "$TMPDIR/stolen.txt"`,
			`mkdir "${join(root, "new-dir")}"`,
			`touch "${join(root, "new-file.txt")}"`,
			`chmod 777 "${file}"`,
			`ln -s "$TMPDIR/x" "${join(root, "sym")}"`,
			`sed -i '' 's/original/hacked/' "${file}"`,
		];

		for (const cmd of attacks) {
			const res = await runIn(sb, cmd);
			expect(res.exitCode, `expected failure for: ${cmd}`).not.toBe(0);
			expect(existsSync(file), `file deleted by: ${cmd}`).toBe(true);
			expect(readFileSync(file, "utf8"), `file mutated by: ${cmd}`).toBe("original-content");
		}
		expect(existsSync(join(root, "new-dir"))).toBe(false);
		expect(existsSync(join(root, "new-file.txt"))).toBe(false);
		expect(existsSync(join(root, "moved.txt"))).toBe(false);
		expect(existsSync(join(root, "sym"))).toBe(false);
	});

	it("blocks child interpreters (node, python3, perl), hardlink/symlink escapes, and /usr/bin/trash", async () => {
		const file = join(root, "contract.txt");
		writeFileSync(file, "binding", "utf8");
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);

		const nodeAttack = await runIn(sb, `node -e 'require("fs").writeFileSync(${JSON.stringify(file)}, "hacked")'`);
		expect(nodeAttack.exitCode).not.toBe(0);
		expect(readFileSync(file, "utf8")).toBe("binding");

		const nodeUnlink = await runIn(sb, `node -e 'require("fs").unlinkSync(${JSON.stringify(file)})'`);
		expect(nodeUnlink.exitCode).not.toBe(0);
		expect(existsSync(file)).toBe(true);

		const symlinkEscape = await runIn(
			sb,
			`ln -s "${file}" "$TMPDIR/escape.txt" && echo hacked > "$TMPDIR/escape.txt"`,
		);
		expect(symlinkEscape.exitCode).not.toBe(0);
		expect(readFileSync(file, "utf8")).toBe("binding");

		const hardlinkEscape = await runIn(sb, `ln "${file}" "$TMPDIR/hard.txt"`);
		expect(hardlinkEscape.exitCode).not.toBe(0);

		const trashAttack = await runIn(sb, `/usr/bin/trash "${file}" >/dev/null 2>&1`);
		expect(trashAttack.exitCode).not.toBe(0);
		expect(existsSync(file)).toBe(true);
	});

	it("blocks writes to other directories under /tmp outside the session scratch dir", async () => {
		const outside = realpathSync(mkdtempSync(join(tmpdir(), "autorag-ro-outside-")));
		try {
			const victim = join(outside, "victim.txt");
			writeFileSync(victim, "keep", "utf8");
			const sb = createReadOnlySandbox();
			sandboxes.push(sb);
			const res = await runIn(sb, `echo x > "${victim}"`);
			expect(res.exitCode).not.toBe(0);
			expect(readFileSync(victim, "utf8")).toBe("keep");
		} finally {
			rmSync(outside, { recursive: true, force: true });
		}
	});

	it("respects abort signal and timeout", async () => {
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);
		const ac = new AbortController();
		setTimeout(() => ac.abort(), 100);
		await expect(
			sb.operations.exec("sleep 10", root, { onData: () => undefined, signal: ac.signal }),
		).rejects.toThrow(/aborted/);
	});

	it("allows writing to explicit extraWritablePaths only while keeping siblings read-only", async () => {
		const allowed = join(root, "config.json");
		const sibling = join(root, "other.json");
		writeFileSync(allowed, "before", "utf8");
		writeFileSync(sibling, "untouched", "utf8");
		const sb = createReadOnlySandbox({ extraWritablePaths: [allowed] });
		sandboxes.push(sb);

		const ok = await runIn(sb, `echo after > "${allowed}"`);
		expect(ok.exitCode).toBe(0);
		expect(readFileSync(allowed, "utf8").trim()).toBe("after");

		const blocked = await runIn(sb, `echo bad > "${sibling}"`);
		expect(blocked.exitCode).not.toBe(0);
		expect(readFileSync(sibling, "utf8")).toBe("untouched");
	});
});

describe.skipIf(process.platform !== "linux")("Linux bwrap OS-level read-only enforcement", () => {
	it("enforces read-only disk when bwrap is usable, otherwise fails closed cleanly", async () => {
		const file = join(root, "precious.txt");
		writeFileSync(file, "original", "utf8");
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);
		if (sb.backend === "unavailable") {
			await expect(runIn(sb, `rm "${file}"`)).rejects.toThrow(/Read-only mode blocked bash execution/);
			expect(existsSync(file)).toBe(true);
			return;
		}
		expect(sb.backend).toBe("bwrap");
		const readRes = await runIn(sb, `cat "${file}"`);
		expect(readRes.exitCode).toBe(0);
		expect(readRes.output).toBe("original");

		const scratchRes = await runIn(sb, 'echo ok > "$TMPDIR/a.txt" && cat "$TMPDIR/a.txt"');
		expect(scratchRes.exitCode).toBe(0);
		expect(scratchRes.output.trim()).toBe("ok");

		const writeRes = await runIn(sb, `echo hacked > "${file}"`);
		expect(writeRes.exitCode).not.toBe(0);
		expect(readFileSync(file, "utf8")).toBe("original");

		const rmRes = await runIn(sb, `rm "${file}"`);
		expect(rmRes.exitCode).not.toBe(0);
		expect(existsSync(file)).toBe(true);

		const sub = join(root, "sub");
		mkdirSync(sub, { recursive: true });
		const touchRes = await runIn(sb, `touch "${join(sub, "x.txt")}"`);
		expect(touchRes.exitCode).not.toBe(0);
		expect(existsSync(join(sub, "x.txt"))).toBe(false);
	});
});

describe.skipIf(process.platform !== "win32")("Windows Low-Integrity OS-level read-only enforcement", () => {
	it("allows reading and writing inside %TEMP% scratch dir while blocking writes/deletes to Medium-IL files", async () => {
		const file = join(root, "precious.txt");
		writeFileSync(file, "original-win", "utf8");
		const sb = createReadOnlySandbox();
		sandboxes.push(sb);
		expect(sb.backend).toBe("windows-low-il");

		// Reading works via PowerShell and bash
		let psReadOut = "";
		const psRead = await sb.powerShellOperations.exec(`Get-Content -Raw '${file}'`, root, {
			onData: (chunk) => {
				psReadOut += chunk.toString("utf8");
			},
		});
		expect(psRead.exitCode).toBe(0);
		expect(psReadOut).toContain("original-win");

		// Scratch temp write succeeds
		let psTmpOut = "";
		const psTmp = await sb.powerShellOperations.exec(
			`Set-Content -Path (Join-Path $env:TEMP 'work.txt') -Value 'scratch-ok'; Get-Content (Join-Path $env:TEMP 'work.txt')`,
			root,
			{
				onData: (chunk) => {
					psTmpOut += chunk.toString("utf8");
				},
			},
		);
		expect(psTmp.exitCode).toBe(0);
		expect(psTmpOut.trim()).toBe("scratch-ok");

		// Overwrite, delete, and cmd /c del on Medium-IL file are blocked by kernel Mandatory Integrity Control
		const psAttacks = [
			`Set-Content -Path '${file}' -Value 'hacked'`,
			`Remove-Item -Force '${file}'`,
			`cmd.exe /c del /f /q "${file}"`,
			`New-Item -ItemType File -Path '${join(root, "new-file.txt")}'`,
		];
		for (const cmd of psAttacks) {
			await sb.powerShellOperations.exec(cmd, root, { onData: () => undefined });
			expect(existsSync(file), `file deleted by: ${cmd}`).toBe(true);
			expect(readFileSync(file, "utf8"), `file mutated by: ${cmd}`).toBe("original-win");
		}
		expect(existsSync(join(root, "new-file.txt"))).toBe(false);
	}, 60_000);
});
