import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

/**
 * Windows OS-level read-only filesystem sandbox backed by Win32 Mandatory Integrity Control (MIC) +
 * Restricted Tokens (`CreateRestrictedToken` + `SetTokenInformation(TokenIntegrityLevel)` = Low IL
 * `S-1-16-4096`) + Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
 *
 * Why this works without admin privileges on any Windows 10/11 machine:
 * 1. Every normal file and directory on NTFS defaults to Medium Integrity (unlabeled objects are treated
 *    as Medium by the Windows Security Reference Monitor).
 * 2. Windows evaluates Mandatory Integrity Control (No-Write-Up) BEFORE DACLs. A Low Integrity process
 *    cannot create, modify, rename, or delete any Medium/High Integrity file or folder — even when the
 *    current user owns it and has Full Control, and even via parent `FILE_DELETE_CHILD` (`cmd /c del`).
 * 3. `CreateRestrictedToken(DISABLE_MAX_PRIVILEGE | LUA_TOKEN)` strips all privileges (except
 *    `SeChangeNotifyPrivilege`) and any elevated admin groups before lowering the token to Low IL.
 * 4. Only the session's dedicated scratch directory (and any explicitly trusted paths) receives a Low
 *    Integrity mandatory label (`icacls <path> /setintegritylevel (OI)(CI)L`), so temporary files under
 *    `$TMPDIR`/`%TEMP%` keep working while the rest of the disk stays kernel-enforced read-only.
 * 5. The child is launched via `CreateProcessAsUserW` with `DETACHED_PROCESS` (never `CREATE_NO_WINDOW`
 *    or `CREATE_NEW_CONSOLE`, which trigger `0xC0000142` `STATUS_DLL_INIT_FAILED` under a restricted
 *    token) and redirected anonymous pipes for stdout/stderr, bound to a Job Object that kills the entire
 *    process tree when the helper exits.
 *
 * Implementation note: the Win32 launcher is compiled once per session scratch directory as a tiny
 * PowerShell-hosted C# (`Add-Type`) runner using only Windows built-in `powershell.exe` / `pwsh.exe`
 * and `icacls.exe` — zero native npm addons, zero admin elevation, zero external downloads.
 */

export const WINDOWS_LOW_INTEGRITY_SID = "S-1-16-4096";

/**
 * PowerShell launcher script that compiles a minimal Win32 P/Invoke helper (`AutoRAGLowIlLauncher`)
 * and spawns the target command line under a Low-Integrity, privilege-stripped primary token with
 * stdout/stderr forwarded verbatim and a kill-on-close Job Object guarding the child tree.
 *
 * Arguments are passed through environment variables (`AUTORAG_WIN_SB_EXE`, `AUTORAG_WIN_SB_ARGS_JSON`,
 * `AUTORAG_WIN_SB_CWD`) to avoid PowerShell/CreateProcess quoting ambiguity.
 */
export const WINDOWS_SANDBOX_LAUNCHER_PS1 = String.raw`
$ErrorActionPreference = 'Stop'

$exe = $env:AUTORAG_WIN_SB_EXE
$argsJson = $env:AUTORAG_WIN_SB_ARGS_JSON
$workDir = $env:AUTORAG_WIN_SB_CWD

if ([string]::IsNullOrEmpty($exe)) {
  [Console]::Error.WriteLine('AutoRAG Windows sandbox: AUTORAG_WIN_SB_EXE is missing.')
  exit 125
}

$argList = @()
if (-not [string]::IsNullOrEmpty($argsJson)) {
  $parsed = ConvertFrom-Json -InputObject $argsJson
  if ($parsed -is [System.Array]) {
    $argList = @($parsed | ForEach-Object { [string]$_ })
  } elseif ($null -ne $parsed) {
    $argList = @([string]$parsed)
  }
}

$csharp = @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

public static class AutoRAGLowIlLauncher
{
    private const uint TOKEN_DUPLICATE = 0x0002;
    private const uint TOKEN_QUERY = 0x0008;
    private const uint TOKEN_ADJUST_DEFAULT = 0x0080;
    private const uint TOKEN_ASSIGN_PRIMARY = 0x0001;

    private const uint DISABLE_MAX_PRIVILEGE = 0x1;
    private const uint LUA_TOKEN = 0x4;

    private const int TokenIntegrityLevel = 25;
    private const uint SE_GROUP_INTEGRITY = 0x00000020;

    private const uint STARTF_USESTDHANDLES = 0x00000100;
    private const uint DETACHED_PROCESS = 0x00000008;
    private const uint CREATE_SUSPENDED = 0x00000004;
    private const uint CREATE_UNICODE_ENVIRONMENT = 0x00000400;
    private const uint HANDLE_FLAG_INHERIT = 0x00000001;
    private const uint INFINITE = 0xFFFFFFFF;
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    private const int JobObjectExtendedLimitInformation = 9;

    [StructLayout(LayoutKind.Sequential)]
    private struct SID_AND_ATTRIBUTES
    {
        public IntPtr Sid;
        public uint Attributes;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct TOKEN_MANDATORY_LABEL
    {
        public SID_AND_ATTRIBUTES Label;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SECURITY_ATTRIBUTES
    {
        public int nLength;
        public IntPtr lpSecurityDescriptor;
        [MarshalAs(UnmanagedType.Bool)]
        public bool bInheritHandle;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct STARTUPINFO
    {
        public int cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public int dwX;
        public int dwY;
        public int dwXSize;
        public int dwYSize;
        public int dwXCountChars;
        public int dwYCountChars;
        public int dwFillAttribute;
        public uint dwFlags;
        public short wShowWindow;
        public short cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_INFORMATION
    {
        public IntPtr hProcess;
        public IntPtr hThread;
        public int dwProcessId;
        public int dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("advapi32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool OpenProcessToken(IntPtr ProcessHandle, uint DesiredAccess, out IntPtr TokenHandle);

    [DllImport("advapi32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateRestrictedToken(
        IntPtr ExistingTokenHandle,
        uint Flags,
        uint DisableSidCount,
        IntPtr SidsToDisable,
        uint DeletePrivilegeCount,
        IntPtr PrivilegesToDelete,
        uint RestrictedSidCount,
        IntPtr SidsToRestrict,
        out IntPtr NewTokenHandle);

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool ConvertStringSidToSidW(string StringSid, out IntPtr Sid);

    [DllImport("advapi32.dll")]
    private static extern uint GetLengthSid(IntPtr pSid);

    [DllImport("advapi32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetTokenInformation(
        IntPtr TokenHandle,
        int TokenInformationClass,
        IntPtr TokenInformation,
        uint TokenInformationLength);

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateProcessAsUserW(
        IntPtr hToken,
        string lpApplicationName,
        StringBuilder lpCommandLine,
        IntPtr lpProcessAttributes,
        IntPtr lpThreadAttributes,
        [MarshalAs(UnmanagedType.Bool)] bool bInheritHandles,
        uint dwCreationFlags,
        IntPtr lpEnvironment,
        string lpCurrentDirectory,
        ref STARTUPINFO lpStartupInfo,
        out PROCESS_INFORMATION lpProcessInformation);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreatePipe(
        out IntPtr hReadPipe,
        out IntPtr hWritePipe,
        ref SECURITY_ATTRIBUTES lpPipeAttributes,
        uint nSize);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetHandleInformation(IntPtr hObject, uint dwMask, uint dwFlags);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateFileW(
        string lpFileName,
        uint dwDesiredAccess,
        uint dwShareMode,
        ref SECURITY_ATTRIBUTES lpSecurityAttributes,
        uint dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool ReadFile(
        IntPtr hFile,
        byte[] lpBuffer,
        uint nNumberOfBytesToRead,
        out uint lpNumberOfBytesRead,
        IntPtr lpOverlapped);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr hHandle, uint dwMilliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetExitCodeProcess(IntPtr hProcess, out uint lpExitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr hThread);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObjectW(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetInformationJobObject(
        IntPtr hJob,
        int JobObjectInformationClass,
        ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInformation,
        uint cbJobObjectInformationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr LocalFree(IntPtr hMem);

    private static string QuoteArg(string arg)
    {
        if (arg.Length == 0) return "\"\"";
        bool needsQuotes = false;
        for (int i = 0; i < arg.Length; i++)
        {
            char c = arg[i];
            if (c == ' ' || c == '\t' || c == '"' || c == '\n' || c == '\r')
            {
                needsQuotes = true;
                break;
            }
        }
        if (!needsQuotes) return arg;
        var sb = new StringBuilder();
        sb.Append('"');
        int backslashes = 0;
        for (int i = 0; i < arg.Length; i++)
        {
            char c = arg[i];
            if (c == '\\')
            {
                backslashes++;
            }
            else if (c == '"')
            {
                sb.Append('\\', backslashes * 2 + 1);
                sb.Append('"');
                backslashes = 0;
            }
            else
            {
                if (backslashes > 0)
                {
                    sb.Append('\\', backslashes);
                    backslashes = 0;
                }
                sb.Append(c);
            }
        }
        if (backslashes > 0) sb.Append('\\', backslashes * 2);
        sb.Append('"');
        return sb.ToString();
    }

    private static void PumpPipe(IntPtr readHandle, Stream target)
    {
        byte[] buf = new byte[8192];
        while (true)
        {
            uint read;
            if (!ReadFile(readHandle, buf, (uint)buf.Length, out read, IntPtr.Zero) || read == 0)
            {
                break;
            }
            target.Write(buf, 0, (int)read);
            target.Flush();
        }
    }

    public static int Run(string exePath, string[] args, string cwd)
    {
        IntPtr currentToken = IntPtr.Zero;
        IntPtr restrictedToken = IntPtr.Zero;
        IntPtr lowSid = IntPtr.Zero;
        IntPtr tmlPtr = IntPtr.Zero;
        IntPtr job = IntPtr.Zero;
        IntPtr outRead = IntPtr.Zero, outWrite = IntPtr.Zero;
        IntPtr errRead = IntPtr.Zero, errWrite = IntPtr.Zero;
        IntPtr nulHandle = IntPtr.Zero;
        PROCESS_INFORMATION pi = new PROCESS_INFORMATION();

        try
        {
            uint access = TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT | TOKEN_ASSIGN_PRIMARY;
            if (!OpenProcessToken(GetCurrentProcess(), access, out currentToken))
            {
                throw new InvalidOperationException("OpenProcessToken failed: win32=" + Marshal.GetLastWin32Error());
            }

            // Strip privileges and any elevated admin groups (LUA_TOKEN). Fall back to DISABLE_MAX_PRIVILEGE
            // alone if LUA_TOKEN is rejected (for example when already running under a restricted token).
            if (!CreateRestrictedToken(currentToken, DISABLE_MAX_PRIVILEGE | LUA_TOKEN, 0, IntPtr.Zero, 0, IntPtr.Zero, 0, IntPtr.Zero, out restrictedToken))
            {
                if (!CreateRestrictedToken(currentToken, DISABLE_MAX_PRIVILEGE, 0, IntPtr.Zero, 0, IntPtr.Zero, 0, IntPtr.Zero, out restrictedToken))
                {
                    throw new InvalidOperationException("CreateRestrictedToken failed: win32=" + Marshal.GetLastWin32Error());
                }
            }

            if (!ConvertStringSidToSidW("S-1-16-4096", out lowSid))
            {
                throw new InvalidOperationException("ConvertStringSidToSidW(Low IL) failed: win32=" + Marshal.GetLastWin32Error());
            }

            TOKEN_MANDATORY_LABEL tml = new TOKEN_MANDATORY_LABEL();
            tml.Label.Sid = lowSid;
            tml.Label.Attributes = SE_GROUP_INTEGRITY;
            int structSize = Marshal.SizeOf(typeof(TOKEN_MANDATORY_LABEL));
            tmlPtr = Marshal.AllocHGlobal(structSize);
            Marshal.StructureToPtr(tml, tmlPtr, false);

            uint totalLen = (uint)structSize + GetLengthSid(lowSid);
            if (!SetTokenInformation(restrictedToken, TokenIntegrityLevel, tmlPtr, totalLen))
            {
                throw new InvalidOperationException("SetTokenInformation(TokenIntegrityLevel=Low) failed: win32=" + Marshal.GetLastWin32Error());
            }

            SECURITY_ATTRIBUTES sa = new SECURITY_ATTRIBUTES();
            sa.nLength = Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES));
            sa.lpSecurityDescriptor = IntPtr.Zero;
            sa.bInheritHandle = true;

            if (!CreatePipe(out outRead, out outWrite, ref sa, 0))
            {
                throw new InvalidOperationException("CreatePipe(stdout) failed: win32=" + Marshal.GetLastWin32Error());
            }
            SetHandleInformation(outRead, HANDLE_FLAG_INHERIT, 0);

            if (!CreatePipe(out errRead, out errWrite, ref sa, 0))
            {
                throw new InvalidOperationException("CreatePipe(stderr) failed: win32=" + Marshal.GetLastWin32Error());
            }
            SetHandleInformation(errRead, HANDLE_FLAG_INHERIT, 0);

            nulHandle = CreateFileW("NUL", 0x80000000u, 3u, ref sa, 3u, 0x80u, IntPtr.Zero);

            STARTUPINFO si = new STARTUPINFO();
            si.cb = Marshal.SizeOf(typeof(STARTUPINFO));
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdInput = nulHandle;
            si.hStdOutput = outWrite;
            si.hStdError = errWrite;

            var cmdLine = new StringBuilder();
            cmdLine.Append(QuoteArg(exePath));
            if (args != null)
            {
                foreach (var a in args)
                {
                    cmdLine.Append(' ');
                    cmdLine.Append(QuoteArg(a));
                }
            }

            // Use DETACHED_PROCESS (never CREATE_NO_WINDOW / CREATE_NEW_CONSOLE, which trigger
            // STATUS_DLL_INIT_FAILED 0xC0000142 under restricted tokens when creating a console).
            uint flags = DETACHED_PROCESS | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT;
            if (!CreateProcessAsUserW(
                restrictedToken,
                null,
                cmdLine,
                IntPtr.Zero,
                IntPtr.Zero,
                true,
                flags,
                IntPtr.Zero,
                cwd,
                ref si,
                out pi))
            {
                throw new InvalidOperationException("CreateProcessAsUserW failed: win32=" + Marshal.GetLastWin32Error());
            }

            job = CreateJobObjectW(IntPtr.Zero, null);
            if (job != IntPtr.Zero)
            {
                var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    ref info,
                    (uint)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)));
                AssignProcessToJobObject(job, pi.hProcess);
            }

            // Close parent-side write ends so EOF propagates once child exits.
            CloseHandle(outWrite); outWrite = IntPtr.Zero;
            CloseHandle(errWrite); errWrite = IntPtr.Zero;
            if (nulHandle != IntPtr.Zero && nulHandle != new IntPtr(-1))
            {
                CloseHandle(nulHandle);
                nulHandle = IntPtr.Zero;
            }

            ResumeThread(pi.hThread);

            Stream stdout = Console.OpenStandardOutput();
            Stream stderr = Console.OpenStandardError();
            var tOut = new Thread(() => PumpPipe(outRead, stdout));
            var tErr = new Thread(() => PumpPipe(errRead, stderr));
            tOut.IsBackground = true;
            tErr.IsBackground = true;
            tOut.Start();
            tErr.Start();

            WaitForSingleObject(pi.hProcess, INFINITE);
            tOut.Join();
            tErr.Join();

            uint exitCode = 1;
            GetExitCodeProcess(pi.hProcess, out exitCode);
            return unchecked((int)exitCode);
        }
        finally
        {
            if (pi.hThread != IntPtr.Zero) CloseHandle(pi.hThread);
            if (pi.hProcess != IntPtr.Zero) CloseHandle(pi.hProcess);
            if (job != IntPtr.Zero) CloseHandle(job);
            if (outRead != IntPtr.Zero) CloseHandle(outRead);
            if (outWrite != IntPtr.Zero) CloseHandle(outWrite);
            if (errRead != IntPtr.Zero) CloseHandle(errRead);
            if (errWrite != IntPtr.Zero) CloseHandle(errWrite);
            if (nulHandle != IntPtr.Zero && nulHandle != new IntPtr(-1)) CloseHandle(nulHandle);
            if (tmlPtr != IntPtr.Zero) Marshal.FreeHGlobal(tmlPtr);
            if (lowSid != IntPtr.Zero) LocalFree(lowSid);
            if (restrictedToken != IntPtr.Zero) CloseHandle(restrictedToken);
            if (currentToken != IntPtr.Zero) CloseHandle(currentToken);
        }
    }
}
'@

Add-Type -TypeDefinition $csharp -Language CSharp | Out-Null
$code = [AutoRAGLowIlLauncher]::Run($exe, [string[]]$argList, $workDir)
exit $code
`.trimStart();

export interface WindowsSandboxSetup {
	readonly ok: true;
	readonly powershellPath: string;
	readonly launcherScriptPath: string;
}

export interface WindowsSandboxSetupFailure {
	readonly ok: false;
	readonly reason: string;
}

export type WindowsSandboxSetupResult = WindowsSandboxSetup | WindowsSandboxSetupFailure;

export interface WindowsSandboxPrepareOptions {
	readonly scratchDir: string;
	readonly extraWritablePaths: readonly string[];
	readonly probeBinary: (name: string) => string | undefined;
	/** Test seam for running `icacls` / probe commands. */
	readonly runCommand?: (
		bin: string,
		args: readonly string[],
	) => { readonly status: number | null; readonly stdout?: string; readonly stderr?: string; readonly error?: Error };
}

function defaultRunCommand(bin: string, args: readonly string[]) {
	const res = spawnSync(bin, [...args], { encoding: "utf8", timeout: 10_000, windowsHide: true });
	return {
		status: res.status,
		stdout: res.stdout,
		stderr: res.stderr,
		...(res.error === undefined ? {} : { error: res.error }),
	};
}

/** Resolve Windows `powershell.exe` or `pwsh.exe` via probe or standard System32 location. */
export function resolveWindowsPowerShell(probeBinary: (name: string) => string | undefined): string | undefined {
	const systemRoot = process.env.SystemRoot || "C:\\Windows";
	const windowsPs = join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
	return probeBinary(windowsPs) ?? probeBinary("powershell.exe") ?? probeBinary("pwsh.exe");
}

/** Resolve Windows `icacls.exe` via probe or standard System32 location. */
export function resolveWindowsIcacls(probeBinary: (name: string) => string | undefined): string | undefined {
	const systemRoot = process.env.SystemRoot || "C:\\Windows";
	const systemIcacls = join(systemRoot, "System32", "icacls.exe");
	return probeBinary(systemIcacls) ?? probeBinary("icacls.exe");
}

/**
 * Prepare the Windows Low-Integrity read-only sandbox:
 * 1. Locate `powershell.exe`/`pwsh.exe` and `icacls.exe`.
 * 2. Label `scratchDir` with Low Integrity `(OI)(CI)L` so Low-IL processes can write temporary files
 *    there while every unlabeled (Medium-IL) path on NTFS rejects writes and deletes at the kernel level.
 * 3. Label any explicit `extraWritablePaths` (e.g. self-config file) with Low Integrity `L`.
 * 4. Write the PowerShell launcher script into a parent-owned location outside the writable scratch dir
 *    (or mark it read-only) so the Low-IL child cannot tamper with it.
 */
export function prepareWindowsReadOnlySandbox(options: WindowsSandboxPrepareOptions): WindowsSandboxSetupResult {
	const run = options.runCommand ?? defaultRunCommand;
	const powershellPath = resolveWindowsPowerShell(options.probeBinary);
	if (powershellPath === undefined) {
		return {
			ok: false,
			reason:
				"Windows read-only sandbox requires PowerShell (powershell.exe or pwsh.exe), which was not found. Switch to `full` permission mode or install PowerShell.",
		};
	}

	const icaclsPath = resolveWindowsIcacls(options.probeBinary);
	if (icaclsPath === undefined) {
		return {
			ok: false,
			reason:
				"Windows read-only sandbox requires icacls.exe to label the session scratch directory as Low Integrity, but icacls.exe was not found.",
		};
	}

	// Keep the launcher script in a dedicated sibling directory that stays Medium Integrity (unlabeled),
	// so the Low-IL child process can read it if needed but can never overwrite it.
	const launcherDir = join(options.scratchDir, "..", `autorag-win-launcher-${Date.now()}-${process.pid}`);
	mkdirSync(launcherDir, { recursive: true });
	const launcherScriptPath = resolve(launcherDir, "run-low-il.ps1");
	writeFileSync(launcherScriptPath, WINDOWS_SANDBOX_LAUNCHER_PS1, "utf8");
	try {
		chmodSync(launcherScriptPath, 0o444);
	} catch {
		// Best-effort read-only attribute.
	}

	// Apply Low Integrity mandatory label `(OI)(CI)L` to the scratch directory so Low-IL processes
	// can create/edit/delete temp files inside scratchDir ONLY.
	const labelScratch = run(icaclsPath, [options.scratchDir, "/setintegritylevel", "(OI)(CI)L"]);
	if (labelScratch.error !== undefined || labelScratch.status !== 0) {
		const detail =
			labelScratch.error?.message ??
			(labelScratch.stderr?.trim() ||
				labelScratch.stdout?.trim() ||
				`exit code ${labelScratch.status ?? "unknown"}`);
		return {
			ok: false,
			reason: `Windows read-only sandbox failed to set Low Integrity label on scratch directory (${options.scratchDir}): ${detail}`,
		};
	}

	for (const extra of options.extraWritablePaths) {
		const resolved = resolve(extra);
		if (!existsSync(resolved)) continue;
		const labelExtra = run(icaclsPath, [resolved, "/setintegritylevel", "L"]);
		if (labelExtra.error !== undefined || labelExtra.status !== 0) {
			const detail =
				labelExtra.error?.message ??
				(labelExtra.stderr?.trim() || labelExtra.stdout?.trim() || `exit code ${labelExtra.status ?? "unknown"}`);
			return {
				ok: false,
				reason: `Windows read-only sandbox failed to set Low Integrity label on trusted path (${resolved}): ${detail}`,
			};
		}
	}

	return {
		ok: true,
		powershellPath,
		launcherScriptPath,
	};
}

/** Build the PowerShell invocation + env overrides that launch `shell + shellArgs + command` at Low Integrity. */
export function buildWindowsLowIlInvocation(
	powershellPath: string,
	launcherScriptPath: string,
	shell: string,
	shellArgs: readonly string[],
	command: string,
	cwd: string,
): {
	readonly bin: string;
	readonly args: readonly string[];
	readonly env: Readonly<Record<string, string>>;
} {
	return {
		bin: powershellPath,
		args: ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", launcherScriptPath],
		env: {
			AUTORAG_WIN_SB_EXE: shell,
			AUTORAG_WIN_SB_ARGS_JSON: JSON.stringify([...shellArgs, command]),
			AUTORAG_WIN_SB_CWD: cwd,
		},
	};
}
