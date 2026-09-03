import { existsSync, mkdtempSync, readFileSync, rmSync, unwatchFile, watchFile } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBashTool } from "../../src/agent/bash-tool.ts";

const workspace = mkdtempSync(join(tmpdir(), "autorag-issue-1500-"));
const pidPath = join(workspace, "child.pid");
let allPassed = true;

const waitForFile = async (path) => {
	if (existsSync(path)) return;
	await new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			unwatchFile(path, onChange);
			reject(new Error(`Timed out waiting for ${path}`));
		}, 2_000);
		const onChange = () => {
			if (!existsSync(path)) return;
			clearTimeout(timeout);
			unwatchFile(path, onChange);
			resolve();
		};
		watchFile(path, { interval: 25 }, onChange);
	});
};

const isAlive = (pid) => {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
};

try {
	const tool = createBashTool({ cwd: workspace, timeoutMs: 120 });
	const timeoutStarted = performance.now();
	const timeoutExecution = tool.execute("manual-timeout", {
		command: "sleep 10 & echo $! > child.pid; wait",
	});
	await waitForFile(pidPath);
	const childPid = Number(readFileSync(pidPath, "utf8"));
	const timeoutResult = await timeoutExecution;
	const timeoutElapsedMs = Math.round(performance.now() - timeoutStarted);
	const timeoutText = timeoutResult.content[0].text;
	const timeoutPass = timeoutResult.details.timedOut && !isAlive(childPid) && timeoutText.includes("command timed out after 120ms");
	allPassed &&= timeoutPass;
	console.log(JSON.stringify({
		scenario: "timeout process tree",
		pass: timeoutPass,
		elapsedMs: timeoutElapsedMs,
		childPid,
		childAlive: isAlive(childPid),
		text: timeoutText,
	}));

	const failedResult = await tool.execute("manual-failure", { command: "printf fail >&2; exit 7" });
	const failedPass = failedResult.details.exitCode === 7 && failedResult.content[0].text.includes("exit code 7");
	allPassed &&= failedPass;
	console.log(JSON.stringify({
		scenario: "failed command",
		pass: failedPass,
		exitCode: failedResult.details.exitCode,
		text: failedResult.content[0].text,
	}));

	const abortPidPath = join(workspace, "abort-child.pid");
	const controller = new AbortController();
	const abortExecution = tool.execute("manual-abort", {
		command: "sleep 10 & echo $! > abort-child.pid; wait",
	}, controller.signal);
	await waitForFile(abortPidPath);
	const abortChildPid = Number(readFileSync(abortPidPath, "utf8"));
	controller.abort();
	const abortResult = await abortExecution;
	const abortPass = !abortResult.details.timedOut && !isAlive(abortChildPid);
	allPassed &&= abortPass;
	console.log(JSON.stringify({
		scenario: "abort process tree",
		pass: abortPass,
		childPid: abortChildPid,
		childAlive: isAlive(abortChildPid),
	}));
} finally {
	rmSync(workspace, { recursive: true, force: true });
	console.log(JSON.stringify({ cleanup: "removed temporary workspace", workspace }));
	if (!allPassed) process.exitCode = 1;
}
