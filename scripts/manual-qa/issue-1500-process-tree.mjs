import { existsSync, mkdtempSync, readFileSync, rmSync, watch } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBashTool } from "../../src/agent/bash-tool.ts";

const workspace = mkdtempSync(join(tmpdir(), "autorag-issue-1500-"));
const pidPath = join(workspace, "child.pid");

const waitForFile = async (path) => {
	if (existsSync(path)) return;
	await new Promise((resolve, reject) => {
		const watcher = watch(workspace, (_event, filename) => {
			if (filename === "child.pid" && existsSync(path)) {
				watcher.close();
				resolve();
			}
		});
		watcher.on("error", (error) => {
			watcher.close();
			reject(error);
		});
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
		command: `sleep 10 & echo $! > ${pidPath}; wait`,
	});
	await waitForFile(pidPath);
	const childPid = Number(readFileSync(pidPath, "utf8"));
	const timeoutResult = await timeoutExecution;
	const timeoutElapsedMs = Math.round(performance.now() - timeoutStarted);
	const timeoutText = timeoutResult.content[0].text;
	console.log(JSON.stringify({
		scenario: "timeout process tree",
		pass: timeoutResult.details.timedOut && !isAlive(childPid) && timeoutText.includes("command timed out after 120ms"),
		elapsedMs: timeoutElapsedMs,
		childPid,
		childAlive: isAlive(childPid),
		text: timeoutText,
	}));

	const failedResult = await tool.execute("manual-failure", { command: "printf fail >&2; exit 7" });
	console.log(JSON.stringify({
		scenario: "failed command",
		pass: failedResult.details.exitCode === 7 && failedResult.content[0].text.includes("exit code 7"),
		exitCode: failedResult.details.exitCode,
		text: failedResult.content[0].text,
	}));

	const abortPidPath = join(workspace, "abort-child.pid");
	const controller = new AbortController();
	const abortExecution = tool.execute("manual-abort", {
		command: `sleep 10 & echo $! > ${abortPidPath}; wait`,
	}, controller.signal);
	await new Promise((resolve, reject) => {
		const watcher = watch(workspace, (_event, filename) => {
			if (filename === "abort-child.pid" && existsSync(abortPidPath)) {
				watcher.close();
				resolve();
			}
		});
		watcher.on("error", reject);
	});
	const abortChildPid = Number(readFileSync(abortPidPath, "utf8"));
	controller.abort();
	const abortResult = await abortExecution;
	console.log(JSON.stringify({
		scenario: "abort process tree",
		pass: !abortResult.details.timedOut && !isAlive(abortChildPid),
		childPid: abortChildPid,
		childAlive: isAlive(abortChildPid),
	}));
} finally {
	rmSync(workspace, { recursive: true, force: true });
	console.log(JSON.stringify({ cleanup: "removed temporary workspace", workspace }));
}
