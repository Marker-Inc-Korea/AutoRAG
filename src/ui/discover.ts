/**
 * Best-effort local account/remote discovery for the datasource UI.
 * Spawns operator CLIs with a short timeout. Failures return empty lists.
 */

import { spawn } from "node:child_process";

export interface Choice {
	readonly value: string;
	readonly label: string;
}

export type DiscoverRun = (binary: string, args: readonly string[]) => Promise<{ ok: boolean; stdout: string }>;

export async function listRcloneRemotes(run: DiscoverRun = runCli): Promise<readonly Choice[]> {
	const result = await run("rclone", ["listremotes"]);
	if (!result.ok) return [];
	return result.stdout
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.length > 0 && !line.startsWith("#"))
		.map((value) => ({ value, label: value }));
}

export async function choicesForType(
	type: string,
	run: DiscoverRun = runCli,
): Promise<{ rcloneRemotes: readonly Choice[]; mailAccounts: readonly Choice[] }> {
	if (type === "cloud-drive") {
		return { rcloneRemotes: await listRcloneRemotes(run), mailAccounts: [] };
	}
	if (type === "gmail") return { rcloneRemotes: [], mailAccounts: [] };
	return { rcloneRemotes: [], mailAccounts: [] };
}

function runCli(binary: string, args: readonly string[]): Promise<{ ok: boolean; stdout: string }> {
	return new Promise((resolve) => {
		let settled = false;
		const child = spawn(binary, [...args], { stdio: ["ignore", "pipe", "pipe"] });
		const chunks: Buffer[] = [];
		const finish = (ok: boolean) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolve({ ok, stdout: Buffer.concat(chunks).toString("utf8") });
		};
		const timer = setTimeout(() => {
			child.kill("SIGKILL");
			finish(false);
		}, 2000);
		child.stdout?.on("data", (chunk: Buffer) => {
			chunks.push(chunk);
		});
		child.on("error", () => finish(false));
		child.on("close", (code) => finish(code === 0));
	});
}
