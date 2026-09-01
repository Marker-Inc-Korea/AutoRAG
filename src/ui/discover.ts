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

const MAIL_PRESETS: readonly Choice[] = [
	{ value: "gmail", label: "Gmail" },
	{ value: "outlook", label: "Outlook" },
	{ value: "icloud", label: "iCloud" },
];

export async function listRcloneRemotes(run: DiscoverRun = runCli): Promise<readonly Choice[]> {
	const result = await run("rclone", ["listremotes"]);
	if (!result.ok) return [];
	return result.stdout
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.length > 0 && !line.startsWith("#"))
		.map((value) => ({ value, label: value }));
}

export async function listMailAccounts(run: DiscoverRun = runCli): Promise<readonly Choice[]> {
	const discovered = await listHimalayaAccounts(run);
	const seen = new Set(discovered.map((item) => item.value.toLowerCase()));
	const presets = MAIL_PRESETS.filter(
		(item) => !seen.has(item.value.toLowerCase()) && !seen.has(item.label.toLowerCase()),
	);
	return [...discovered, ...presets, { value: "other", label: "Other…" }];
}

export async function listHimalayaAccounts(run: DiscoverRun = runCli): Promise<readonly Choice[]> {
	const result = await run("himalaya", ["account", "list"]);
	if (!result.ok) return [];
	const choices: Choice[] = [];
	for (const raw of result.stdout.split(/\r?\n/)) {
		const line = raw.replace(/^\s*[*-]\s*/, "").trim();
		if (
			line.length === 0 ||
			line.toLowerCase().startsWith("account") ||
			line.startsWith("─") ||
			line.startsWith("-")
		) {
			continue;
		}
		const name = line.split(/\s{2,}|\t/)[0]?.trim();
		if (name && name !== "NAME" && name !== "Name") choices.push({ value: name, label: name });
	}
	return choices;
}

export async function choicesForType(
	type: string,
	run: DiscoverRun = runCli,
): Promise<{ rcloneRemotes: readonly Choice[]; mailAccounts: readonly Choice[] }> {
	if (type === "cloud-drive") {
		return { rcloneRemotes: await listRcloneRemotes(run), mailAccounts: [] };
	}
	if (type === "gmail") {
		return { rcloneRemotes: [], mailAccounts: await listMailAccounts(run) };
	}
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
