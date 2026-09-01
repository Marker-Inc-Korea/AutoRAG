import { chmodSync, mkdirSync } from "node:fs";

export function validateMailcrawlInstanceId(instanceId: string): void {
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(instanceId)) {
		throw new Error("mailcrawl instanceId must be a safe single path segment");
	}
}

export function ensurePrivateMailcrawlDataDir(path: string): void {
	mkdirSync(path, { recursive: true, mode: 0o700 });
	if (process.platform !== "win32") chmodSync(path, 0o700);
}
