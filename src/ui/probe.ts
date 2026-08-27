/**
 * Local, side-effect-free connection probes for the datasource UI.
 *
 * Checks env-var presence, path existence, and CLI binaries on PATH.
 * Never returns secret values.
 */

import { existsSync } from "node:fs";
import { delimiter, join } from "node:path";
import { getDatasourceType } from "./catalog.ts";

export type ProbeStatus =
	| "ready"
	| "auth-missing"
	| "binary-missing"
	| "path-missing"
	| "not-configured"
	| "unknown-type";

export interface ProbeResult {
	readonly ok: boolean;
	readonly status: ProbeStatus;
	readonly detail: string;
}

export interface ProbeConnectionInput {
	readonly alias: string;
	readonly type: string;
	readonly enabled?: boolean;
	readonly connector?: Record<string, unknown>;
}

export interface ProbeDeps {
	readonly env?: NodeJS.ProcessEnv;
	readonly pathExists?: (path: string) => boolean;
	readonly binaryExists?: (binaryName: string, binaryPath?: string) => boolean;
}

export function probeConnection(input: ProbeConnectionInput, deps: ProbeDeps = {}): ProbeResult {
	const catalog = getDatasourceType(input.type);
	if (catalog === undefined) {
		return { ok: false, status: "unknown-type", detail: `Unknown datasource type: ${input.type}` };
	}
	const env = deps.env ?? process.env;
	const pathExists = deps.pathExists ?? existsSync;
	const binaryExists = deps.binaryExists ?? defaultBinaryExists;
	const connector = input.connector ?? {};

	const fail = (status: ProbeStatus, detail: string): ProbeResult => ({ ok: false, status, detail });

	if (input.type === "github") {
		const repos = asStringList(connector.repos);
		if (repos.length === 0) return fail("not-configured", "Add at least one owner/repo.");
		const tokenEnv = envName(connector.tokenEnv, "GITHUB_TOKEN");
		if (!envHas(env, tokenEnv)) return fail("auth-missing", `${tokenEnv} is not set in the environment.`);
		return ok("Repository list saved. Token is read from the environment.");
	}

	if (input.type === "rss") {
		const feeds = asFeedUrls(connector.feeds);
		if (feeds.length === 0) return fail("not-configured", "Add at least one feed URL.");
		return ok(`${feeds.length} feed(s) configured.`);
	}

	if (input.type === "obsidian") {
		const vault = asString(connector.vaultPath);
		if (vault === undefined) return fail("not-configured", "Choose an Obsidian vault folder.");
		if (!pathExists(vault)) return fail("path-missing", "Vault folder was not found.");
		const binary = asString(connector.binaryPath);
		if (!binaryExists(catalog.binaryName ?? "qmd", binary)) {
			return fail("binary-missing", binary ? "qmd CLI was not found at that path." : "qmd CLI is not on PATH.");
		}
		return ok("Vault folder is reachable.");
	}

	if (input.type === "mail-export") {
		const paths = asStringList(connector.paths);
		if (paths.length === 0) return fail("not-configured", "Add at least one .eml/.mbox path.");
		for (const path of paths) {
			if (!pathExists(path)) return fail("path-missing", "A mail export path was not found.");
		}
		return ok("Export paths are reachable.");
	}

	if (input.type === "spotlight") {
		const queries = asStringList(connector.queries);
		if (queries.length === 0) return fail("not-configured", "Add at least one Spotlight query.");
		const onlyIn = asString(connector.onlyIn);
		if (onlyIn !== undefined && !pathExists(onlyIn))
			return fail("path-missing", "The “only in” folder was not found.");
		return ok("Queries saved.");
	}

	if (input.type === "gdrive") {
		if (connector.backend === "rclone") {
			const remote = asString(connector.remote);
			if (remote === undefined) return fail("not-configured", "Enter an rclone remote (for example gdrive:).");
			if (!binaryExists("rclone", asString(connector.binaryPath))) {
				return fail("binary-missing", "rclone is not installed.");
			}
			return ok("rclone remote saved. Authenticate with `rclone config` if needed.");
		}
		const tokenEnv = envName(connector.tokenEnv, "GDRIVE_ACCESS_TOKEN");
		if (!envHas(env, tokenEnv)) return fail("auth-missing", `${tokenEnv} is not set in the environment.`);
		return ok("Drive token will be read from the environment.");
	}

	if (input.type === "gmail") {
		if (connector.backend === "himalaya") {
			if (!binaryExists("himalaya", asString(connector.binaryPath))) {
				return fail("binary-missing", "himalaya is not installed.");
			}
			return ok("himalaya will use its own account config.");
		}
		const tokenEnv = envName(connector.tokenEnv, "GMAIL_ACCESS_TOKEN");
		if (!envHas(env, tokenEnv)) return fail("auth-missing", `${tokenEnv} is not set in the environment.`);
		return ok("Gmail token will be read from the environment.");
	}

	if (input.type === "cloud-drive") {
		const remote = asString(connector.remote);
		if (remote === undefined) return fail("not-configured", "Enter an rclone remote.");
		if (!binaryExists(catalog.binaryName ?? "rclone", asString(connector.binaryPath))) {
			return fail("binary-missing", "rclone is not installed.");
		}
		return ok("rclone remote saved.");
	}

	if (catalog.binaryName !== undefined) {
		const binary = asString(connector.binaryPath);
		if (!binaryExists(catalog.binaryName, binary)) {
			return fail(
				"binary-missing",
				binary ? `${catalog.binaryName} was not found at that path.` : `${catalog.binaryName} is not on PATH.`,
			);
		}
		return ok(`${catalog.title} CLI is available.`);
	}

	return ok("Saved.");
}

function ok(detail: string): ProbeResult {
	return { ok: true, status: "ready", detail };
}

function envName(value: unknown, fallback: string): string {
	return typeof value === "string" && value.trim().length > 0 ? value.trim() : fallback;
}

function envHas(env: NodeJS.ProcessEnv, name: string): boolean {
	const value = env[name];
	return typeof value === "string" && value.length > 0;
}

function asString(value: unknown): string | undefined {
	if (typeof value !== "string") return undefined;
	const trimmed = value.trim();
	return trimmed.length > 0 ? trimmed : undefined;
}

function asStringList(value: unknown): string[] {
	if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter((item) => item.length > 0);
	if (typeof value === "string") {
		return value
			.split(/\r?\n/)
			.map((item) => item.trim())
			.filter((item) => item.length > 0);
	}
	return [];
}

function asFeedUrls(value: unknown): string[] {
	if (!Array.isArray(value)) return asStringList(value);
	const urls: string[] = [];
	for (const item of value) {
		if (typeof item === "string" && item.trim().length > 0) urls.push(item.trim());
		else if (item && typeof item === "object" && typeof (item as { url?: unknown }).url === "string") {
			const url = (item as { url: string }).url.trim();
			if (url.length > 0) urls.push(url);
		}
	}
	return urls;
}

function defaultBinaryExists(binaryName: string, binaryPath?: string): boolean {
	if (binaryPath !== undefined && binaryPath.length > 0) return existsSync(binaryPath);
	const pathEnv = process.env.PATH ?? "";
	const extensions = process.platform === "win32" ? [".exe", ".cmd", ".bat", ""] : [""];
	for (const dir of pathEnv.split(delimiter)) {
		if (dir.length === 0) continue;
		for (const extension of extensions) {
			if (existsSync(join(dir, `${binaryName}${extension}`))) return true;
		}
	}
	return false;
}
