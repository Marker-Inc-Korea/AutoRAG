import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";
import { resolveAutoRAGHome } from "../../config/home.ts";
import { resolveP2pWorkspace } from "./p2p.ts";
import type { CommandContext } from "./types.ts";

const VALID_TIERS = new Set(["private", "never", "always", "peers"]);

interface PolicyEntry {
	tier: string;
	peers?: readonly string[];
}

function workspacePolicyPath(cwd: string): string {
	return join(cwd, ".autorag", "p2p", "policy.toml");
}

function normalizeEntry(value: unknown, key: string): PolicyEntry {
	if (typeof value === "string") {
		if (!VALID_TIERS.has(value)) {
			throw new Error(`Invalid tier for ${key}: "${value}"`);
		}
		return { tier: value };
	}
	if (typeof value === "object" && value !== null && !Array.isArray(value)) {
		const obj = value as Record<string, unknown>;
		const tier = obj.tier;
		if (typeof tier !== "string" || !VALID_TIERS.has(tier)) {
			throw new Error(`Invalid tier for ${key}: ${JSON.stringify(tier)}`);
		}
		const peers = obj.peers;
		if (peers !== undefined) {
			if (!Array.isArray(peers)) {
				throw new Error(`Invalid peers for ${key}: must be an array`);
			}
			return { tier, peers: peers as readonly string[] };
		}
		return { tier };
	}
	throw new Error(`Invalid policy entry for ${key}: ${JSON.stringify(value)}`);
}

function readPolicyToml(policyPath: string): Record<string, PolicyEntry> {
	if (!existsSync(policyPath)) return {};
	const parsed = parseToml(readFileSync(policyPath, "utf8")) as Record<string, unknown>;
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return {};

	// policy.toml can have entries directly at root or under [policy]
	let raw: Record<string, unknown>;
	if (parsed.policy !== null && typeof parsed.policy === "object" && !Array.isArray(parsed.policy)) {
		raw = parsed.policy as Record<string, unknown>;
	} else {
		// Check if entries are at root with known keys
		raw = {};
		for (const [key, value] of Object.entries(parsed)) {
			if (key === "quotas" || key === "newFilesPublic" || key === "p2p") continue;
			raw[key] = value;
		}
	}

	const entries: Record<string, PolicyEntry> = {};
	for (const [key, value] of Object.entries(raw)) {
		entries[key] = normalizeEntry(value, key);
	}
	return entries;
}

function writePolicyToml(policyPath: string, entries: Record<string, PolicyEntry>): void {
	mkdirSync(dirname(policyPath), { recursive: true });

	if (Object.keys(entries).length === 0) {
		// Write empty file
		writeFileSync(policyPath, "", { mode: 0o600 });
		return;
	}

	const data: Record<string, unknown> = { policy: entries };
	const toml = stringifyToml(data);
	writeFileSync(policyPath, toml, { mode: 0o600 });
}

function getGlobalPolicy(): Record<string, PolicyEntry> {
	const home = resolveAutoRAGHome();
	const configPath = join(home, "config.json");
	if (!existsSync(configPath)) return {};
	try {
		const raw = readFileSync(configPath, "utf8");
		const parsed = JSON.parse(raw) as Record<string, unknown>;
		if (parsed.p2p && typeof parsed.p2p === "object" && !Array.isArray(parsed.p2p)) {
			const p2p = parsed.p2p as Record<string, unknown>;
			if (p2p.policy && typeof p2p.policy === "object" && !Array.isArray(p2p.policy)) {
				return p2p.policy as Record<string, PolicyEntry>;
			}
		}
	} catch {
		// ignore
	}
	return {};
}

function validateTier(tier: string): tier is "private" | "never" | "always" | "peers" {
	return VALID_TIERS.has(tier);
}

export function runP2pPolicy(ctx: CommandContext): number {
	const subcommand = ctx.positionals[0];
	const policyPath = workspacePolicyPath(resolveP2pWorkspace(ctx));

	switch (subcommand) {
		case "list": {
			const workspaceEntries = readPolicyToml(policyPath);
			const globalEntries = getGlobalPolicy();
			// Merge: workspace overrides global
			const merged = { ...globalEntries, ...workspaceEntries };
			ctx.stdout(JSON.stringify(merged));
			return 0;
		}
		case "set": {
			const key = ctx.positionals[1];
			const tier = ctx.positionals[2];
			if (!key) {
				throw new Error("Source glob required: autorag p2p policy set <source-glob> <tier> [--peer fp...]");
			}
			if (!tier) {
				throw new Error("Tier required: private, never, always, or peers");
			}
			if (!validateTier(tier)) {
				throw new Error(`Invalid tier "${tier}". Must be one of: ${[...VALID_TIERS].join(", ")}`);
			}
			const peerFlags = ctx.flags.peer;
			const peers = Array.isArray(peerFlags) ? peerFlags : typeof peerFlags === "string" ? [peerFlags] : [];

			if (tier === "peers" && peers.length === 0) {
				throw new Error("Tier 'peers' requires at least one --peer <fingerprint>");
			}
			if (tier !== "peers" && peers.length > 0) {
				throw new Error("--peer flags are only valid for 'peers' tier");
			}

			const entries = readPolicyToml(policyPath);
			entries[key] = { tier, ...(peers.length > 0 ? { peers } : {}) };
			writePolicyToml(policyPath, entries);

			const result: Record<string, unknown> = { ok: true, key, tier };
			if (peers.length > 0) {
				result.peers = peers;
			}
			ctx.stdout(JSON.stringify(result));
			return 0;
		}
		case "unset": {
			const key = ctx.positionals[1];
			if (!key) {
				throw new Error("Key required: autorag p2p policy unset <key>");
			}
			const entries = readPolicyToml(policyPath);
			if (key in entries) {
				// eslint-disable-next-line @typescript-eslint/no-dynamic-delete
				delete entries[key];
			}
			writePolicyToml(policyPath, entries);
			ctx.stdout(JSON.stringify({ ok: true, key }));
			return 0;
		}
		default:
			throw new Error(`Unknown subcommand "${subcommand}". Expected: list, set, or unset`);
	}
}
