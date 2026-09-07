import { loadSimplexPeerRegistry, type SimplexPeerRecord, saveSimplexPeerRegistry } from "../../p2p/simplex-server.ts";
import { ConfigError, resolveConfig, resolveConfigPath } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/** Resolve the workspace used for P2P identity/policy, matching `autorag serve`. */
export function resolveP2pWorkspace(ctx: CommandContext): string {
	const resolved = resolveConfigPath({ flags: ctx.flags, cwd: ctx.cwd });
	// Only honor an explicit config (`--config` / AUTORAG_CONFIG). A default home
	// config must not steal identity/policy writes away from the caller's cwd.
	if (resolved.explicit) {
		return resolveConfig({ flags: ctx.flags, cwd: ctx.cwd, readOnly: true }).workspacePath;
	}
	return ctx.cwd;
}

/**
 * `autorag p2p` — SimpleX peer trust management.
 */
export async function runP2p(ctx: CommandContext): Promise<number> {
	const subcommand = ctx.positionals[0];
	const flags = ctx.flags;

	if (subcommand === undefined || subcommand === "help") {
		ctx.stdout(`Usage: autorag p2p <subcommand> [options]

Subcommands:
  peers                     List trusted peers (alias, contactId, addedAt)
  peers --add <alias> --contact-id <n>   Trust a peer's SimpleX contact id
  peers --remove <alias>    Remove a peer from the registry

  help                      Show this help

Peers connect via SimpleX addresses printed by \`autorag serve\`.
`);
		return 0;
	}

	let workspace: string;
	try {
		workspace = resolveP2pWorkspace(ctx);
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	switch (subcommand) {
		case "peers": {
			const remove = flags.remove;
			const add = flags.add;
			if (remove !== undefined) {
				if (typeof remove !== "string" || remove.length === 0) {
					ctx.stderr(renderError(new ConfigError("--remove requires a peer alias."), { json: ctx.json }));
					return 2;
				}
				const registry = loadSimplexPeerRegistry(workspace);
				if (!(remove in registry)) {
					ctx.stderr(renderError(new ConfigError(`Peer not found: ${remove}`), { json: ctx.json }));
					return 2;
				}
				delete registry[remove];
				saveSimplexPeerRegistry(workspace, registry);
				if (ctx.json) ctx.stdout(JSON.stringify({ ok: true, removed: remove }));
				else ctx.stdout(`Removed peer: ${remove}`);
				return 0;
			}
			if (add !== undefined) {
				if (typeof add !== "string" || add.length === 0) {
					ctx.stderr(renderError(new ConfigError("--add requires a peer alias."), { json: ctx.json }));
					return 2;
				}
				const contactIdRaw = flags["contact-id"];
				const contactId = typeof contactIdRaw === "string" ? Number(contactIdRaw) : NaN;
				if (!Number.isSafeInteger(contactId) || contactId < 1) {
					ctx.stderr(
						renderError(new ConfigError("--contact-id requires the peer's SimpleX contact id (integer)."), {
							json: ctx.json,
						}),
					);
					return 2;
				}
				const registry = loadSimplexPeerRegistry(workspace);
				const record: SimplexPeerRecord = { contactId, addedAt: new Date().toISOString() };
				registry[add] = record;
				saveSimplexPeerRegistry(workspace, registry);
				if (ctx.json) ctx.stdout(JSON.stringify({ ok: true, alias: add, contactId }));
				else ctx.stdout(`Added peer: ${add} (contactId ${contactId})`);
				return 0;
			}

			const registry = loadSimplexPeerRegistry(workspace);
			const entries = Object.entries(registry);
			if (ctx.json) {
				ctx.stdout(
					JSON.stringify({
						ok: true,
						peers: entries.map(([alias, peer]) => ({ alias, contactId: peer.contactId, addedAt: peer.addedAt })),
					}),
				);
			} else if (entries.length === 0) {
				ctx.stdout("No peers registered.");
				ctx.stdout("Use `autorag p2p peers --add <alias> --contact-id <n>` to trust a peer.");
			} else {
				ctx.stdout(`${"Alias".padEnd(24)} ${"Contact ID".padEnd(12)} Added At`);
				ctx.stdout("-".repeat(72));
				for (const [alias, peer] of entries) {
					ctx.stdout(`${alias.padEnd(24)} ${String(peer.contactId).padEnd(12)} ${peer.addedAt}`);
				}
			}
			return 0;
		}

		default: {
			ctx.stderr(
				renderError(new ConfigError(`Unknown p2p subcommand: ${subcommand}. Use 'autorag p2p help'.`), {
					json: ctx.json,
				}),
			);
			return 2;
		}
	}
}
