import {
	decodePairingCode,
	loadOrCreateIdentity,
	loadPeerRegistry,
	registerPeer,
	removePeer,
} from "../../p2p/identity.ts";
import { ConfigError, resolveConfigPath } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag p2p pair` and `autorag p2p peers` — peer identity and registry management.
 */
export async function runP2p(ctx: CommandContext): Promise<number> {
	const subcommand = ctx.positionals[0];
	const flags = ctx.flags;
	const _resolved = resolveConfigPath({ flags, cwd: ctx.cwd });

	// Load or create identity (lazy init if not yet present)
	const identity = loadOrCreateIdentity(ctx.cwd);

	switch (subcommand) {
		case undefined:
		case "help": {
			ctx.stdout(`Usage: autorag p2p <subcommand> [options]

Subcommands:
  pair                      Print this installation's pairing code
  pair --accept <code>      Accept a peer pairing code and add to registry
    [--alias name]          Optional alias for the peer (default: from code)
    [--endpoint host:port]  Local endpoint advertised in pairing code
    [--alias name]          Local alias in pairing code (default: hostname)

  peers                     List registered peers (alias, fingerprint, endpoint, addedAt)
  peers --remove <alias>    Remove a peer from the registry

  help                      Show this help
`);
			return 0;
		}

		case "pair": {
			const accept = flags.accept;
			if (accept !== undefined) {
				// --accept <code> path
				if (typeof accept !== "string" || accept.length === 0) {
					ctx.stderr(renderError(new ConfigError("--accept requires a pairing code value."), { json: ctx.json }));
					return 2;
				}
				let codePayload: ReturnType<typeof decodePairingCode>;
				try {
					codePayload = decodePairingCode(accept);
				} catch (error) {
					ctx.stderr(
						renderError(
							new ConfigError(
								`Invalid pairing code: ${error instanceof Error ? error.message : "decode failed"}.`,
							),
							{ json: ctx.json },
						),
					);
					return 2;
				}

				// Determine alias: explicit --alias flag wins, fall back to code's alias
				const alias = typeof flags.alias === "string" && flags.alias.length > 0 ? flags.alias : codePayload.alias;

				let peer: import("../../p2p/identity.ts").PeerRecord;
				try {
					peer = registerPeer(ctx.cwd, alias, {
						endpoint: codePayload.endpoint,
						pubkey: codePayload.pubkey,
					});
				} catch (error) {
					ctx.stderr(
						renderError(
							new ConfigError(
								`Failed to register peer: ${error instanceof Error ? error.message : "unknown error"}.`,
							),
							{ json: ctx.json },
						),
					);
					return 2;
				}

				if (ctx.json) {
					ctx.stdout(
						JSON.stringify({
							ok: true,
							fingerprint: peer.fingerprint,
							endpoint: peer.endpoint,
							alias,
						}),
					);
				} else {
					ctx.stdout(`Accepted peer: ${alias}`);
					ctx.stdout(`  fingerprint: ${peer.fingerprint}`);
					ctx.stdout(`  endpoint:    ${peer.endpoint}`);
				}
				return 0;
			}

			// Generate a pairing code for this installation
			const endpoint =
				typeof flags.endpoint === "string" && flags.endpoint.length > 0 ? flags.endpoint : "127.0.0.1:9470";
			const alias =
				typeof flags.alias === "string" && flags.alias.length > 0
					? flags.alias
					: `autorag-${identity.fingerprint.slice(0, 8)}`;

			const code = identity.pairingCode(endpoint, alias);

			if (ctx.json) {
				ctx.stdout(
					JSON.stringify({
						ok: true,
						code,
						fingerprint: identity.fingerprint,
						endpoint,
					}),
				);
			} else {
				ctx.stdout(`Pairing code for this installation:`);
				ctx.stdout(`  code=${code}`);
				ctx.stdout(`  fingerprint=${identity.fingerprint}`);
				ctx.stdout(`  endpoint=${endpoint}`);
				ctx.stdout(`  alias=${alias}`);
				ctx.stdout(``);
				ctx.stdout(`Give this code to a friend, or run:`);
				ctx.stdout(`  autorag p2p pair --accept <code> [--alias name]`);
			}
			return 0;
		}

		case "peers": {
			const remove = flags.remove;
			if (remove !== undefined) {
				// --remove <alias>
				if (typeof remove !== "string" || remove.length === 0) {
					ctx.stderr(renderError(new ConfigError("--remove requires a peer alias."), { json: ctx.json }));
					return 2;
				}
				const removed = removePeer(ctx.cwd, remove);
				if (!removed) {
					ctx.stderr(renderError(new ConfigError(`Peer not found: ${remove}`), { json: ctx.json }));
					return 2;
				}
				if (ctx.json) {
					ctx.stdout(JSON.stringify({ ok: true, removed: remove }));
				} else {
					ctx.stdout(`Removed peer: ${remove}`);
				}
				return 0;
			}

			const registry = loadPeerRegistry(ctx.cwd);
			const entries = Object.entries(registry);

			if (ctx.json) {
				ctx.stdout(
					JSON.stringify({
						ok: true,
						peers: entries.map(([alias, peer]) => ({
							alias,
							fingerprint: peer.fingerprint,
							endpoint: peer.endpoint,
							addedAt: peer.addedAt,
						})),
					}),
				);
			} else if (entries.length === 0) {
				ctx.stdout("No peers registered.");
				ctx.stdout("Use `autorag p2p pair --accept <code>` to add a peer.");
			} else {
				// Column header
				ctx.stdout(`${"Alias".padEnd(24)} ${"Fingerprint".padEnd(64)} ${"Endpoint".padEnd(24)} Added At`);
				ctx.stdout("-".repeat(140));
				for (const [alias, peer] of entries) {
					ctx.stdout(
						`${alias.padEnd(24)} ${peer.fingerprint.padEnd(64)} ${peer.endpoint.padEnd(24)} ${peer.addedAt}`,
					);
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
