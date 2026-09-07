import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadSignalPeerRegistry, type SignalPeerRecord, type SignalPeerRegistry } from "../../p2p/signal-server.ts";
import { registerAccount, verifyAccount } from "../../p2p/signal-transport.ts";
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

function peersPath(workspace: string): string {
	return join(workspace, ".autorag", "p2p", "signal-peers.json");
}

function savePeerRegistry(workspace: string, registry: SignalPeerRegistry): void {
	const path = peersPath(workspace);
	mkdirSync(join(workspace, ".autorag", "p2p"), { recursive: true });
	writeFileSync(path, JSON.stringify(registry, null, 2), { mode: 0o600 });
}

/**
 * `autorag p2p` — Signal account registration and peer registry management.
 */
export async function runP2p(ctx: CommandContext): Promise<number> {
	const subcommand = ctx.positionals[0];
	const flags = ctx.flags;

	if (subcommand === undefined || subcommand === "help") {
		ctx.stdout(`Usage: autorag p2p <subcommand> [options]

Subcommands:
  register <+E164>          Register this installation's Signal account (SMS)
    [--voice]               Request a voice call instead (only after an SMS attempt)
    [--captcha <token>]     signalcaptcha:// token when Signal challenges registration
  verify <+E164> <code>     Complete registration with the SMS/voice code
    [--pin <pin>]           Registration-lock PIN, if set

  peers                     List registered peers (alias, signalId, addedAt)
  peers --add <alias> --signal-id <+E164|uuid>   Trust a peer's Signal id
  peers --remove <alias>    Remove a peer from the registry

  help                      Show this help
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
		case "register": {
			const number = ctx.positionals[1];
			if (typeof number !== "string" || !/^\+[1-9][0-9]{6,14}$/.test(number)) {
				ctx.stderr(
					renderError(new ConfigError("register requires an E.164 phone number like +821012345678."), {
						json: ctx.json,
					}),
				);
				return 2;
			}
			const captcha = typeof flags.captcha === "string" ? flags.captcha : undefined;
			try {
				await registerAccount({
					number,
					voice: flags.voice === true,
					...(captcha !== undefined ? { captcha } : {}),
				});
			} catch (error) {
				ctx.stderr(
					renderError(error instanceof Error ? error : new ConfigError("registration failed"), {
						json: ctx.json,
						debug: ctx.debug,
					}),
				);
				return 1;
			}
			ctx.stdout(`Verification code sent to ${number}. Complete with: autorag p2p verify ${number} <code>`);
			return 0;
		}

		case "verify": {
			const number = ctx.positionals[1];
			const code = ctx.positionals[2];
			if (typeof number !== "string" || typeof code !== "string" || code.length === 0) {
				ctx.stderr(renderError(new ConfigError("verify requires <+E164> <code>."), { json: ctx.json }));
				return 2;
			}
			const pin = typeof flags.pin === "string" ? flags.pin : undefined;
			try {
				await verifyAccount({ number, code, ...(pin !== undefined ? { pin } : {}) });
			} catch (error) {
				ctx.stderr(
					renderError(error instanceof Error ? error : new ConfigError("verification failed"), {
						json: ctx.json,
						debug: ctx.debug,
					}),
				);
				return 1;
			}
			ctx.stdout(
				`Registered Signal account ${number}. Set p2p.account = "${number}" in config and run autorag serve.`,
			);
			return 0;
		}

		case "peers": {
			const remove = flags.remove;
			const add = flags.add;
			if (remove !== undefined) {
				if (typeof remove !== "string" || remove.length === 0) {
					ctx.stderr(renderError(new ConfigError("--remove requires a peer alias."), { json: ctx.json }));
					return 2;
				}
				const registry = loadSignalPeerRegistry(workspace);
				if (!(remove in registry)) {
					ctx.stderr(renderError(new ConfigError(`Peer not found: ${remove}`), { json: ctx.json }));
					return 2;
				}
				delete registry[remove];
				savePeerRegistry(workspace, registry);
				if (ctx.json) ctx.stdout(JSON.stringify({ ok: true, removed: remove }));
				else ctx.stdout(`Removed peer: ${remove}`);
				return 0;
			}
			if (add !== undefined) {
				if (typeof add !== "string" || add.length === 0) {
					ctx.stderr(renderError(new ConfigError("--add requires a peer alias."), { json: ctx.json }));
					return 2;
				}
				const signalId = flags["signal-id"];
				if (typeof signalId !== "string" || signalId.length === 0) {
					ctx.stderr(
						renderError(new ConfigError("--signal-id requires the peer's E.164 number or ACI UUID."), {
							json: ctx.json,
						}),
					);
					return 2;
				}
				const registry = loadSignalPeerRegistry(workspace);
				const record: SignalPeerRecord = { signalId, addedAt: new Date().toISOString() };
				registry[add] = record;
				savePeerRegistry(workspace, registry);
				if (ctx.json) ctx.stdout(JSON.stringify({ ok: true, alias: add, signalId }));
				else ctx.stdout(`Added peer: ${add} (${signalId})`);
				return 0;
			}

			const registry = loadSignalPeerRegistry(workspace);
			const entries = Object.entries(registry);
			if (ctx.json) {
				ctx.stdout(
					JSON.stringify({
						ok: true,
						peers: entries.map(([alias, peer]) => ({ alias, signalId: peer.signalId, addedAt: peer.addedAt })),
					}),
				);
			} else if (entries.length === 0) {
				ctx.stdout("No peers registered.");
				ctx.stdout("Use `autorag p2p peers --add <alias> --signal-id <+E164|uuid>` to trust a peer.");
			} else {
				ctx.stdout(`${"Alias".padEnd(24)} ${"Signal ID".padEnd(40)} Added At`);
				ctx.stdout("-".repeat(96));
				for (const [alias, peer] of entries) {
					ctx.stdout(`${alias.padEnd(24)} ${peer.signalId.padEnd(40)} ${peer.addedAt}`);
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
