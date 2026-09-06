import { existsSync } from "node:fs";
import { loadOrCreateIdentity } from "../../p2p/identity.ts";
import { type P2pServer, type StartP2pServerOptions, startP2pServer } from "../../p2p/server.ts";
import { ConfigError, resolveConfig, resolveConfigPath } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export interface ServeCommandDeps {
	readonly startP2pServer?: typeof startP2pServer;
	readonly waitUntilStopped?: (server: P2pServer) => Promise<void>;
	readonly getFingerprint?: () => Promise<string>;
}

/**
 * `autorag serve` — start the P2P peer query server.
 */
export async function runServe(ctx: CommandContext, deps: ServeCommandDeps = {}): Promise<number> {
	const flags = ctx.flags;
	const resolved = resolveConfigPath({ flags, cwd: ctx.cwd });
	if (!existsSync(resolved.configPath)) {
		ctx.stderr(
			renderError(new ConfigError(`Config file not found: ${resolved.configPath}. Run autorag init first.`), {
				json: ctx.json,
			}),
		);
		return 2;
	}

	let config: ReturnType<typeof resolveConfig>;
	try {
		config = resolveConfig({ flags, cwd: ctx.cwd });
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	const p2p = config.p2p ?? { enabled: false, host: "0.0.0.0", port: 9470 };
	const force = flags.force === true;
	if (p2p.enabled !== true && !force) {
		ctx.stderr(
			renderError(
				new ConfigError(
					"P2P sharing is disabled in config. Use --force to start anyway, or set p2p.enabled to true in config.",
				),
				{ json: ctx.json },
			),
		);
		return 2;
	}

	const host = typeof flags.host === "string" && flags.host.length > 0 ? flags.host : (p2p.host ?? "0.0.0.0");
	let port = p2p.port ?? 9470;
	if (typeof flags.port === "string" && flags.port.length > 0) {
		const parsed = Number(flags.port);
		if (!Number.isInteger(parsed) || parsed < 0) {
			ctx.stderr(renderError(new ConfigError("--port must be a non-negative integer."), { json: ctx.json }));
			return 2;
		}
		port = parsed;
	}

	// Load identity to get fingerprint (never print secrets)
	const getFingerprint =
		deps.getFingerprint ??
		(async () => {
			const identity = loadOrCreateIdentity(config.workspacePath);
			return identity.fingerprint;
		});
	const fingerprint = await getFingerprint();
	const start = deps.startP2pServer ?? startP2pServer;

	let server: P2pServer;
	try {
		server = await start({
			host,
			port,
			workspacePath: config.workspacePath,
			agent: {
				searchDocuments: () => {
					throw new Error("P2P server requires a configured search agent; use autorag search instead.");
				},
			},
		} as StartP2pServerOptions);
	} catch (error) {
		const status = error instanceof ConfigError ? 2 : 1;
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return status;
	}

	const payload = {
		ok: true,
		host: server.host,
		port: server.port,
		fingerprint,
	};
	ctx.stdout(
		ctx.json ? JSON.stringify(payload) : `P2P server on ${server.url} | fingerprint: ${fingerprint.slice(0, 16)}...`,
	);

	const wait = deps.waitUntilStopped ?? defaultWaitUntilStopped;
	try {
		await wait(server);
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

async function defaultWaitUntilStopped(server: P2pServer): Promise<void> {
	await new Promise<void>((resolve) => {
		let stopped = false;
		const stop = () => {
			if (stopped) return;
			stopped = true;
			void server.close().finally(resolve);
		};
		process.once("SIGINT", stop);
		process.once("SIGTERM", stop);
	});
}
