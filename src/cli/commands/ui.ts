import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { isLoopbackHost, startUiServer, type UiServer } from "../../ui/server.ts";
import { ConfigError, resolveConfig, resolveConfigPath } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export interface UiCommandDeps {
	readonly startServer?: typeof startUiServer;
	readonly openBrowser?: (url: string) => Promise<void>;
	readonly waitUntilStopped?: (server: UiServer) => Promise<void>;
	readonly createToken?: () => string;
}

/**
 * `autorag ui` — local loopback dashboard for folders and datasource connections.
 */
export async function runUi(ctx: CommandContext, deps: UiCommandDeps = {}): Promise<number> {
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
	const ui = config.ui ?? {};
	const host = typeof flags.host === "string" && flags.host.length > 0 ? flags.host : (ui.host ?? "127.0.0.1");
	const allowRemote = ui.allowRemote === true || flags["allow-remote"] === true;
	if (!isLoopbackHost(host) && !allowRemote) {
		ctx.stderr(
			renderError(new ConfigError("UI host must be a loopback address unless ui.allowRemote is true."), {
				json: ctx.json,
			}),
		);
		return 2;
	}

	let port = ui.port ?? 8787;
	if (typeof flags.port === "string" && flags.port.length > 0) {
		const parsed = Number(flags.port);
		if (!Number.isInteger(parsed) || parsed < 0) {
			ctx.stderr(renderError(new ConfigError("--port must be a non-negative integer."), { json: ctx.json }));
			return 2;
		}
		port = parsed;
	}

	const tokenEnv = ui.tokenEnv ?? "AUTORAG_UI_TOKEN";
	const configuredToken = process.env[tokenEnv];
	if (allowRemote && (configuredToken === undefined || configuredToken.length < 16)) {
		ctx.stderr(
			renderError(new ConfigError(`Remote UI requires ${tokenEnv} with at least 16 characters.`), {
				json: ctx.json,
			}),
		);
		return 2;
	}
	const token = configuredToken ?? deps.createToken?.() ?? randomBytes(24).toString("hex");
	const start = deps.startServer ?? startUiServer;
	let server: UiServer;
	try {
		server = await start({
			configPath: resolved.configPath,
			host,
			port,
			token,
			allowRemote,
			...(ui.publicOrigin !== undefined ? { publicOrigin: ui.publicOrigin } : {}),
			...(ui.corsOrigins !== undefined ? { corsOrigins: ui.corsOrigins } : {}),
		});
	} catch (error) {
		const status = error instanceof ConfigError ? 2 : 1;
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return status;
	}

	const payload = { ok: true, url: server.url, host: server.host, port: server.port };
	ctx.stdout(ctx.json ? JSON.stringify(payload) : `AutoRAG UI on ${server.url}`);

	if (flags["no-open"] !== true) {
		try {
			await (deps.openBrowser ?? openBrowser)(server.url);
		} catch (error) {
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		}
	}

	const wait = deps.waitUntilStopped ?? defaultWaitUntilStopped;
	try {
		await wait(server);
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

async function defaultWaitUntilStopped(server: UiServer): Promise<void> {
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

export async function openBrowser(url: string): Promise<void> {
	const command = process.platform === "darwin" ? "open" : process.platform === "win32" ? "cmd" : "xdg-open";
	const args = process.platform === "win32" ? ["/c", "start", "", url] : [url];
	spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true }).unref();
}
