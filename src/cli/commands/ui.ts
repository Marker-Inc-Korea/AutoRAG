import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { isLoopbackHost, startUiServer, type UiServer } from "../../ui/server.ts";
import { ConfigError, resolveConfigPath } from "../config.ts";
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
	const host = typeof flags.host === "string" && flags.host.length > 0 ? flags.host : "127.0.0.1";
	if (!isLoopbackHost(host)) {
		ctx.stderr(
			renderError(new ConfigError("UI host must be a loopback address (127.0.0.1 or ::1)."), { json: ctx.json }),
		);
		return 2;
	}

	let port = 8787;
	if (typeof flags.port === "string" && flags.port.length > 0) {
		const parsed = Number(flags.port);
		if (!Number.isInteger(parsed) || parsed < 0) {
			ctx.stderr(renderError(new ConfigError("--port must be a non-negative integer."), { json: ctx.json }));
			return 2;
		}
		port = parsed;
	}

	const resolved = resolveConfigPath({ flags, cwd: ctx.cwd });
	if (!existsSync(resolved.configPath)) {
		ctx.stderr(
			renderError(new ConfigError(`Config file not found: ${resolved.configPath}. Run autorag init first.`), {
				json: ctx.json,
			}),
		);
		return 2;
	}

	const token = deps.createToken?.() ?? randomBytes(24).toString("hex");
	const start = deps.startServer ?? startUiServer;
	let server: UiServer;
	try {
		server = await start({ configPath: resolved.configPath, host, port, token });
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
