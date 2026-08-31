/**
 * Loopback-only HTTP server for the datasource UI.
 *
 * Binds 127.0.0.1 / ::1, requires a session token, and never returns secret
 * connector values. This is a trusted local control plane — the same trust
 * boundary as editing config.json by hand.
 */

import { timingSafeEqual } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { ConfigError } from "../cli/config.ts";
import { browseDirectory } from "./browse.ts";
import { listUiState, removeConnection, setSearchPaths, toggleConnection, upsertConnection } from "./config-store.ts";
import { choicesForType } from "./discover.ts";
import { renderUiPage } from "./html.ts";
import { probeConnection } from "./probe.ts";
import { buildRegistrationPrompt } from "./prompt.ts";

export interface StartUiServerOptions {
	readonly configPath: string;
	readonly host?: string;
	readonly port?: number;
	readonly token: string;
	readonly env?: NodeJS.ProcessEnv;
	readonly allowRemote?: boolean;
	readonly publicOrigin?: string;
	readonly corsOrigins?: readonly string[];
}

export interface UiServer {
	readonly url: string;
	readonly origin: string;
	readonly host: string;
	readonly port: number;
	readonly token: string;
	close(): Promise<void>;
}

const COOKIE = "autorag_ui";

export function isLoopbackHost(host: string): boolean {
	const normalized = host.trim().toLowerCase();
	return normalized === "127.0.0.1" || normalized === "::1" || normalized === "localhost";
}

export function isLoopbackAddress(address: string | undefined): boolean {
	if (address === undefined || address.length === 0) return false;
	return address === "127.0.0.1" || address === "::1" || address === "::ffff:127.0.0.1";
}

export async function startUiServer(options: StartUiServerOptions): Promise<UiServer> {
	const host = options.host ?? "127.0.0.1";
	const allowRemote = options.allowRemote === true;
	if (!isLoopbackHost(host) && !allowRemote) {
		throw new ConfigError("UI host must be a loopback address (127.0.0.1 or ::1).");
	}
	const port = options.port ?? 8787;
	const env = options.env ?? process.env;
	const token = options.token;
	if (token.length < 16) throw new ConfigError("UI session token is too short.");
	const publicOrigin = options.publicOrigin === undefined ? undefined : normalizeOrigin(options.publicOrigin);
	const corsOrigins = normalizeOrigins(options.corsOrigins ?? []);
	if (allowRemote && isWildcardHost(host) && publicOrigin === undefined) {
		throw new ConfigError("Remote wildcard binds require ui.publicOrigin.");
	}

	const httpServer = createServer((req, res) => {
		void handleRequest(req, res, {
			configPath: options.configPath,
			token,
			env,
			allowRemote,
			publicOrigin,
			corsOrigins,
		});
	});

	await listen(httpServer, port, host);
	const address = httpServer.address();
	if (address === null || typeof address === "string") {
		httpServer.close();
		throw new ConfigError("UI server failed to bind a loopback port.");
	}
	const boundHost = formatHost(host);
	const origin = publicOrigin ?? `http://${boundHost}:${address.port}`;
	return {
		origin,
		url: `${origin}/?token=${encodeURIComponent(token)}`,
		host: boundHost === "[::1]" ? "::1" : boundHost,
		port: address.port,
		token,
		close: () =>
			new Promise((resolve, reject) => {
				httpServer.close((error) => (error ? reject(error) : resolve()));
			}),
	};
}

async function handleRequest(
	req: IncomingMessage,
	res: ServerResponse,
	ctx: {
		configPath: string;
		token: string;
		env: NodeJS.ProcessEnv;
		allowRemote: boolean;
		publicOrigin: string | undefined;
		corsOrigins: readonly string[];
	},
): Promise<void> {
	try {
		const requestOrigin = typeof req.headers.origin === "string" ? normalizeOrigin(req.headers.origin) : undefined;
		const sameOrigin = requestOrigin !== undefined && requestOrigin === `http://${req.headers.host ?? "127.0.0.1"}`;
		const allowedOrigin =
			requestOrigin === undefined ||
			sameOrigin ||
			requestOrigin === ctx.publicOrigin ||
			ctx.corsOrigins.includes(requestOrigin);
		if (!allowedOrigin) {
			send(res, 403, { error: "Origin is not allowed." });
			return;
		}
		applyCorsHeaders(res, requestOrigin, sameOrigin);
		if (req.method === "OPTIONS") {
			if (requestOrigin === undefined) {
				send(res, 400, { error: "CORS preflight requires an Origin header." });
				return;
			}
			res.statusCode = 204;
			res.end();
			return;
		}
		if (!ctx.allowRemote && !isLoopbackAddress(req.socket.remoteAddress)) {
			send(res, 403, { error: "Loopback only." });
			return;
		}
		const hostHeader = req.headers.host ?? "127.0.0.1";
		const url = new URL(req.url ?? "/", `http://${hostHeader}`);
		const authorized = tokensEqual(readToken(req, url), ctx.token);
		if (!authorized) {
			send(res, 401, { error: "Missing or invalid UI token." });
			return;
		}

		if (url.pathname === "/" && (req.method === "GET" || req.method === "HEAD")) {
			const state = listUiState(ctx.configPath, ctx.env);
			const html = renderUiPage(state);
			res.statusCode = 200;
			res.setHeader("content-type", "text/html; charset=utf-8");
			res.setHeader("cache-control", "no-store");
			res.setHeader("set-cookie", `${COOKIE}=${ctx.token}; HttpOnly; SameSite=Strict; Path=/`);
			res.end(req.method === "HEAD" ? undefined : html);
			return;
		}

		if (url.pathname === "/api/state" && req.method === "GET") {
			send(res, 200, publicState(listUiState(ctx.configPath, ctx.env)));
			return;
		}

		if (url.pathname === "/api/folders" && req.method === "POST") {
			const body = await readJson(req);
			const searchPaths = Array.isArray(body.searchPaths) ? body.searchPaths.map(String) : [];
			send(res, 200, publicState(setSearchPaths(ctx.configPath, searchPaths)));
			return;
		}

		if (url.pathname === "/api/connections" && req.method === "POST") {
			const body = await readJson(req);
			const state = upsertConnection(ctx.configPath, {
				alias: String(body.alias ?? ""),
				type: String(body.type ?? ""),
				enabled: body.enabled !== false,
				...(typeof body.instanceId === "string" ? { instanceId: body.instanceId } : {}),
				connector: isRecord(body.connector) ? body.connector : {},
			});
			const alias = String(body.alias ?? "").trim();
			const connection = state.connections.find((item) => item.alias === alias);
			send(res, 200, { ...publicState(state), connection });
			return;
		}

		if (url.pathname === "/api/browse" && req.method === "GET") {
			send(res, 200, browseDirectory(url.searchParams.get("path") ?? undefined));
			return;
		}

		if (url.pathname === "/api/prompt" && req.method === "GET") {
			send(
				res,
				200,
				buildRegistrationPrompt({
					type: url.searchParams.get("type") ?? "",
					...(url.searchParams.get("alias") ? { alias: url.searchParams.get("alias") ?? undefined } : {}),
					...(url.searchParams.get("note") ? { note: url.searchParams.get("note") ?? undefined } : {}),
					extras: parseExtras(url.searchParams.get("extras")),
				}),
			);
			return;
		}

		if (url.pathname === "/api/choices" && req.method === "GET") {
			const type = url.searchParams.get("type") ?? "";
			send(res, 200, await choicesForType(type));
			return;
		}

		const connectionMatch = /^\/api\/connections\/([^/]+)(?:\/(test|toggle))?$/.exec(url.pathname);
		if (connectionMatch) {
			const alias = decodeURIComponent(connectionMatch[1] ?? "");
			const action = connectionMatch[2];
			if (req.method === "DELETE" && action === undefined) {
				send(res, 200, publicState(removeConnection(ctx.configPath, alias)));
				return;
			}
			if (req.method === "POST" && action === "toggle") {
				const body = await readJson(req);
				send(res, 200, publicState(toggleConnection(ctx.configPath, alias, body.enabled !== false)));
				return;
			}
			if (req.method === "POST" && action === "test") {
				const state = listUiState(ctx.configPath, ctx.env);
				const connection = state.connections.find((item) => item.alias === alias);
				if (connection === undefined) {
					send(res, 404, { error: `Unknown connection: ${alias}` });
					return;
				}
				send(res, 200, probeConnection(connection, { env: ctx.env }));
				return;
			}
		}

		send(res, 404, { error: "Not found." });
	} catch (error) {
		const message = error instanceof Error ? error.message : "UI request failed.";
		const status = error instanceof ConfigError ? 400 : 500;
		send(res, status, { error: message });
	}
}

function publicState(state: ReturnType<typeof listUiState>): Record<string, unknown> {
	return {
		searchPaths: state.searchPaths,
		connections: state.connections,
		catalog: state.catalog.map((entry) => ({
			type: entry.type,
			title: entry.title,
			summary: entry.summary,
			binaryName: entry.binaryName,
			installHint: entry.installHint,
			fields: entry.fields,
			defaultTags: entry.defaultTags,
		})),
		picker: state.picker,
		access: state.access,
	};
}

function readToken(req: IncomingMessage, url: URL): string {
	const header = req.headers["x-autorag-token"];
	if (typeof header === "string" && header.length > 0) return header;
	if (Array.isArray(header) && header[0]) return header[0];
	const query = url.searchParams.get("token");
	if (query) return query;
	const cookie = req.headers.cookie;
	if (typeof cookie !== "string") return "";
	for (const part of cookie.split(";")) {
		const trimmed = part.trim();
		if (trimmed.startsWith(`${COOKIE}=`)) return trimmed.slice(COOKIE.length + 1);
	}
	return "";
}

function tokensEqual(provided: string, expected: string): boolean {
	const left = Buffer.from(provided);
	const right = Buffer.from(expected);
	if (left.length !== right.length) return false;
	return timingSafeEqual(left, right);
}

function send(res: ServerResponse, status: number, body: unknown): void {
	res.statusCode = status;
	res.setHeader("content-type", "application/json; charset=utf-8");
	res.setHeader("cache-control", "no-store");
	res.end(JSON.stringify(body));
}

function applyCorsHeaders(res: ServerResponse, origin: string | undefined, sameOrigin: boolean): void {
	res.setHeader("vary", "Origin");
	if (origin === undefined || sameOrigin) return;
	res.setHeader("access-control-allow-origin", origin);
	res.setHeader("access-control-allow-credentials", "true");
	res.setHeader("access-control-allow-methods", "GET, POST, DELETE, OPTIONS");
	res.setHeader("access-control-allow-headers", "content-type, x-autorag-token");
}

function normalizeOrigins(origins: readonly string[]): string[] {
	const normalized: string[] = [];
	for (const origin of origins) {
		const value = normalizeOrigin(origin);
		if (!normalized.includes(value)) normalized.push(value);
	}
	return normalized;
}

function normalizeOrigin(value: string): string {
	let origin: URL;
	try {
		origin = new URL(value);
	} catch {
		throw new ConfigError("UI origins must be valid http(s) origins.");
	}
	if (!["http:", "https:"].includes(origin.protocol) || origin.pathname !== "/" || origin.search || origin.hash) {
		throw new ConfigError("UI origins must be valid http(s) origins.");
	}
	return origin.origin;
}

function formatHost(host: string): string {
	return host.includes(":") ? `[${host}]` : host;
}

function isWildcardHost(host: string): boolean {
	return host === "0.0.0.0" || host === "::";
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
	const chunks: Buffer[] = [];
	for await (const chunk of req) {
		chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
		if (chunks.reduce((sum, item) => sum + item.length, 0) > 1_000_000) {
			throw new ConfigError("Request body is too large.");
		}
	}
	if (chunks.length === 0) return {};
	const text = Buffer.concat(chunks).toString("utf8");
	if (text.trim().length === 0) return {};
	const parsed: unknown = JSON.parse(text);
	if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
		throw new ConfigError("JSON body must be an object.");
	}
	return parsed as Record<string, unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseExtras(raw: string | null): Record<string, string> {
	if (raw === null || raw.trim().length === 0) return {};
	try {
		const parsed: unknown = JSON.parse(raw);
		if (!isRecord(parsed)) return {};
		const extras: Record<string, string> = {};
		for (const [key, value] of Object.entries(parsed)) {
			if (typeof value === "string") extras[key] = value;
		}
		return extras;
	} catch {
		return {};
	}
}

function listen(server: Server, port: number, host: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const onError = (error: Error) => {
			server.off("error", onError);
			reject(error);
		};
		server.once("error", onError);
		server.listen(port, host, () => {
			server.off("error", onError);
			resolve();
		});
	});
}
