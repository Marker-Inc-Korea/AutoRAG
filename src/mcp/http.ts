import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { createMcpHandler, validateHostHeader } from "@modelcontextprotocol/server";
import { z } from "zod";
import type { AutoRAGLite } from "../core.ts";
import { createAutoRAGMcpServer } from "./server.ts";

export interface McpHttpOptions {
	readonly lite: AutoRAGLite;
	/** Interface to bind. Anything but loopback needs `publicUrl`, because it exposes the corpus to a network. */
	readonly host?: string;
	/** `0` picks a free port. */
	readonly port?: number;
	readonly path?: string;
	/**
	 * The HTTPS origin remote agents reach this server at (a tunnel or reverse proxy). Setting it turns on the OAuth
	 * endpoints connector-only clients need (ChatGPT, Claude connectors, Meta Muse) and widens the Host/Origin allowlist.
	 */
	readonly publicUrl?: string;
	/**
	 * The one secret that opens everything: sent as `Authorization: Bearer`, typed on the OAuth consent page, and the key
	 * every issued token is signed with. Rotating it revokes every token.
	 */
	readonly ownerToken: string;
	/** Default `true`: a remote agent gets search and evidence, never refresh, report or convert. */
	readonly readOnly?: boolean;
	readonly tools?: readonly string[];
	readonly maxBodyBytes?: number;
	readonly now?: () => number;
	readonly onError?: (error: Error) => void;
}

export interface McpHttpServer {
	/** The MCP endpoint on the local listener. */
	readonly url: string;
	/** The MCP endpoint remote agents are given, when `publicUrl` is set. */
	readonly publicMcpUrl?: string;
	readonly close: () => Promise<void>;
}

const MIN_OWNER_TOKEN_LENGTH = 32;
const DEFAULT_MAX_BODY_BYTES = 1024 * 1024;
const CODE_TTL_SECONDS = 60;
const ACCESS_TTL_SECONDS = 60 * 60;
const REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60;
const MAX_REDIRECT_URIS = 10;
const AUTHORIZE_FAILURE_LIMIT = 5;
const AUTHORIZE_FAILURE_WINDOW_MS = 60_000;
const LOOPBACK_HOSTS: Record<string, true> = { "127.0.0.1": true, localhost: true, "::1": true, "[::1]": true };

const clientPayload = z.object({
	k: z.literal("client"),
	name: z.string(),
	uris: z.array(z.string()).min(1).max(MAX_REDIRECT_URIS),
});
const codePayload = z.object({
	k: z.literal("code"),
	cid: z.string(),
	ru: z.string(),
	cc: z.string(),
	res: z.string(),
	n: z.string(),
	exp: z.number(),
});
const accessPayload = z.object({ k: z.literal("access"), cid: z.string(), res: z.string(), exp: z.number() });
const refreshPayload = z.object({ k: z.literal("refresh"), cid: z.string(), res: z.string(), exp: z.number() });

const registrationBody = z.object({
	client_name: z.string().max(200).optional(),
	redirect_uris: z.array(z.string().max(2000)).max(MAX_REDIRECT_URIS),
});

function isLoopbackHost(hostname: string): boolean {
	return LOOPBACK_HOSTS[hostname] === true;
}

function digest(value: string): Buffer {
	return createHash("sha256").update(value).digest();
}

/** Compares secrets through fixed-length digests so neither length nor content leaks through timing. */
function secretsEqual(a: string, b: string): boolean {
	return timingSafeEqual(digest(a), digest(b));
}

function normalizeResource(value: string): string {
	return value.replace(/\/+$/, "");
}

function isAllowedRedirect(uri: string): boolean {
	let parsed: URL;
	try {
		parsed = new URL(uri);
	} catch {
		return false;
	}
	if (parsed.hash !== "") return false;
	if (parsed.protocol === "https:") return true;
	return parsed.protocol === "http:" && isLoopbackHost(parsed.hostname);
}

function escapeHtml(value: string): string {
	return value
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&#39;");
}

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "content-type": "application/json", "cache-control": "no-store", ...headers },
	});
}

function oauthError(status: number, error: string, description?: string): Response {
	return jsonResponse(status, { error, ...(description === undefined ? {} : { error_description: description }) });
}

function htmlResponse(status: number, body: string): Response {
	return new Response(body, {
		status,
		headers: {
			"content-type": "text/html; charset=utf-8",
			"cache-control": "no-store",
			"x-frame-options": "DENY",
			"referrer-policy": "same-origin",
			// Submitting the consent form redirects to the client's callback, which the form-action directive also governs.
			"content-security-policy":
				"default-src 'none'; style-src 'unsafe-inline'; form-action 'self' https: http://127.0.0.1:* http://localhost:*; frame-ancestors 'none'",
		},
	});
}

class ReadError extends Error {
	readonly status: number;
	constructor(status: number) {
		super(`request body rejected with ${status}`);
		this.status = status;
	}
}

async function readBody(req: IncomingMessage, limit: number): Promise<Buffer | undefined> {
	if (req.method === "GET" || req.method === "HEAD" || req.method === "DELETE") return undefined;
	const declared = Number(req.headers["content-length"]);
	if (Number.isFinite(declared) && declared > limit) throw new ReadError(413);
	const chunks: Buffer[] = [];
	let total = 0;
	for await (const chunk of req) {
		const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
		total += buffer.length;
		if (total > limit) throw new ReadError(413);
		chunks.push(buffer);
	}
	return Buffer.concat(chunks);
}

async function writeResponse(res: ServerResponse, response: Response): Promise<void> {
	res.writeHead(response.status, Object.fromEntries(response.headers));
	if (response.body === null) {
		res.end();
		return;
	}
	const reader = response.body.getReader();
	res.on("close", () => {
		reader.cancel().catch(() => {});
	});
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		res.write(value);
	}
	res.end();
}

function parseForm(body: Buffer | undefined, contentType: string | null): Record<string, string> {
	const raw = body?.toString("utf8") ?? "";
	if (raw === "") return {};
	if (contentType?.includes("application/json")) {
		const parsed: unknown = JSON.parse(raw);
		if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
		return Object.fromEntries(
			Object.entries(parsed).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
		);
	}
	return Object.fromEntries(new URLSearchParams(raw));
}

export async function startMcpHttpServer(options: McpHttpOptions): Promise<McpHttpServer> {
	const ownerToken = options.ownerToken;
	if (ownerToken.trim() === "") throw new Error("An owner token is required to serve MCP over HTTP.");
	if (ownerToken.length < MIN_OWNER_TOKEN_LENGTH) {
		throw new Error(
			`The owner token must be at least ${MIN_OWNER_TOKEN_LENGTH} characters (try: openssl rand -base64 32).`,
		);
	}
	const bindHost = options.host ?? "127.0.0.1";
	const publicOrigin = options.publicUrl === undefined ? undefined : new URL(options.publicUrl).origin;
	if (publicOrigin !== undefined) {
		const parsed = new URL(publicOrigin);
		if (parsed.protocol !== "https:" && !isLoopbackHost(parsed.hostname)) {
			throw new Error("The public URL must be https (or loopback http for local testing).");
		}
	}
	if (!isLoopbackHost(bindHost) && publicOrigin === undefined) {
		throw new Error("Binding a non-loopback host exposes your corpus; set a public URL (HTTPS tunnel or proxy) too.");
	}
	const path = options.path ?? "/mcp";
	const readOnly = options.readOnly !== false;
	const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
	const now = options.now ?? Date.now;
	const onError = options.onError ?? (() => {});
	const resourceUrl = publicOrigin === undefined ? undefined : `${publicOrigin}${path}`;
	const resourceMetadataUrl =
		publicOrigin === undefined ? undefined : `${publicOrigin}/.well-known/oauth-protected-resource${path}`;
	const allowedHostnames = ["127.0.0.1", "localhost", "[::1]"];
	if (publicOrigin !== undefined) allowedHostnames.push(new URL(publicOrigin).hostname);

	const signingKey = createHmac("sha256", ownerToken).update("autorag-mcp-oauth-v1").digest();
	const usedCodes = new Map<string, number>();
	let authorizeFailures: number[] = [];

	const sign = (kind: string, payload: object): string => {
		const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
		const mac = createHmac("sha256", signingKey).update(`${kind}.${body}`).digest("base64url");
		return `${kind}.${body}.${mac}`;
	};
	const verify = <T extends z.ZodType>(kind: string, token: string, schema: T): z.infer<T> | undefined => {
		const parts = token.split(".");
		if (parts.length !== 3 || parts[0] !== kind) return undefined;
		const [, body, mac] = parts;
		if (body === undefined || mac === undefined) return undefined;
		const expected = createHmac("sha256", signingKey).update(`${kind}.${body}`).digest("base64url");
		if (!secretsEqual(mac, expected)) return undefined;
		try {
			const parsed = schema.safeParse(JSON.parse(Buffer.from(body, "base64url").toString("utf8")));
			return parsed.success ? parsed.data : undefined;
		} catch {
			return undefined;
		}
	};
	const nowSeconds = (): number => Math.floor(now() / 1000);

	const isBearerAuthorized = (header: string | null): boolean => {
		if (header === null) return false;
		const match = /^Bearer (\S+)$/.exec(header);
		if (match === null || match[1] === undefined) return false;
		const token = match[1];
		if (secretsEqual(token, ownerToken)) return true;
		const access = verify("ar-at", token, accessPayload);
		return access !== undefined && access.exp > nowSeconds() && access.res === resourceUrl;
	};

	const handler = createMcpHandler(() => createAutoRAGMcpServer(options.lite, { readOnly, tools: options.tools }), {
		legacy: "stateless",
		onerror: onError,
	});

	const challenge = (): Response => {
		const metadata = resourceMetadataUrl === undefined ? "" : `, resource_metadata="${resourceMetadataUrl}"`;
		return new Response(JSON.stringify({ error: "invalid_token" }), {
			status: 401,
			headers: {
				"content-type": "application/json",
				"www-authenticate": `Bearer realm="autorag"${metadata}`,
				"cache-control": "no-store",
			},
		});
	};

	const originAllowed = (origin: string | null, localOrigins: readonly string[]): boolean => {
		if (origin === null) return true;
		return origin === publicOrigin || localOrigins.includes(origin);
	};

	function validateAuthorizeParams(
		params: Record<string, string>,
	):
		| { ok: true; clientId: string; redirectUri: string; challenge: string; resource: string; clientName: string }
		| { ok: false } {
		const clientId = params.client_id ?? "";
		const redirectUri = params.redirect_uri ?? "";
		const codeChallenge = params.code_challenge ?? "";
		const client = verify("ar-c", clientId, clientPayload);
		if (client === undefined || !client.uris.includes(redirectUri)) return { ok: false };
		if (params.response_type !== "code") return { ok: false };
		if (params.code_challenge_method !== "S256" || !/^[A-Za-z0-9_-]{43,128}$/.test(codeChallenge))
			return { ok: false };
		const resource = params.resource === undefined ? resourceUrl : params.resource;
		if (resourceUrl === undefined || resource === undefined || normalizeResource(resource) !== resourceUrl) {
			return { ok: false };
		}
		return {
			ok: true,
			clientId,
			redirectUri,
			challenge: codeChallenge,
			resource: resourceUrl,
			clientName: client.name,
		};
	}

	function consentPage(
		params: Record<string, string>,
		valid: { redirectUri: string; clientName: string },
		error?: string,
	): Response {
		const hidden = [
			"response_type",
			"client_id",
			"redirect_uri",
			"code_challenge",
			"code_challenge_method",
			"state",
			"resource",
		]
			.filter((key) => params[key] !== undefined)
			.map((key) => `<input type="hidden" name="${key}" value="${escapeHtml(params[key] ?? "")}">`)
			.join("\n");
		const target = new URL(valid.redirectUri).host;
		return htmlResponse(
			error === undefined ? 200 : 401,
			`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Authorize AutoRAG</title>
<style>body{font:16px system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem}input{font:inherit;width:100%;padding:.5rem;margin:.5rem 0}button{font:inherit;padding:.5rem 1rem}.err{color:#b00020}</style>
<h1>Authorize AutoRAG</h1>
<p><b>${escapeHtml(valid.clientName || "An agent")}</b> (returning to <code>${escapeHtml(target)}</code>) wants to search the files and archives this AutoRAG server indexes.</p>
<p>Only continue if you started this connection. Enter the owner token you set when you started the server.</p>
${error === undefined ? "" : `<p class="err">${escapeHtml(error)}</p>`}
<form method="post" action="/authorize">
${hidden}
<input type="password" name="owner_token" autocomplete="off" autofocus required placeholder="owner token">
<button type="submit">Authorize</button>
</form>`,
		);
	}

	async function handleOAuth(req: Request, url: URL, body: Buffer | undefined): Promise<Response | undefined> {
		if (resourceUrl === undefined || publicOrigin === undefined) return undefined;
		if (req.method === "GET" && url.pathname === `/.well-known/oauth-protected-resource${path}`) {
			return jsonResponse(200, {
				resource: resourceUrl,
				authorization_servers: [publicOrigin],
				bearer_methods_supported: ["header"],
				resource_name: "AutoRAG",
			});
		}
		if (req.method === "GET" && url.pathname === "/.well-known/oauth-authorization-server") {
			return jsonResponse(200, {
				issuer: publicOrigin,
				authorization_endpoint: `${publicOrigin}/authorize`,
				token_endpoint: `${publicOrigin}/token`,
				registration_endpoint: `${publicOrigin}/register`,
				response_types_supported: ["code"],
				grant_types_supported: ["authorization_code", "refresh_token"],
				code_challenge_methods_supported: ["S256"],
				token_endpoint_auth_methods_supported: ["none"],
				authorization_response_iss_parameter_supported: true,
			});
		}
		if (req.method === "POST" && url.pathname === "/register") {
			let parsed: z.infer<typeof registrationBody>;
			try {
				parsed = registrationBody.parse(JSON.parse(body?.toString("utf8") ?? ""));
			} catch {
				return oauthError(400, "invalid_client_metadata");
			}
			if (parsed.redirect_uris.length === 0 || !parsed.redirect_uris.every(isAllowedRedirect)) {
				return oauthError(400, "invalid_redirect_uri", "Redirect URIs must be https (or loopback http).");
			}
			const name = parsed.client_name ?? "";
			return jsonResponse(201, {
				client_id: sign("ar-c", { k: "client", name, uris: parsed.redirect_uris }),
				client_name: name,
				redirect_uris: parsed.redirect_uris,
				token_endpoint_auth_method: "none",
				grant_types: ["authorization_code", "refresh_token"],
				response_types: ["code"],
			});
		}
		if (url.pathname === "/authorize" && req.method === "GET") {
			const params = Object.fromEntries(url.searchParams);
			const valid = validateAuthorizeParams(params);
			if (!valid.ok)
				return htmlResponse(
					400,
					"<!doctype html><title>Invalid request</title><p>Invalid authorization request.</p>",
				);
			return consentPage(params, valid);
		}
		if (url.pathname === "/authorize" && req.method === "POST") {
			const params = parseForm(body, req.headers.get("content-type"));
			const valid = validateAuthorizeParams(params);
			if (!valid.ok)
				return htmlResponse(
					400,
					"<!doctype html><title>Invalid request</title><p>Invalid authorization request.</p>",
				);
			const current = now();
			authorizeFailures = authorizeFailures.filter((at) => current - at < AUTHORIZE_FAILURE_WINDOW_MS);
			if (authorizeFailures.length >= AUTHORIZE_FAILURE_LIMIT) {
				return new Response("Too many failed attempts. Wait a minute and try again.", {
					status: 429,
					headers: { "retry-after": "60", "content-type": "text/plain" },
				});
			}
			if (!secretsEqual(params.owner_token ?? "", ownerToken)) {
				authorizeFailures.push(current);
				return consentPage(params, valid, "That is not the owner token.");
			}
			const code = sign("ar-code", {
				k: "code",
				cid: valid.clientId,
				ru: valid.redirectUri,
				cc: valid.challenge,
				res: valid.resource,
				n: randomBytes(16).toString("base64url"),
				exp: nowSeconds() + CODE_TTL_SECONDS,
			});
			const target = new URL(valid.redirectUri);
			target.searchParams.set("code", code);
			if (params.state !== undefined) target.searchParams.set("state", params.state);
			target.searchParams.set("iss", publicOrigin);
			return new Response(null, {
				status: 302,
				headers: { location: target.toString(), "cache-control": "no-store" },
			});
		}
		if (url.pathname === "/token" && req.method === "POST") {
			let params: Record<string, string>;
			try {
				params = parseForm(body, req.headers.get("content-type"));
			} catch {
				return oauthError(400, "invalid_request");
			}
			if (params.resource !== undefined && normalizeResource(params.resource) !== resourceUrl) {
				return oauthError(400, "invalid_target");
			}
			const issue = (clientId: string): Response =>
				jsonResponse(
					200,
					{
						token_type: "Bearer",
						access_token: sign("ar-at", {
							k: "access",
							cid: clientId,
							res: resourceUrl,
							exp: nowSeconds() + ACCESS_TTL_SECONDS,
						}),
						expires_in: ACCESS_TTL_SECONDS,
						refresh_token: sign("ar-rt", {
							k: "refresh",
							cid: clientId,
							res: resourceUrl,
							exp: nowSeconds() + REFRESH_TTL_SECONDS,
						}),
					},
					{ pragma: "no-cache" },
				);
			if (params.grant_type === "authorization_code") {
				const code = verify("ar-code", params.code ?? "", codePayload);
				const current = nowSeconds();
				for (const [nonce, exp] of usedCodes) if (exp <= current) usedCodes.delete(nonce);
				if (
					code === undefined ||
					code.exp <= current ||
					usedCodes.has(code.n) ||
					code.cid !== params.client_id ||
					code.ru !== params.redirect_uri ||
					!secretsEqual(
						createHash("sha256")
							.update(params.code_verifier ?? "")
							.digest("base64url"),
						code.cc,
					)
				) {
					return oauthError(400, "invalid_grant");
				}
				usedCodes.set(code.n, code.exp);
				return issue(code.cid);
			}
			if (params.grant_type === "refresh_token") {
				const refresh = verify("ar-rt", params.refresh_token ?? "", refreshPayload);
				if (
					refresh === undefined ||
					refresh.exp <= nowSeconds() ||
					refresh.cid !== params.client_id ||
					refresh.res !== resourceUrl
				) {
					return oauthError(400, "invalid_grant");
				}
				return issue(refresh.cid);
			}
			return oauthError(400, "unsupported_grant_type");
		}
		return undefined;
	}

	const server = createServer((nodeReq, nodeRes) => {
		void (async () => {
			try {
				const host = nodeReq.headers.host ?? "";
				const url = new URL(nodeReq.url ?? "/", `http://${host === "" ? "localhost" : host}`);
				if (url.pathname === "/healthz" && nodeReq.method === "GET") {
					await writeResponse(nodeRes, jsonResponse(200, { ok: true }));
					return;
				}
				const hostCheck = validateHostHeader(nodeReq.headers.host, allowedHostnames);
				if (!hostCheck.ok) {
					await writeResponse(nodeRes, jsonResponse(403, { error: "forbidden_host" }));
					return;
				}
				const port = (server.address() as AddressInfo).port;
				const localOrigins = [`http://127.0.0.1:${port}`, `http://localhost:${port}`, `http://[::1]:${port}`];
				const originHeader = nodeReq.headers.origin;
				const origin = Array.isArray(originHeader) ? (originHeader[0] ?? null) : (originHeader ?? null);
				if (!originAllowed(origin, localOrigins)) {
					await writeResponse(nodeRes, jsonResponse(403, { error: "forbidden_origin" }));
					return;
				}
				const isMcp = url.pathname === path;
				if (isMcp && !isBearerAuthorized(nodeReq.headers.authorization ?? null)) {
					await writeResponse(nodeRes, challenge());
					return;
				}
				let body: Buffer | undefined;
				try {
					body = await readBody(nodeReq, isMcp ? maxBodyBytes : Math.min(maxBodyBytes, 64 * 1024));
				} catch (error) {
					if (error instanceof ReadError) {
						await writeResponse(nodeRes, jsonResponse(error.status, { error: "payload_too_large" }));
						return;
					}
					throw error;
				}
				const headers = new Headers();
				for (const [name, value] of Object.entries(nodeReq.headers)) {
					if (value === undefined) continue;
					headers.set(name, Array.isArray(value) ? value.join(", ") : value);
				}
				const request = new Request(url, {
					method: nodeReq.method,
					headers,
					...(body === undefined ? {} : { body: new Uint8Array(body) }),
				});
				if (isMcp) {
					await writeResponse(nodeRes, await handler.fetch(request));
					return;
				}
				const oauth = await handleOAuth(request, url, body);
				await writeResponse(nodeRes, oauth ?? jsonResponse(404, { error: "not_found" }));
			} catch (error) {
				onError(error instanceof Error ? error : new Error(String(error)));
				if (!nodeRes.headersSent) await writeResponse(nodeRes, jsonResponse(500, { error: "internal_error" }));
				else nodeRes.end();
			}
		})();
	});

	await new Promise<void>((resolve, reject) => {
		server.once("error", reject);
		server.listen(options.port ?? 0, bindHost, () => {
			server.off("error", reject);
			resolve();
		});
	});
	const address = server.address() as AddressInfo;
	const hostForUrl = bindHost.includes(":") && !bindHost.startsWith("[") ? `[${bindHost}]` : bindHost;

	return {
		url: `http://${hostForUrl}:${address.port}${path}`,
		...(resourceUrl === undefined ? {} : { publicMcpUrl: resourceUrl }),
		close: async () => {
			await handler.close();
			server.closeAllConnections();
			await new Promise<void>((resolve) => server.close(() => resolve()));
		},
	};
}
