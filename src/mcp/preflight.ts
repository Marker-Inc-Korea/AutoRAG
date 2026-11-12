import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
import { z } from "zod";

export type PreflightStatus = "pass" | "warn" | "fail" | "skip";

export interface PreflightStep {
	readonly name: string;
	readonly status: PreflightStatus;
	readonly detail: string;
}

export interface PreflightResult {
	/** False when any step failed. Warnings and skipped steps do not fail a run. */
	readonly ok: boolean;
	readonly steps: readonly PreflightStep[];
}

export interface PreflightOptions {
	/** The owner token. Without it the authenticated steps are skipped, so a URL can be checked before a secret is handy. */
	readonly token?: string;
	/** Test seam. */
	readonly fetch?: typeof fetch;
}

const WRITE_TOOLS = ["autorag.refresh", "autorag.report", "autorag.convert"];

const resourceMetadata = z.object({ resource: z.string(), authorization_servers: z.array(z.string()).min(1) });
const serverMetadata = z.object({
	issuer: z.string(),
	authorization_endpoint: z.string(),
	token_endpoint: z.string(),
	registration_endpoint: z.string().optional(),
	code_challenge_methods_supported: z.array(z.string()).optional(),
});
const registration = z.object({ client_id: z.string() });

function normalize(value: string): string {
	return value.replace(/\/+$/, "");
}

/**
 * Replays what a connector-only agent (ChatGPT/dots, Claude Duet, Meta Muse, Grok Bot) does before it ever calls a
 * tool: provoke the 401 challenge, follow it to the protected-resource metadata, read the authorization server's
 * metadata, register a client, open the consent page, and finally list tools with the owner token. It talks to the
 * URL exactly as a vendor cloud would, so it catches a tunnel pointing at the wrong port, a `--public-url` that does
 * not match the tunnel, and a proxy that strips headers, before the vendor's connect dialog does with a vaguer error.
 *
 * It never completes the consent form (that needs the owner token typed by a person) and never prints the token.
 */
export async function runPreflight(mcpUrl: string, options: PreflightOptions = {}): Promise<PreflightResult> {
	let endpoint: URL;
	try {
		endpoint = new URL(mcpUrl);
	} catch {
		throw new Error(`Not a valid URL: ${mcpUrl}`);
	}
	if (endpoint.protocol !== "https:" && endpoint.protocol !== "http:") {
		throw new Error(`The MCP URL must be http(s), got ${endpoint.protocol}`);
	}
	const doFetch = options.fetch ?? fetch;
	const steps: PreflightStep[] = [];
	const add = (name: string, status: PreflightStatus, detail: string): void => {
		steps.push({ name, status, detail });
	};
	const describeError = (error: unknown): string => (error instanceof Error ? error.message : String(error));

	let metadataUrl: string | undefined;
	try {
		const response = await doFetch(endpoint, {
			method: "POST",
			headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
			body: "{}",
		});
		const header = response.headers.get("www-authenticate") ?? "";
		metadataUrl = /resource_metadata="([^"]+)"/.exec(header)?.[1];
		if (response.status !== 401)
			add("challenge", "fail", `An unauthenticated request got HTTP ${response.status}, expected 401.`);
		else if (!/^Bearer/i.test(header))
			add("challenge", "fail", "The 401 has no `WWW-Authenticate: Bearer` challenge.");
		else if (metadataUrl === undefined)
			add(
				"challenge",
				"warn",
				"The 401 does not point at resource metadata; only clients that send a header can connect.",
			);
		else add("challenge", "pass", `401 with resource_metadata ${metadataUrl}`);
	} catch (error) {
		add("challenge", "fail", `Could not reach ${endpoint.origin}: ${describeError(error)}`);
		return { ok: false, steps };
	}

	let authServer: string | undefined;
	{
		const wanted = metadataUrl ?? `${endpoint.origin}/.well-known/oauth-protected-resource${endpoint.pathname}`;
		try {
			const response = await doFetch(wanted);
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			const body = resourceMetadata.parse(await response.json());
			if (normalize(body.resource) !== normalize(endpoint.toString())) {
				add(
					"resource-metadata",
					"fail",
					`Metadata says the resource is ${body.resource}, not ${endpoint.toString()}. Start the server with --public-url set to the URL agents use.`,
				);
			} else {
				authServer = body.authorization_servers[0];
				add("resource-metadata", "pass", `resource ${body.resource}, authorization server ${authServer}`);
			}
		} catch (error) {
			add(
				"resource-metadata",
				"fail",
				`No usable OAuth resource metadata at ${wanted} (${describeError(error)}). ChatGPT, Muse and Claude connectors need it; start the server with --public-url.`,
			);
		}
	}

	let registrationEndpoint: string | undefined;
	let authorizationEndpoint: string | undefined;
	if (authServer === undefined) {
		for (const name of ["authorization-server-metadata", "dynamic-registration", "consent-page"]) {
			add(name, "skip", "Needs the resource metadata step to pass.");
		}
	} else {
		try {
			const response = await doFetch(`${normalize(authServer)}/.well-known/oauth-authorization-server`);
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			const body = serverMetadata.parse(await response.json());
			const problems: string[] = [];
			if (normalize(body.issuer) !== normalize(authServer))
				problems.push(`issuer ${body.issuer} differs from ${authServer}`);
			if (!body.code_challenge_methods_supported?.includes("S256")) problems.push("S256 PKCE not advertised");
			if (body.registration_endpoint === undefined)
				problems.push("no registration endpoint (clients cannot self-register)");
			if (problems.length > 0) add("authorization-server-metadata", "fail", problems.join("; "));
			else {
				registrationEndpoint = body.registration_endpoint;
				authorizationEndpoint = body.authorization_endpoint;
				add("authorization-server-metadata", "pass", `issuer ${body.issuer}`);
			}
		} catch (error) {
			add(
				"authorization-server-metadata",
				"fail",
				`Authorization server metadata unavailable (${describeError(error)}).`,
			);
		}

		let clientId: string | undefined;
		if (registrationEndpoint === undefined)
			add("dynamic-registration", "skip", "Needs authorization server metadata.");
		else {
			const redirect = "https://preflight.invalid/callback";
			try {
				const response = await doFetch(registrationEndpoint, {
					method: "POST",
					headers: { "content-type": "application/json" },
					body: JSON.stringify({ client_name: "autorag-preflight", redirect_uris: [redirect] }),
				});
				if (response.status !== 201) throw new Error(`HTTP ${response.status}`);
				clientId = registration.parse(await response.json()).client_id;
				add("dynamic-registration", "pass", "A public client registered.");
			} catch (error) {
				add("dynamic-registration", "fail", `Client registration failed (${describeError(error)}).`);
			}
			if (clientId === undefined || authorizationEndpoint === undefined)
				add("consent-page", "skip", "Needs a registered client.");
			else {
				try {
					const query = new URLSearchParams({
						response_type: "code",
						client_id: clientId,
						redirect_uri: redirect,
						code_challenge: "A".repeat(43),
						code_challenge_method: "S256",
						state: "preflight",
						resource: endpoint.toString(),
					});
					const response = await doFetch(`${authorizationEndpoint}?${query}`, { redirect: "manual" });
					const type = response.headers.get("content-type") ?? "";
					if (response.status === 200 && type.includes("text/html"))
						add("consent-page", "pass", "The consent page renders; a person must type the owner token there.");
					else add("consent-page", "fail", `The authorize endpoint answered HTTP ${response.status} (${type}).`);
				} catch (error) {
					add("consent-page", "fail", `Authorize endpoint unreachable (${describeError(error)}).`);
				}
			}
		}
	}

	if (options.token === undefined || options.token === "") {
		add("tools", "skip", "No owner token given (set AUTORAG_MCP_TOKEN to check the authenticated path).");
	} else {
		const client = new Client({ name: "autorag-preflight", version: "1.0.0" });
		try {
			await client.connect(
				new StreamableHTTPClientTransport(endpoint, {
					requestInit: { headers: { authorization: `Bearer ${options.token}` } },
					fetch: doFetch,
				}),
			);
			const names = (await client.listTools()).tools.map((tool) => tool.name);
			const writers = names.filter((name) => WRITE_TOOLS.includes(name));
			if (names.length === 0) add("tools", "fail", "Connected, but the server offers no tools.");
			else if (writers.length > 0)
				add(
					"tools",
					"warn",
					`Write tools are exposed to remote agents: ${writers.join(", ")}. Drop --allow-write unless you mean it.`,
				);
			else add("tools", "pass", `${names.length} read-only tools: ${names.join(", ")}`);
		} catch (error) {
			// The SDK error can quote the request; never echo the credential.
			add(
				"tools",
				"fail",
				`The owner token did not open ${endpoint.pathname}: ${describeError(error).split(options.token).join("<token>")}`,
			);
		} finally {
			await client.close().catch(() => {});
		}
	}

	return { ok: steps.every((step) => step.status !== "fail"), steps };
}
