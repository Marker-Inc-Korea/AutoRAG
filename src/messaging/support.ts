import { MessagingConfigError, MessagingError } from "./types.ts";

/** Upper-case only: a pasted key such as `am_us_AbC123` is mixed case and must not pass for a variable name. */
const ENV_NAME = /^[A-Z_][A-Z0-9_]*$/;
const REDACTED = "[redacted]";

/**
 * Config stores the name of the variable that holds a secret, never the secret. A value that is not an
 * upper-case variable name is almost certainly the secret itself pasted into the wrong field, so it is refused
 * without being echoed.
 */
export function assertEnvName(value: unknown, path: string): string {
	if (typeof value !== "string" || !ENV_NAME.test(value)) {
		throw new MessagingConfigError(
			`${path} must be an upper-case environment variable name matching ${ENV_NAME}; secrets never go in the config`,
		);
	}
	return value;
}

export function requireEnv(env: Readonly<Record<string, string | undefined>>, name: string, what: string): string {
	const value = env[name]?.trim();
	if (value === undefined || value === "") {
		throw new MessagingError(`${what} is missing: set the environment variable ${name}.`);
	}
	return value;
}

export function scrubSecrets(text: string, secrets: readonly string[]): string {
	let out = text;
	for (const secret of secrets) {
		if (secret !== "") out = out.split(secret).join(REDACTED);
	}
	return out;
}

/**
 * Cuts `text` into parts of at most `limit` UTF-16 units, breaking at a newline, else a space, else mid-word,
 * and never inside a surrogate pair. The whitespace a part breaks on is dropped.
 */
export function splitText(text: string, limit: number): string[] {
	if (limit < 2) throw new RangeError("splitText limit must be at least 2");
	if (text.length <= limit) return [text];
	const parts: string[] = [];
	let rest = text;
	while (rest.length > limit) {
		let cut = Math.max(rest.lastIndexOf("\n", limit), 0);
		if (cut < limit / 2) cut = Math.max(rest.lastIndexOf(" ", limit), cut);
		let drop = 1;
		if (cut < limit / 2) {
			cut = limit;
			drop = 0;
			const high = rest.charCodeAt(cut - 1);
			if (high >= 0xd800 && high <= 0xdbff) cut -= 1;
		}
		parts.push(rest.slice(0, cut));
		rest = rest.slice(cut + drop);
	}
	if (rest !== "") parts.push(rest);
	return parts;
}

export interface PostJsonRequest {
	/** Provider name for error messages. */
	readonly label: string;
	readonly url: string;
	readonly headers?: Readonly<Record<string, string>>;
	readonly body: unknown;
	/** Values to scrub from any error text (tokens, webhook URLs). */
	readonly secrets: readonly string[];
	readonly signal?: AbortSignal;
	readonly timeoutMs?: number;
}

export const DEFAULT_SEND_TIMEOUT_MS = 20_000;
const ERROR_BODY_CHARS = 300;

/** POSTs JSON and returns the parsed reply (or the text when it is not JSON). Every failure is a scrubbed {@link MessagingError}. */
export async function postJson(fetchImpl: typeof fetch, request: PostJsonRequest): Promise<unknown> {
	const { label, secrets } = request;
	const timeoutMs = request.timeoutMs ?? DEFAULT_SEND_TIMEOUT_MS;
	const timeout = AbortSignal.timeout(timeoutMs);
	const signal = request.signal === undefined ? timeout : AbortSignal.any([request.signal, timeout]);
	let response: Response;
	try {
		response = await fetchImpl(request.url, {
			method: "POST",
			headers: { "content-type": "application/json", ...request.headers },
			body: JSON.stringify(request.body),
			signal,
		});
	} catch (error) {
		if (request.signal?.aborted === true) throw error;
		if (timeout.aborted) throw new MessagingError(`${label} request timed out after ${timeoutMs} ms.`);
		const reason = error instanceof Error ? error.message : String(error);
		throw new MessagingError(`${label} request failed: ${scrubSecrets(reason, secrets)}`);
	}
	const raw = await response.text();
	if (!response.ok) {
		throw new MessagingError(
			`${label} rejected the message (HTTP ${response.status}): ${scrubSecrets(raw.slice(0, ERROR_BODY_CHARS), secrets)}`,
		);
	}
	try {
		return JSON.parse(raw);
	} catch {
		return raw;
	}
}
