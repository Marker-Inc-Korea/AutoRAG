import { createServer } from "node:http";
import { AutoragOAuthError } from "./errors.ts";

/** Path the browser is redirected to on the loopback listener. */
export const CALLBACK_PATH = "/callback";

/** Loopback listeners are ephemeral; abandon a login nobody completes. */
export const DEFAULT_LOGIN_TIMEOUT_MS = 5 * 60 * 1000;

export interface CallbackServer {
	/** `http://127.0.0.1:<ephemeral port>/callback`. */
	redirectUri: string;
	/** Resolves with the first callback's query parameters. */
	waitForCallback(): Promise<URLSearchParams>;
	/** Stop listening and abandon any pending wait. */
	close(): void;
}

export interface CallbackServerOptions {
	/** Loopback interface to bind. Defaults to `127.0.0.1`. */
	host?: string;
	/**
	 * Expected OAuth `state`. A browser callback is only accepted when it
	 * echoes this value, so a stray or forged request cannot win the race.
	 */
	state: string;
	timeoutMs?: number;
	signal?: AbortSignal;
	onListening?: (redirectUri: string) => void;
}

interface Deferred<T> {
	promise: Promise<T>;
	resolve: (value: T) => void;
	reject: (error: Error) => void;
}

function createDeferred<T>(): Deferred<T> {
	let resolve!: (value: T) => void;
	let reject!: (error: Error) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

function renderCallbackPage(failed: boolean): string {
	const message = failed
		? "AutoRAG sign-in did not complete. You can close this tab and try again."
		: "Signed in to AutoRAG. You can close this tab.";
	return [
		"<!doctype html>",
		'<html lang="en">',
		"<head>",
		'<meta charset="utf-8">',
		'<meta name="viewport" content="width=device-width, initial-scale=1">',
		"<title>AutoRAG</title>",
		"</head>",
		'<body style="font-family: system-ui, sans-serif; margin: 4rem auto; max-width: 32rem; text-align: center;">',
		`<p>${message}</p>`,
		"</body>",
		"</html>",
	].join("\n");
}

/**
 * Start a loopback HTTP listener for the OAuth redirect.
 *
 * The server closes itself as soon as the first valid callback — a request
 * that echoes {@link CallbackServerOptions.state} and carries a `code` or
 * `error` — arrives; malformed or forged requests get a `400` and are ignored.
 * It also closes on abort and on timeout. Requests to other paths are ignored.
 */
export function startCallbackServer(options: CallbackServerOptions): Promise<CallbackServer> {
	const host = options.host ?? "127.0.0.1";
	const timeoutMs = options.timeoutMs ?? DEFAULT_LOGIN_TIMEOUT_MS;

	const server = createServer();
	const started = createDeferred<CallbackServer>();
	const callback = createDeferred<URLSearchParams>();
	// The wait may be attached after an early abort/timeout; keep that rejection handled.
	callback.promise.catch(() => {});

	let finished = false;
	let startSettled = false;
	let timer: NodeJS.Timeout | undefined;

	function cleanup(): void {
		clearTimeout(timer);
		options.signal?.removeEventListener("abort", onAbort);
		if (server.listening) {
			server.closeAllConnections();
			server.close();
		}
	}
	function fail(error: Error): void {
		if (finished) return;
		finished = true;
		cleanup();
		callback.reject(error);
	}
	function onAbort(): void {
		fail(new AutoragOAuthError("AutoRAG sign-in was cancelled."));
	}

	server.on("request", (req, res) => {
		const url = new URL(req.url ?? "/", `http://${host}`);
		if (req.method !== "GET" || url.pathname !== CALLBACK_PATH) {
			res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
			res.end("Not found");
			return;
		}
		if (finished) {
			res.writeHead(410, { "content-type": "text/plain; charset=utf-8" });
			res.end("Callback already handled");
			return;
		}
		// Only a well-formed authorization response that echoes the state we
		// started with may consume the pending callback: a stray request, a
		// forged response, or a missing/empty `code`/`error` must leave the
		// listener open for the genuine redirect.
		const params = url.searchParams;
		const code = params.get("code");
		const oauthError = params.get("error");
		const hasResponse = (code !== null && code !== "") || (oauthError !== null && oauthError !== "");
		if (params.get("state") !== options.state || !hasResponse) {
			res.writeHead(400, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
			res.end(renderCallbackPage(true));
			return;
		}
		finished = true;
		res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
		res.end(renderCallbackPage(params.has("error")));
		cleanup();
		callback.resolve(params);
	});

	server.on("error", (error) => {
		const wrapped = error instanceof Error ? error : new Error(String(error));
		if (!startSettled) {
			startSettled = true;
			cleanup();
			started.reject(wrapped);
			return;
		}
		fail(wrapped);
	});

	server.listen(0, host, () => {
		const address = server.address();
		const port = typeof address === "object" && address !== null ? address.port : 0;
		const redirectUri = `http://${host}:${port}${CALLBACK_PATH}`;
		timer = setTimeout(
			() => fail(new AutoragOAuthError("AutoRAG sign-in timed out waiting for the browser callback.")),
			timeoutMs,
		);
		if (options.signal?.aborted) onAbort();
		else options.signal?.addEventListener("abort", onAbort, { once: true });
		startSettled = true;
		options.onListening?.(redirectUri);
		started.resolve({
			redirectUri,
			waitForCallback: () => callback.promise,
			close: () => {
				if (!finished) fail(new AutoragOAuthError("AutoRAG sign-in was cancelled."));
			},
		});
	});

	return started.promise;
}
