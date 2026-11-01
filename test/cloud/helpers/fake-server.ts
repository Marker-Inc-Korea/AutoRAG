import { createServer, type IncomingHttpHeaders } from "node:http";

export interface FakeRequest {
	method: string;
	url: string;
	headers: IncomingHttpHeaders;
	body: string;
	json: unknown;
}

export interface FakeResponse {
	status?: number;
	body?: unknown;
	rawBody?: string;
	headers?: Record<string, string>;
}

export interface FakeServer {
	/** `http://127.0.0.1:<port>` */
	url: string;
	requests: FakeRequest[];
	close(): Promise<void>;
}

/** Local HTTP server standing in for the hosted Dazzi Models API in unit tests. */
export async function startFakeServer(handler: (request: FakeRequest) => FakeResponse): Promise<FakeServer> {
	const requests: FakeRequest[] = [];
	const server = createServer((req, res) => {
		const chunks: Buffer[] = [];
		req.on("data", (chunk: Buffer) => chunks.push(chunk));
		req.on("end", () => {
			const body = Buffer.concat(chunks).toString("utf8");
			let json: unknown;
			try {
				json = body === "" ? undefined : JSON.parse(body);
			} catch {
				json = undefined;
			}
			const request: FakeRequest = {
				method: req.method ?? "GET",
				url: req.url ?? "/",
				headers: req.headers,
				body,
				json,
			};
			requests.push(request);
			const response = handler(request);
			res.writeHead(response.status ?? 200, {
				"content-type": "application/json",
				...(response.headers ?? {}),
			});
			res.end(response.rawBody ?? JSON.stringify(response.body ?? {}));
		});
	});

	let resolveListening!: () => void;
	const listening = new Promise<void>((resolve) => {
		resolveListening = resolve;
	});
	server.listen(0, "127.0.0.1", () => resolveListening());
	await listening;
	const address = server.address();
	const port = typeof address === "object" && address !== null ? address.port : 0;

	return {
		url: `http://127.0.0.1:${port}`,
		requests,
		close: () =>
			new Promise<void>((resolve, reject) => {
				server.closeAllConnections();
				server.close((error) => (error ? reject(error) : resolve()));
			}),
	};
}

/** The question keys Cloudflare Clef Omni accepts; any other key is an HTTP 422. */
const STRICT_QUESTION_KEY = /^[A-Za-z0-9_.-]{1,100}$/u;

/**
 * A decision provider as strict as Cloudflare Clef Omni: a question key outside
 * {@link STRICT_QUESTION_KEY} is rejected with Clef Omni's HTTP 422, and otherwise each noul
 * question gets the probability `probabilities` names for its instructions.
 */
export function strictKeyDecisionHandler(
	probabilities: Readonly<Record<string, number>>,
): (request: FakeRequest) => FakeResponse {
	return (request) => {
		const body = request.json as { questions?: Record<string, { instructions?: string }> } | undefined;
		const questions = Object.entries(body?.questions ?? {});
		const rejected = questions.find(([key]) => !STRICT_QUESTION_KEY.test(key));
		if (rejected !== undefined) {
			return {
				status: 422,
				body: {
					errors: [
						{
							message: `AiError: Request body failed validation: questions.${rejected[0]}.[key]: String should match pattern '^[A-Za-z0-9_.-]{1,100}$'`,
							code: 3030,
						},
					],
				},
			};
		}
		return {
			body: {
				model: "cloudflare/clef-omni-20261009",
				answers: Object.fromEntries(
					questions.map(([key, question]) => [
						key,
						{ type: "noul", noul: probabilities[question.instructions ?? ""] ?? 0.5 },
					]),
				),
			},
		};
	};
}

/** Retry a request until it fails (the listener closed) or the budget runs out. */
export async function expectConnectionRefused(url: string, attempts = 20): Promise<void> {
	let lastError: unknown;
	for (let attempt = 0; attempt < attempts; attempt += 1) {
		try {
			await fetch(url, { signal: AbortSignal.timeout(500) });
		} catch (error) {
			lastError = error;
			return;
		}
		await new Promise<void>((resolve) => setTimeout(resolve, 25));
	}
	throw new Error(`Expected ${url} to be unreachable, but it kept accepting requests: ${String(lastError)}`);
}
