/**
 * Route-based mock `fetch` for connector tests. Routes are matched by
 * substring against `${method} ${url}`; the first match wins. Unmatched
 * requests return 404. Also records every requested URL for assertions.
 */

export interface MockRoute {
	/** Substring matched against `${METHOD} ${url}`. */
	readonly match: string;
	readonly status?: number;
	readonly json?: unknown;
	readonly text?: string;
	/** When set, the route throws (simulates network failure). */
	readonly networkError?: boolean;
}

export interface MockFetch {
	readonly fetchImpl: typeof fetch;
	readonly requests: readonly string[];
}

export function createMockFetch(routes: readonly MockRoute[]): MockFetch {
	const requests: string[] = [];
	const fetchImpl = (async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
		const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
		const method = (init?.method ?? "GET").toUpperCase();
		const key = `${method} ${url}`;
		requests.push(key);
		for (const route of routes) {
			if (!key.includes(route.match)) continue;
			if (route.networkError === true) throw new TypeError("fetch failed");
			const body = route.json !== undefined ? JSON.stringify(route.json) : (route.text ?? "");
			return new Response(body, { status: route.status ?? 200 });
		}
		return new Response("not found", { status: 404 });
	}) as typeof fetch;
	return { fetchImpl, requests };
}
