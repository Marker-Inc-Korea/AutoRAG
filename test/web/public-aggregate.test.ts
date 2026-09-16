import { afterEach, describe, expect, it, vi } from "vitest";
import {
	clearRegisteredSearchProviders,
	registerSearchProvider,
	setExcludedSearchProviders,
} from "../../src/web/search/provider.ts";
import { SearchProvider } from "../../src/web/search/providers/base.ts";
import { searchPublicWeb } from "../../src/web/search/providers/public.ts";
import type { SearchProviderId, SearchResponse, SearchSource } from "../../src/web/search/types.ts";
import { SearchProviderError } from "../../src/web/search/types.ts";

const ENGINE_IDS = ["startpage", "google", "duckduckgo", "ecosia", "mojeek"] as const;

class FakeProvider extends SearchProvider {
	readonly label: string;

	constructor(
		readonly id: SearchProviderId,
		private readonly run: (signal: AbortSignal | undefined) => Promise<SearchResponse>,
	) {
		super();
		this.label = id;
	}

	isAvailable(): boolean {
		return true;
	}

	search(params: { signal?: AbortSignal }): Promise<SearchResponse> {
		return this.run(params.signal);
	}
}

function response(provider: SearchProviderId, sources: SearchSource[]): SearchResponse {
	return { provider, sources };
}

function register(id: SearchProviderId, run: (signal: AbortSignal | undefined) => Promise<SearchResponse>): void {
	registerSearchProvider(new FakeProvider(id, run));
}

function registerEmptyExcept(overrides: Partial<Record<(typeof ENGINE_IDS)[number], SearchSource[]>>): void {
	for (const id of ENGINE_IDS) {
		register(id, () => Promise.resolve(response(id, overrides[id] ?? [])));
	}
}

afterEach(() => {
	vi.useRealTimers();
	setExcludedSearchProviders([]);
	clearRegisteredSearchProviders();
});

describe("Public Web aggregate", () => {
	it("deduplicates URL variants, ranks by consensus, and keeps the longest snippet", async () => {
		registerEmptyExcept({
			startpage: [
				{ title: "Shared startpage", url: "https://www.example.com/shared/", snippet: "short" },
				{ title: "Alpha", url: "https://alpha.example/one", snippet: "alpha" },
			],
			google: [
				{
					title: "Shared google",
					url: "https://example.com/shared",
					snippet: "a much longer consensus snippet",
				},
				{ title: "Gamma", url: "https://gamma.example/three", snippet: "gamma" },
			],
			duckduckgo: [{ title: "Gamma duplicate", url: "https://gamma.example/three/", snippet: "gamma longer" }],
		});

		const result = await searchPublicWeb({ query: "consensus" });

		expect(result.sources).toEqual([
			{
				title: "Shared startpage",
				url: "https://www.example.com/shared/",
				snippet: "a much longer consensus snippet",
			},
			{ title: "Gamma duplicate", url: "https://gamma.example/three/", snippet: "gamma longer" },
			{ title: "Alpha", url: "https://alpha.example/one", snippet: "alpha" },
		]);
	});

	it("returns at the soft deadline with delivered results and aborts stragglers", async () => {
		vi.useFakeTimers();
		let aborted = false;
		register("startpage", () =>
			Promise.resolve(response("startpage", [{ title: "Fast", url: "https://example.com/fast" }])),
		);
		for (const id of ENGINE_IDS.slice(1)) {
			register(
				id,
				(signal) =>
					new Promise((_resolve, reject) => {
						signal?.addEventListener(
							"abort",
							() => {
								aborted = true;
								reject(new Error("aborted"));
							},
							{ once: true },
						);
					}),
			);
		}

		const pending = searchPublicWeb({ query: "deadline" }, { softMs: 50, hardMs: 500 });
		await vi.advanceTimersByTimeAsync(50);
		const result = await pending;

		expect(result.sources).toEqual([{ title: "Fast", url: "https://example.com/fast" }]);
		expect(aborted).toBe(true);
	});

	it("waits past the soft deadline for the first success", async () => {
		vi.useFakeTimers();
		let resolveDelivered: (value: SearchResponse) => void = () => {};
		const delivered = new Promise<SearchResponse>((resolve) => {
			resolveDelivered = resolve;
		});
		register("startpage", () => delivered);
		for (const id of ENGINE_IDS.slice(1)) register(id, () => Promise.reject(new Error("blocked")));

		const pending = searchPublicWeb({ query: "slow first" }, { softMs: 10, hardMs: 100 });
		await vi.advanceTimersByTimeAsync(10);
		let settled = false;
		void pending.finally(() => {
			settled = true;
		});
		await Promise.resolve();
		expect(settled).toBe(false);

		resolveDelivered(response("startpage", [{ title: "Late", url: "https://example.com/late" }]));
		await expect(pending).resolves.toMatchObject({
			sources: [{ title: "Late", url: "https://example.com/late" }],
		});
	});

	it("returns empty at the hard deadline when pending engines have not succeeded", async () => {
		vi.useFakeTimers();
		for (const id of ENGINE_IDS) register(id, () => new Promise(() => undefined));

		const pending = searchPublicWeb({ query: "hard cap" }, { softMs: 10, hardMs: 40 });
		await vi.advanceTimersByTimeAsync(40);

		await expect(pending).resolves.toEqual({ provider: "public", sources: [] });
	});

	it("throws an aggregated provider-tagged error when every engine fails", async () => {
		for (const id of ENGINE_IDS) register(id, () => Promise.reject(new Error(`${id} blocked`)));

		await expect(searchPublicWeb({ query: "all fail" })).rejects.toSatisfy(
			(error: unknown) =>
				error instanceof SearchProviderError &&
				error.provider === "public" &&
				error.status === 503 &&
				ENGINE_IDS.every((id) => error.message.includes(`${id}:`)),
		);
	});

	it("respects exclusions and rejects when every public engine is excluded", async () => {
		setExcludedSearchProviders([...ENGINE_IDS]);
		await expect(searchPublicWeb({ query: "nothing" })).rejects.toMatchObject({ provider: "public", status: 400 });
	});
});
