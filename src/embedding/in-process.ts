import type { Embedder } from "./embedder.ts";

/** MinSync's Node-API addon: the same engine, weights and cache as `minsync`'s `native:` embedder. */
export const NATIVE_ADDON_PACKAGE = "@nomadamas/minsync-native";
export const DEFAULT_NATIVE_MODEL_ID = "native:Qwen/Qwen3-Embedding-0.6B";
/** Release the ~2.8 GB of weights after this much idle time in a long-lived host such as the TUI. */
const IDLE_UNLOAD_SECONDS = 300;

export interface InProcessEmbedder extends Embedder {
	/** Releases the model; later calls reject. */
	close(): Promise<void>;
}

interface NativeAddon {
	NativeEmbedder: new (
		options?: Record<string, unknown>,
	) => {
		identity(): Promise<{ id: string; dimension: number }>;
		embed(texts: string[], options?: { signal?: AbortSignal }): Promise<Float32Array[]>;
		dispose(): Promise<void>;
	};
}

export class NativeAddonMissingError extends Error {
	constructor(cause: unknown) {
		super(
			`MinSync's in-process embedder (${NATIVE_ADDON_PACKAGE}) is not installed, so no embedding backend is available without a running \`minsync serve\`: ${
				cause instanceof Error ? cause.message : String(cause)
			}`,
			{ cause },
		);
		this.name = "NativeAddonMissingError";
	}
}

/**
 * Loads MinSync's in-process embedder. AutoRAG ships no model and no inference runtime; the addon is an
 * optional peer that brings MinSync's engine into this process, and MinSync provisions the weights into its
 * own cache on first use.
 */
export async function loadNativeEmbedder(options: { readonly id?: string } = {}): Promise<InProcessEmbedder> {
	let addon: NativeAddon;
	// A plain string keeps the specifier opaque to the type checker and the bundler: the addon is optional.
	const specifier: string = NATIVE_ADDON_PACKAGE;
	try {
		addon = (await import(/* @vite-ignore */ specifier)) as NativeAddon;
	} catch (error) {
		throw new NativeAddonMissingError(error);
	}
	const engine = new addon.NativeEmbedder({
		id: options.id ?? DEFAULT_NATIVE_MODEL_ID,
		idleUnloadSeconds: IDLE_UNLOAD_SECONDS,
	});
	return {
		async identity() {
			const info = await engine.identity();
			return { provider: "minsync", model: info.id.replace(/^native:/, ""), dimension: info.dimension };
		},
		async embed(texts) {
			if (texts.length === 0) return [];
			const vectors = await engine.embed([...texts]);
			return vectors.map((vector) => Array.from(vector));
		},
		close: () => engine.dispose(),
	};
}
