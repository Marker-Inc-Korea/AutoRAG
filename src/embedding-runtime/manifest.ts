import type { BackendKind, PlatformId, ProfileId, RuntimeAsset, RuntimeProfile } from "./types.ts";

const RUNTIME_BUILD = "b10951";
const protocolCapabilities = { healthz: true, embed: true, openaiCompatible: true, ollamaCompatible: true } as const;

const profiles: Readonly<Record<ProfileId, RuntimeProfile>> = {
	"qwen3-embedding-0.6b": {
		profileId: "qwen3-embedding-0.6b",
		provider: "qwen",
		model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
		dimension: 1024,
		queryPrefix: "",
		passagePrefix: "",
		runtimeBuild: RUNTIME_BUILD,
		modelRevision: "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
		artifactSha256: "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
		backend: "auto",
		protocolCapabilities,
	},
	"embeddinggemma-300m": {
		profileId: "embeddinggemma-300m",
		provider: "google",
		model: "embeddinggemma-300M-Q8_0.gguf",
		dimension: 768,
		queryPrefix: "task: search result | query: ",
		passagePrefix: "title: none | text: ",
		runtimeBuild: RUNTIME_BUILD,
		modelRevision: "0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
		artifactSha256: "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
		backend: "auto",
		protocolCapabilities,
	},
};

const runtimeAssets: Readonly<Record<PlatformId, RuntimeAsset>> = {
	"darwin-arm64-metal": {
		id: "llama-macos-arm64",
		kind: "runtime",
		platform: "darwin-arm64-metal",
		version: RUNTIME_BUILD,
		filename: "llama-b10951-bin-macos-arm64.tar.gz",
		url: "https://github.com/ggml-org/llama.cpp/releases/download/b10951/llama-b10951-bin-macos-arm64.tar.gz",
		sha256: "93d024186f1e6ff1d221f5e0b03567f74dc27a49f5bd42c83066d30af4fbbfec",
		archiveMembers: ["llama-b10951/llama-server"],
	},
	"win-x64-cpu": {
		id: "llama-win-cpu-x64",
		kind: "runtime",
		platform: "win-x64-cpu",
		version: RUNTIME_BUILD,
		filename: "llama-b10951-bin-win-cpu-x64.zip",
		url: "https://github.com/ggml-org/llama.cpp/releases/download/b10951/llama-b10951-bin-win-cpu-x64.zip",
		sha256: "ec79f36abd0545ebfad5f61a0c965605058022567659deb21bec711268c3421f",
		archiveMembers: ["llama-server.exe"],
	},
	"win-x64-vulkan": {
		id: "llama-win-vulkan-x64",
		kind: "runtime",
		platform: "win-x64-vulkan",
		version: RUNTIME_BUILD,
		filename: "llama-b10951-bin-win-vulkan-x64.zip",
		url: "https://github.com/ggml-org/llama.cpp/releases/download/b10951/llama-b10951-bin-win-vulkan-x64.zip",
		sha256: "1e36655f134ab7b94790e3a6311a833dd603c6535b45588ff5b12bdfa79da385",
		archiveMembers: ["llama-server.exe"],
	},
};

export function resolveProfile(profileId: ProfileId): RuntimeProfile {
	const profile = profiles[profileId];
	if (!profile) throw new Error(`unknown profile: ${String(profileId)}`);
	return profile;
}

export function platformRuntimeAsset(platform: PlatformId): RuntimeAsset {
	const asset = runtimeAssets[platform];
	if (!asset) throw new Error(`Unknown runtime platform: ${String(platform)}`);
	return asset;
}

export function selectPlatformAsset(platform: PlatformId, backend: BackendKind, vulkanAvailable = false): RuntimeAsset {
	if (backend === "vulkan" || (backend === "auto" && platform === "win-x64-cpu" && vulkanAvailable))
		return runtimeAssets["win-x64-vulkan"];
	return platformRuntimeAsset(platform);
}

export const MODEL_ASSETS = {
	qwen3: {
		url: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf",
		revision: "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
		sha256: "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
		license: "Apache-2.0",
		noticeReference: "Qwen3 model card and Apache-2.0 license",
		intendedUse:
			"Local loopback embedding for AutoRAG MinSync and native-datasource semantic retrieval. Not a generative model.",
		unsuitableUse:
			"Chat or generation, remote embedding of corpus text, shipping weights in the npm package, any GGUF other than the pinned SHA-256.",
		upstreamCardUrl: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
		modelCard: "docs/model-cards/qwen3-embedding-0.6b.md",
	},
	embeddinggemma: {
		url: "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf",
		revision: "0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
		sha256: "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
		license: "Gemma Terms of Use",
		noticeReference: "Google EmbeddingGemma model card and Gemma Terms of Use",
		intendedUse:
			"Local loopback embedding for AutoRAG MinSync and native-datasource semantic retrieval. Not a generative model.",
		unsuitableUse:
			"Chat or generation, remote embedding of corpus text, shipping weights in the npm package, any GGUF other than the pinned SHA-256, uses restricted by the Gemma Prohibited Use Policy.",
		upstreamCardUrl: "https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
		modelCard: "docs/model-cards/embeddinggemma-300m.md",
	},
} as const;

export const MODEL_ASSET_URLS = {
	qwen3: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf",
	embeddinggemma:
		"https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf",
} as const;
