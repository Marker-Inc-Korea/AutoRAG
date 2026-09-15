import { describe, expect, it } from "vitest";
import { platformRuntimeAsset, resolveProfile, selectPlatformAsset } from "../../src/embedding-runtime/manifest.ts";
import type { PlatformId } from "../../src/embedding-runtime/types.ts";

describe("embedding runtime manifest", () => {
	it("resolves both pinned profiles", () => {
		expect(resolveProfile("qwen3-embedding-0.6b")).toMatchObject({
			profileId: "qwen3-embedding-0.6b",
			provider: "qwen",
			dimension: 1024,
		});
		expect(resolveProfile("embeddinggemma-300m")).toMatchObject({
			profileId: "embeddinggemma-300m",
			provider: "google",
			dimension: 768,
			queryPrefix: "task: search result | query: ",
		});
		expect(() => resolveProfile("missing" as never)).toThrow(/unknown profile/i);
	});

	it.each<[PlatformId, string]>([
		["darwin-arm64-metal", "macos-arm64"],
		["win-x64-cpu", "win-cpu-x64"],
		["win-x64-vulkan", "win-vulkan-x64"],
	])("selects %s runtime asset", (platform, marker) => {
		expect(platformRuntimeAsset(platform).url).toContain(marker);
		expect(platformRuntimeAsset(platform).sha256).toMatch(/^[a-f0-9]{64}$/);
	});

	it("resolves auto backend to Metal on macOS and probes Vulkan before CPU on Windows", () => {
		expect(selectPlatformAsset("darwin-arm64-metal", "auto").platform).toBe("darwin-arm64-metal");
		expect(selectPlatformAsset("win-x64-cpu", "auto", true).platform).toBe("win-x64-vulkan");
		expect(selectPlatformAsset("win-x64-cpu", "auto", false).platform).toBe("win-x64-cpu");
		expect(selectPlatformAsset("win-x64-cpu", "vulkan").platform).toBe("win-x64-vulkan");
	});
});
