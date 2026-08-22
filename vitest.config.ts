import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		include: ["test/**/*.test.ts"],
		// Child-process and Pi-session tests exceed the 5s default on slow CI runners.
		testTimeout: 60_000,
		hookTimeout: 60_000,
		// Bun's Windows fs-event implementation can abort fork workers while
		// Vitest runs files in parallel. Keep the complete suite, but serialize
		// files on Windows until the runtime assertion is resolved upstream.
		fileParallelism: process.platform !== "win32",
	},
});
