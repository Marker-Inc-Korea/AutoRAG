import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		include: ["test/**/*.test.ts"],
		// Puts an inert `minsync` first on PATH: MinSync is required, and no test may depend on a real one.
		setupFiles: ["test/setup/fake-minsync-on-path.ts"],
		// Child-process and Pi-session tests exceed the 5s default on slow CI runners.
		testTimeout: 60_000,
		hookTimeout: 60_000,
		// Bun's Windows fs-event implementation can abort parallel Vitest forks.
		fileParallelism: process.platform !== "win32",
	},
});
