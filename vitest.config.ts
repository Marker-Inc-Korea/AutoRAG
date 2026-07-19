import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		include: ["test/**/*.test.ts"],
		// Child-process and Pi-session tests exceed the 5s default on slow CI runners.
		testTimeout: 60_000,
		hookTimeout: 60_000,
	},
});
