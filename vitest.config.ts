import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		include: ["test/**/*.test.ts"],
		// Child-process and Pi-session tests exceed the 5s default on slow CI runners.
		testTimeout: 60_000,
		hookTimeout: 60_000,
		// Use the same worker-thread pool on every supported platform. Bun's
		// Windows fs-event implementation can abort fork workers, while these
		// process-heavy tests mutate process-global environment variables.
		pool: "threads",
		fileParallelism: false,
		maxWorkers: 1,
	},
});
