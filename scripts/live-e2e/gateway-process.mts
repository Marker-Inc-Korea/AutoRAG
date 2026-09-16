import { createEmbeddingRuntime } from "../../src/embedding-runtime/index.ts";

const profileId = "qwen3-embedding-0.6b" as const;
const runtime = createEmbeddingRuntime({ cacheRoot: process.env.AUTORAG_HOME });
const ensured = await runtime.ensureRuntime({ profileId, cachedOnly: true });
const health = await fetch(`${ensured.baseUrl}/healthz`);
if (!health.ok) throw new Error(`gateway health check failed: HTTP ${health.status}`);
const body = (await health.json()) as { profileId?: string; dimension?: number };
if (body.profileId !== profileId || body.dimension !== 1024) throw new Error("gateway profile identity mismatch");
process.stdout.write(`${ensured.baseUrl}\n`);

let stopping = false;
async function stop(): Promise<void> {
	if (stopping) return;
	stopping = true;
	await runtime.stopRuntime();
	process.exit(0);
}
process.once("SIGINT", () => void stop());
process.once("SIGTERM", () => void stop());
await new Promise<void>(() => undefined);
