import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { join, resolve } from "node:path";
import type { Model } from "@earendil-works/pi-ai";

const [agentDirArgument, cwdArgument, workerId, workerCountArgument] = process.argv.slice(2);
if (!agentDirArgument || !cwdArgument || !workerId || !workerCountArgument || process.send === undefined) {
	throw new Error("runtime models process fixture requires agentDir, cwd, worker id, worker count, and IPC");
}

const agentDir = resolve(agentDirArgument);
const cwd = resolve(cwdArgument);
const workerCount = Number(workerCountArgument);
if (!Number.isInteger(workerCount) || workerCount < 2) {
	throw new Error("runtime models process fixture requires at least two workers");
}

const modelsPath = join(agentDir, "models.json");
const lockPath = `${modelsPath}.lock`;
const barrierDir = join(agentDir, ".models-registry-race-barrier");
const barrierReleasePath = join(barrierDir, "release");
const barrierWaitBuffer = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
const originalReadFileSync = fs.readFileSync;
let observedFirstModelsRead = false;

function hasErrorCode(error: unknown, code: string): boolean {
	return error instanceof Error && "code" in error && error.code === code;
}

function releaseReadBarrierWhenAllWorkersArrive(): void {
	fs.mkdirSync(barrierDir, { recursive: true });
	fs.writeFileSync(join(barrierDir, `${workerId}.ready`), "", { flag: "wx" });
	const deadline = Date.now() + 10_000;
	while (!fs.existsSync(barrierReleasePath)) {
		const readyWorkers = fs.readdirSync(barrierDir).filter((name) => name.endsWith(".ready"));
		if (readyWorkers.length >= workerCount) {
			try {
				fs.writeFileSync(barrierReleasePath, "", { flag: "wx" });
			} catch (error) {
				if (!hasErrorCode(error, "EEXIST")) throw error;
			}
			continue;
		}
		if (Date.now() >= deadline) {
			throw new Error(`Timed out synchronizing models.json reads for worker ${workerId}`);
		}
		Atomics.wait(barrierWaitBuffer, 0, 0, 10);
	}
}

fs.readFileSync = ((...args: Parameters<typeof fs.readFileSync>) => {
	const contents = originalReadFileSync(...args);
	const requestedPath = resolve(String(args[0]));
	if (!observedFirstModelsRead && requestedPath === modelsPath) {
		observedFirstModelsRead = true;
		if (!fs.existsSync(lockPath)) releaseReadBarrierWhenAllWorkersArrive();
	}
	return contents;
}) as typeof fs.readFileSync;
syncBuiltinESMExports();

const provider = `race-provider-${workerId}`;
const credential = `AUTORAG_RUNTIME_MODELS_PROCESS_SECRET_${workerId}`;
const orchestratorModel: Model<"openai-responses"> = {
	id: `race-orchestrator-${workerId}`,
	name: `Race Orchestrator ${workerId}`,
	api: "openai-responses",
	provider,
	baseUrl: `https://${workerId}.example.invalid/v1`,
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400_000,
	maxTokens: 128_000,
};
const explorerModel: Model<"openai-responses"> = {
	...orchestratorModel,
	id: `race-explorer-${workerId}`,
	name: `Race Explorer ${workerId}`,
};

let started = false;
process.on("message", async (message: unknown) => {
	if (message !== "start" || started) return;
	started = true;
	let runtime: Awaited<ReturnType<typeof import("../../src/subagents/runtime.ts")["createMandatorySubagentSession"]>>;
	try {
		const { createMandatorySubagentSession } = await import("../../src/subagents/runtime.ts");
		runtime = await createMandatorySubagentSession({
			cwd,
			agentDir,
			sessionDir: join(agentDir, "sessions", workerId),
			model: orchestratorModel,
			explorerModel,
			apiKey: credential,
			systemPrompt: `process fixture ${workerId}`,
			tools: [],
		});
		try {
			const modelRuntime = runtime.session.modelRuntime;
			const modelsJson = String(originalReadFileSync(modelsPath, "utf8"));
			const resolvedCredential = (await modelRuntime.getAuth(provider))?.auth.apiKey;
			process.send?.(
				{
					type: "complete",
					workerId,
					provider,
					orchestratorId: orchestratorModel.id,
					explorerId: explorerModel.id,
					resolvedOrchestrator: modelRuntime.getModel(provider, orchestratorModel.id) !== undefined,
					resolvedExplorer: modelRuntime.getModel(provider, explorerModel.id) !== undefined,
					resolvedCredential: resolvedCredential === credential,
					credentialPersisted: modelsJson.includes(credential),
				},
				() => process.exit(0),
			);
		} finally {
			runtime.session.dispose();
		}
	} catch (error) {
		process.send?.(
			{
				type: "failed",
				workerId,
				message: error instanceof Error ? (error.stack ?? error.message) : String(error),
			},
			() => process.exit(1),
		);
	}
});

process.send({ type: "ready", workerId });
