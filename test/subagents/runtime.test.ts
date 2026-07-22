import { type ChildProcess, fork, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	rmSync,
	statSync,
	utimesSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { Value } from "typebox/value";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveAutoRAGHome } from "../../src/config/home.ts";
import { acquireFileLock } from "../../src/filesystem/file-lock.ts";
import {
	AUTORAG_EXPLORER_AGENT_DEFINITION,
	createHealthSubagentProbeSession,
	createMandatorySubagentSession,
	EXPLORER_TOOLS_EXTENSION_PATH,
} from "../../src/subagents/runtime.ts";

const runtimeFsMock = vi.hoisted(() => ({
	realRenameSync: undefined as typeof import("node:fs").renameSync | undefined,
	realRmdirSync: undefined as typeof import("node:fs").rmdirSync | undefined,
	renameSyncHook: undefined as
		| ((...args: Parameters<typeof import("node:fs").renameSync>) => ReturnType<typeof import("node:fs").renameSync>)
		| undefined,
	rmdirSyncHook: undefined as
		| ((...args: Parameters<typeof import("node:fs").rmdirSync>) => ReturnType<typeof import("node:fs").rmdirSync>)
		| undefined,
}));

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
	runtimeFsMock.realRenameSync = actual.renameSync;
	runtimeFsMock.realRmdirSync = actual.rmdirSync;
	return {
		...actual,
		renameSync: (...args: Parameters<typeof actual.renameSync>) => {
			if (runtimeFsMock.renameSyncHook) return runtimeFsMock.renameSyncHook(...args);
			return actual.renameSync(...args);
		},
		rmdirSync: (...args: Parameters<typeof actual.rmdirSync>) => {
			if (runtimeFsMock.rmdirSyncHook) return runtimeFsMock.rmdirSyncHook(...args);
			return actual.rmdirSync(...args);
		},
	};
});

const RUNTIME_MODELS_PROCESS_FIXTURE = fileURLToPath(new URL("./runtime-models-process-fixture.ts", import.meta.url));
const RUNTIME_MODULE_URL = new URL("../../src/subagents/runtime.ts", import.meta.url).href;

const model: Model<"openai-responses"> = {
	id: "gpt-5.6-sol",
	name: "GPT-5.6 Sol",
	api: "openai-responses",
	provider: "test-proxy",
	baseUrl: "https://example.invalid/v1",
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400_000,
	maxTokens: 128_000,
};

const explorerModel: Model<"openai-responses"> = {
	...model,
	id: "gpt-5.6-luna",
	name: "GPT-5.6 Luna",
};

const INSTALLED_LEGACY_EXPLORER_DEFINITION = `---
name: autorag-explorer
description: Read-only, high-recall document explorer for AutoRAG evidence collection
tools: read, grep, find, ls
systemPromptMode: replace
inheritProjectContext: false
inheritSkills: false
---

You are the AutoRAG document explorer. Search and read broadly, but never make
the final relevance, sufficiency, conflict, freshness, or curation decision.

Your assignment includes the unchanged original query, a selected retrieval
method, multiple query variants, policy constraints, and possibly a seed pack
from a process-bound retrieval method. Use only the read-only tools provided.

The assigned task \`cwd\` is a hard read boundary. Read, grep, find, and ls only
within that directory and its descendants. Never discover from the workspace
root, a parent directory, a sibling search root, or an absolute path outside
the assigned cwd. If a requested path is outside the assigned cwd, reject it,
do not read it, and report the request as \`outside-assigned-cwd\` in your
evidence handoff.

The parent \`subagent\` invocation sets \`artifacts: false\` once at the top level.
Nested task items in single, tasks, chain, or parallel dispatches omit the
\`artifacts\` field.

Return candidate findings with source, method, query variant, relevance
(strong/moderate/weak), exact evidence and location context, retrievedAt,
source temporal metadata or explicit unknown status, temporal basis, and
uncertainty. Include weak candidates that could explain a conflict or gap.
`;

const PREVIOUS_MANAGED_EXPLORER_DEFINITION = `---
name: autorag-explorer
description: Read-only, high-recall document explorer for AutoRAG evidence collection
tools: read, grep, find, ls
thinking: high
systemPromptMode: replace
inheritProjectContext: false
inheritSkills: false
---

You are the AutoRAG document explorer. Search and read broadly, but never make
the final relevance, sufficiency, conflict, freshness, or curation decision.

Your assignment includes the unchanged original query, a selected retrieval
method, multiple query variants, policy constraints, and possibly a seed pack
from a process-bound retrieval method. Use only the read-only tools provided.

Return candidate findings with source, method, query variant, relevance
(strong/moderate/weak), exact evidence and location context, retrievedAt,
source temporal metadata or explicit unknown status, temporal basis, and
uncertainty. Include weak candidates that could explain a conflict or gap.
`;

const customTool: AgentTool = {
	name: "custom_search",
	label: "Custom search",
	description: "Test search tool",
	parameters: Type.Object({ query: Type.String() }),
	execute: async () => ({ content: [{ type: "text", text: "ok" }], details: {} }),
};

const tempDirs: string[] = [];

async function waitForPath(path: string, timeoutMs = 5_000): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (!existsSync(path)) {
		if (Date.now() >= deadline) throw new Error(`Timed out waiting for path: ${path}`);
		await new Promise((resolveWait) => setTimeout(resolveWait, 10));
	}
}

function expectedSessionNamespace(cwd: string): string {
	const canonicalCwd = resolve(cwd);
	const readablePath = canonicalCwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-");
	const cwdHash = createHash("sha256").update(canonicalCwd).digest("hex").slice(0, 12);
	return `--${readablePath}-${cwdHash}--`;
}

interface RuntimeModelsWorkerReady {
	readonly type: "ready";
	readonly workerId: string;
}

interface RuntimeModelsWorkerComplete {
	readonly type: "complete";
	readonly workerId: string;
	readonly provider: string;
	readonly orchestratorId: string;
	readonly explorerId: string;
	readonly resolvedOrchestrator: boolean;
	readonly resolvedExplorer: boolean;
	readonly resolvedCredential: boolean;
	readonly credentialPersisted: boolean;
}

interface RuntimeModelsWorkerFailed {
	readonly type: "failed";
	readonly workerId: string;
	readonly message: string;
}

type RuntimeModelsWorkerMessage = RuntimeModelsWorkerReady | RuntimeModelsWorkerComplete | RuntimeModelsWorkerFailed;

interface RuntimeModelsWorker {
	readonly child: ChildProcess;
	readonly ready: Promise<RuntimeModelsWorkerReady>;
	readonly completed: Promise<RuntimeModelsWorkerComplete>;
	readonly exited: Promise<void>;
}

function isRuntimeModelsWorkerMessage(value: unknown): value is RuntimeModelsWorkerMessage {
	if (typeof value !== "object" || value === null || !("type" in value) || !("workerId" in value)) return false;
	if (typeof value.workerId !== "string") return false;
	if (value.type === "ready") return true;
	if (value.type === "failed") return "message" in value && typeof value.message === "string";
	return (
		value.type === "complete" &&
		"provider" in value &&
		typeof value.provider === "string" &&
		"orchestratorId" in value &&
		typeof value.orchestratorId === "string" &&
		"explorerId" in value &&
		typeof value.explorerId === "string" &&
		"resolvedOrchestrator" in value &&
		typeof value.resolvedOrchestrator === "boolean" &&
		"resolvedExplorer" in value &&
		typeof value.resolvedExplorer === "boolean" &&
		"resolvedCredential" in value &&
		typeof value.resolvedCredential === "boolean" &&
		"credentialPersisted" in value &&
		typeof value.credentialPersisted === "boolean"
	);
}

function waitForRuntimeModelsWorkerMessage<TType extends "ready" | "complete">(
	child: ChildProcess,
	expectedType: TType,
	readStderr: () => string,
): Promise<Extract<RuntimeModelsWorkerMessage, { type: TType }>> {
	return new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			cleanup();
			reject(new Error(`Timed out waiting for runtime worker ${expectedType}: ${readStderr()}`));
		}, 15_000);
		const onMessage = (message: unknown): void => {
			if (!isRuntimeModelsWorkerMessage(message)) return;
			if (message.type === "failed") {
				cleanup();
				reject(new Error(`Runtime worker ${message.workerId} failed: ${message.message}`));
				return;
			}
			if (message.type !== expectedType) return;
			cleanup();
			resolve(message as Extract<RuntimeModelsWorkerMessage, { type: TType }>);
		};
		const onError = (error: Error): void => {
			cleanup();
			reject(error);
		};
		const onExit = (code: number | null, signal: NodeJS.Signals | null): void => {
			cleanup();
			reject(
				new Error(`Runtime worker exited before ${expectedType}: code=${code} signal=${signal} ${readStderr()}`),
			);
		};
		const cleanup = (): void => {
			clearTimeout(timeout);
			child.off("message", onMessage);
			child.off("error", onError);
			child.off("exit", onExit);
		};
		child.on("message", onMessage);
		child.once("error", onError);
		child.once("exit", onExit);
	});
}

function spawnRuntimeModelsWorker(
	agentDir: string,
	cwd: string,
	workerId: string,
	workerCount: number,
): RuntimeModelsWorker {
	const child = fork(RUNTIME_MODELS_PROCESS_FIXTURE, [agentDir, cwd, workerId, String(workerCount)], {
		execArgv: ["--experimental-strip-types", "--disable-warning=ExperimentalWarning"],
		silent: true,
	});
	let stderr = "";
	child.stderr?.on("data", (chunk: Buffer | string) => {
		stderr += String(chunk);
	});
	const readStderr = (): string => stderr;
	const exited = new Promise<void>((resolve, reject) => {
		child.once("error", reject);
		child.once("exit", (code, signal) => {
			if (code === 0) resolve();
			else reject(new Error(`Runtime worker failed: code=${code} signal=${signal} ${stderr}`));
		});
	});
	return {
		child,
		ready: waitForRuntimeModelsWorkerMessage(child, "ready", readStderr),
		completed: waitForRuntimeModelsWorkerMessage(child, "complete", readStderr),
		exited,
	};
}

async function stopRuntimeModelsWorker(worker: RuntimeModelsWorker): Promise<void> {
	if (worker.child.exitCode !== null || worker.child.signalCode !== null) return;
	await new Promise<void>((resolve) => {
		worker.child.once("exit", () => resolve());
		worker.child.kill();
	});
}

function registryModelDefinition(source: Model<"openai-responses">): Record<string, unknown> {
	return {
		id: source.id,
		name: source.name,
		api: source.api,
		baseUrl: source.baseUrl,
		reasoning: source.reasoning,
		input: source.input,
		cost: source.cost,
		contextWindow: source.contextWindow,
		maxTokens: source.maxTokens,
	};
}

afterEach(() => {
	runtimeFsMock.renameSyncHook = undefined;
	runtimeFsMock.rmdirSyncHook = undefined;
	vi.restoreAllMocks();
	for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("mandatory pi-subagents runtime", () => {
	it("preserves explicit session layout and custom tools", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-explicit-layout-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const sessionDir = join(root, "sessions");
		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			sessionDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(runtime.session.sessionFile?.startsWith(sessionDir)).toBe(true);
			expect(runtime.session.getActiveToolNames()).toContain("custom_search");
		} finally {
			runtime.session.dispose();
		}
	});

	it("uses a deterministic session namespace for repeated sessions from the same cwd", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-stable-session-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const sessionDirectories: string[] = [];

		for (const _attempt of [0, 1]) {
			const runtime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			try {
				const sessionFile = runtime.session.sessionFile;
				if (sessionFile === undefined) throw new Error("The runtime did not expose a session file");
				sessionDirectories.push(dirname(sessionFile));
			} finally {
				runtime.session.dispose();
			}
		}

		expect(sessionDirectories[1]).toBe(sessionDirectories[0]);
		expect(sessionDirectories[0]).toBe(join(agentDir, "sessions", expectedSessionNamespace(root)));
	});

	it("does not collide when cwd path components contain separators and hyphens", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-session-collision-"));
		tempDirs.push(root);
		const firstCwd = join(root, "a-b", "c");
		const secondCwd = join(root, "a", "b-c");
		mkdirSync(firstCwd, { recursive: true });
		mkdirSync(secondCwd, { recursive: true });
		const agentDir = join(root, "agent");
		const sessionDirectories: string[] = [];

		for (const cwd of [firstCwd, secondCwd]) {
			const runtime = await createMandatorySubagentSession({
				cwd,
				agentDir,
				model,
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			try {
				const sessionFile = runtime.session.sessionFile;
				if (sessionFile === undefined) throw new Error("The runtime did not expose a session file");
				sessionDirectories.push(dirname(sessionFile));
			} finally {
				runtime.session.dispose();
			}
		}

		expect(sessionDirectories[0]).toBe(join(agentDir, "sessions", expectedSessionNamespace(firstCwd)));
		expect(sessionDirectories[1]).toBe(join(agentDir, "sessions", expectedSessionNamespace(secondCwd)));
		expect(sessionDirectories[0]).not.toBe(sessionDirectories[1]);
	});

	it("creates an empty Pi settings object when the agent settings file is absent", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-settings-create-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const settingsPath = join(agentDir, "settings.json");
		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(existsSync(settingsPath)).toBe(true);
			expect(readFileSync(settingsPath, "utf8")).toBe("{}");
			expect(JSON.parse(readFileSync(settingsPath, "utf8"))).toEqual({});
		} finally {
			runtime.session.dispose();
		}
	});

	it("preserves an existing Pi settings file byte-for-byte", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-settings-preserve-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const settingsPath = join(agentDir, "settings.json");
		const original = '{\n  "quietStartup": true,\n  "retry": {\n    "enabled": false\n  }\n}\n';
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(settingsPath, original, "utf8");

		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(readFileSync(settingsPath, "utf8")).toBe(original);
		} finally {
			runtime.session.dispose();
		}
	});

	it("preserves existing provider and model metadata while merging role models in one process", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-model-registry-single-process-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const siblingModel = {
			...registryModelDefinition({
				...model,
				id: "existing-sibling-model",
				name: "Existing Sibling Model",
			}),
			thinkingLevelMap: { high: "xhigh" },
		};
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(
			modelsPath,
			JSON.stringify({
				providers: {
					"test-proxy": {
						name: "Pinned Existing Provider",
						baseUrl: "https://existing-provider.example.invalid/v1",
						api: "openai-responses",
						apiKey: "TEST_PROXY_API_KEY",
						authHeader: true,
						models: [
							{
								...registryModelDefinition(model),
								name: "Stale Orchestrator Name",
								thinkingLevelMap: { high: "xhigh" },
							},
							siblingModel,
						],
					},
				},
			}),
			"utf8",
		);

		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			explorerModel,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			const config = JSON.parse(readFileSync(modelsPath, "utf8")) as {
				providers: Record<
					string,
					{
						name?: string;
						baseUrl?: string;
						authHeader?: boolean;
						models?: Array<{ id: string; name?: string; thinkingLevelMap?: unknown }>;
					}
				>;
			};
			const provider = config.providers["test-proxy"];
			expect(provider?.name).toBe("Pinned Existing Provider");
			expect(provider?.baseUrl).toBe("https://existing-provider.example.invalid/v1");
			expect(provider?.authHeader).toBe(true);
			expect(provider?.models?.map((entry) => entry.id)).toEqual([
				model.id,
				"existing-sibling-model",
				explorerModel.id,
			]);
			expect(provider?.models?.find((entry) => entry.id === model.id)).toMatchObject({
				name: model.name,
				thinkingLevelMap: { high: "xhigh" },
			});
			expect(provider?.models?.find((entry) => entry.id === "existing-sibling-model")).toEqual(siblingModel);
		} finally {
			runtime.session.dispose();
		}
	});

	it("merges overlapping models.json writes from independent processes", { timeout: 20_000 }, async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-model-registry-process-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const barrierDir = join(agentDir, ".models-registry-race-barrier");
		mkdirSync(barrierDir, { recursive: true });
		writeFileSync(modelsPath, '{"providers":{}}\n', "utf8");
		const workerIds = ["alpha", "beta"];
		const workers = workerIds.map((workerId) => spawnRuntimeModelsWorker(agentDir, root, workerId, workerIds.length));

		try {
			await Promise.all(workers.map((worker) => worker.ready));
			for (const worker of workers) worker.child.send("start");
			const results = await Promise.all(workers.map((worker) => worker.completed));
			await Promise.all(workers.map((worker) => worker.exited));

			for (const result of results) {
				expect(result.resolvedOrchestrator).toBe(true);
				expect(result.resolvedExplorer).toBe(true);
				expect(result.resolvedCredential).toBe(true);
				expect(result.credentialPersisted).toBe(false);
			}

			const modelsJson = readFileSync(modelsPath, "utf8");
			const config = JSON.parse(modelsJson) as {
				providers: Record<string, { models?: Array<{ id: string }> }>;
			};
			for (const result of results) {
				expect(config.providers[result.provider]?.models?.map((entry) => entry.id)).toEqual([
					result.orchestratorId,
					result.explorerId,
				]);
				expect(modelsJson).not.toContain(`AUTORAG_RUNTIME_MODELS_PROCESS_SECRET_${result.workerId}`);
			}
		} finally {
			await Promise.all(workers.map(stopRuntimeModelsWorker));
			rmSync(barrierDir, { recursive: true, force: true });
		}

		expect(
			readdirSync(agentDir).filter(
				(name) =>
					name === "models.json.lock" ||
					(name.startsWith("models.json.") && (name.endsWith(".tmp") || name.endsWith(".stale"))),
			),
		).toEqual([]);
	});

	it("reclaims a stale models.json lock owned by a dead process", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-model-registry-stale-lock-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const lockPath = `${modelsPath}.lock`;
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(modelsPath, '{"providers":{}}\n', "utf8");
		writeFileSync(lockPath, `${JSON.stringify({ token: "abandoned", pid: 999_999, createdAt: 0 })}\n`, "utf8");
		const staleTime = new Date(Date.now() - 60_000);
		utimesSync(lockPath, staleTime, staleTime);

		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			explorerModel,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		runtime.session.dispose();

		expect(existsSync(lockPath)).toBe(false);
		expect(readdirSync(agentDir).filter((name) => name.endsWith(".tmp") || name.endsWith(".stale"))).toEqual([]);
		const config = JSON.parse(readFileSync(modelsPath, "utf8")) as {
			providers: Record<string, { models?: Array<{ id: string }> }>;
		};
		expect(config.providers["test-proxy"]?.models?.map((entry) => entry.id)).toEqual([model.id, explorerModel.id]);
	});

	it("preserves a fresh models.json owner update during stale-lock turnover", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-model-registry-turnover-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const lockPath = `${modelsPath}.lock`;
		const staleContents = `${JSON.stringify({ token: "stale-owner", pid: 999_999, createdAt: 0 })}\n`;
		const freshProvider = "turnover-provider";
		const freshModelId = "turnover-model";
		const freshRegistry = `${JSON.stringify({
			providers: {
				[freshProvider]: {
					baseUrl: "https://turnover.example.invalid/v1",
					api: "openai-responses",
					apiKey: "TURNOVER_PROVIDER_API_KEY",
					models: [{ id: freshModelId }],
				},
			},
		})}\n`;
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(modelsPath, '{"providers":{}}\n', "utf8");
		mkdirSync(lockPath, { mode: 0o700 });
		const staleMarkerPath = join(lockPath, "owner-stale-owner.json");
		writeFileSync(staleMarkerPath, staleContents, "utf8");
		const staleTime = new Date(Date.now() - 60_000);
		utimesSync(staleMarkerPath, staleTime, staleTime);
		let turnoverInjected = false;
		let freshOwnerAssertions = 0;
		let freshOwnerCommitted = false;
		let staleReaperBlocked = false;
		let competingOwnerRejected = false;
		runtimeFsMock.rmdirSyncHook = (...args) => {
			if (!turnoverInjected && String(args[0]) === lockPath) {
				turnoverInjected = true;
				if (!runtimeFsMock.realRmdirSync) throw new Error("real rmdirSync is unavailable");
				runtimeFsMock.realRmdirSync(...args);
				const freshOwner = acquireFileLock(lockPath, {
					timeoutMs: 1_000,
					staleMs: 30_000,
					retryMs: 1,
					timeoutError: () => new Error("fresh turnover owner could not acquire the models.json lock"),
				});
				let delayedReaperError: unknown;
				try {
					freshOwner.assertOwned();
					freshOwnerAssertions++;
					try {
						runtimeFsMock.realRmdirSync(lockPath);
					} catch (error) {
						if (
							!(
								error instanceof Error &&
								"code" in error &&
								(error.code === "ENOTEMPTY" || error.code === "EEXIST")
							)
						) {
							throw error;
						}
						staleReaperBlocked = true;
						delayedReaperError = error;
					}
					freshOwner.assertOwned();
					freshOwnerAssertions++;
					expect(() =>
						acquireFileLock(lockPath, {
							timeoutMs: 0,
							staleMs: 30_000,
							retryMs: 1,
							timeoutError: () => {
								competingOwnerRejected = true;
								return new Error("turnover competitor could not acquire the models.json lock");
							},
						}),
					).toThrow("turnover competitor could not acquire the models.json lock");
					writeFileSync(modelsPath, freshRegistry, "utf8");
					freshOwnerCommitted = true;
				} finally {
					freshOwner.release();
				}
				if (delayedReaperError !== undefined) throw delayedReaperError;
				throw new Error("delayed stale reaper unexpectedly removed the fresh models.json lock");
			}
			return runtimeFsMock.realRmdirSync?.(...args);
		};

		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			explorerModel,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		runtime.session.dispose();

		expect(turnoverInjected).toBe(true);
		expect(freshOwnerAssertions).toBe(2);
		expect(freshOwnerCommitted).toBe(true);
		expect(staleReaperBlocked).toBe(true);
		expect(competingOwnerRejected).toBe(true);
		const config = JSON.parse(readFileSync(modelsPath, "utf8")) as {
			providers: Record<string, { models?: Array<{ id: string }> }>;
		};
		expect(config.providers[freshProvider]?.models?.map((entry) => entry.id)).toEqual([freshModelId]);
		expect(config.providers["test-proxy"]?.models?.map((entry) => entry.id)).toEqual([model.id, explorerModel.id]);
		expect(
			readdirSync(agentDir).filter(
				(name) => name === "models.json.lock" || name.includes(".quarantine") || name.endsWith(".tmp"),
			),
		).toEqual([]);
	});

	it("times out on a live models.json lock without deleting the owner's lock", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-model-registry-live-lock-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const lockPath = `${modelsPath}.lock`;
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(modelsPath, '{"providers":{}}\n', "utf8");
		const realNow = Date.now();
		const lockContents = `${JSON.stringify({ token: "live-owner", pid: process.pid, createdAt: realNow })}\n`;
		writeFileSync(lockPath, lockContents, "utf8");
		let clock = realNow;
		vi.spyOn(Date, "now").mockImplementation(() => {
			clock += 20_000;
			return clock;
		});

		await expect(
			createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				explorerModel,
				systemPrompt: "test prompt",
				tools: [customTool],
			}),
		).rejects.toThrow("Timed out waiting for AutoRAG Pi models.json lock");
		expect(readFileSync(lockPath, "utf8")).toBe(lockContents);
		expect(readFileSync(modelsPath, "utf8")).toBe('{"providers":{}}\n');
		expect(readdirSync(agentDir).filter((name) => name.endsWith(".tmp") || name.endsWith(".stale"))).toEqual([]);
	});

	it("merges both role models without persisting the credential and exposes the same registry to Pi children", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-model-registry-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const sentinel = "AUTORAG_TEST_SENTINEL_SECRET";
		const previousApiKey = process.env.TEST_PROXY_API_KEY;
		const previousPiAgentDir = process.env.PI_CODING_AGENT_DIR;
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(
			modelsPath,
			JSON.stringify({
				providers: {
					"custom-role-provider": {
						baseUrl: "https://custom.example/v1",
						api: "openai-responses",
						apiKey: "CUSTOM_ROLE_PROVIDER_API_KEY",
						models: [{ id: "custom-role-model" }],
					},
				},
			}),
		);

		try {
			const runtimeOptions = Object.assign(
				{
					cwd: root,
					agentDir,
					model,
					apiKey: sentinel,
					systemPrompt: "test prompt",
					tools: [customTool],
				},
				{ explorerModel },
			);
			const runtime = await createMandatorySubagentSession(runtimeOptions);
			try {
				expect(existsSync(modelsPath)).toBe(true);
				const modelsJson = readFileSync(modelsPath, "utf8");
				const config = JSON.parse(modelsJson) as {
					providers: Record<string, { apiKey?: string; models?: Array<{ id: string }> }>;
				};
				expect(config.providers["test-proxy"]?.models?.map((entry) => entry.id)).toEqual(
					expect.arrayContaining([model.id, explorerModel.id]),
				);
				expect(config.providers["custom-role-provider"]?.models?.map((entry) => entry.id)).toContain(
					"custom-role-model",
				);
				expect(config.providers["test-proxy"]?.apiKey).toBe("$TEST_PROXY_API_KEY");
				expect(modelsJson).not.toContain(sentinel);
				expect(process.env.PI_CODING_AGENT_DIR).toBe(previousPiAgentDir);
				expect(process.env.TEST_PROXY_API_KEY).toBe(previousApiKey);

				const piBinary = join(
					process.cwd(),
					"node_modules",
					"@earendil-works",
					"pi-coding-agent",
					"dist",
					"cli.js",
				);
				const child = spawnSync(process.execPath, [piBinary, "--list-models", "gpt-5.6-luna"], {
					encoding: "utf8",
					env: {
						HOME: process.env.HOME ?? "",
						PATH: process.env.PATH ?? "",
						PI_CODING_AGENT_DIR: agentDir,
						// Pi >= 0.80 hides models whose provider auth is unconfigured from
						// --list-models; satisfy the $TEST_PROXY_API_KEY reference.
						TEST_PROXY_API_KEY: "pi-child-visibility-check",
					},
				});
				expect(child.status).toBe(0);
				const childOutput = `${child.stdout}\n${child.stderr}`;
				expect(childOutput).toContain("test-proxy");
				expect(childOutput).toContain("gpt-5.6-luna");
			} finally {
				runtime.session.dispose();
			}
			expect(process.env.PI_CODING_AGENT_DIR).toBe(previousPiAgentDir);
			expect(process.env.TEST_PROXY_API_KEY).toBe(previousApiKey);
		} finally {
			if (previousApiKey === undefined) delete process.env.TEST_PROXY_API_KEY;
			else process.env.TEST_PROXY_API_KEY = previousApiKey;
			if (previousPiAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousPiAgentDir;
		}
	});

	it("resolves a different-provider orchestrator credential through parent auth without persisting role secrets", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-parent-credential-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const orchestratorSecret = "AUTORAG_PARENT_ORCHESTRATOR_SECRET";
		const explorerSecret = "AUTORAG_PARENT_EXPLORER_SECRET";
		const orchestratorModel: Model<"openai-responses"> = {
			...model,
			provider: "openai",
			baseUrl: "https://api.openai.com/v1",
		};
		const differingExplorerModel: Model<"openai-responses"> = {
			...explorerModel,
			provider: "explorer-provider",
			baseUrl: "https://explorer.example/v1",
		};
		const previousOrchestratorKey = process.env.OPENAI_API_KEY;
		process.env.OPENAI_API_KEY = orchestratorSecret;
		try {
			const runtime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model: orchestratorModel,
				explorerModel: differingExplorerModel,
				providerApiKeys: { "explorer-provider": explorerSecret },
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			try {
				const orchestratorAuth = await runtime.session.modelRuntime.getAuth(orchestratorModel.provider);
				expect(orchestratorAuth?.auth.apiKey === orchestratorSecret).toBe(true);
				const persisted = `${readFileSync(join(agentDir, "models.json"), "utf8")}\n${readFileSync(
					join(agentDir, "auth.json"),
					"utf8",
				)}`;
				expect(persisted).not.toContain(orchestratorSecret);
				expect(persisted).not.toContain(explorerSecret);
			} finally {
				runtime.session.dispose();
			}
		} finally {
			if (previousOrchestratorKey === undefined) delete process.env.OPENAI_API_KEY;
			else process.env.OPENAI_API_KEY = previousOrchestratorKey;
		}
	});

	it("gives only the explorer credential to Pi while a concurrent generic child keeps the parent environment", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-provider-credentials-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const piChildCapturePath = join(root, "pi-child-env.json");
		const piChildReadyPath = join(root, "pi-child-ready");
		const piChildReleasePath = join(root, "pi-child-release");
		const fakePiBinary = join(root, "fake-pi.mjs");
		const explorerSecret = "AUTORAG_EXPLORER_PROVIDER_SECRET";
		const orchestratorSecret = "AUTORAG_ORCHESTRATOR_PROVIDER_SECRET";
		const unlistedParentSecret = "AUTORAG_UNLISTED_PARENT_SECRET";
		const previousPiAgentDir = process.env.PI_CODING_AGENT_DIR;
		const previousPiBinary = process.env.PI_SUBAGENT_PI_BINARY;
		const previousOrchestratorKey = process.env.OPENAI_API_KEY;
		const previousExplorerKey = process.env.OTHER_PROVIDER_API_KEY;
		const previousUnlistedKey = process.env.UNLISTED_PARENT_SECRET;
		const orchestratorModel: Model<"openai-responses"> = {
			...model,
			provider: "openai",
			baseUrl: "https://api.openai.com/v1",
		};
		const differingExplorerModel: Model<"openai-responses"> = {
			...explorerModel,
			provider: "other-provider",
			baseUrl: "https://other.example/v1",
		};
		process.env.OPENAI_API_KEY = orchestratorSecret;
		delete process.env.OTHER_PROVIDER_API_KEY;
		process.env.UNLISTED_PARENT_SECRET = unlistedParentSecret;
		writeFileSync(
			fakePiBinary,
			`#!/usr/bin/env node
	import { existsSync, writeFileSync } from "node:fs";

	writeFileSync(${JSON.stringify(piChildCapturePath)}, JSON.stringify({
		agentDir: process.env.PI_CODING_AGENT_DIR,
		orchestratorSecretVisible: process.env.OPENAI_API_KEY === ${JSON.stringify(orchestratorSecret)},
		explorerSecretVisible: process.env.OTHER_PROVIDER_API_KEY === ${JSON.stringify(explorerSecret)},
		unlistedParentSecretVisible: process.env.UNLISTED_PARENT_SECRET === ${JSON.stringify(unlistedParentSecret)},
	}), "utf8");
	writeFileSync(${JSON.stringify(piChildReadyPath)}, "ready", "utf8");
	const waitBuffer = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
	while (!existsSync(${JSON.stringify(piChildReleasePath)})) Atomics.wait(waitBuffer, 0, 0, 10);
	process.stdout.write(JSON.stringify({
	type: "message_end",
	message: {
		role: "assistant",
		content: [{ type: "text", text: "captured child environment" }],
		api: "openai-responses",
		provider: "other-provider",
		model: "gpt-5.6-luna",
		usage: {
			input: 0,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 1,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	},
}) + "\\n");
`,
			"utf8",
		);
		chmodSync(fakePiBinary, 0o700);
		process.env.PI_SUBAGENT_PI_BINARY = fakePiBinary;

		try {
			const runtime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model: orchestratorModel,
				explorerModel: differingExplorerModel,
				providerApiKeys: { "other-provider": explorerSecret },
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			try {
				expect(process.env.PI_CODING_AGENT_DIR).toBe(previousPiAgentDir);
				expect(process.env.OPENAI_API_KEY).toBe(orchestratorSecret);
				expect(process.env.OTHER_PROVIDER_API_KEY).toBeUndefined();
				expect(process.env.UNLISTED_PARENT_SECRET).toBe(unlistedParentSecret);
				const modelsJson = readFileSync(join(agentDir, "models.json"), "utf8");
				const authJson = readFileSync(join(agentDir, "auth.json"), "utf8");
				const config = JSON.parse(modelsJson) as {
					providers: Record<string, { apiKey?: string }>;
				};
				expect(config.providers[orchestratorModel.provider]?.apiKey).toBe("$OPENAI_API_KEY");
				expect(config.providers["other-provider"]?.apiKey).toBe("$OTHER_PROVIDER_API_KEY");
				for (const secret of [orchestratorSecret, explorerSecret, unlistedParentSecret]) {
					expect(modelsJson).not.toContain(secret);
					expect(authJson).not.toContain(secret);
				}
				const subagentTool = runtime.session.getToolDefinition("subagent");
				if (subagentTool === undefined) throw new Error("The mandatory subagent tool was not registered");
				const dispatch = subagentTool.execute(
					"credential-isolation",
					{
						agent: "autorag-explorer",
						model: "other-provider/gpt-5.6-luna",
						task: "Capture the child environment for the credential-isolation integration test.",
						cwd: root,
						agentScope: "user",
						artifacts: false,
					},
					undefined,
					undefined,
					runtime.session.extensionRunner.createContext(),
				);
				await waitForPath(piChildReadyPath);

				const genericChild = spawnSync(
					process.execPath,
					[
						"-e",
						`process.stdout.write(JSON.stringify({ agentDir: process.env.PI_CODING_AGENT_DIR, orchestratorSecretVisible: process.env.OPENAI_API_KEY === ${JSON.stringify(
							orchestratorSecret,
						)}, explorerSecretVisible: process.env.OTHER_PROVIDER_API_KEY === ${JSON.stringify(
							explorerSecret,
						)}, unlistedParentSecretVisible: process.env.UNLISTED_PARENT_SECRET === ${JSON.stringify(
							unlistedParentSecret,
						)} }))`,
					],
					{ encoding: "utf8", env: { ...process.env } },
				);
				expect(genericChild.status).toBe(0);
				expect(JSON.parse(genericChild.stdout)).toEqual({
					orchestratorSecretVisible: true,
					explorerSecretVisible: false,
					unlistedParentSecretVisible: true,
				});
				writeFileSync(piChildReleasePath, "release", "utf8");
				await dispatch;
				expect(JSON.parse(readFileSync(piChildCapturePath, "utf8"))).toEqual({
					agentDir,
					orchestratorSecretVisible: false,
					explorerSecretVisible: true,
					unlistedParentSecretVisible: false,
				});
			} finally {
				if (!existsSync(piChildReleasePath)) writeFileSync(piChildReleasePath, "release", "utf8");
				runtime.session.dispose();
			}
			expect(process.env.PI_CODING_AGENT_DIR).toBe(previousPiAgentDir);
			expect(process.env.OPENAI_API_KEY).toBe(orchestratorSecret);
			expect(process.env.OTHER_PROVIDER_API_KEY).toBeUndefined();
			expect(process.env.UNLISTED_PARENT_SECRET).toBe(unlistedParentSecret);
		} finally {
			if (previousPiAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousPiAgentDir;
			if (previousPiBinary === undefined) delete process.env.PI_SUBAGENT_PI_BINARY;
			else process.env.PI_SUBAGENT_PI_BINARY = previousPiBinary;
			if (previousOrchestratorKey === undefined) delete process.env.OPENAI_API_KEY;
			else process.env.OPENAI_API_KEY = previousOrchestratorKey;
			if (previousExplorerKey === undefined) delete process.env.OTHER_PROVIDER_API_KEY;
			else process.env.OTHER_PROVIDER_API_KEY = previousExplorerKey;
			if (previousUnlistedKey === undefined) delete process.env.UNLISTED_PARENT_SECRET;
			else process.env.UNLISTED_PARENT_SECRET = previousUnlistedKey;
		}
	});

	it("sanitizes an existing literal provider credential without mutating the parent environment", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-existing-credential-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		const existingCredential = "literal-existing-secret";
		const runtimeCredential = "AUTORAG_NEW_RUNTIME_SECRET";
		const apiKeyEnvName = "LITERAL_PROVIDER_API_KEY";
		const previousApiKey = process.env[apiKeyEnvName];
		const roleModel: Model<"openai-responses"> = {
			...model,
			provider: "literal-provider",
			id: "orchestrator-model",
			baseUrl: "https://literal.example/v1",
		};
		const roleExplorerModel: Model<"openai-responses"> = {
			...roleModel,
			id: "explorer-model",
		};
		writeFileSync(
			join(agentDir, "models.json"),
			JSON.stringify({
				providers: {
					"literal-provider": {
						baseUrl: roleModel.baseUrl,
						api: roleModel.api,
						apiKey: existingCredential,
						models: [],
					},
				},
			}),
		);

		try {
			const runtime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model: roleModel,
				explorerModel: roleExplorerModel,
				apiKey: runtimeCredential,
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			try {
				const modelsJson = readFileSync(join(agentDir, "models.json"), "utf8");
				const config = JSON.parse(modelsJson) as { providers: Record<string, { apiKey?: string }> };
				expect(config.providers[roleModel.provider]?.apiKey).toBe(`$${apiKeyEnvName}`);
				expect(modelsJson).not.toContain(existingCredential);
				expect(modelsJson).not.toContain(runtimeCredential);
				expect((await runtime.session.modelRuntime.getAuth(roleModel.provider))?.auth.apiKey).toBe(
					runtimeCredential,
				);
				expect(process.env[apiKeyEnvName]).toBe(previousApiKey);

				const child = spawnSync(
					process.execPath,
					["-e", `process.stdout.write(process.env.${apiKeyEnvName} ?? "")`],
					{
						encoding: "utf8",
						env: { ...process.env },
					},
				);
				expect(child.status).toBe(0);
				expect(child.stdout).toBe(previousApiKey ?? "");
			} finally {
				runtime.session.dispose();
			}
			expect(process.env[apiKeyEnvName]).toBe(previousApiKey);
		} finally {
			if (previousApiKey === undefined) delete process.env[apiKeyEnvName];
			else process.env[apiKeyEnvName] = previousApiKey;
		}
	});

	it("rejects active providers that collide on their derived API-key environment name before mutating runtime state", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-provider-env-collision-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const firstCredential = "AUTORAG_FOO_BAR_FIRST_SECRET";
		const secondCredential = "AUTORAG_FOO_BAR_SECOND_SECRET";
		const previousApiKey = process.env.FOO_BAR_API_KEY;
		const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
		const originalModels = JSON.stringify({
			providers: {
				"foo-bar": {
					baseUrl: "https://foo-bar.example/v1",
					api: "openai-responses",
					models: [{ id: "existing-foo-bar" }],
				},
				foo_bar: {
					baseUrl: "https://foo-bar-underscore.example/v1",
					api: "openai-responses",
					models: [{ id: "existing-foo_bar" }],
				},
			},
		});
		const firstModel: Model<"openai-responses"> = {
			...model,
			provider: "foo-bar",
			baseUrl: "https://foo-bar.example/v1",
		};
		const secondModel: Model<"openai-responses"> = {
			...explorerModel,
			provider: "foo_bar",
			baseUrl: "https://foo-bar-underscore.example/v1",
		};
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(modelsPath, originalModels);
		process.env.FOO_BAR_API_KEY = "preexisting-foo-bar-env";
		process.env.PI_CODING_AGENT_DIR = "preexisting-pi-agent-dir";

		let runtime: Awaited<ReturnType<typeof createMandatorySubagentSession>> | undefined;
		let rejection: unknown;
		try {
			try {
				runtime = await createMandatorySubagentSession({
					cwd: root,
					agentDir,
					model: firstModel,
					explorerModel: secondModel,
					providerApiKeys: { "foo-bar": firstCredential, foo_bar: secondCredential },
					systemPrompt: "test prompt",
					tools: [customTool],
				});
			} catch (error) {
				rejection = error;
			} finally {
				runtime?.session.dispose();
			}

			expect(rejection).toBeInstanceOf(Error);
			if (!(rejection instanceof Error)) return;
			expect(rejection.message).toContain("FOO_BAR_API_KEY");
			expect(rejection.message).toContain("foo-bar");
			expect(rejection.message).toContain("foo_bar");
			expect(rejection.message).not.toContain(firstCredential);
			expect(rejection.message).not.toContain(secondCredential);
			expect(readFileSync(modelsPath, "utf8")).toBe(originalModels);
			expect(process.env.FOO_BAR_API_KEY).toBe("preexisting-foo-bar-env");
			expect(process.env.PI_CODING_AGENT_DIR).toBe("preexisting-pi-agent-dir");
		} finally {
			if (previousApiKey === undefined) delete process.env.FOO_BAR_API_KEY;
			else process.env.FOO_BAR_API_KEY = previousApiKey;
			if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
		}
	});

	it("rejects incompatible role models before changing an active same-agentDir registry", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-incompatible-registry-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const activeRuntime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			explorerModel,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			const activeRegistry = readFileSync(modelsPath, "utf8");
			const incompatibleExplorerModel: Model<"openai-responses"> = {
				...explorerModel,
				id: "gpt-5.6-luna-incompatible",
				name: "GPT-5.6 Luna Incompatible",
			};
			let rejection: Error | undefined;
			let unexpectedRuntime: Awaited<ReturnType<typeof createMandatorySubagentSession>> | undefined;
			try {
				unexpectedRuntime = await createMandatorySubagentSession({
					cwd: root,
					agentDir,
					model,
					explorerModel: incompatibleExplorerModel,
					systemPrompt: "test prompt",
					tools: [customTool],
				});
			} catch (error) {
				if (!(error instanceof Error)) throw error;
				rejection = error;
			} finally {
				unexpectedRuntime?.session.dispose();
			}

			expect(rejection?.message).toMatch(/different child environment\/registry routing/i);
			expect(readFileSync(modelsPath, "utf8")).toBe(activeRegistry);
		} finally {
			activeRuntime.session.dispose();
		}
	});

	it("allows compatible concurrent sessions with different parent-only provider credentials", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-incompatible-mask-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const sharedCredential = "AUTORAG_SHARED_EXPLORER_SECRET";
		const firstUnrelatedSecret = "AUTORAG_FIRST_UNRELATED_SECRET";
		const secondUnrelatedSecret = "AUTORAG_SECOND_UNRELATED_SECRET";
		const previousFirstKey = process.env.FIRST_UNRELATED_PROVIDER_API_KEY;
		const previousSecondKey = process.env.SECOND_UNRELATED_PROVIDER_API_KEY;
		const previousProviderKey = process.env.TEST_PROXY_API_KEY;
		process.env.FIRST_UNRELATED_PROVIDER_API_KEY = firstUnrelatedSecret;
		process.env.SECOND_UNRELATED_PROVIDER_API_KEY = secondUnrelatedSecret;
		let activeRuntime: Awaited<ReturnType<typeof createMandatorySubagentSession>> | undefined;
		let secondRuntime: Awaited<ReturnType<typeof createMandatorySubagentSession>> | undefined;

		try {
			activeRuntime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				explorerModel,
				providerApiKeys: {
					"test-proxy": sharedCredential,
					"first-unrelated-provider": firstUnrelatedSecret,
				},
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			secondRuntime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				explorerModel,
				providerApiKeys: {
					"test-proxy": sharedCredential,
					"second-unrelated-provider": secondUnrelatedSecret,
				},
				systemPrompt: "test prompt",
				tools: [customTool],
			});

			expect(process.env.FIRST_UNRELATED_PROVIDER_API_KEY).toBe(firstUnrelatedSecret);
			expect(process.env.SECOND_UNRELATED_PROVIDER_API_KEY).toBe(secondUnrelatedSecret);
			expect(process.env.TEST_PROXY_API_KEY).toBe(previousProviderKey);
		} finally {
			secondRuntime?.session.dispose();
			activeRuntime?.session.dispose();
			if (previousFirstKey === undefined) delete process.env.FIRST_UNRELATED_PROVIDER_API_KEY;
			else process.env.FIRST_UNRELATED_PROVIDER_API_KEY = previousFirstKey;
			if (previousSecondKey === undefined) delete process.env.SECOND_UNRELATED_PROVIDER_API_KEY;
			else process.env.SECOND_UNRELATED_PROVIDER_API_KEY = previousSecondKey;
		}
	});

	it("keeps compatible routing leases without changing the parent environment", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-compatible-lease-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const runtimeCredential = "AUTORAG_COMPATIBLE_RUNTIME_SECRET";
		const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
		const previousApiKey = process.env.TEST_PROXY_API_KEY;
		let firstRuntime: Awaited<ReturnType<typeof createMandatorySubagentSession>> | undefined;
		let secondRuntime: Awaited<ReturnType<typeof createMandatorySubagentSession>> | undefined;

		try {
			firstRuntime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				explorerModel,
				apiKey: runtimeCredential,
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			secondRuntime = await createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				explorerModel,
				apiKey: runtimeCredential,
				systemPrompt: "test prompt",
				tools: [customTool],
			});

			firstRuntime.session.dispose();
			firstRuntime = undefined;
			expect(process.env.PI_CODING_AGENT_DIR).toBe(previousAgentDir);
			expect(process.env.TEST_PROXY_API_KEY).toBe(previousApiKey);

			secondRuntime.session.dispose();
			secondRuntime = undefined;
			expect(process.env.PI_CODING_AGENT_DIR).toBe(previousAgentDir);
			expect(process.env.TEST_PROXY_API_KEY).toBe(previousApiKey);
		} finally {
			firstRuntime?.session.dispose();
			secondRuntime?.session.dispose();
			if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
			if (previousApiKey === undefined) delete process.env.TEST_PROXY_API_KEY;
			else process.env.TEST_PROXY_API_KEY = previousApiKey;
		}
	});

	it("rejects missing role metadata before writing a registry", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-invalid-model-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const invalidModel = { ...model, baseUrl: "" } as Model<"openai-responses">;

		await expect(
			createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model: invalidModel,
				systemPrompt: "test prompt",
				tools: [customTool],
			}),
		).rejects.toThrow(/orchestrator model baseUrl/);
		expect(existsSync(join(agentDir, "models.json"))).toBe(false);
	});

	it("rejects malformed provider metadata without replacing the existing registry", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-invalid-registry-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const modelsPath = join(agentDir, "models.json");
		const original = JSON.stringify({ providers: { "test-proxy": { models: "invalid" } } });
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(modelsPath, original);

		await expect(
			createMandatorySubagentSession({
				cwd: root,
				agentDir,
				model,
				systemPrompt: "test prompt",
				tools: [customTool],
			}),
		).rejects.toThrow(/provider "test-proxy" has invalid "models" metadata/);
		expect(readFileSync(modelsPath, "utf8")).toBe(original);
	});

	it("uses AUTORAG_HOME for default agent state while preserving Pi session layout", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-home-"));
		tempDirs.push(root);
		const previousAutoragHome = process.env.AUTORAG_HOME;
		const autoragHome = join(root, "autorag-home");
		process.env.AUTORAG_HOME = autoragHome;

		try {
			const runtime = await createMandatorySubagentSession({
				cwd: root,
				model,
				systemPrompt: "test prompt",
				tools: [customTool],
			});
			try {
				expect(
					runtime.session.sessionFile?.startsWith(
						join(autoragHome, "pi-agent", "sessions", expectedSessionNamespace(root)),
					),
				).toBe(true);
			} finally {
				runtime.session.dispose();
			}
		} finally {
			if (previousAutoragHome === undefined) delete process.env.AUTORAG_HOME;
			else process.env.AUTORAG_HOME = previousAutoragHome;
		}
	});

	it("persists a Pi-compatible JSONL session under the configured agent directory", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-session-"));
		tempDirs.push(root);
		const agentDir = join(root, ".autorag", "pi-agent");
		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			const sessionFile = runtime.session.sessionFile;
			expect(sessionFile).toBeDefined();
			expect(sessionFile?.startsWith(join(agentDir, "sessions"))).toBe(true);
			expect(runtime.session.sessionManager.isPersisted()).toBe(true);
			// Pi delays creating the JSONL until the first assistant message so
			// abandoned sessions do not leave empty files.
			expect(existsSync(sessionFile as string)).toBe(false);
		} finally {
			runtime.session.dispose();
		}
	});

	it("loads pi-subagents and exposes its tools beside AutoRAG tools", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-tools-"));
		tempDirs.push(root);
		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(runtime.session.getActiveToolNames()).toEqual(
				expect.arrayContaining(["custom_search", "subagent", "subagent_wait"]),
			);
			expect(runtime.extensionPath).toContain("pi-subagents/src/extension/index.ts");
		} finally {
			runtime.session.dispose();
		}
	});

	it("does not mutate PI_CODING_AGENT_DIR during the synchronous list execute interval", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-list-env-"));
		tempDirs.push(root);
		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			const subagentTool = runtime.session.getToolDefinition("subagent");
			if (subagentTool === undefined) throw new Error("The mandatory subagent tool was not registered");
			const originalEnv = process.env;
			const mutations: string[] = [];
			const observedEnv = new Proxy(originalEnv, {
				set(target, property, value) {
					if (property === "PI_CODING_AGENT_DIR") mutations.push(`set:${String(value)}`);
					return Reflect.set(target, property, value);
				},
				deleteProperty(target, property) {
					if (property === "PI_CODING_AGENT_DIR") mutations.push("delete");
					return Reflect.deleteProperty(target, property);
				},
			});
			let execution: ReturnType<typeof subagentTool.execute> | undefined;
			try {
				process.env = observedEnv;
				execution = subagentTool.execute(
					"list-without-global-env",
					{ action: "list", agentScope: "user" },
					undefined,
					undefined,
					runtime.session.extensionRunner.createContext(),
				);
			} finally {
				process.env = originalEnv;
			}
			if (execution === undefined) throw new Error("The mandatory subagent list action did not execute");
			const result = await execution;
			const text = result.content
				.filter((item): item is { type: "text"; text: string } => item.type === "text")
				.map((item) => item.text)
				.join("\n");
			expect(text).toContain("autorag-explorer");
			expect(mutations).toEqual([]);
		} finally {
			runtime.session.dispose();
		}
	});

	it("keeps child_process spawn identities unchanged while a runtime is active", () => {
		const childEnv = { ...process.env };
		delete childEnv.PI_SUBAGENT_PI_BINARY;
		const child = spawnSync(
			process.execPath,
			[
				"--experimental-strip-types",
				"--disable-warning=ExperimentalWarning",
				"--input-type=module",
				"-e",
				`import childProcess, { spawn as namedSpawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const originalGlobalSpawn = childProcess.spawn;
const originalNamedSpawn = namedSpawn;
const root = mkdtempSync(join(tmpdir(), "autorag-pi-spawn-identity-"));
let runtime;
try {
	const { createMandatorySubagentSession } = await import(${JSON.stringify(RUNTIME_MODULE_URL)});
	runtime = await createMandatorySubagentSession({
		cwd: root,
		agentDir: join(root, "agent"),
		model: ${JSON.stringify(model)},
		systemPrompt: "test prompt",
		tools: [],
	});
	const active = {
		global: childProcess.spawn === originalGlobalSpawn,
		named: namedSpawn === originalNamedSpawn,
		namedMatchesGlobal: namedSpawn === childProcess.spawn,
	};
	runtime.session.dispose();
	runtime = undefined;
	const disposed = {
		global: childProcess.spawn === originalGlobalSpawn,
		named: namedSpawn === originalNamedSpawn,
		namedMatchesGlobal: namedSpawn === childProcess.spawn,
	};
	process.stdout.write(JSON.stringify({ active, disposed }));
} finally {
	runtime?.session.dispose();
	rmSync(root, { recursive: true, force: true });
}`,
			],
			{
				cwd: process.cwd(),
				encoding: "utf8",
				env: childEnv,
				timeout: 20_000,
			},
		);
		if (child.status !== 0) {
			throw new Error(`Spawn identity fixture failed: ${child.error?.message ?? child.stderr}`);
		}
		expect(JSON.parse(child.stdout)).toEqual({
			active: { global: true, named: true, namedMatchesGlobal: true },
			disposed: { global: true, named: true, namedMatchesGlobal: true },
		});
	});

	it("installs the mandatory explorer into persistent agent state and exposes it through the real list action", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-explorer-install-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const workspace = join(root, "arbitrary-workspace");
		const projectExplorerPath = join(workspace, ".pi", "agents", "autorag-explorer.md");
		const conflictingProjectExplorer = `---
name: autorag-explorer
description: Conflicting project override
tools: read
---
Project override body.
`;
		mkdirSync(join(workspace, ".pi", "agents"), { recursive: true });
		writeFileSync(projectExplorerPath, conflictingProjectExplorer, "utf8");
		const runtime = await createMandatorySubagentSession({
			cwd: workspace,
			agentDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			const explorerPath = join(agentDir, "agents", "autorag-explorer.md");
			expect(readFileSync(explorerPath, "utf8")).toBe(AUTORAG_EXPLORER_AGENT_DEFINITION);
			expect(readFileSync(projectExplorerPath, "utf8")).toBe(conflictingProjectExplorer);
			const subagentTool = runtime.session.getToolDefinition("subagent");
			if (subagentTool === undefined) throw new Error("The mandatory subagent tool was not registered");

			const result = await subagentTool.execute(
				"list-explorer-install",
				{ action: "list", agentScope: "user" },
				undefined,
				undefined,
				runtime.session.extensionRunner.createContext(),
			);
			const text = result.content
				.filter((item): item is { type: "text"; text: string } => item.type === "text")
				.map((item) => item.text)
				.join("\n");
			expect(text).toContain("autorag-explorer");
			expect(text).toContain("Read-only, high-recall document explorer for AutoRAG evidence collection");
			expect(text).not.toContain("Conflicting project override");
			expect(existsSync(join(workspace, ".pi-subagents"))).toBe(false);
		} finally {
			runtime.session.dispose();
		}
	});

	it.each([
		["installed legacy", INSTALLED_LEGACY_EXPLORER_DEFINITION],
		["previous managed", PREVIOUS_MANAGED_EXPLORER_DEFINITION],
	])("atomically migrates the %s explorer definition", async (_label, legacyDefinition) => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-explorer-legacy-migrate-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const agentsDir = join(agentDir, "agents");
		const explorerPath = join(agentsDir, "autorag-explorer.md");
		mkdirSync(agentsDir, { recursive: true });
		writeFileSync(explorerPath, legacyDefinition, { encoding: "utf8", mode: 0o644 });
		chmodSync(explorerPath, 0o644);

		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(readFileSync(explorerPath, "utf8")).toBe(AUTORAG_EXPLORER_AGENT_DEFINITION);
			expect(statSync(explorerPath).mode & 0o777).toBe(0o600);
			expect(readdirSync(agentsDir).filter((name) => name.startsWith(".autorag-explorer.")).length).toBe(0);
		} finally {
			runtime.session.dispose();
		}
	});

	it("migrates a managed explorer after its package extension path changes", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-explorer-managed-upgrade-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const agentsDir = join(agentDir, "agents");
		const explorerPath = join(agentsDir, "autorag-explorer.md");
		const relocatedDefinition = AUTORAG_EXPLORER_AGENT_DEFINITION.replace(
			`subagentOnlyExtensions: ${EXPLORER_TOOLS_EXTENSION_PATH}`,
			"subagentOnlyExtensions: /relocated/autorag/explorer-tools-extension.ts",
		);
		expect(relocatedDefinition).not.toBe(AUTORAG_EXPLORER_AGENT_DEFINITION);
		mkdirSync(agentsDir, { recursive: true });
		writeFileSync(explorerPath, relocatedDefinition, { encoding: "utf8", mode: 0o644 });
		chmodSync(explorerPath, 0o644);

		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(readFileSync(explorerPath, "utf8")).toBe(AUTORAG_EXPLORER_AGENT_DEFINITION);
			expect(statSync(explorerPath).mode & 0o777).toBe(0o600);
			expect(readdirSync(agentsDir).filter((name) => name.startsWith(".autorag-explorer.")).length).toBe(0);
		} finally {
			runtime.session.dispose();
		}
	});

	it("rejects a noncanonical persistent explorer before changing child env or models", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-explorer-reject-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const explorerPath = join(agentDir, "agents", "autorag-explorer.md");
		const modelsPath = join(agentDir, "models.json");
		const original =
			"---\nname: autorag-explorer\ndescription: User-owned explorer\ntools: read\n---\nUser-owned body.\n";
		const originalModels = '{"providers":{"existing":{"models":[]}}}';
		const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
		const previousPiBinary = process.env.PI_SUBAGENT_PI_BINARY;
		const sentinelAgentDir = "preexisting-pi-agent-dir";
		const sentinelPiBinary = "preexisting-pi-subagent-binary";
		mkdirSync(join(agentDir, "agents"), { recursive: true });
		writeFileSync(explorerPath, original, "utf8");
		writeFileSync(modelsPath, originalModels, "utf8");
		process.env.PI_CODING_AGENT_DIR = sentinelAgentDir;
		process.env.PI_SUBAGENT_PI_BINARY = sentinelPiBinary;
		try {
			await expect(
				createMandatorySubagentSession({
					cwd: root,
					agentDir,
					model,
					systemPrompt: "test prompt",
					tools: [customTool],
				}),
			).rejects.toThrow(/persistent explorer definition.*not canonical/i);
			expect(readFileSync(explorerPath, "utf8")).toBe(original);
			expect(readFileSync(modelsPath, "utf8")).toBe(originalModels);
			expect(process.env.PI_CODING_AGENT_DIR).toBe(sentinelAgentDir);
			expect(process.env.PI_SUBAGENT_PI_BINARY).toBe(sentinelPiBinary);
		} finally {
			if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
			if (previousPiBinary === undefined) delete process.env.PI_SUBAGENT_PI_BINARY;
			else process.env.PI_SUBAGENT_PI_BINARY = previousPiBinary;
		}
	});

	it("matches the installed subagent schema with top-level artifacts and no nested artifacts", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-artifacts-schema-"));
		tempDirs.push(root);
		const runtime = await createMandatorySubagentSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			const subagentTool = runtime.session.getAllTools().find((tool) => tool.name === "subagent");
			if (subagentTool === undefined) throw new Error("The mandatory subagent tool was not registered");

			const task = {
				agent: "autorag-explorer",
				model: "test-proxy/gpt-5.6-luna",
				task: "Original query: test Selected retrieval method: POSIX query variants: test retrievedAt temporal metadata",
				cwd: root,
			};
			const validInvocations = [
				{ ...task, agentScope: "user", artifacts: false },
				{ agentScope: "user", artifacts: false, tasks: [task] },
				{ agentScope: "user", artifacts: false, chain: [{ ...task }] },
				{ agentScope: "user", artifacts: false, chain: [{ parallel: [{ ...task }] }] },
			];
			for (const invocation of validInvocations) {
				expect(Value.Check(subagentTool.parameters, invocation)).toBe(true);
			}

			const nestedArtifacts = { ...task, artifacts: false };
			expect(
				Value.Check(subagentTool.parameters, {
					agentScope: "user",
					artifacts: false,
					chain: [nestedArtifacts],
				}),
			).toBe(false);
		} finally {
			runtime.session.dispose();
		}
	});

	it("fails closed when the mandatory extension path cannot load", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-pi-missing-extension-"));
		tempDirs.push(root);
		await expect(
			createMandatorySubagentSession({
				cwd: root,
				agentDir: join(root, "agent"),
				model,
				systemPrompt: "test prompt",
				tools: [customTool],
				extensionPath: "/definitely/missing/pi-subagents.ts",
			}),
		).rejects.toThrow(/mandatory pi-subagents extension/i);
	});
});
describe("createHealthSubagentProbeSession", () => {
	it("requires an absolute agentDir and sessionDir", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-health-probe-req-"));
		tempDirs.push(root);
		await expect(
			createHealthSubagentProbeSession({
				cwd: root,
				model,
				systemPrompt: "health probe",
				tools: [],
				agentDir: "relative/agent",
				sessionDir: join(root, "sessions"),
			}),
		).rejects.toThrow(/absolute agentDir/);
		await expect(
			createHealthSubagentProbeSession({
				cwd: root,
				model,
				systemPrompt: "health probe",
				tools: [],
				agentDir: join(root, "agent"),
				sessionDir: "relative/sessions",
			}),
		).rejects.toThrow(/absolute sessionDir/);
		await expect(
			createHealthSubagentProbeSession({
				cwd: root,
				model,
				systemPrompt: "health probe",
				tools: [],
				agentDir: "",
				sessionDir: join(root, "sessions"),
			}),
		).rejects.toThrow(/absolute agentDir/);
	});

	it("exposes subagent and wait tools when the extension loads", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-health-probe-tools-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const sessionDir = join(root, "sessions");
		const probe = await createHealthSubagentProbeSession({
			cwd: root,
			model,
			systemPrompt: "health probe",
			tools: [],
			agentDir,
			sessionDir,
		});
		try {
			const toolNames = new Set(probe.session.getAllTools().map((tool) => tool.name));
			expect(toolNames.has("subagent")).toBe(true);
			expect(toolNames.has("subagent_wait")).toBe(true);
			expect(probe.extensionPath).toContain("pi-subagents/src/extension/index.ts");
		} finally {
			probe.dispose();
		}
	});

	it("maps a concurrent lease conflict to a busy message", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-health-probe-lease-"));
		tempDirs.push(root);
		const agentDir = join(root, "agent");
		const sessionDir = join(root, "sessions");

		// Hold an active session with one model, then try a health probe with
		// an incompatible explorer model to trigger the lease conflict.
		const active = await createMandatorySubagentSession({
			cwd: root,
			agentDir,
			model,
			explorerModel,
			systemPrompt: "active session",
			tools: [customTool],
		});
		try {
			const incompatibleExplorer: Model<"openai-responses"> = {
				...explorerModel,
				id: "gpt-5.6-luna-incompatible",
				name: "GPT-5.6 Luna Incompatible",
			};
			await expect(
				createHealthSubagentProbeSession({
					cwd: root,
					model,
					explorerModel: incompatibleExplorer,
					systemPrompt: "health probe",
					tools: [],
					agentDir,
					sessionDir,
				}),
			).rejects.toThrow("concurrent AutoRAG session busy with different child environment routing");
		} finally {
			active.session.dispose();
		}
	});

	it("does not write under a durable home path when agentDir is temp", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-health-probe-nondestructive-"));
		tempDirs.push(root);
		const durableHome = resolveAutoRAGHome();
		const durableAgentDir = join(durableHome, "pi-agent");
		// Snapshot durable home state before the probe.
		const durableExisted = existsSync(durableAgentDir);
		const durableSnapshot = durableExisted
			? readdirSync(durableAgentDir, { withFileTypes: true })
					.map((entry) => `${entry.name}:${entry.isDirectory() ? "dir" : "file"}`)
					.sort()
			: [];

		const agentDir = join(root, "agent");
		const sessionDir = join(root, "sessions");
		const probe = await createHealthSubagentProbeSession({
			cwd: root,
			model,
			systemPrompt: "health probe",
			tools: [],
			agentDir,
			sessionDir,
		});
		try {
			expect(probe.session.sessionFile?.startsWith(sessionDir)).toBe(true);
		} finally {
			probe.dispose();
		}

		// The durable home path must be unchanged.
		const durableAfter = existsSync(durableAgentDir)
			? readdirSync(durableAgentDir, { withFileTypes: true })
					.map((entry) => `${entry.name}:${entry.isDirectory() ? "dir" : "file"}`)
					.sort()
			: [];
		expect(existsSync(durableAgentDir)).toBe(durableExisted);
		expect(durableAfter).toEqual(durableSnapshot);
		// The temp agentDir was used, not the durable one.
		expect(existsSync(join(agentDir, "models.json"))).toBe(true);
	});
});
