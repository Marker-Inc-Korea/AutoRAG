import childProcess from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
	chmodSync,
	existsSync,
	linkSync,
	mkdirSync,
	readFileSync,
	realpathSync,
	renameSync,
	statSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	AuthStorage,
	createAgentSession,
	DefaultResourceLoader,
	type ExtensionFactory,
	ModelRegistry,
	SessionManager,
	SettingsManager,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { resolveAutoRAGHome } from "../config/home.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";

const runtimePath = fileURLToPath(import.meta.url);
const MODELS_LOCK_WAIT_TIMEOUT_MS = 5_000;
const MODELS_LOCK_STALE_MS = 30_000;
const MODELS_LOCK_RETRY_MS = 10;
const PI_SUBAGENTS_CHILD_PROCESS_MODULE = "@autorag/librarian/pi-subagents-child-process";

const EXPLORER_RUNTIME_ENV_KEYS = [
	"PATH",
	"Path",
	"HOME",
	"USERPROFILE",
	"HOMEDRIVE",
	"HOMEPATH",
	"TMPDIR",
	"TMP",
	"TEMP",
	"LANG",
	"LC_ALL",
	"LC_CTYPE",
	"TZ",
	"SystemRoot",
	"SYSTEMROOT",
	"WINDIR",
	"COMSPEC",
	"PATHEXT",
	"NODE_EXTRA_CA_CERTS",
	"SSL_CERT_FILE",
	"SSL_CERT_DIR",
] as const;

const EXPLORER_PI_ENV_KEYS = new Set([
	"MCP_DIRECT_TOOLS",
	"PI_INTERCOM_ASK_TIMEOUT_MS",
	"PI_SUBAGENTS_PI_CODING_AGENT_PACKAGE_ROOT",
	"PI_SUBAGENT_ASYNC_EVENTS_MAX_BYTES",
	"PI_SUBAGENT_CHILD",
	"PI_SUBAGENT_CHILD_AGENT",
	"PI_SUBAGENT_CHILD_INDEX",
	"PI_SUBAGENT_DEPTH",
	"PI_SUBAGENT_FANOUT_CHILD",
	"PI_SUBAGENT_INHERIT_PROJECT_CONTEXT",
	"PI_SUBAGENT_INHERIT_SKILLS",
	"PI_SUBAGENT_INTERCOM_SESSION_NAME",
	"PI_SUBAGENT_MAX_DEPTH",
	"PI_SUBAGENT_MAX_SPAWNS_PER_SESSION",
	"PI_SUBAGENT_NESTED_PARENT_RUN_ID",
	"PI_SUBAGENT_NESTED_ROOT_RUN_ID",
	"PI_SUBAGENT_ORCHESTRATOR_SESSION_ID",
	"PI_SUBAGENT_ORCHESTRATOR_TARGET",
	"PI_SUBAGENT_PARENT_CAPABILITY_TOKEN",
	"PI_SUBAGENT_PARENT_CHILD_INDEX",
	"PI_SUBAGENT_PARENT_CONTROL_INBOX",
	"PI_SUBAGENT_PARENT_DEPTH",
	"PI_SUBAGENT_PARENT_EVENT_SINK",
	"PI_SUBAGENT_PARENT_PATH",
	"PI_SUBAGENT_PARENT_ROOT_RUN_ID",
	"PI_SUBAGENT_PARENT_RUN_ID",
	"PI_SUBAGENT_PARENT_SESSION",
	"PI_SUBAGENT_RUN_ID",
	"PI_SUBAGENT_STEER_INBOX",
	"PI_SUBAGENT_STRUCTURED_OUTPUT_CAPTURE",
	"PI_SUBAGENT_STRUCTURED_OUTPUT_SCHEMA",
	"PI_SUBAGENT_SUPERVISOR_CHANNEL_DIR",
	"PI_SUBAGENT_TOOL_BUDGET",
	"PI_SUBAGENT_WAIT_TOOL_ENABLED",
]);

function resolveExplorerToolsExtensionPath(): string {
	const dir = dirname(runtimePath);
	const candidates = [
		// Source layout: src/subagents/explorer-tools-extension.ts next to runtime.ts
		join(dir, `explorer-tools-extension${extname(runtimePath)}`),
		// Bundled library entry (dist/index.js): dist/subagents/explorer-tools-extension.js
		join(dir, "subagents", "explorer-tools-extension.js"),
		// Bundled CLI entry (dist/cli/index.js): dist/subagents/explorer-tools-extension.js
		join(dir, "..", "subagents", "explorer-tools-extension.js"),
	];
	for (const candidate of candidates) {
		if (existsSync(candidate)) return realpathSync(candidate);
	}
	throw new Error(`AutoRAG explorer tools extension not found; looked in: ${candidates.join(", ")}`);
}

export const EXPLORER_TOOLS_EXTENSION_PATH = resolveExplorerToolsExtensionPath();

export const AUTORAG_EXPLORER_AGENT_MANAGED_VERSION = 2;

export const AUTORAG_EXPLORER_AGENT_DEFINITION = `---
name: autorag-explorer
description: Read-only, high-recall document explorer for AutoRAG evidence collection
tools: read, grep, find, ls
autoragManagedVersion: ${AUTORAG_EXPLORER_AGENT_MANAGED_VERSION}
extensions:
subagentOnlyExtensions: ${EXPLORER_TOOLS_EXTENSION_PATH}
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
It also sets \`agentScope: "user"\` at the top level so project agent overrides
cannot replace this canonical persistent explorer.
Nested task items in single, tasks, chain, or parallel dispatches omit the
\`artifacts\` and \`agentScope\` fields.

Return candidate findings with source, method, query variant, relevance
(strong/moderate/weak), exact evidence and location context, retrievedAt,
source temporal metadata or explicit unknown status, temporal basis, and
uncertainty. Include weak candidates that could explain a conflict or gap.

Required handoff: include retrievedAt.
Required handoff: include temporal metadata.
`;

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

export interface MandatorySubagentSessionOptions {
	readonly cwd: string;
	readonly model: Model<Api>;
	readonly explorerModel?: Model<Api>;
	readonly systemPrompt: string;
	readonly tools: readonly AgentTool[];
	readonly extensionPath?: string;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly agentDir?: string;
	readonly sessionDir?: string;
}

export interface MandatorySubagentSession {
	readonly session: AgentSession;
	readonly extensionPath: string;
}

export interface HealthSubagentProbeSessionOptions {
	readonly cwd: string;
	readonly model: Model<Api>;
	readonly explorerModel?: Model<Api>;
	readonly systemPrompt: string;
	readonly tools: readonly AgentTool[];
	readonly extensionPath?: string;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly agentDir: string;
	readonly sessionDir: string;
}

export interface HealthSubagentProbeSession {
	readonly session: AgentSession;
	readonly extensionPath: string;
	readonly dispose: () => void;
}

function resolvePiSubagentsPackageJson(): string {
	const require = createRequire(import.meta.url);
	return require.resolve("pi-subagents/package.json");
}

function resolvePiSubagentsExtension(): string {
	return join(dirname(resolvePiSubagentsPackageJson()), "src", "extension", "index.ts");
}

function resolveSubagentPiBinary(): string {
	const configured = process.env.PI_SUBAGENT_PI_BINARY?.trim();
	if (configured) return configured;
	const require = createRequire(import.meta.url);
	const subagentsPackage = resolvePiSubagentsPackageJson();
	const packageJsonPath = join(
		dirname(dirname(subagentsPackage)),
		"@earendil-works",
		"pi-coding-agent",
		"package.json",
	);
	const packageJson = require(packageJsonPath) as { bin?: string | Record<string, string> };
	const bin = typeof packageJson.bin === "string" ? packageJson.bin : packageJson.bin?.pi;
	if (!bin) throw new Error("Mandatory pi-subagents runtime could not resolve the Pi CLI binary");
	return join(dirname(packageJsonPath), bin);
}

function asToolDefinition(tool: AgentTool): ToolDefinition {
	return {
		...tool,
		label: tool.label ?? tool.name,
	} as ToolDefinition;
}

function defaultAgentDir(): string {
	return join(resolveAutoRAGHome(), "pi-agent");
}

function ensurePiSettingsFile(agentDir: string): void {
	const settingsPath = join(agentDir, "settings.json");
	try {
		writeFileSync(settingsPath, "{}", { encoding: "utf8", flag: "wx" });
	} catch (error) {
		if (typeof error === "object" && error !== null && "code" in error && error.code === "EEXIST") return;
		throw error;
	}
}

function hasManagedExplorerVersionMarker(definition: string): boolean {
	const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(definition)?.[1];
	if (frontmatter === undefined) return false;
	const lines = frontmatter.split(/\r?\n/);
	if (!lines.includes("name: autorag-explorer")) return false;
	const marker = /^autoragManagedVersion:\s*(\d+)\s*$/.exec(
		lines.find((line) => line.startsWith("autoragManagedVersion:")) ?? "",
	);
	return marker !== null && Number.isSafeInteger(Number(marker[1])) && Number(marker[1]) > 0;
}

function isRecognizedManagedExplorerDefinition(definition: string): boolean {
	return (
		definition === INSTALLED_LEGACY_EXPLORER_DEFINITION ||
		definition === PREVIOUS_MANAGED_EXPLORER_DEFINITION ||
		hasManagedExplorerVersionMarker(definition)
	);
}

function replaceExplorerDefinitionAtomically(explorerPath: string): void {
	const temporaryPath = `${explorerPath}.${randomUUID()}.tmp`;
	try {
		writeFileSync(temporaryPath, AUTORAG_EXPLORER_AGENT_DEFINITION, {
			encoding: "utf8",
			flag: "wx",
			mode: 0o600,
		});
		chmodSync(temporaryPath, 0o600);
		renameSync(temporaryPath, explorerPath);
	} finally {
		if (existsSync(temporaryPath)) unlinkSync(temporaryPath);
	}
}

function ensureMandatoryExplorerAgent(agentDir: string): void {
	const agentsDir = join(agentDir, "agents");
	const explorerPath = join(agentsDir, "autorag-explorer.md");
	if (existsSync(explorerPath)) {
		const existingDefinition = readFileSync(explorerPath, "utf8");
		if (existingDefinition === AUTORAG_EXPLORER_AGENT_DEFINITION) return;
		if (!isRecognizedManagedExplorerDefinition(existingDefinition)) {
			throw new Error(`AutoRAG persistent explorer definition at ${explorerPath} is not canonical`);
		}
		replaceExplorerDefinitionAtomically(explorerPath);
		if (
			readFileSync(explorerPath, "utf8") !== AUTORAG_EXPLORER_AGENT_DEFINITION ||
			(statSync(explorerPath).mode & 0o777) !== 0o600
		) {
			throw new Error(`AutoRAG persistent explorer definition at ${explorerPath} is not canonical`);
		}
		return;
	}

	mkdirSync(agentsDir, { recursive: true });
	const temporaryPath = join(agentsDir, `.autorag-explorer.${randomUUID()}.tmp`);
	try {
		writeFileSync(temporaryPath, AUTORAG_EXPLORER_AGENT_DEFINITION, {
			encoding: "utf8",
			flag: "wx",
			mode: 0o600,
		});
		try {
			linkSync(temporaryPath, explorerPath);
		} catch (error) {
			if (!(typeof error === "object" && error !== null && "code" in error && error.code === "EEXIST")) {
				throw error;
			}
		}
	} finally {
		if (existsSync(temporaryPath)) unlinkSync(temporaryPath);
	}
	if (readFileSync(explorerPath, "utf8") !== AUTORAG_EXPLORER_AGENT_DEFINITION) {
		throw new Error(`AutoRAG persistent explorer definition at ${explorerPath} is not canonical`);
	}
}

function defaultSessionDir(cwd: string, agentDir: string): string {
	const canonicalCwd = resolve(cwd);
	const readablePath = canonicalCwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-");
	const cwdHash = createHash("sha256").update(canonicalCwd).digest("hex").slice(0, 12);
	return join(agentDir, "sessions", `--${readablePath}-${cwdHash}--`);
}

type JsonObject = Record<string, unknown>;

interface PiModelsConfig {
	providers: Record<string, JsonObject>;
}

class PiModelsLockTimeoutError extends Error {
	constructor() {
		super("Timed out waiting for AutoRAG Pi models.json lock");
		this.name = "PiModelsLockTimeoutError";
	}
}

interface ModelRegistryReference {
	readonly provider: string;
	readonly id: string;
	readonly apiKeyReference: string;
}

type SubagentModelRole = "orchestrator" | "explorer";

interface ActiveRoleModel {
	readonly role: SubagentModelRole;
	readonly model: Model<Api>;
	readonly reference: ModelRegistryReference;
}

interface ChildEnvironmentLeaseRequest {
	readonly agentDir: string;
	readonly registryIdentity: string;
	readonly environmentIdentity: string;
}

interface ChildEnvironmentLease {
	readonly agentDir: string;
	readonly registryIdentity: string;
	readonly environmentIdentity: string;
	owners: number;
}

interface ExplorerChildEnvironment {
	readonly agentDir: string;
	readonly piBinary: string;
	readonly apiKeyName: string;
	readonly apiKey?: string;
}

type SpawnFunction = (...args: unknown[]) => ReturnType<typeof childProcess.spawn>;

interface JitiTransformOptions {
	readonly source: string;
	readonly filename?: string;
	readonly [key: string]: unknown;
}

interface JitiInstance {
	import<T>(id: string, options?: { readonly default?: true }): Promise<T>;
	transform(options: JitiTransformOptions): string;
}

interface JitiOptions {
	readonly moduleCache?: boolean;
	readonly fsCache?: boolean;
	readonly virtualModules?: Readonly<Record<string, unknown>>;
	readonly transform?: (options: JitiTransformOptions) => { readonly code: string };
}

type CreateJiti = (id: string, options?: JitiOptions) => JitiInstance;

let childEnvironmentLease: ChildEnvironmentLease | undefined;

function childEnvironmentFromSpawnOptions(args: readonly unknown[]): NodeJS.ProcessEnv {
	const optionsIndex = Array.isArray(args[1]) ? 2 : 1;
	const options = args[optionsIndex];
	if (!isJsonObject(options) || !isJsonObject(options.env)) return {};
	return options.env as NodeJS.ProcessEnv;
}

function buildExplorerChildEnvironment(
	sourceEnvironment: NodeJS.ProcessEnv,
	context: ExplorerChildEnvironment,
): NodeJS.ProcessEnv {
	const childEnvironment: NodeJS.ProcessEnv = {};
	for (const key of EXPLORER_RUNTIME_ENV_KEYS) {
		const value = sourceEnvironment[key];
		if (typeof value === "string") childEnvironment[key] = value;
	}
	for (const [key, value] of Object.entries(sourceEnvironment)) {
		if (EXPLORER_PI_ENV_KEYS.has(key) && typeof value === "string") childEnvironment[key] = value;
	}
	childEnvironment.PI_CODING_AGENT_DIR = context.agentDir;
	childEnvironment.PI_SUBAGENT_PI_BINARY = context.piBinary;
	if (context.apiKey !== undefined) childEnvironment[context.apiKeyName] = context.apiKey;
	else delete childEnvironment[context.apiKeyName];
	return childEnvironment;
}

function childArgsFromSpawnArguments(args: readonly unknown[]): readonly unknown[] {
	return Array.isArray(args[1]) ? args[1] : [];
}

function isPiChildSpawn(args: readonly unknown[], context: ExplorerChildEnvironment): boolean {
	const command = args[0];
	if (command === "pi" || command === context.piBinary) return true;
	return childArgsFromSpawnArguments(args)[0] === context.piBinary;
}

function isAsyncSubagentRunnerSpawn(args: readonly unknown[]): boolean {
	return childArgsFromSpawnArguments(args).some(
		(argument) => typeof argument === "string" && basename(argument) === "subagent-runner.ts",
	);
}

function createScopedChildProcessModule(context: ExplorerChildEnvironment): Record<string, unknown> {
	const originalSpawn = childProcess.spawn.bind(childProcess) as SpawnFunction;
	const spawnWithExplorerEnvironment: SpawnFunction = (...args) => {
		if (!isPiChildSpawn(args, context) && !isAsyncSubagentRunnerSpawn(args)) return originalSpawn(...args);
		const nextArgs = [...args];
		if (nextArgs[0] === "pi") nextArgs[0] = context.piBinary;
		const optionsIndex = Array.isArray(nextArgs[1]) ? 2 : 1;
		const currentOptions = isJsonObject(nextArgs[optionsIndex]) ? nextArgs[optionsIndex] : {};
		nextArgs[optionsIndex] = {
			...currentOptions,
			env: buildExplorerChildEnvironment(childEnvironmentFromSpawnOptions(nextArgs), context),
		};
		return originalSpawn(...nextArgs);
	};
	const scopedModule = { ...childProcess, spawn: spawnWithExplorerEnvironment };
	return { ...scopedModule, default: scopedModule };
}

function isWithinDirectory(root: string, candidate: string): boolean {
	const relativePath = relative(root, candidate);
	return (
		relativePath === "" ||
		(relativePath !== ".." && !relativePath.startsWith(`..${sep}`) && !isAbsolute(relativePath))
	);
}

async function loadScopedPiSubagentsExtension(
	extensionPath: string,
	environment: ExplorerChildEnvironment,
): Promise<ExtensionFactory> {
	const packageJsonPath = resolvePiSubagentsPackageJson();
	const packageRoot = dirname(packageJsonPath);
	const packageRequire = createRequire(packageJsonPath);
	const jitiModule = packageRequire("jiti") as { readonly createJiti?: unknown };
	if (typeof jitiModule.createJiti !== "function") {
		throw new Error("pi-subagents did not expose its required Jiti loader");
	}
	const createJiti = jitiModule.createJiti as CreateJiti;
	const baseJiti = createJiti(packageJsonPath, { moduleCache: false, fsCache: false });
	const utilsPath = join(packageRoot, "src", "shared", "utils.ts");
	const upstreamUtils = await baseJiti.import<Record<string, unknown>>(utilsPath);
	if (typeof upstreamUtils.getAgentDir !== "function") {
		throw new Error("pi-subagents did not expose its expected getAgentDir utility");
	}
	const scopedUtils = { ...upstreamUtils, getAgentDir: () => environment.agentDir };
	const virtualModules: Record<string, unknown> = {
		"../shared/utils.ts": scopedUtils,
		"../../shared/utils.ts": scopedUtils,
		[PI_SUBAGENTS_CHILD_PROCESS_MODULE]: createScopedChildProcessModule(environment),
	};
	const scopedJiti = createJiti(packageJsonPath, {
		moduleCache: false,
		fsCache: false,
		virtualModules,
		transform: (options) => {
			const source =
				options.filename !== undefined && isWithinDirectory(packageRoot, options.filename)
					? options.source.replace(
							/(["'])node:child_process\1/g,
							JSON.stringify(PI_SUBAGENTS_CHILD_PROCESS_MODULE),
						)
					: options.source;
			return { code: baseJiti.transform({ ...options, source }) };
		},
	});
	const factory = await scopedJiti.import<unknown>(extensionPath, { default: true });
	if (typeof factory !== "function") {
		throw new Error(`pi-subagents extension at ${extensionPath} did not export a factory function`);
	}
	return factory as ExtensionFactory;
}

function isJsonObject(value: unknown): value is JsonObject {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readPiModelsConfig(modelsPath: string): PiModelsConfig {
	if (!existsSync(modelsPath)) return { providers: {} };

	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(modelsPath, "utf8"));
	} catch (error) {
		throw new Error(`AutoRAG could not parse Pi models.json at ${modelsPath}: ${(error as Error).message}`, {
			cause: error,
		});
	}
	if (!isJsonObject(parsed) || !isJsonObject(parsed.providers)) {
		throw new Error(`AutoRAG Pi models.json at ${modelsPath} must contain an object named "providers"`);
	}
	for (const [provider, config] of Object.entries(parsed.providers)) {
		if (!isJsonObject(config)) {
			throw new Error(`AutoRAG Pi models.json provider "${provider}" must be an object`);
		}
	}
	return { providers: parsed.providers as Record<string, JsonObject> };
}

function providerApiKeyEnvName(provider: string): string {
	const normalized = provider.replace(/[^A-Za-z0-9_]/g, "_").toUpperCase();
	const safeName = /^[A-Z_]/.test(normalized) ? normalized : `PROVIDER_${normalized}`;
	return `${safeName || "AUTORAG"}_API_KEY`;
}

function requireNonEmptyString(value: unknown, label: string): string {
	if (typeof value !== "string" || value.trim().length === 0) {
		throw new Error(`AutoRAG cannot persist Pi model metadata: ${label} must be a non-empty string`);
	}
	return value;
}

function validateRoleModel(model: Model<Api>, role: string): void {
	requireNonEmptyString(model.provider, `${role} provider`);
	requireNonEmptyString(model.id, `${role} model id`);
	requireNonEmptyString(model.name, `${role} model name`);
	requireNonEmptyString(model.api, `${role} model api`);
	requireNonEmptyString(model.baseUrl, `${role} model baseUrl`);
	if (typeof model.reasoning !== "boolean") {
		throw new Error(`AutoRAG cannot persist Pi model metadata: ${role} model reasoning must be boolean`);
	}
	if (
		!Array.isArray(model.input) ||
		model.input.length === 0 ||
		model.input.some((input) => input !== "text" && input !== "image")
	) {
		throw new Error(`AutoRAG cannot persist Pi model metadata: ${role} model input must contain only text or image`);
	}
	if (
		!Number.isFinite(model.contextWindow) ||
		model.contextWindow <= 0 ||
		!Number.isFinite(model.maxTokens) ||
		model.maxTokens <= 0
	) {
		throw new Error(`AutoRAG cannot persist Pi model metadata: ${role} model limits must be positive numbers`);
	}
	const costs = model.cost;
	if (
		!costs ||
		!Number.isFinite(costs.input) ||
		!Number.isFinite(costs.output) ||
		!Number.isFinite(costs.cacheRead) ||
		!Number.isFinite(costs.cacheWrite)
	) {
		throw new Error(`AutoRAG cannot persist Pi model metadata: ${role} model cost must contain finite numbers`);
	}
}

function activeRoleModel(model: Model<Api>, role: SubagentModelRole): ActiveRoleModel {
	validateRoleModel(model, role);
	return {
		role,
		model,
		reference: {
			provider: model.provider,
			id: model.id,
			apiKeyReference: providerApiKeyEnvName(model.provider),
		},
	};
}

function validateProviderApiKeyEnvNames(roleModels: readonly ActiveRoleModel[]): void {
	const providersByEnvName = new Map<string, string>();
	for (const { reference } of roleModels) {
		const existingProvider = providersByEnvName.get(reference.apiKeyReference);
		if (existingProvider === undefined) {
			providersByEnvName.set(reference.apiKeyReference, reference.provider);
			continue;
		}
		if (existingProvider !== reference.provider) {
			throw new Error(
				`AutoRAG cannot use active providers "${existingProvider}" and "${reference.provider}": both map to API-key environment variable "${reference.apiKeyReference}"`,
			);
		}
	}
}

function toPiModelDefinition(model: Model<Api>): JsonObject {
	const definition: JsonObject = {
		id: model.id,
		name: model.name,
		api: model.api,
		baseUrl: model.baseUrl,
		reasoning: model.reasoning,
		input: model.input,
		cost: model.cost,
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
	};
	if (model.thinkingLevelMap !== undefined) definition.thinkingLevelMap = model.thinkingLevelMap;
	if (model.compat !== undefined) definition.compat = model.compat;
	// Model headers may contain credentials; provider auth stays environment-backed.
	return definition;
}

function mergeRoleModel(config: PiModelsConfig, roleModel: ActiveRoleModel): void {
	const { model, reference } = roleModel;
	const providerConfig = config.providers[model.provider] ?? {};
	if (!isJsonObject(providerConfig)) {
		throw new Error(`AutoRAG Pi models.json provider "${model.provider}" must be an object`);
	}
	const baseUrl = providerConfig.baseUrl;
	if (baseUrl !== undefined) requireNonEmptyString(baseUrl, `provider "${model.provider}" baseUrl`);
	const api = providerConfig.api;
	if (api !== undefined) requireNonEmptyString(api, `provider "${model.provider}" api`);
	const configuredModels = providerConfig.models;
	if (configuredModels !== undefined && !Array.isArray(configuredModels)) {
		throw new Error(`AutoRAG Pi models.json provider "${model.provider}" has invalid "models" metadata`);
	}
	const models = [...((configuredModels as unknown[] | undefined) ?? [])];
	for (const existingModel of models) {
		if (!isJsonObject(existingModel)) {
			throw new Error(`AutoRAG Pi models.json provider "${model.provider}" contains a non-object model entry`);
		}
	}
	const modelDefinition = toPiModelDefinition(model);
	const existingIndex = models.findIndex(
		(existingModel) => isJsonObject(existingModel) && existingModel.id === model.id,
	);
	if (existingIndex >= 0) {
		models[existingIndex] = { ...(models[existingIndex] as JsonObject), ...modelDefinition };
	} else {
		models.push(modelDefinition);
	}
	config.providers[model.provider] = {
		...providerConfig,
		baseUrl: baseUrl ?? model.baseUrl,
		api: api ?? model.api,
		apiKey: reference.apiKeyReference,
		models,
	};
}

function acquirePiModelsLock(modelsPath: string): FileLockHandle {
	return acquireFileLock(`${modelsPath}.lock`, {
		timeoutMs: MODELS_LOCK_WAIT_TIMEOUT_MS,
		staleMs: MODELS_LOCK_STALE_MS,
		retryMs: MODELS_LOCK_RETRY_MS,
		timeoutError: () => new PiModelsLockTimeoutError(),
	});
}

function persistPiModels(
	modelsPath: string,
	authStorage: AuthStorage,
	roleModels: readonly ActiveRoleModel[],
	providerCredentials: ReadonlyMap<string, string>,
): ModelRegistry {
	const lock = acquirePiModelsLock(modelsPath);
	let temporaryPath: string | undefined;
	try {
		const config = readPiModelsConfig(modelsPath);
		for (const roleModel of roleModels) mergeRoleModel(config, roleModel);
		let content: string;
		try {
			content = `${JSON.stringify(config, null, 2)}\n`;
		} catch (error) {
			throw new Error(`AutoRAG could not serialize Pi models.json at ${modelsPath}: ${(error as Error).message}`, {
				cause: error,
			});
		}
		for (const credential of providerCredentials.values()) {
			if (content.includes(credential)) {
				throw new Error("AutoRAG refused to persist a runtime API credential in Pi models.json");
			}
		}

		temporaryPath = `${modelsPath}.${randomUUID()}.tmp`;
		writeFileSync(temporaryPath, content, { encoding: "utf8", flag: "wx", mode: 0o600 });
		const candidateRegistry = ModelRegistry.create(authStorage, temporaryPath);
		const registryError = candidateRegistry.getError();
		if (registryError) {
			throw new Error(`AutoRAG generated invalid Pi models.json: ${registryError}`);
		}
		for (const { reference } of roleModels) {
			if (!candidateRegistry.find(reference.provider, reference.id)) {
				throw new Error(`AutoRAG generated Pi models.json without ${reference.provider}/${reference.id}`);
			}
		}
		lock.assertOwned();
		renameSync(temporaryPath, modelsPath);
		const modelRegistry = ModelRegistry.create(authStorage, modelsPath);
		const persistedRegistryError = modelRegistry.getError();
		if (persistedRegistryError) {
			throw new Error(`AutoRAG persisted invalid Pi models.json: ${persistedRegistryError}`);
		}
		for (const { reference } of roleModels) {
			if (!modelRegistry.find(reference.provider, reference.id)) {
				throw new Error(`AutoRAG persisted Pi models.json without ${reference.provider}/${reference.id}`);
			}
		}
		return modelRegistry;
	} finally {
		try {
			if (temporaryPath && existsSync(temporaryPath)) unlinkSync(temporaryPath);
		} finally {
			lock.release();
		}
	}
}

function modelRegistryIdentity(modelsPath: string, roleModels: readonly ActiveRoleModel[]): string {
	try {
		return JSON.stringify({
			modelsPath,
			roles: roleModels.map(({ role, model, reference }) => ({
				role,
				reference,
				definition: toPiModelDefinition(model),
			})),
		});
	} catch (error) {
		throw new Error(
			`AutoRAG could not identify the active Pi model registry at ${modelsPath}: ${(error as Error).message}`,
			{
				cause: error,
			},
		);
	}
}

function acquireChildEnvironment(request: ChildEnvironmentLeaseRequest): () => void {
	if (childEnvironmentLease !== undefined) {
		const sameValues =
			childEnvironmentLease.agentDir === request.agentDir &&
			childEnvironmentLease.registryIdentity === request.registryIdentity &&
			childEnvironmentLease.environmentIdentity === request.environmentIdentity;
		if (!sameValues) {
			throw new Error(
				"AutoRAG cannot create concurrent Pi sessions with different child environment/registry routing",
			);
		}
		childEnvironmentLease.owners += 1;
		let released = false;
		return () => {
			if (released) return;
			released = true;
			if (childEnvironmentLease === undefined) return;
			childEnvironmentLease.owners -= 1;
			if (childEnvironmentLease.owners === 0) childEnvironmentLease = undefined;
		};
	}

	childEnvironmentLease = {
		agentDir: request.agentDir,
		registryIdentity: request.registryIdentity,
		environmentIdentity: request.environmentIdentity,
		owners: 1,
	};
	let released = false;
	return () => {
		if (released) return;
		released = true;
		if (childEnvironmentLease === undefined) return;
		childEnvironmentLease.owners -= 1;
		if (childEnvironmentLease.owners === 0) childEnvironmentLease = undefined;
	};
}

function explorerEnvironmentIdentity(environment: ExplorerChildEnvironment): string {
	return JSON.stringify({
		agentDir: environment.agentDir,
		piBinary: environment.piBinary,
		apiKeyName: environment.apiKeyName,
		apiKeyHash:
			environment.apiKey === undefined ? undefined : createHash("sha256").update(environment.apiKey).digest("hex"),
	});
}

function defaultExplorerModel(model: Model<Api>): Model<Api> {
	return { ...model, id: "gpt-5.6-luna", name: "GPT-5.6 Luna" };
}

export async function createMandatorySubagentSession(
	options: MandatorySubagentSessionOptions,
): Promise<MandatorySubagentSession> {
	const agentDir = resolve(options.agentDir ?? defaultAgentDir());
	mkdirSync(agentDir, { recursive: true });
	ensureMandatoryExplorerAgent(agentDir);
	const piBinary = resolveSubagentPiBinary();
	const extensionPath = options.extensionPath ?? resolvePiSubagentsExtension();
	const explorerModel = options.explorerModel ?? defaultExplorerModel(options.model);
	const parentProviderCredentials = new Map<string, string>();
	for (const [provider, credential] of Object.entries(options.providerApiKeys ?? {})) {
		if (credential.length === 0) throw new Error(`AutoRAG received an empty credential for provider "${provider}"`);
		parentProviderCredentials.set(provider, credential);
	}
	const explicitOrchestratorCredential = parentProviderCredentials.get(options.model.provider);
	if (
		explicitOrchestratorCredential !== undefined &&
		options.apiKey !== undefined &&
		explicitOrchestratorCredential !== options.apiKey
	) {
		throw new Error(`AutoRAG received different credentials for provider "${options.model.provider}"`);
	}
	if (options.apiKey !== undefined) parentProviderCredentials.set(options.model.provider, options.apiKey);
	const customTools = options.tools.map(asToolDefinition);
	const orchestratorRoleModel = activeRoleModel(options.model, "orchestrator");
	const explorerRoleModel = activeRoleModel(explorerModel, "explorer");
	const roleModels: readonly ActiveRoleModel[] = [orchestratorRoleModel, explorerRoleModel];
	validateProviderApiKeyEnvNames(roleModels);
	const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
	for (const [provider, credential] of parentProviderCredentials) authStorage.setRuntimeApiKey(provider, credential);
	for (const { reference } of roleModels) {
		const credential = await authStorage.getApiKey(reference.provider);
		if (credential === undefined) continue;
		authStorage.setRuntimeApiKey(reference.provider, credential);
		parentProviderCredentials.set(reference.provider, credential);
	}
	const modelsPath = join(agentDir, "models.json");
	const explorerCredential = parentProviderCredentials.get(explorerRoleModel.reference.provider);
	const explorerEnvironment: ExplorerChildEnvironment = {
		agentDir,
		piBinary,
		apiKeyName: explorerRoleModel.reference.apiKeyReference,
		...(explorerCredential === undefined ? {} : { apiKey: explorerCredential }),
	};
	let extensionFactory: ExtensionFactory;
	try {
		extensionFactory = await loadScopedPiSubagentsExtension(extensionPath, explorerEnvironment);
	} catch (error) {
		throw new Error(`Mandatory pi-subagents extension failed to load: ${(error as Error).message}`, {
			cause: error,
		});
	}
	const resourceLoader = new DefaultResourceLoader({
		cwd: options.cwd,
		agentDir,
		extensionFactories: [extensionFactory],
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
		systemPrompt: options.systemPrompt,
	});
	try {
		await resourceLoader.reload();
	} catch (error) {
		throw new Error(`Mandatory pi-subagents extension failed to load: ${(error as Error).message}`, {
			cause: error,
		});
	}
	const extensionResult = resourceLoader.getExtensions();
	if (extensionResult.errors.length > 0) {
		const messages = extensionResult.errors.map((error) => error.error).join("; ");
		throw new Error(`Mandatory pi-subagents extension failed to load: ${messages}`);
	}
	const releaseChildEnvironment = acquireChildEnvironment({
		agentDir,
		registryIdentity: modelRegistryIdentity(modelsPath, roleModels),
		environmentIdentity: explorerEnvironmentIdentity(explorerEnvironment),
	});
	try {
		const modelRegistry = persistPiModels(modelsPath, authStorage, roleModels, parentProviderCredentials);
		ensurePiSettingsFile(agentDir);
		const settingsManager = SettingsManager.create(options.cwd, agentDir);
		const sessionManager = SessionManager.create(
			options.cwd,
			options.sessionDir ?? defaultSessionDir(options.cwd, agentDir),
		);
		sessionManager.appendSessionInfo("AutoRAG orchestrator");
		const { session } = await createAgentSession({
			cwd: options.cwd,
			agentDir,
			model: options.model,
			authStorage,
			modelRegistry,
			settingsManager,
			thinkingLevel: "high",
			resourceLoader,
			sessionManager,
			noTools: "builtin",
			customTools,
		});
		const dispose = session.dispose.bind(session);
		let disposed = false;
		session.dispose = () => {
			if (disposed) return;
			disposed = true;
			try {
				dispose();
			} finally {
				releaseChildEnvironment();
			}
		};
		const requiredTools = ["subagent", "wait"];
		const allToolNames = new Set(session.getAllTools().map((tool) => tool.name));
		const missing = requiredTools.filter((name) => !allToolNames.has(name));
		if (missing.length > 0) {
			session.dispose();
			throw new Error(`Mandatory pi-subagents extension did not register tools: ${missing.join(", ")}`);
		}
		session.setActiveToolsByName([...customTools.map((tool) => tool.name), ...requiredTools]);
		return { session, extensionPath };
	} catch (error) {
		releaseChildEnvironment();
		throw error;
	}
}

/**
 * Create a short-lived subagent session for the `autorag health` explorer probe.
 *
 * Unlike {@link createMandatorySubagentSession}, this requires absolute
 * `agentDir` and `sessionDir` (typically under a caller-created temp
 * directory) so the probe never touches durable home state. On a lease
 * conflict — i.e. when another live session holds a different child
 * environment/registry routing — the error is rethrown with a stable
 * "concurrent AutoRAG session busy" message so the health command can map
 * it to `subagent_failed`.
 */
export async function createHealthSubagentProbeSession(
	options: HealthSubagentProbeSessionOptions,
): Promise<HealthSubagentProbeSession> {
	if (typeof options.agentDir !== "string" || options.agentDir.trim().length === 0) {
		throw new Error("AutoRAG health probe requires an absolute agentDir");
	}
	if (typeof options.sessionDir !== "string" || options.sessionDir.trim().length === 0) {
		throw new Error("AutoRAG health probe requires an absolute sessionDir");
	}
	if (!isAbsolute(options.agentDir)) {
		throw new Error("AutoRAG health probe requires an absolute agentDir");
	}
	if (!isAbsolute(options.sessionDir)) {
		throw new Error("AutoRAG health probe requires an absolute sessionDir");
	}
	try {
		const result = await createMandatorySubagentSession({
			cwd: options.cwd,
			model: options.model,
			...(options.explorerModel !== undefined ? { explorerModel: options.explorerModel } : {}),
			systemPrompt: options.systemPrompt,
			tools: options.tools,
			...(options.extensionPath !== undefined ? { extensionPath: options.extensionPath } : {}),
			...(options.apiKey !== undefined ? { apiKey: options.apiKey } : {}),
			...(options.providerApiKeys !== undefined ? { providerApiKeys: options.providerApiKeys } : {}),
			agentDir: options.agentDir,
			sessionDir: options.sessionDir,
		});
		return {
			session: result.session,
			extensionPath: result.extensionPath,
			dispose: () => result.session.dispose(),
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		if (message.includes("concurrent Pi sessions")) {
			throw new Error("concurrent AutoRAG session busy with different child environment routing");
		}
		throw error;
	}
}
