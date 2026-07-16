import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Api, completeSimple, type Model } from "@earendil-works/pi-ai";
import type { LoadLocalAutoRAGModelsOptions } from "../../subagents/local-models.ts";
import { createHealthSubagentProbeSession } from "../../subagents/runtime.ts";
import {
	type CliConfig,
	type ResolvedAgentModelDetailed,
	resolveAgentModelDetailed,
	resolveConfigReadOnly,
} from "../config.ts";
import { renderHealth } from "../output.ts";
import type { CommandContext } from "./types.ts";

// ---------------------------------------------------------------------------
// Health report types (healthSchemaVersion: 1)
// ---------------------------------------------------------------------------

export type HealthCategory =
	| "config"
	| "model_resolution"
	| "auth_missing"
	| "provider_unreachable"
	| "completion_failed"
	| "subagent_failed"
	| "timeout"
	| "ok";

export type HealthRole = "orchestrator" | "explorer";

export interface HealthCoverage {
	modelProvider: boolean;
	subagentDispatch: boolean;
	retrievalTools: false;
	searchCuration: false;
	indexHealth: false;
}

export interface HealthRoleAuth {
	present: boolean;
	source: "env" | "local_runtime" | "pi_auth" | "catalog" | "none" | "unknown";
	envName?: string;
}

export interface HealthRoleReport {
	role: HealthRole;
	provider: string;
	modelId: string;
	displayName?: string;
	api: string;
	routeFamily?: string;
	baseUrl?: string;
	contextWindow?: number;
	maxTokens?: number;
	capabilities: { text: boolean; image: boolean; reasoning?: boolean };
	auth: HealthRoleAuth;
	resolutionSource: "config" | "flags" | "env" | "local_runtime" | "catalog" | "configured_alias" | "mixed";
}

export interface HealthProbeReport {
	role: HealthRole;
	skipped: boolean;
	ok: boolean;
	category: HealthCategory;
	durationMs?: number;
	message?: string;
}

export interface HealthConfigReport {
	ok: boolean;
	source: "explicit" | "home" | "legacy" | "defaults";
	message?: string;
}

export interface HealthReportV1 {
	healthSchemaVersion: 1;
	ok: boolean;
	category: HealthCategory;
	command: "health";
	probesSkipped: boolean;
	coverage: HealthCoverage;
	config: HealthConfigReport;
	models: { orchestrator?: HealthRoleReport; explorer?: HealthRoleReport };
	probes: { orchestrator?: HealthProbeReport; explorer?: HealthProbeReport };
	indexHealth: { separate: true; command: "autorag status"; included: false };
}

// ---------------------------------------------------------------------------
// Dependency injection
// ---------------------------------------------------------------------------

/**
 * Probe input passed to injected orchestrator/explorer probes. Credential
 * values are present for the probe to use but must never be surfaced in
 * rendered output.
 */
export interface ProbeInput {
	role: HealthRole;
	model: Model<Api>;
	apiKey?: string;
	providerApiKeys?: Readonly<Record<string, string>>;
	timeoutMs: number;
	cwd: string;
}

/**
 * Probe outcome returned by injected probes. `category` is the per-role
 * health category; `message` is already sanitized (no secrets/paths/stacks).
 */
export interface ProbeOutput {
	ok: boolean;
	category: HealthCategory;
	message?: string;
}

/**
 * Detailed role metadata produced by the model resolver. The default
 * adapter derives this from {@link resolveAgentModel}; when config exports a
 * dedicated `resolveAgentModelDetailed`, the health command prefers it.
 */
export interface ResolvedHealthRole {
	readonly provider: string;
	readonly modelId: string;
	readonly displayName?: string;
	readonly api: Api;
	readonly baseUrl?: string;
	readonly contextWindow?: number;
	readonly maxTokens?: number;
	readonly capabilities: { input: readonly string[]; reasoning: boolean };
	readonly auth: {
		present: boolean;
		source: "env" | "local_runtime" | "pi_auth" | "catalog" | "none" | "unknown";
		envName?: string;
	};
	readonly resolutionSource: "config" | "flags" | "env" | "local_runtime" | "catalog" | "configured_alias" | "mixed";
}

export interface ResolvedHealthModel {
	readonly model: Model<Api>;
	readonly explorerModel: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly roles: { orchestrator: ResolvedHealthRole; explorer: ResolvedHealthRole };
}

export interface HealthDeps {
	configResolver?: (input: {
		flags: Record<string, string | boolean | undefined>;
		cwd?: string;
		env?: NodeJS.ProcessEnv;
	}) => CliConfig;
	modelResolver?: (config: CliConfig, localOptions?: LoadLocalAutoRAGModelsOptions) => ResolvedHealthModel;
	orchestratorProbe?: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>;
	explorerProbe?: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>;
	now?: () => number;
}

// ---------------------------------------------------------------------------
// Default adapters (shared config resolution)
// ---------------------------------------------------------------------------

function defaultResolveConfig(input: {
	flags: Record<string, string | boolean | undefined>;
	cwd?: string;
	env?: NodeJS.ProcessEnv;
}): CliConfig {
	return resolveConfigReadOnly({ flags: input.flags, cwd: input.cwd, env: input.env });
}

function defaultResolveModel(config: CliConfig, localOptions: LoadLocalAutoRAGModelsOptions = {}): ResolvedHealthModel {
	const resolved: ResolvedAgentModelDetailed = resolveAgentModelDetailed(config, localOptions);
	return {
		model: resolved.model,
		explorerModel: resolved.explorerModel,
		...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
		...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
		roles: {
			orchestrator: {
				provider: resolved.roles.orchestrator.provider,
				modelId: resolved.roles.orchestrator.modelId,
				displayName: resolved.roles.orchestrator.displayName,
				api: resolved.roles.orchestrator.api,
				baseUrl: resolved.roles.orchestrator.baseUrl,
				contextWindow: resolved.roles.orchestrator.contextWindow,
				maxTokens: resolved.roles.orchestrator.maxTokens,
				capabilities: {
					input: resolved.roles.orchestrator.capabilities.input,
					reasoning: resolved.roles.orchestrator.capabilities.reasoning,
				},
				auth: resolved.roles.orchestrator.auth,
				resolutionSource: resolved.roles.orchestrator.resolutionSource,
			},
			explorer: {
				provider: resolved.roles.explorer.provider,
				modelId: resolved.roles.explorer.modelId,
				displayName: resolved.roles.explorer.displayName,
				api: resolved.roles.explorer.api,
				baseUrl: resolved.roles.explorer.baseUrl,
				contextWindow: resolved.roles.explorer.contextWindow,
				maxTokens: resolved.roles.explorer.maxTokens,
				capabilities: {
					input: resolved.roles.explorer.capabilities.input,
					reasoning: resolved.roles.explorer.capabilities.reasoning,
				},
				auth: resolved.roles.explorer.auth,
				resolutionSource: resolved.roles.explorer.resolutionSource,
			},
		},
	};
}

// ---------------------------------------------------------------------------
// Category precedence + exit code mapping
// ---------------------------------------------------------------------------

const CATEGORY_PRECEDENCE: readonly HealthCategory[] = [
	"config",
	"model_resolution",
	"auth_missing",
	"timeout",
	"provider_unreachable",
	"completion_failed",
	"subagent_failed",
	"ok",
];

function categoryRank(category: HealthCategory): number {
	return CATEGORY_PRECEDENCE.indexOf(category);
}

function aggregateCategory(categories: readonly HealthCategory[]): HealthCategory {
	let best: HealthCategory = "ok";
	for (const category of categories) {
		if (categoryRank(category) < categoryRank(best)) best = category;
	}
	return best;
}

function exitCodeFor(category: HealthCategory): number {
	switch (category) {
		case "config":
		case "model_resolution":
			return 2;
		case "auth_missing":
		case "provider_unreachable":
		case "completion_failed":
		case "subagent_failed":
		case "timeout":
			return 1;
		case "ok":
			return 0;
	}
}

// ---------------------------------------------------------------------------
// Sanitization
// ---------------------------------------------------------------------------

const SECRET_PATTERNS = [
	/[A-Za-z0-9_-]{20,}/g, // long token-like substrings
];

/**
 * Sanitize a message for safe rendering: strip stack frames, absolute paths,
 * and credential-like substrings. Never emits env values or temp paths.
 */
function sanitizeMessage(raw: string): string {
	let out = raw;
	// Drop stack-trace frames (everything after the first newline if it looks
	// like a stack, and explicit "at " frames).
	const newlineIdx = out.indexOf("\n");
	if (newlineIdx !== -1) {
		const tail = out.slice(newlineIdx + 1);
		if (tail.includes("    at ") || tail.includes("\tat ")) {
			out = out.slice(0, newlineIdx);
		}
	}
	out = out.replace(/\s+at\s+.*/g, "");
	// Strip absolute filesystem paths (unix + windows).
	out = out.replace(/(?:^|[^A-Za-z0-9])(\/(?:[^/\s]+\/)+[^/\s]+)/g, " <path>");
	out = out.replace(/[A-Za-z]:\\[^\s]+/g, "<path>");
	// Strip credential-like long tokens.
	for (const pattern of SECRET_PATTERNS) out = out.replace(pattern, "<redacted>");
	// Collapse repeated whitespace.
	out = out.replace(/\s{2,}/g, " ").trim();
	return out;
}

// ---------------------------------------------------------------------------
// Timeout parsing
// ---------------------------------------------------------------------------

const DEFAULT_TIMEOUT_MS = 10_000;

function parseTimeoutMs(flags: Record<string, string | boolean | undefined>): number | { error: string } {
	const raw = flags["timeout-ms"];
	if (raw === undefined || raw === true) return DEFAULT_TIMEOUT_MS;
	const text = typeof raw === "string" ? raw.trim() : "";
	if (text.length === 0) return { error: "--timeout-ms requires a value" };
	if (!/^\d+$/.test(text)) return { error: `--timeout-ms must be a positive integer (got "${text}")` };
	const value = Number(text);
	if (!Number.isFinite(value) || value <= 0) {
		return { error: `--timeout-ms must be a positive integer (got "${text}")` };
	}
	return value;
}

// ---------------------------------------------------------------------------
// Report builders
// ---------------------------------------------------------------------------

function emptyCoverage(): HealthCoverage {
	return {
		modelProvider: false,
		subagentDispatch: false,
		retrievalTools: false,
		searchCuration: false,
		indexHealth: false,
	};
}

function buildConfigReport(error?: unknown): HealthConfigReport {
	if (error === undefined) {
		return { ok: true, source: "defaults" };
	}
	const message = sanitizeMessage(error instanceof Error ? error.message : String(error));
	return { ok: false, source: "defaults", message };
}

function buildRoleReport(role: HealthRole, resolved: ResolvedHealthRole): HealthRoleReport {
	const input = resolved.capabilities.input;
	return {
		role,
		provider: resolved.provider,
		modelId: resolved.modelId,
		...(resolved.displayName !== undefined ? { displayName: resolved.displayName } : {}),
		api: resolved.api,
		...(resolved.baseUrl !== undefined ? { baseUrl: sanitizeBaseUrl(resolved.baseUrl) } : {}),
		...(resolved.contextWindow !== undefined ? { contextWindow: resolved.contextWindow } : {}),
		...(resolved.maxTokens !== undefined ? { maxTokens: resolved.maxTokens } : {}),
		capabilities: {
			text: input.includes("text"),
			image: input.includes("image"),
			...(resolved.capabilities.reasoning ? { reasoning: true } : {}),
		},
		auth: {
			present: resolved.auth.present,
			source: resolved.auth.source,
			...(resolved.auth.envName !== undefined ? { envName: resolved.auth.envName } : {}),
		},
		resolutionSource: resolved.resolutionSource,
	};
}

function sanitizeBaseUrl(url: string): string {
	try {
		const parsed = new URL(url);
		// Strip userinfo, query, hash — keep scheme/host/path only.
		return `${parsed.protocol}//${parsed.host}${parsed.pathname.replace(/\/$/, "")}`;
	} catch {
		return "<redacted>";
	}
}

function skippedProbe(role: HealthRole): HealthProbeReport {
	return { role, skipped: true, ok: false, category: "ok" };
}

// ---------------------------------------------------------------------------
// Default probes
// ---------------------------------------------------------------------------

const AUTH_ERROR_PATTERNS: readonly string[] = [
	"api key",
	"apikey",
	"401",
	"403",
	"unauthorized",
	"forbidden",
	"authentication",
];

const NETWORK_ERROR_CODES: readonly string[] = ["ENOTFOUND", "ECONNREFUSED", "ECONNRESET", "ETIMEDOUT"];

const NETWORK_ERROR_PATTERNS: readonly string[] = [
	"fetch failed",
	"connection error",
	"network error",
	"socket hang up",
];

/**
 * Classify a completion error (thrown or returned via stopReason "error"/"aborted")
 * into a health category. Never includes secrets in the returned message.
 */
function classifyCompletionError(error: unknown): { category: HealthCategory; message: string } {
	const name = error instanceof Error ? error.name : "";
	const rawMessage = error instanceof Error ? error.message : String(error);
	const lowerMessage = rawMessage.toLowerCase();

	// AbortError / timeout → timeout.
	if (name === "AbortError" || /abort|timeout|timed out/i.test(rawMessage)) {
		return { category: "timeout", message: "orchestrator probe timed out" };
	}

	// Auth / 401 / 403 / missing key → auth_missing.
	for (const pattern of AUTH_ERROR_PATTERNS) {
		if (lowerMessage.includes(pattern)) {
			return { category: "auth_missing", message: "orchestrator API key rejected or missing" };
		}
	}

	// Network / provider unreachable → provider_unreachable.
	if (typeof error === "object" && error !== null && "code" in error) {
		const code = String((error as { code?: unknown }).code ?? "");
		if (NETWORK_ERROR_CODES.includes(code)) {
			return { category: "provider_unreachable", message: "orchestrator provider network error" };
		}
	}
	for (const pattern of NETWORK_ERROR_PATTERNS) {
		if (lowerMessage.includes(pattern)) {
			return { category: "provider_unreachable", message: "orchestrator provider network error" };
		}
	}

	return { category: "completion_failed", message: "orchestrator completion failed" };
}

/**
 * Default orchestrator probe: issues a minimal completion request via
 * `complete` from @earendil-works/pi-ai. Success = any non-error completion.
 * Never includes secrets in the returned message.
 */
async function defaultOrchestratorProbe(input: ProbeInput, signal: AbortSignal): Promise<ProbeOutput> {
	try {
		const apiKey =
			input.apiKey ??
			(input.providerApiKeys !== undefined ? input.providerApiKeys[input.model.provider] : undefined);
		const result = await completeSimple(
			input.model,
			{ messages: [{ role: "user", content: "Reply with OK.", timestamp: Date.now() }] },
			{
				...(apiKey !== undefined ? { apiKey } : {}),
				signal,
				maxTokens: 64,
				...(input.model.reasoning === true ? { reasoning: "low" as const } : {}),
			},
		);
		if (result.stopReason === "error" || result.stopReason === "aborted") {
			const classified = classifyCompletionError(
				new Error(result.errorMessage ?? `completion ${result.stopReason}`),
			);
			return { ok: false, category: classified.category, message: classified.message };
		}
		// Any non-error completion counts as provider reachability success.
		return { ok: true, category: "ok" };
	} catch (error) {
		const classified = classifyCompletionError(error);
		return { ok: false, category: classified.category, message: classified.message };
	}
}

const HEALTH_EXPLORER_SYSTEM_PROMPT = "You are a health probe. Reply with HEALTH_OK. Do not read any files.";

/**
 * Default explorer probe: creates a short-lived subagent session under a temp
 * directory and invokes the `autorag-explorer` agent with a trivial task.
 * Cleans up the temp directory in a finally block. Never prints temp paths.
 * Maps lease conflicts to `subagent_failed` with a "concurrent AutoRAG session
 * busy" message.
 */
async function defaultExplorerProbe(input: ProbeInput, signal: AbortSignal): Promise<ProbeOutput> {
	const tempRoot = mkdtempSync(join(tmpdir(), "autorag-health-"));
	try {
		const agentDir = join(tempRoot, "agent");
		const sessionDir = join(tempRoot, "sessions");
		const providerApiKeys = input.providerApiKeys;
		const modelRef = `${input.model.provider}/${input.model.id}`;
		let probeSession: Awaited<ReturnType<typeof createHealthSubagentProbeSession>> | undefined;
		try {
			probeSession = await createHealthSubagentProbeSession({
				cwd: input.cwd,
				model: input.model,
				explorerModel: input.model,
				systemPrompt: HEALTH_EXPLORER_SYSTEM_PROMPT,
				tools: [],
				...(input.apiKey !== undefined ? { apiKey: input.apiKey } : {}),
				...(providerApiKeys !== undefined ? { providerApiKeys } : {}),
				agentDir,
				sessionDir,
			});
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			if (message.includes("concurrent AutoRAG session busy")) {
				return { ok: false, category: "subagent_failed", message: "concurrent AutoRAG session busy" };
			}
			return { ok: false, category: "subagent_failed", message: "explorer subagent session failed" };
		}
		try {
			const subagentTool = probeSession.session.getToolDefinition("subagent");
			if (subagentTool === undefined) {
				return { ok: false, category: "subagent_failed", message: "explorer subagent tool not registered" };
			}
			const dispatch = subagentTool.execute(
				"health-probe",
				{
					agent: "autorag-explorer",
					agentScope: "user",
					model: modelRef,
					artifacts: false,
					task: "Return the exact token HEALTH_OK and nothing else. Do not read files.",
				},
				signal,
				undefined,
				probeSession.session.extensionRunner.createContext(),
			);
			const result = await dispatch;
			const text = result.content
				.filter((item): item is { type: "text"; text: string } => item.type === "text")
				.map((item) => item.text)
				.join(" ");
			// Real subagent dispatch success is the health gate; exact marker is best-effort.
			const details = (result as { details?: { status?: string; error?: string } }).details;
			const failed =
				(typeof details?.status === "string" && /fail|error/i.test(details.status)) ||
				(typeof details?.error === "string" && details.error.length > 0);
			if (!failed) {
				return {
					ok: true,
					category: "ok",
					...(text.includes("HEALTH_OK") ? {} : { message: "explorer subagent dispatched" }),
				};
			}
			return { ok: false, category: "subagent_failed", message: "explorer subagent dispatch failed" };
		} finally {
			probeSession.dispose();
		}
	} catch (error) {
		if (signal.aborted) {
			return { ok: false, category: "timeout", message: "explorer probe timed out" };
		}
		const message = error instanceof Error ? error.message : String(error);
		if (message.includes("concurrent AutoRAG session busy")) {
			return { ok: false, category: "subagent_failed", message: "concurrent AutoRAG session busy" };
		}
		return { ok: false, category: "subagent_failed", message: "explorer subagent failed" };
	} finally {
		rmSync(tempRoot, { recursive: true, force: true });
	}
}

// ---------------------------------------------------------------------------
// Probe execution
// ---------------------------------------------------------------------------

async function runProbe(
	role: HealthRole,
	probe: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>,
	input: ProbeInput,
	now: () => number,
): Promise<HealthProbeReport> {
	const start = now();
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), input.timeoutMs);
	try {
		const out = await probe(input, controller.signal);
		return {
			role,
			skipped: false,
			ok: out.ok,
			category: out.category,
			durationMs: now() - start,
			...(out.message !== undefined ? { message: sanitizeMessage(out.message) } : {}),
		};
	} catch (error) {
		const isAbort = error instanceof Error && (error.name === "AbortError" || /abort/i.test(error.message));
		return {
			role,
			skipped: false,
			ok: false,
			category: isAbort ? "timeout" : "completion_failed",
			durationMs: now() - start,
			message: sanitizeMessage(error instanceof Error ? error.message : String(error)),
		};
	} finally {
		clearTimeout(timer);
	}
}

// ---------------------------------------------------------------------------
// runHealth
// ---------------------------------------------------------------------------

/**
 * Run the `autorag health` command. Returns an exit code and writes the
 * rendered health report to `ctx.stdout` (or an error to `ctx.stderr`).
 *
 * All network/runtime behavior is injectable via `deps`; unit tests inject
 * fake resolvers/probes and never touch live networks.
 */
export async function runHealth(ctx: CommandContext, deps: HealthDeps = {}): Promise<number> {
	const now = deps.now ?? (() => Date.now());
	const configResolver = deps.configResolver ?? defaultResolveConfig;
	const modelResolver = deps.modelResolver ?? defaultResolveModel;
	const skipProbes = ctx.flags["skip-probes"] === true;

	// 1. Parse --timeout-ms (invalid => config, exit 2).
	const timeoutResult = parseTimeoutMs(ctx.flags);
	if (typeof timeoutResult !== "number") {
		const report: HealthReportV1 = {
			healthSchemaVersion: 1,
			ok: false,
			category: "config",
			command: "health",
			probesSkipped: true,
			coverage: emptyCoverage(),
			config: { ok: false, source: "defaults", message: timeoutResult.error },
			models: {},
			probes: {},
			indexHealth: { separate: true, command: "autorag status", included: false },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return exitCodeFor("config");
	}
	const timeoutMs = timeoutResult;

	// 2. Resolve config (read-only).
	let config: CliConfig;
	let configReport: HealthConfigReport;
	try {
		config = configResolver({ flags: ctx.flags, cwd: ctx.cwd });
		configReport = { ok: true, source: "defaults" };
	} catch (error) {
		configReport = buildConfigReport(error);
		const report: HealthReportV1 = {
			healthSchemaVersion: 1,
			ok: false,
			category: "config",
			command: "health",
			probesSkipped: true,
			coverage: emptyCoverage(),
			config: configReport,
			models: {},
			probes: {},
			indexHealth: { separate: true, command: "autorag status", included: false },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return exitCodeFor("config");
	}

	// 3. Resolve detailed role models.
	let resolvedModel: ResolvedHealthModel;
	let modelReports: { orchestrator?: HealthRoleReport; explorer?: HealthRoleReport } = {};
	try {
		resolvedModel = modelResolver(config);
		modelReports = {
			orchestrator: buildRoleReport("orchestrator", resolvedModel.roles.orchestrator),
			explorer: buildRoleReport("explorer", resolvedModel.roles.explorer),
		};
	} catch (error) {
		const message = sanitizeMessage(error instanceof Error ? error.message : String(error));
		const report: HealthReportV1 = {
			healthSchemaVersion: 1,
			ok: false,
			category: "model_resolution",
			command: "health",
			probesSkipped: true,
			coverage: emptyCoverage(),
			config: configReport,
			models: {},
			probes: {
				orchestrator: {
					role: "orchestrator",
					skipped: true,
					ok: false,
					category: "model_resolution",
					message,
				},
			},
			indexHealth: { separate: true, command: "autorag status", included: false },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return exitCodeFor("model_resolution");
	}

	// 4. Check auth presence for both roles.
	const orchAuthMissing = !resolvedModel.roles.orchestrator.auth.present;
	const explorerAuthMissing = !resolvedModel.roles.explorer.auth.present;
	const anyAuthMissing = orchAuthMissing || explorerAuthMissing;

	const coverage = emptyCoverage();
	coverage.modelProvider = !anyAuthMissing;

	if (anyAuthMissing) {
		const report: HealthReportV1 = {
			healthSchemaVersion: 1,
			ok: false,
			category: "auth_missing",
			command: "health",
			probesSkipped: skipProbes,
			coverage,
			config: configReport,
			models: modelReports,
			probes: {
				orchestrator: orchAuthMissing
					? { role: "orchestrator", skipped: true, ok: false, category: "auth_missing" }
					: skippedProbe("orchestrator"),
				explorer: explorerAuthMissing
					? { role: "explorer", skipped: true, ok: false, category: "auth_missing" }
					: skippedProbe("explorer"),
			},
			indexHealth: { separate: true, command: "autorag status", included: false },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return exitCodeFor("auth_missing");
	}

	// 5. --skip-probes: exit 0 only if config/model/auth all ok.
	if (skipProbes) {
		const report: HealthReportV1 = {
			healthSchemaVersion: 1,
			ok: true,
			category: "ok",
			command: "health",
			probesSkipped: true,
			coverage,
			config: configReport,
			models: modelReports,
			probes: {
				orchestrator: skippedProbe("orchestrator"),
				explorer: skippedProbe("explorer"),
			},
			indexHealth: { separate: true, command: "autorag status", included: false },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return exitCodeFor("ok");
	}

	// 6. Run probes serially: orchestrator then explorer.
	const orchProbe = await runProbe(
		"orchestrator",
		deps.orchestratorProbe ?? defaultOrchestratorProbe,
		{
			role: "orchestrator",
			model: resolvedModel.model,
			...(resolvedModel.apiKey !== undefined ? { apiKey: resolvedModel.apiKey } : {}),
			...(resolvedModel.providerApiKeys !== undefined ? { providerApiKeys: resolvedModel.providerApiKeys } : {}),
			timeoutMs,
			cwd: ctx.cwd,
		},
		now,
	);

	const probeCwd = config.searchPaths.length > 0 ? config.searchPaths[0] : process.cwd();

	const explorerProbe = await runProbe(
		"explorer",
		deps.explorerProbe ?? defaultExplorerProbe,
		{
			role: "explorer",
			model: resolvedModel.explorerModel,
			...(resolvedModel.providerApiKeys !== undefined ? { providerApiKeys: resolvedModel.providerApiKeys } : {}),
			timeoutMs,
			cwd: probeCwd,
		},
		now,
	);

	// coverage.subagentDispatch is true only when the explorer probe ran and passed.
	coverage.subagentDispatch = !explorerProbe.skipped && explorerProbe.ok;

	const categories: HealthCategory[] = [orchProbe.category, explorerProbe.category];
	const topCategory = aggregateCategory(categories);

	const report: HealthReportV1 = {
		healthSchemaVersion: 1,
		ok: topCategory === "ok",
		category: topCategory,
		command: "health",
		probesSkipped: false,
		coverage,
		config: configReport,
		models: modelReports,
		probes: { orchestrator: orchProbe, explorer: explorerProbe },
		indexHealth: { separate: true, command: "autorag status", included: false },
	};
	ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
	return exitCodeFor(topCategory);
}
