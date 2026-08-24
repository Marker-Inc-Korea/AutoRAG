import type { Api, Model } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import type { LoadLocalAutoRAGModelOptions } from "../../agent/local-model.ts";
import {
	type CliConfig,
	type ResolvedAgentModelDetailed,
	resolveAgentModelDetailed,
	resolveConfigReadOnly,
} from "../config.ts";
import { renderHealth } from "../output.ts";
import type { CommandContext } from "./types.ts";

export type HealthCategory =
	| "config"
	| "model_resolution"
	| "auth_missing"
	| "provider_unreachable"
	| "completion_failed"
	| "timeout"
	| "ok";

export interface HealthCoverage {
	modelProvider: boolean;
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
	provider: string;
	modelId: string;
	displayName?: string;
	api: string;
	baseUrl?: string;
	contextWindow?: number;
	maxTokens?: number;
	capabilities: { text: boolean; image: boolean; reasoning?: boolean };
	auth: HealthRoleAuth;
	resolutionSource: "config" | "flags" | "env" | "local_runtime" | "catalog" | "configured_alias" | "mixed";
}

export interface HealthProbeReport {
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
	model?: HealthRoleReport;
	probe?: HealthProbeReport;
	indexHealth: { separate: true; command: "autorag status"; included: false };
}

export interface ProbeInput {
	model: Model<Api>;
	apiKey?: string;
	providerApiKeys?: Readonly<Record<string, string>>;
	timeoutMs: number;
	cwd: string;
}

export interface ProbeOutput {
	ok: boolean;
	category: HealthCategory;
	message?: string;
}

export type ResolvedHealthRole = ResolvedAgentModelDetailed["role"];

export interface ResolvedHealthModel {
	readonly model: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly role: ResolvedHealthRole;
}

export interface HealthDeps {
	configResolver?: (input: {
		flags: Record<string, string | boolean | undefined>;
		cwd?: string;
		env?: NodeJS.ProcessEnv;
	}) => CliConfig;
	modelResolver?: (config: CliConfig, localOptions?: LoadLocalAutoRAGModelOptions) => ResolvedHealthModel;
	probe?: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>;
	now?: () => number;
}

const DEFAULT_TIMEOUT_MS = 10_000;

function sanitizeMessage(raw: string): string {
	let out = raw.split(/\n\s+at\s/)[0] ?? raw;
	out = out.replace(/(?:^|[^A-Za-z0-9])(\/(?:[^/\s]+\/)+[^/\s]+)/g, " <path>");
	out = out.replace(/[A-Za-z]:\\[^\s]+/g, "<path>");
	out = out.replace(/[A-Za-z0-9_-]{20,}/g, "<redacted>");
	return out.replace(/\s{2,}/g, " ").trim();
}

function parseTimeoutMs(flags: Record<string, string | boolean | undefined>): number | string {
	const raw = flags["timeout-ms"];
	if (raw === undefined || raw === true) return DEFAULT_TIMEOUT_MS;
	const text = typeof raw === "string" ? raw.trim() : "";
	if (!/^\d+$/.test(text) || Number(text) <= 0) {
		return `--timeout-ms must be a positive integer (got "${text}")`;
	}
	return Number(text);
}

function coverage(modelProvider = false): HealthCoverage {
	return { modelProvider, retrievalTools: false, searchCuration: false, indexHealth: false };
}

function exitCode(category: HealthCategory): number {
	if (category === "ok") return 0;
	if (category === "config" || category === "model_resolution") return 2;
	return 1;
}

function roleReport(role: ResolvedHealthRole): HealthRoleReport {
	return {
		provider: role.provider,
		modelId: role.modelId,
		displayName: role.displayName,
		api: role.api,
		baseUrl: role.baseUrl,
		contextWindow: role.contextWindow,
		maxTokens: role.maxTokens,
		capabilities: {
			text: role.capabilities.input.includes("text"),
			image: role.capabilities.input.includes("image"),
			...(role.capabilities.reasoning ? { reasoning: true } : {}),
		},
		auth: role.auth,
		resolutionSource: role.resolutionSource,
	};
}

function classifyError(error: unknown): ProbeOutput {
	const message = error instanceof Error ? error.message : String(error);
	const lower = message.toLowerCase();
	if ((error instanceof Error && error.name === "AbortError") || /abort|timeout|timed out/.test(lower)) {
		return { ok: false, category: "timeout", message: "model probe timed out" };
	}
	if (/api key|apikey|401|403|unauthorized|forbidden|authentication/.test(lower)) {
		return { ok: false, category: "auth_missing", message: "model API key rejected or missing" };
	}
	if (/enotfound|econnrefused|econnreset|etimedout|fetch failed|network error|socket hang up/.test(lower)) {
		return { ok: false, category: "provider_unreachable", message: "model provider network error" };
	}
	return { ok: false, category: "completion_failed", message: "model completion failed" };
}

async function defaultProbe(input: ProbeInput, signal: AbortSignal): Promise<ProbeOutput> {
	try {
		const apiKey = input.apiKey ?? input.providerApiKeys?.[input.model.provider];
		const result = await completeSimple(
			input.model,
			{ messages: [{ role: "user", content: "Reply with OK.", timestamp: Date.now() }] },
			{ ...(apiKey !== undefined ? { apiKey } : {}), signal, maxTokens: 64 },
		);
		if (result.stopReason === "error" || result.stopReason === "aborted") {
			return classifyError(new Error(result.errorMessage ?? `completion ${result.stopReason}`));
		}
		return { ok: true, category: "ok" };
	} catch (error) {
		return classifyError(error);
	}
}

async function runProbe(
	probe: (input: ProbeInput, signal: AbortSignal) => Promise<ProbeOutput>,
	input: ProbeInput,
	now: () => number,
): Promise<HealthProbeReport> {
	const started = now();
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), input.timeoutMs);
	try {
		const result = await probe(input, controller.signal);
		return {
			skipped: false,
			ok: result.ok,
			category: result.category,
			durationMs: now() - started,
			...(result.message !== undefined ? { message: sanitizeMessage(result.message) } : {}),
		};
	} finally {
		clearTimeout(timer);
	}
}

export async function runHealth(ctx: CommandContext, deps: HealthDeps = {}): Promise<number> {
	const timeout = parseTimeoutMs(ctx.flags);
	const base = {
		healthSchemaVersion: 1 as const,
		command: "health" as const,
		indexHealth: { separate: true as const, command: "autorag status" as const, included: false as const },
	};
	if (typeof timeout === "string") {
		const report: HealthReportV1 = {
			...base,
			ok: false,
			category: "config",
			probesSkipped: true,
			coverage: coverage(),
			config: { ok: false, source: "defaults", message: timeout },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	let config: CliConfig;
	try {
		config = (deps.configResolver ?? resolveConfigReadOnly)({ flags: ctx.flags, cwd: ctx.cwd });
	} catch (error) {
		const report: HealthReportV1 = {
			...base,
			ok: false,
			category: "config",
			probesSkipped: true,
			coverage: coverage(),
			config: { ok: false, source: "defaults", message: sanitizeMessage(String(error)) },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	let resolved: ResolvedHealthModel;
	try {
		const detailed = (deps.modelResolver ?? resolveAgentModelDetailed)(config);
		resolved = detailed;
	} catch (error) {
		const report: HealthReportV1 = {
			...base,
			ok: false,
			category: "model_resolution",
			probesSkipped: true,
			coverage: coverage(),
			config: { ok: true, source: "defaults" },
			probe: {
				skipped: true,
				ok: false,
				category: "model_resolution",
				message: sanitizeMessage(error instanceof Error ? error.message : String(error)),
			},
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	const model = roleReport(resolved.role);
	if (!resolved.role.auth.present) {
		const report: HealthReportV1 = {
			...base,
			ok: false,
			category: "auth_missing",
			probesSkipped: true,
			coverage: coverage(),
			config: { ok: true, source: "defaults" },
			model,
			probe: { skipped: true, ok: false, category: "auth_missing" },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}

	if (ctx.flags["skip-probes"] === true) {
		const report: HealthReportV1 = {
			...base,
			ok: true,
			category: "ok",
			probesSkipped: true,
			coverage: coverage(true),
			config: { ok: true, source: "defaults" },
			model,
			probe: { skipped: true, ok: false, category: "ok" },
		};
		ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
		return 0;
	}

	const probe = await runProbe(
		deps.probe ?? defaultProbe,
		{
			model: resolved.model,
			apiKey: resolved.apiKey,
			providerApiKeys: resolved.providerApiKeys,
			timeoutMs: timeout,
			cwd: ctx.cwd,
		},
		deps.now ?? Date.now,
	);
	const report: HealthReportV1 = {
		...base,
		ok: probe.category === "ok",
		category: probe.category,
		probesSkipped: false,
		coverage: coverage(probe.ok),
		config: { ok: true, source: "defaults" },
		model,
		probe,
	};
	ctx.stdout(renderHealth(report, { json: ctx.json, debug: ctx.debug }));
	return exitCode(probe.category);
}
