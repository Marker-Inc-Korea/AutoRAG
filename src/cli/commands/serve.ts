import { existsSync } from "node:fs";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import { AutoRAGAgent } from "../../agent/agent.ts";
import { loadOrCreateIdentity } from "../../p2p/identity.ts";
import type { InjectionClassifierModel } from "../../p2p/injection-classifier.ts";
import { PolicyStore } from "../../p2p/policy.ts";
import { type P2pServer, type StartP2pServerOptions, startP2pServer } from "../../p2p/server.ts";
import {
	buildAgentOptions,
	ConfigError,
	type ResolvedAgentModel,
	resolveAgentModel,
	resolveConfig,
	resolveConfigPath,
} from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export interface ServeCommandDeps {
	readonly startP2pServer?: typeof startP2pServer;
	readonly waitUntilStopped?: (server: P2pServer) => Promise<void>;
	readonly getFingerprint?: () => Promise<string>;
	readonly modelResolver?: typeof resolveAgentModel;
}

function classifierText(message: AssistantMessage): string {
	return message.content
		.filter((part): part is Extract<(typeof message.content)[number], { type: "text" }> => part.type === "text")
		.map((part) => part.text)
		.join("");
}

export function createInjectionClassifierModel(resolved: ResolvedAgentModel): InjectionClassifierModel {
	return async (prompt) => {
		const apiKey = resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider];
		const result = await completeSimple(
			resolved.model,
			{ messages: [{ role: "user", content: prompt, timestamp: Date.now() }] },
			{ ...(apiKey !== undefined ? { apiKey } : {}), maxTokens: 256 },
		);
		if (result.stopReason === "error" || result.stopReason === "aborted") {
			throw new Error(result.errorMessage ?? `classifier ${result.stopReason}`);
		}
		return classifierText(result);
	};
}

/**
 * `autorag serve` — start the P2P peer query server.
 */
export async function runServe(ctx: CommandContext, deps: ServeCommandDeps = {}): Promise<number> {
	const flags = ctx.flags;
	const resolved = resolveConfigPath({ flags, cwd: ctx.cwd });
	if (!existsSync(resolved.configPath)) {
		ctx.stderr(
			renderError(new ConfigError(`Config file not found: ${resolved.configPath}. Run autorag init first.`), {
				json: ctx.json,
			}),
		);
		return 2;
	}

	let config: ReturnType<typeof resolveConfig>;
	try {
		config = resolveConfig({ flags, cwd: ctx.cwd });
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	const p2p = config.p2p ?? { enabled: false, host: "0.0.0.0", port: 9470 };
	const force = flags.force === true;
	if (p2p.enabled !== true && !force) {
		ctx.stderr(
			renderError(
				new ConfigError(
					"P2P sharing is disabled in config. Use --force to start anyway, or set p2p.enabled to true in config.",
				),
				{ json: ctx.json },
			),
		);
		return 2;
	}

	const host = typeof flags.host === "string" && flags.host.length > 0 ? flags.host : (p2p.host ?? "0.0.0.0");
	let port = p2p.port ?? 9470;
	if (typeof flags.port === "string" && flags.port.length > 0) {
		const parsed = Number(flags.port);
		if (!Number.isInteger(parsed) || parsed < 0) {
			ctx.stderr(renderError(new ConfigError("--port must be a non-negative integer."), { json: ctx.json }));
			return 2;
		}
		port = parsed;
	}

	const injectionClassifier = p2p.injectionClassifier !== false;
	let resolvedModel: ResolvedAgentModel | undefined;
	if (injectionClassifier) {
		try {
			resolvedModel = (deps.modelResolver ?? resolveAgentModel)(config);
		} catch (error) {
			ctx.stderr(
				renderError(
					error instanceof ConfigError
						? error
						: new ConfigError(
								`P2P injection classifier requires a configured model: ${error instanceof Error ? error.message : "resolve failed"}.`,
							),
					{ json: ctx.json, debug: ctx.debug },
				),
			);
			return 2;
		}
	}

	const getFingerprint =
		deps.getFingerprint ??
		(async () => {
			const identity = loadOrCreateIdentity(config.workspacePath);
			return identity.fingerprint;
		});
	const fingerprint = await getFingerprint();
	const start = deps.startP2pServer ?? startP2pServer;

	let server: P2pServer;
	try {
		const agent = new AutoRAGAgent({
			...buildAgentOptions(config),
			remoteSession: true,
			searchTimeoutMs: p2p.searchTimeoutMs,
			...(resolvedModel !== undefined
				? {
						model: resolvedModel.model,
						...(resolvedModel.apiKey !== undefined ? { apiKey: resolvedModel.apiKey } : {}),
						...(resolvedModel.providerApiKeys !== undefined
							? { providerApiKeys: resolvedModel.providerApiKeys }
							: {}),
					}
				: {}),
		});
		const policyStore = new PolicyStore({
			workspacePath: config.workspacePath,
			globalConfigPath: resolved.configPath,
			...(p2p.newFilesPublic !== undefined ? { newFilesPublic: p2p.newFilesPublic } : {}),
		});
		server = await start({
			host,
			port,
			workspacePath: config.workspacePath,
			workspaceRoots: config.searchPaths,
			agent,
			policyStore,
			injectionClassifier,
			...(injectionClassifier && resolvedModel !== undefined
				? { injectionClassifierModel: createInjectionClassifierModel(resolvedModel) }
				: {}),
			piiNer: p2p.piiNer,
			pseudonymize: p2p.piiNer,
			searchTimeoutMs: p2p.searchTimeoutMs,
			maxBodyBytes: p2p.maxBodyBytes,
			maxFileBytes: p2p.maxFileBytes,
			...(p2p.quotas !== undefined ? { quotas: p2p.quotas } : {}),
		} as StartP2pServerOptions);
	} catch (error) {
		const status = error instanceof ConfigError ? 2 : 1;
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return status;
	}

	const payload = {
		ok: true,
		host: server.host,
		port: server.port,
		fingerprint,
	};
	ctx.stdout(
		ctx.json ? JSON.stringify(payload) : `P2P server on ${server.url} | fingerprint: ${fingerprint.slice(0, 16)}...`,
	);

	const wait = deps.waitUntilStopped ?? defaultWaitUntilStopped;
	try {
		await wait(server);
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

async function defaultWaitUntilStopped(server: P2pServer): Promise<void> {
	await new Promise<void>((resolve) => {
		let stopped = false;
		const stop = () => {
			if (stopped) return;
			stopped = true;
			void server.close().finally(resolve);
		};
		process.once("SIGINT", stop);
		process.once("SIGTERM", stop);
	});
}
