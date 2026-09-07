import { existsSync } from "node:fs";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import { AutoRAGAgent } from "../../agent/agent.ts";
import type { InjectionClassifierModel } from "../../p2p/injection-classifier.ts";
import { PolicyStore } from "../../p2p/policy.ts";
import {
	type SignalPeerServer,
	type StartSignalPeerServerOptions,
	startSignalPeerServer,
} from "../../p2p/signal-server.ts";
import { type SignalTransport, type StartSignalDaemonOptions, startSignalDaemon } from "../../p2p/signal-transport.ts";
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
	readonly startSignalPeerServer?: typeof startSignalPeerServer;
	readonly startSignalDaemon?: (options: StartSignalDaemonOptions) => Promise<SignalTransport>;
	readonly waitUntilStopped?: (server: SignalPeerServer) => Promise<void>;
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
 * `autorag serve` — start the P2P peer query server over the Signal
 * transport (signal-cli JSON-RPC daemon). Peers reach this installation
 * through Signal; all security gates run unchanged.
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

	const p2p = config.p2p ?? { enabled: false };
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

	const account = typeof flags.account === "string" && flags.account.length > 0 ? flags.account : p2p.account;
	if (account === undefined || account.length === 0) {
		ctx.stderr(
			renderError(
				new ConfigError(
					"A Signal account is required. Set p2p.account in config or pass --account +E164 (register first with `autorag p2p register`).",
				),
				{ json: ctx.json },
			),
		);
		return 2;
	}

	// signal-cli daemon bind; loopback-only — the daemon is a local control
	// surface, peers arrive through the Signal network.
	const host = typeof flags.host === "string" && flags.host.length > 0 ? flags.host : (p2p.host ?? "127.0.0.1");
	let port = p2p.port ?? 7583;
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

	const startDaemon = deps.startSignalDaemon ?? startSignalDaemon;
	const startServer = deps.startSignalPeerServer ?? startSignalPeerServer;

	let server: SignalPeerServer;
	let transport: SignalTransport;
	try {
		transport = await startDaemon({
			account,
			host,
			port,
			dataDir: p2p.signalDataDir,
		});
	} catch (error) {
		ctx.stderr(
			renderError(
				error instanceof Error
					? error
					: new ConfigError("signal-cli daemon failed to start. Is the account registered?"),
				{ json: ctx.json, debug: ctx.debug },
			),
		);
		return 1;
	}

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
		server = await startServer({
			transport,
			agent,
			policyStore,
			workspacePath: config.workspacePath,
			workspaceRoots: config.searchPaths,
			injectionClassifier,
			...(injectionClassifier && resolvedModel !== undefined
				? { injectionClassifierModel: createInjectionClassifierModel(resolvedModel) }
				: {}),
			pseudonymize: p2p.piiNer,
			searchTimeoutMs: p2p.searchTimeoutMs,
			...(p2p.quotas !== undefined ? { quotas: p2p.quotas } : {}),
		} as StartSignalPeerServerOptions);
	} catch (error) {
		await transport.close().catch(() => {});
		const status = error instanceof ConfigError ? 2 : 1;
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return status;
	}

	const payload = {
		ok: true,
		account,
		daemon: { host, port },
	};
	ctx.stdout(
		ctx.json
			? JSON.stringify(payload)
			: `P2P server over Signal | account: ${account} | signal-cli daemon: http://${host}:${port}`,
	);

	const wait = deps.waitUntilStopped ?? defaultWaitUntilStopped;
	try {
		await wait(server);
		await transport.close().catch(() => {});
		return 0;
	} catch (error) {
		await transport.close().catch(() => {});
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

async function defaultWaitUntilStopped(server: SignalPeerServer): Promise<void> {
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
