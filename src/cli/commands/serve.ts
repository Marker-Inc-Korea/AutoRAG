import { existsSync } from "node:fs";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import { AutoRAGAgent } from "../../agent/agent.ts";
import { planSourceRoots, sourceIdentifier } from "../../filesystem/source-paths.ts";
import type { InjectionClassifierModel } from "../../p2p/injection-classifier.ts";
import { PolicyStore } from "../../p2p/policy.ts";
import {
	type SimplexPeerServer,
	type StartSimplexPeerServerOptions,
	startSimplexPeerServer,
} from "../../p2p/simplex-server.ts";
import { type SimplexTransport, type StartSimplexOptions, startSimplexChat } from "../../p2p/simplex-transport.ts";
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
	readonly startSimplexPeerServer?: typeof startSimplexPeerServer;
	readonly startSimplexChat?: (options: StartSimplexOptions) => Promise<SimplexTransport>;
	readonly waitUntilStopped?: (server: SimplexPeerServer) => Promise<void>;
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
 * `autorag serve` — start the P2P peer query server over SimpleX Chat.
 * Peers reach this installation through SimpleX; all security gates run
 * unchanged.
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

	// SimpleX WebSocket bind; loopback-only — the CLI is a local control
	// surface, peers arrive through the SimpleX network.
	let port = p2p.port ?? 5225;
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
	try {
		resolvedModel = (deps.modelResolver ?? resolveAgentModel)(config);
	} catch (error) {
		if (!injectionClassifier) {
			// Model resolution still required for the search agent itself; classifier-only failures are non-fatal.
			resolvedModel = undefined;
		} else {
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

	const startChat = deps.startSimplexChat ?? startSimplexChat;
	const startServer = deps.startSimplexPeerServer ?? startSimplexPeerServer;

	let server: SimplexPeerServer;
	let transport: SimplexTransport;
	try {
		transport = await startChat({
			dbPrefix: p2p.simplexDbPrefix ?? `${config.workspacePath}/.autorag/p2p/simplex`,
			displayName: `autorag-${config.workspacePath.split("/").pop() ?? "node"}`,
			port,
		});
	} catch (error) {
		ctx.stderr(
			renderError(
				error instanceof Error ? error : new ConfigError("simplex-chat failed to start. Is the CLI installed?"),
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
		// Resolve policy against the canonical virtual path. Retrieval may
		// return real absolute paths (e.g. MinSync source fields); convert
		// them to the virtual path before matching policy globs.
		const sourceRoots = planSourceRoots(config.searchPaths);
		const resolveVirtualPathPolicy = (source: string, peer?: string) => {
			if (!source.startsWith("/")) return policyStore.resolvePolicy(source, peer);
			// Absolute real path -> virtual path
			for (const root of sourceRoots) {
				if (source.startsWith(root.rootPath)) {
					const virtual = sourceIdentifier(root, source);
					return policyStore.resolvePolicy(virtual, peer);
				}
			}
			// Already a virtual path (starts with root prefix)
			return policyStore.resolvePolicy(source, peer);
		};
		server = await startServer({
			transport,
			agent,
			policyStore,
			resolvePolicy: resolveVirtualPathPolicy,
			workspacePath: config.workspacePath,
			workspaceRoots: config.searchPaths,
			injectionClassifier,
			...(injectionClassifier && resolvedModel !== undefined
				? { injectionClassifierModel: createInjectionClassifierModel(resolvedModel) }
				: {}),
			pseudonymize: p2p.piiNer,
			searchTimeoutMs: p2p.searchTimeoutMs,
			...(p2p.quotas !== undefined ? { quotas: p2p.quotas } : {}),
		} as StartSimplexPeerServerOptions);
	} catch (error) {
		await transport.close().catch(() => {});
		const status = error instanceof ConfigError ? 2 : 1;
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return status;
	}

	let address: string;
	try {
		address = await transport.getOrCreateAddress();
	} catch (error) {
		await transport.close().catch(() => {});
		ctx.stderr(
			renderError(error instanceof Error ? error : new ConfigError("address creation failed"), {
				json: ctx.json,
				debug: ctx.debug,
			}),
		);
		return 1;
	}

	const payload = {
		ok: true,
		address,
		port,
	};
	ctx.stdout(
		ctx.json
			? JSON.stringify(payload)
			: `P2P server over SimpleX | address: ${address.slice(0, 40)}... | WebSocket: 127.0.0.1:${port}`,
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

async function defaultWaitUntilStopped(server: SimplexPeerServer): Promise<void> {
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
