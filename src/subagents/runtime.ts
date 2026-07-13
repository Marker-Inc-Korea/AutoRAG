import { createRequire } from "node:module";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	AuthStorage,
	createAgentSession,
	DefaultResourceLoader,
	SessionManager,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";

export interface MandatorySubagentSessionOptions {
	readonly cwd: string;
	readonly model: Model<Api>;
	readonly systemPrompt: string;
	readonly tools: readonly AgentTool[];
	readonly extensionPath?: string;
	readonly apiKey?: string;
}

export interface MandatorySubagentSession {
	readonly session: AgentSession;
	readonly extensionPath: string;
}

function resolvePiSubagentsExtension(): string {
	const require = createRequire(import.meta.url);
	const packageJson = require.resolve("pi-subagents/package.json");
	return join(dirname(packageJson), "src", "extension", "index.ts");
}

function configureSubagentPiBinary(): void {
	if (process.env.PI_SUBAGENT_PI_BINARY) return;
	const require = createRequire(import.meta.url);
	const subagentsPackage = require.resolve("pi-subagents/package.json");
	const packageJsonPath = join(
		dirname(dirname(subagentsPackage)),
		"@earendil-works",
		"pi-coding-agent",
		"package.json",
	);
	const packageJson = require(packageJsonPath) as { bin?: string | Record<string, string> };
	const bin = typeof packageJson.bin === "string" ? packageJson.bin : packageJson.bin?.pi;
	if (!bin) throw new Error("Mandatory pi-subagents runtime could not resolve the Pi CLI binary");
	process.env.PI_SUBAGENT_PI_BINARY = join(dirname(packageJsonPath), bin);
}

function asToolDefinition(tool: AgentTool): ToolDefinition {
	return {
		...tool,
		label: tool.label ?? tool.name,
	} as ToolDefinition;
}

export async function createMandatorySubagentSession(
	options: MandatorySubagentSessionOptions,
): Promise<MandatorySubagentSession> {
	configureSubagentPiBinary();
	if (options.apiKey && options.model.provider === "myproxy" && !process.env.MYPROXY_API_KEY) {
		process.env.MYPROXY_API_KEY = options.apiKey;
	}
	const extensionPath = options.extensionPath ?? resolvePiSubagentsExtension();
	const resourceLoader = new DefaultResourceLoader({
		cwd: options.cwd,
		agentDir: join(homedir(), ".autorag", "pi-agent"),
		additionalExtensionPaths: [extensionPath],
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

	const customTools = options.tools.map(asToolDefinition);
	const authStorage = AuthStorage.create(join(homedir(), ".autorag", "pi-agent", "auth.json"));
	if (options.apiKey) authStorage.setRuntimeApiKey(options.model.provider, options.apiKey);
	const { session } = await createAgentSession({
		cwd: options.cwd,
		agentDir: join(homedir(), ".autorag", "pi-agent"),
		model: options.model,
		authStorage,
		thinkingLevel: "high",
		resourceLoader,
		sessionManager: SessionManager.inMemory(options.cwd),
		noTools: "builtin",
		customTools,
	});
	const requiredTools = ["subagent", "wait"];
	const allToolNames = new Set(session.getAllTools().map((tool) => tool.name));
	const missing = requiredTools.filter((name) => !allToolNames.has(name));
	if (missing.length > 0) {
		session.dispose();
		throw new Error(`Mandatory pi-subagents extension did not register tools: ${missing.join(", ")}`);
	}
	session.setActiveToolsByName([...customTools.map((tool) => tool.name), ...requiredTools]);
	return { session, extensionPath };
}
