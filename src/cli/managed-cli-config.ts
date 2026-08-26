import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";

export type ManagedCliOwnership = "managed" | "external";

export interface ManagedCliContext {
	readonly workspace: string;
	readonly tool: string;
	readonly instance: string;
	readonly ownership: ManagedCliOwnership;
	readonly configPath: string;
	readonly externalConfigPath?: string;
	readonly config: unknown;
	readonly existingConfig?: unknown;
}

export interface ManagedCliLaunchContext {
	readonly ownership: ManagedCliOwnership;
	readonly cwd?: string;
	readonly env: Readonly<Record<string, string>>;
	readonly prefixArgs: readonly string[];
	readonly configPath: string;
}

export interface ManagedCliConfigStatus {
	readonly ownership: ManagedCliOwnership;
	readonly configPath: string;
	readonly appliedBy: string;
	readonly missingRequirements: readonly string[];
	readonly drift: readonly string[];
}

export interface ManagedCliConfigProvider {
	readonly tool: string;
	readonly aliases?: readonly string[];
	readonly binaryPaths?: readonly string[];
	readonly managedConfigPath?: (context: Omit<ManagedCliContext, "configPath" | "existingConfig">) => string;
	readonly readConfig?: (path: string) => unknown;
	readonly materialize: (context: ManagedCliContext) => Promise<ManagedCliLaunchContext>;
	readonly inspect: (context: ManagedCliContext) => Promise<ManagedCliConfigStatus>;
	/** Provider-owned serialization keeps the shared layer format-agnostic. */
	readonly renderConfig?: (config: unknown, existingConfig: unknown) => string;
}

export class ManagedCliRegistry {
	private readonly providers = new Map<string, ManagedCliConfigProvider>();
	private readonly aliases = new Map<string, ManagedCliConfigProvider>();

	register(provider: ManagedCliConfigProvider): void {
		if (this.providers.has(provider.tool)) throw new Error(`Managed CLI "${provider.tool}" is already registered`);
		const names = [provider.tool, ...(provider.aliases ?? []), ...(provider.binaryPaths ?? [])];
		for (const name of names) {
			if (this.aliases.has(name)) throw new Error(`Managed CLI alias "${name}" is already registered`);
		}
		this.providers.set(provider.tool, provider);
		for (const name of names) this.aliases.set(name, provider);
	}

	resolve(binary: string): ManagedCliConfigProvider | undefined {
		return this.aliases.get(binary) ?? this.aliases.get(binary.split(/[\\/]/).pop() ?? binary);
	}

	list(): readonly ManagedCliConfigProvider[] {
		return [...this.providers.values()];
	}
}

export interface ManagedCliConfigRequest {
	readonly instance?: string;
	readonly ownership?: ManagedCliOwnership;
	readonly configPath?: string;
	readonly config?: unknown;
	readonly [key: string]: unknown;
}

export class ManagedCliConfigManager {
	private readonly workspace: string;
	private readonly registry: ManagedCliRegistry;

	constructor(options: { workspace: string; registry: ManagedCliRegistry }) {
		this.workspace = resolve(options.workspace);
		this.registry = options.registry;
	}

	async materialize(tool: string, request: ManagedCliConfigRequest = {}): Promise<ManagedCliLaunchContext> {
		const provider = this.registry.resolve(tool);
		if (!provider) throw new Error(`Managed CLI "${tool}" is not registered`);
		const instance = request.instance ?? "default";
		const ownership = request.ownership ?? "managed";
		const managedDir = join(this.workspace, ".autorag", "tools", provider.tool, instance);
		const pathContext = {
			workspace: this.workspace,
			tool: provider.tool,
			instance,
			ownership,
			config: request.config ?? withoutControlFields(request),
		};
		const managedPath = provider.managedConfigPath?.(pathContext) ?? join(managedDir, "config.json");
		const externalPath = request.configPath;
		const configPath = ownership === "external" ? this.requireExternalPath(externalPath, managedDir) : managedPath;
		const config = request.config ?? withoutControlFields(request);
		rejectSecrets(config);
		const existingConfig = ownership === "external" ? undefined : readConfig(configPath, provider, ownership);
		if (ownership === "managed") {
			mkdirSync(dirname(configPath), { recursive: true });
			const rendered = provider.renderConfig?.(config, existingConfig);
			if (rendered === undefined) throw new Error(`Managed CLI "${provider.tool}" has no config renderer`);
			atomicWrite(configPath, rendered);
		}
		return provider.materialize({
			workspace: this.workspace,
			tool: provider.tool,
			instance,
			ownership,
			configPath,
			...(ownership === "external" ? { externalConfigPath: configPath } : {}),
			config,
			...(existingConfig === undefined ? {} : { existingConfig }),
		});
	}

	async inspect(tool: string, request: ManagedCliConfigRequest = {}): Promise<ManagedCliConfigStatus> {
		const provider = this.registry.resolve(tool);
		if (!provider) throw new Error(`Managed CLI "${tool}" is not registered`);
		const instance = request.instance ?? "default";
		const ownership = request.ownership ?? "managed";
		const managedDir = join(this.workspace, ".autorag", "tools", provider.tool, instance);
		const configPath = ownership === "external" ? this.requireExternalPath(request.configPath, managedDir) : join(managedDir, "config.json");
		return provider.inspect({
			workspace: this.workspace, tool: provider.tool, instance, ownership, configPath,
			config: request.config ?? withoutControlFields(request),
		});
	}

	private requireExternalPath(path: string | undefined, managedDir: string): string {
		if (!path || !isAbsolute(path)) throw new Error("External managed CLI configuration requires an absolute path");
		const resolved = resolve(path);
		const rel = relative(resolve(managedDir), resolved);
		if (rel === "" || (!rel.startsWith(`..${sep}`) && rel !== "..")) {
			throw new Error("External configuration cannot be inside managed CLI state; ownership conflict");
		}
		return resolved;
	}
}

function readConfig(path: string, provider: ManagedCliConfigProvider, ownership: ManagedCliOwnership): unknown {
	if (ownership === "external" || !existsSync(path)) return undefined;
	if (provider.readConfig) return provider.readConfig(path);
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch {
		throw new Error(`Managed CLI "${provider.tool}" configuration is not valid JSON`);
	}
}

function withoutControlFields(request: ManagedCliConfigRequest): Record<string, unknown> {
	const { instance: _instance, ownership: _ownership, configPath: _configPath, config: _config, ...config } = request;
	return config;
}

function rejectSecrets(value: unknown, path = "config"): void {
	if (!value || typeof value !== "object") return;
	for (const [key, child] of Object.entries(value)) {
		if (/(secret|token|password|cookie|credential|refresh.?oauth)/i.test(key)) {
			throw new Error(`Secret value rejected at ${path}.${key}; use an environment or keychain reference`);
		}
		rejectSecrets(child, `${path}.${key}`);
	}
}

function atomicWrite(path: string, content: string): void {
	const temporary = `${path}.tmp-${process.pid}-${Date.now()}`;
	writeFileSync(temporary, content, { encoding: "utf8", mode: 0o600 });
	renameSync(temporary, path);
}
