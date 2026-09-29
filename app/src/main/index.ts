import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { app, BrowserWindow, clipboard, shell } from "electron";
import { BUILTIN_DATASOURCE_SKILL_NAMES, writeConfigObject, writeDefaultConfig } from "@autorag/librarian";
import type { DataSourceRow } from "../shared/settings-contract";
import { createChatStore } from "./chat-store";
import { createFsService } from "./fs-service";
import { createRecentsStore } from "./recents-store";
import { createDefaultAgentFactory, createSearchService } from "./search-service";
import { createSettingsService } from "./settings-service";
import { registerFsIpcHandlers, registerSearchIpcHandlers, registerSettingsIpcHandlers } from "./ipc";

function autoragConfigPath(): string {
	const explicit = process.env.AUTORAG_CONFIG?.trim();
	if (explicit) return explicit;
	const home = process.env.AUTORAG_HOME?.trim();
	const autoragHome = home ? resolve(home) : join(process.env.HOME && isAbsolute(process.env.HOME) ? process.env.HOME : homedir(), ".autorag");
	return join(autoragHome, "config.json");
}

function readConfigObject(path: string): Record<string, unknown> {
	const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("AutoRAG config must be a JSON object");
	}
	return parsed as Record<string, unknown>;
}

function ensureAutoRAGConfig(): void {
	const configPath = autoragConfigPath();
	if (existsSync(configPath)) return;
	try {
		writeDefaultConfig(
			configPath,
			{ searchPaths: [process.cwd()], workspacePath: process.cwd() },
			{ atomicCreate: true, cwd: process.cwd(), env: process.env },
		);
	} catch (error) {
		console.error("AutoRAG config initialization failed", error);
	}
}

function configuredDataSources(): readonly DataSourceRow[] {
	const configPath = autoragConfigPath();
	if (!existsSync(configPath)) return [];
	try {
		const raw = readConfigObject(configPath);
		const configured = raw.datasources;
		if (configured === null || typeof configured !== "object" || Array.isArray(configured)) return [];
		return Object.entries(configured as Record<string, unknown>)
			.filter(([, value]) => value !== false)
			.map(([id, value]) => {
				const entry = value !== null && typeof value === "object" && !Array.isArray(value)
					? value as Record<string, unknown>
					: {};
				const type = typeof entry.type === "string" ? entry.type : id;
				const enabled = entry.enabled !== false;
				return {
					id,
					name: id,
					kind: BUILTIN_DATASOURCE_SKILL_NAMES.includes(type) ? type : "custom",
					detail: typeof entry.description === "string" ? entry.description : type,
					status: enabled ? "indexed" : "paused",
					progress: null,
					enabled,
					description: typeof entry.description === "string" ? entry.description : `AutoRAG datasource: ${type}`,
				} satisfies DataSourceRow;
			});
	} catch {
		return [];
	}
}

async function setConfiguredDatasourceEnabled(id: string, enabled: boolean): Promise<void> {
	const configPath = autoragConfigPath();
	if (!existsSync(configPath)) return;
	const raw = readConfigObject(configPath);
	const configured = raw.datasources;
	if (configured === null || typeof configured !== "object" || Array.isArray(configured)) return;
	const entry = (configured as Record<string, unknown>)[id];
	if (entry === undefined) return;
	(configured as Record<string, unknown>)[id] =
		entry !== null && typeof entry === "object" && !Array.isArray(entry)
			? { ...(entry as Record<string, unknown>), enabled }
			: { enabled, type: id };
	writeConfigObject(configPath, raw);
}

function createMainWindow(): BrowserWindow {
	const window = new BrowserWindow({
		width: 1440,
		height: 900,
		minWidth: 1360,
		minHeight: 820,
		webPreferences: {
			preload: join(import.meta.dirname, "../preload/index.mjs"),
			contextIsolation: true,
			nodeIntegration: false,
			sandbox: false,
		},
	});

	const rendererUrl = process.env.ELECTRON_RENDERER_URL;
	if (rendererUrl) {
		void window.loadURL(rendererUrl);
	} else {
		void window.loadFile(join(import.meta.dirname, "../renderer/index.html"));
	}
	return window;
}

app.whenReady().then(() => {
	ensureAutoRAGConfig();
	registerFsIpcHandlers(
		createFsService({
			shell,
			clipboard,
			recents: createRecentsStore({ directory: join(app.getPath("userData"), "recents") }),
		}),
	);
	const searchService = createSearchService({
		agentFactory: createDefaultAgentFactory(),
		chatStore: createChatStore({ directory: join(app.getPath("userData"), "chat-history") }),
		send: (channel, payload) => BrowserWindow.getAllWindows()[0]?.webContents.send(channel, payload),
	});
	registerSearchIpcHandlers(searchService);
	const agentFactory = createDefaultAgentFactory();
	registerSettingsIpcHandlers(
		createSettingsService({
			directory: join(app.getPath("userData"), "settings"),
			initialSources: configuredDataSources(),
			setSourceEnabled: setConfiguredDatasourceEnabled,
			createChatSession: (surface) => {
				const agent = agentFactory();
				if (agent.createChatSession === undefined) {
					throw new Error("AutoRAG chat sessions are unavailable");
				}
				return agent.createChatSession({
					systemPrompt: `You are AutoRAG Settings Assistant for the ${surface} surface. Explain and configure the existing AutoRAG application using its actual settings and data sources. Never invent unavailable state.`,
				});
			},
			send: (channel, payload) => BrowserWindow.getAllWindows()[0]?.webContents.send(channel, payload),
		}),
	);
	createMainWindow();
	app.on("activate", () => {
		if (BrowserWindow.getAllWindows().length === 0) {
			createMainWindow();
		}
	});
});

app.on("window-all-closed", () => {
	if (process.platform !== "darwin") {
		app.quit();
	}
});
