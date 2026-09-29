import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type {
	AccessRequest,
	AppSettings,
	AssistantEvent,
	AssistantSurface,
	Contact,
	DataSourceRow,
	Permission,
	ResolvedPermission,
	SettingsBridge,
} from "../shared/settings-contract";
import { DEFAULT_SETTINGS, SETTINGS_CHANNELS } from "../shared/settings-contract";

interface SettingsState {
	settings: AppSettings;
	permissions: Record<string, Permission>;
	contacts: Contact[];
	requests: AccessRequest[];
	sources: DataSourceRow[];
}

interface ChatAgent {
	readonly agent: {
		subscribe(listener: (event: unknown) => void): () => void;
	};
	prompt(text: string): Promise<void>;
	dispose(): void;
}

export interface SettingsServiceDeps {
	readonly directory: string;
	readonly createChatSession: (surface: AssistantSurface) => ChatAgent;
	readonly send: (channel: string, payload: AssistantEvent) => void;
	/** Datasources resolved from the real AutoRAG config at app startup. */
	readonly initialSources?: readonly DataSourceRow[];
	/** Persists a source toggle back into AutoRAG's trusted config. */
	readonly setSourceEnabled?: (id: string, enabled: boolean) => Promise<void>;
	/** Fired after every successful set() so live services can adopt the change. */
	readonly onSettingsChanged?: (settings: AppSettings) => void;
	readonly now?: () => Date;
}

const emptyState = (): SettingsState => ({
	settings: DEFAULT_SETTINGS,
	permissions: {},
	contacts: [],
	requests: [],
	sources: [],
});

function isTextDelta(event: unknown): event is { readonly type: "message_update"; readonly assistantMessageEvent: { readonly type: "text_delta"; readonly delta: string } } {
	if (typeof event !== "object" || event === null) return false;
	const candidate = event as { readonly type?: unknown; readonly assistantMessageEvent?: unknown };
	if (candidate.type !== "message_update" || typeof candidate.assistantMessageEvent !== "object" || candidate.assistantMessageEvent === null) {
		return false;
	}
	const update = candidate.assistantMessageEvent as { readonly type?: unknown; readonly delta?: unknown };
	return update.type === "text_delta" && typeof update.delta === "string";
}

async function loadState(path: string, initialSources: readonly DataSourceRow[]): Promise<SettingsState> {
	try {
		const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
		if (typeof parsed !== "object" || parsed === null) return { ...emptyState(), sources: [...initialSources] };
		const state = parsed as Partial<SettingsState>;
		return {
			...emptyState(),
			...state,
			settings: { ...DEFAULT_SETTINGS, ...(state.settings ?? {}) },
			permissions: state.permissions ?? {},
			contacts: state.contacts ?? [],
			requests: state.requests ?? [],
			sources: state.sources?.length ? state.sources : [...initialSources],
		};
	} catch {
		return { ...emptyState(), sources: [...initialSources] };
	}
}

export function createSettingsService(deps: SettingsServiceDeps): SettingsBridge {
	const path = join(deps.directory, "settings-state.json");
	let statePromise = loadState(path, deps.initialSources ?? []);
	const now = deps.now ?? (() => new Date());
	const persist = async (): Promise<void> => {
		const state = await statePromise;
		await mkdir(dirname(path), { recursive: true });
		await writeFile(path, `${JSON.stringify(state, null, 2)}\n`, "utf8");
	};
	const mutate = async (update: (state: SettingsState) => void): Promise<SettingsState> => {
		const state = await statePromise;
		update(state);
		statePromise = Promise.resolve(state);
		await persist();
		return state;
	};
	const resolvedPermission = (state: SettingsState, pathName: string): ResolvedPermission => {
		const candidates = Object.entries(state.permissions)
			.filter(([source]) => pathName === source || pathName.startsWith(`${source}/`))
			.sort(([a], [b]) => b.length - a.length);
		const [sourcePath, value] = candidates[0] ?? [];
		const explicitDescendants = Object.keys(state.permissions).some(
			(source) => source !== pathName && source.startsWith(`${pathName}/`),
		);
		return {
			value: value ?? "ask",
			origin: sourcePath === undefined ? "default" : sourcePath === pathName ? "explicit" : "inherited",
			sourcePath: sourcePath ?? null,
			mixed: explicitDescendants,
		};
	};

	return {
		get: async () => (await statePromise).settings,
		set: async (patch) => {
			const state = await mutate((current) => {
				current.settings = { ...current.settings, ...patch };
			});
			deps.onSettingsChanged?.(state.settings);
			return state.settings;
		},
		clearHistory: async () => undefined,
		permGet: async (pathName) => resolvedPermission(await statePromise, pathName),
		permSet: async (pathName, value, cascade) => {
			await mutate((current) => {
				current.permissions[pathName] = value;
				if (cascade === "apply-all") {
					for (const child of Object.keys(current.permissions)) {
						if (child.startsWith(`${pathName}/`)) current.permissions[child] = value;
					}
				}
			});
		},
		permClear: async (pathName) => {
			await mutate((current) => {
				delete current.permissions[pathName];
			});
		},
		permExplicitUnder: async (pathName) => {
			const state = await statePromise;
			return Object.entries(state.permissions)
				.filter(([source]) => source === pathName || source.startsWith(`${pathName}/`))
				.map(([source, value]) => ({ path: source, value }));
		},
		contactsList: async () => (await statePromise).contacts,
		contactsAdd: async (contact) => {
			const state = await statePromise;
			if (state.contacts.some((candidate) => candidate.id === contact.id)) return { ok: false, error: "duplicate-id" };
			await mutate((current) => {
				current.contacts.push(contact);
			});
			return { ok: true };
		},
		contactsUpdate: async (id, patch) => {
			await mutate((current) => {
				current.contacts = current.contacts.map((contact) =>
					contact.id === id ? { ...contact, ...patch, id: contact.id } : contact,
				);
			});
		},
		contactsRemove: async (id) => {
			await mutate((current) => {
				current.contacts = current.contacts.filter((contact) => contact.id !== id);
			});
		},
		contactsTest: async (id) => {
			const started = now().getTime();
			const exists = (await statePromise).contacts.some((contact) => contact.id === id);
			return { ok: exists, latencyMs: exists ? Math.max(0, now().getTime() - started) : null };
		},
		requestsList: async (status) => {
			const state = await statePromise;
			return state.requests.filter((request) => (status === "pending" ? request.status === "pending" : request.status !== "pending"));
		},
		requestsRespond: async (id, allow) => {
			await mutate((current) => {
				current.requests = current.requests.map((request) =>
					request.id === id ? { ...request, status: allow ? "allowed" : "denied" } : request,
				);
			});
		},
		sourcesList: async () => (await statePromise).sources,
		sourcesSetEnabled: async (id, enabled) => {
			await deps.setSourceEnabled?.(id, enabled);
			await mutate((current) => {
				current.sources = current.sources.map((source) =>
					source.id === id ? { ...source, enabled } : source,
				);
			});
		},
		assistantSend: async (surface, text) => {
			let session: ChatAgent | undefined;
			let unsubscribe: (() => void) | undefined;
			try {
				session = deps.createChatSession(surface);
				unsubscribe = session.agent.subscribe((event) => {
					if (isTextDelta(event)) {
						deps.send(SETTINGS_CHANNELS.assistantEvent, {
							type: "text",
							surface,
							delta: event.assistantMessageEvent.delta,
						});
					}
				});
				await session.prompt(text);
				deps.send(SETTINGS_CHANNELS.assistantEvent, { type: "done", surface });
			} catch (error) {
				deps.send(SETTINGS_CHANNELS.assistantEvent, {
					type: "error",
					surface,
					message: error instanceof Error ? error.message : String(error),
				});
			} finally {
				unsubscribe?.();
				session?.dispose();
			}
		},
		onEvent: () => () => undefined,
	};
}
