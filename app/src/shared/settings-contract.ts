/**
 * Settings, access-rule, contacts, and requests contract shared by the
 * Electron main process and the renderer.
 *
 * Persistence lives in the main process under Electron's userData dir.
 * The settings assistant chat is driven by a persona-configured
 * AutoRAGAgent (see persona options) — never a separately written agent.
 */

export const SETTINGS_CHANNELS = {
	get: "settings:get",
	set: "settings:set",
	clearHistory: "settings:clearHistory",
	permGet: "access:get",
	permSet: "access:set",
	permClear: "access:clear",
	permExplicitUnder: "access:explicitUnder",
	contactsList: "contacts:list",
	contactsAdd: "contacts:add",
	contactsUpdate: "contacts:update",
	contactsRemove: "contacts:remove",
	contactsTest: "contacts:test",
	requestsList: "requests:list",
	requestsRespond: "requests:respond",
	sourcesList: "sources:list",
	sourcesSetEnabled: "sources:setEnabled",
	assistantSend: "assistant:send",
	assistantEvent: "assistant:event",
} as const;

export type Permission = "ask" | "allow" | "deny";

export interface ResolvedPermission {
	readonly value: Permission;
	/** explicit = set on this exact path; inherited = from an ancestor; default = nothing set anywhere. */
	readonly origin: "explicit" | "inherited" | "default";
	/** The path the value came from (itself when explicit). */
	readonly sourcePath: string | null;
	/** True when explicit descendant values differ from this folder's resolved value. */
	readonly mixed: boolean;
}

export interface Contact {
	readonly id: string;
	readonly name: string;
	readonly role: string;
	readonly description: string;
}

export type RequestStatus = "pending" | "allowed" | "denied";

export interface AccessRequest {
	readonly id: string;
	readonly contactName: string;
	readonly contactRole: string;
	readonly question: string;
	readonly files: readonly { readonly path: string; readonly name: string }[];
	readonly status: RequestStatus;
	readonly requestedAt: string;
	/** True when an access rule handled it automatically. */
	readonly automatic: boolean;
}

export interface AppSettings {
	readonly language: "ko" | "en" | "ja" | "zh";
	readonly launchAtLogin: boolean;
	readonly showInMenuBar: boolean;
	readonly autoInstallUpdates: boolean;
	readonly telemetry: boolean;
	/** Off: dot-prefixed files never appear. On: they appear, rendered dimmed. */
	readonly showHiddenFiles: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
	language: "ko",
	launchAtLogin: false,
	showInMenuBar: true,
	autoInstallUpdates: true,
	telemetry: false,
	showHiddenFiles: false,
};

export interface DataSourceRow {
	readonly id: string;
	readonly name: string;
	readonly kind: string;
	readonly detail: string;
	readonly status: "indexed" | "syncing" | "paused" | "error";
	readonly progress: number | null;
	readonly enabled: boolean;
	readonly description: string;
}

export type AssistantSurface = "models" | "sources";

export type AssistantEvent =
	| { readonly type: "text"; readonly surface: AssistantSurface; readonly delta: string }
	| { readonly type: "done"; readonly surface: AssistantSurface }
	| { readonly type: "error"; readonly surface: AssistantSurface; readonly message: string };

export interface SettingsBridge {
	get(): Promise<AppSettings>;
	set(patch: Partial<AppSettings>): Promise<AppSettings>;
	clearHistory(): Promise<void>;

	permGet(path: string): Promise<ResolvedPermission>;
	/** Sets an explicit value. For folders the renderer runs the confirm modal first. */
	permSet(path: string, value: Permission, cascade: "apply-all" | "keep-individual"): Promise<void>;
	permClear(path: string): Promise<void>;
	/** Explicit values at or under a folder path (drives Mixed detection and the confirm modal counts). */
	permExplicitUnder(path: string): Promise<readonly { readonly path: string; readonly value: Permission }[]>;

	contactsList(): Promise<readonly Contact[]>;
	contactsAdd(contact: Contact): Promise<{ readonly ok: true } | { readonly ok: false; readonly error: "duplicate-id" }>;
	contactsUpdate(id: string, patch: Partial<Omit<Contact, "id">>): Promise<void>;
	contactsRemove(id: string): Promise<void>;
	contactsTest(id: string): Promise<{ readonly ok: boolean; readonly latencyMs: number | null }>;

	requestsList(status: "pending" | "done"): Promise<readonly AccessRequest[]>;
	requestsRespond(id: string, allow: boolean): Promise<void>;

	sourcesList(): Promise<readonly DataSourceRow[]>;
	sourcesSetEnabled(id: string, enabled: boolean): Promise<void>;

	/** Send one message to the settings assistant; events stream via assistantEvent. */
	assistantSend(surface: AssistantSurface, text: string): Promise<void>;
	readonly onEvent?: (listener: (event: AssistantEvent) => void) => () => void;
}
