import { ipcMain } from "electron";
import { FS_CHANNELS, type FsClipboard, type FsCoreBridge } from "../shared/fs-contract";
import { SEARCH_CHANNELS, type SearchBridge } from "../shared/search-contract";
import { SETTINGS_CHANNELS, type SettingsBridge } from "../shared/settings-contract";
import type { VersionFamilyService } from "./version-family-service";

/** Extra work a channel runs after the filesystem operation itself succeeded. */
export interface FsIpcHooks {
	/**
	 * Paths that actually reached the Trash. A deleted duplicate must leave the
	 * version-family snapshot at once, so the user can see it is gone.
	 */
	readonly onTrashed?: (paths: readonly string[]) => void | Promise<void>;
}

/** Register one ipcMain.handle per FS_CHANNELS channel, in contract order. */
export function registerFsIpcHandlers(service: FsCoreBridge, hooks: FsIpcHooks = {}): void {
	ipcMain.handle(FS_CHANNELS.listDir, (_event, path: string) => service.listDir(path));
	ipcMain.handle(FS_CHANNELS.stat, (_event, path: string) => service.stat(path));
	ipcMain.handle(FS_CHANNELS.search, (_event, query: string) => service.search(query));
	ipcMain.handle(FS_CHANNELS.locations, () => service.locations());
	ipcMain.handle(FS_CHANNELS.copy, (_event, paths: readonly string[], destDir: string) => service.copy(paths, destDir));
	ipcMain.handle(FS_CHANNELS.move, (_event, paths: readonly string[], destDir: string) => service.move(paths, destDir));
	ipcMain.handle(FS_CHANNELS.duplicate, (_event, paths: readonly string[]) => service.duplicate(paths));
	ipcMain.handle(FS_CHANNELS.rename, (_event, path: string, newName: string) => service.rename(path, newName));
	ipcMain.handle(FS_CHANNELS.trash, async (_event, paths: readonly string[]) => {
		const result = await service.trash(paths);
		if (result.ok.length > 0) {
			try {
				await hooks.onTrashed?.(result.ok);
			} catch (error) {
				// The move to the Trash already happened; a stale snapshot must not
				// turn into a failed delete for the caller.
				console.error("trash follow-up failed", error);
			}
		}
		return result;
	});
	ipcMain.handle(FS_CHANNELS.reveal, (_event, path: string) => service.reveal(path));
	ipcMain.handle(FS_CHANNELS.quickLook, (_event, path: string) => service.quickLook(path));
	ipcMain.handle(FS_CHANNELS.open, (_event, path: string) => service.open(path));
	ipcMain.handle(FS_CHANNELS.clipboardSet, (_event, clipboard: FsClipboard) => service.clipboardSet(clipboard));
	ipcMain.handle(FS_CHANNELS.clipboardGet, () => service.clipboardGet());
	ipcMain.handle(FS_CHANNELS.copyPathsToClipboard, (_event, paths: readonly string[]) =>
		service.copyPathsToClipboard(paths),
	);
}

/** The persisted-snapshot version-family service owns its own two channels. */
export function registerVersionFamilyIpcHandlers(service: VersionFamilyService): void {
	ipcMain.handle(FS_CHANNELS.versionFamilies, () => service.result());
	ipcMain.handle(FS_CHANNELS.versionFamiliesRefresh, () => service.refresh());
}

export function registerSearchIpcHandlers(service: SearchBridge): void {
	ipcMain.handle(SEARCH_CHANNELS.start, (_event, searchId: string, chatId: string, query: string, attachments) =>
		service.start(searchId, chatId, query, attachments),
	);
	ipcMain.handle(SEARCH_CHANNELS.cancel, (_event, searchId: string) => service.cancel(searchId));
	ipcMain.handle(SEARCH_CHANNELS.feedback, (_event, sessionId: string, useful: readonly number[], notUseful: readonly number[]) =>
		service.feedback(sessionId, useful, notUseful),
	);
	ipcMain.handle(SEARCH_CHANNELS.historyList, () => service.historyList());
	ipcMain.handle(SEARCH_CHANNELS.historyGet, (_event, chatId: string) => service.historyGet(chatId));
	ipcMain.handle(SEARCH_CHANNELS.historySearch, (_event, query: string) => service.historySearch(query));
	ipcMain.handle(SEARCH_CHANNELS.historyClear, () => service.historyClear());
}

export function registerSettingsIpcHandlers(service: SettingsBridge): void {
	ipcMain.handle(SETTINGS_CHANNELS.get, () => service.get());
	ipcMain.handle(SETTINGS_CHANNELS.set, (_event, patch) => service.set(patch));
	ipcMain.handle(SETTINGS_CHANNELS.clearHistory, () => service.clearHistory());
	ipcMain.handle(SETTINGS_CHANNELS.permGet, (_event, path: string) => service.permGet(path));
	ipcMain.handle(SETTINGS_CHANNELS.permSet, (_event, path: string, value, cascade) => service.permSet(path, value, cascade));
	ipcMain.handle(SETTINGS_CHANNELS.permClear, (_event, path: string) => service.permClear(path));
	ipcMain.handle(SETTINGS_CHANNELS.permExplicitUnder, (_event, path: string) => service.permExplicitUnder(path));
	ipcMain.handle(SETTINGS_CHANNELS.contactsList, () => service.contactsList());
	ipcMain.handle(SETTINGS_CHANNELS.contactsAdd, (_event, contact) => service.contactsAdd(contact));
	ipcMain.handle(SETTINGS_CHANNELS.contactsUpdate, (_event, id: string, patch) => service.contactsUpdate(id, patch));
	ipcMain.handle(SETTINGS_CHANNELS.contactsRemove, (_event, id: string) => service.contactsRemove(id));
	ipcMain.handle(SETTINGS_CHANNELS.contactsTest, (_event, id: string) => service.contactsTest(id));
	ipcMain.handle(SETTINGS_CHANNELS.requestsList, (_event, status) => service.requestsList(status));
	ipcMain.handle(SETTINGS_CHANNELS.requestsRespond, (_event, id: string, allow: boolean) => service.requestsRespond(id, allow));
	ipcMain.handle(SETTINGS_CHANNELS.sourcesList, () => service.sourcesList());
	ipcMain.handle(SETTINGS_CHANNELS.sourcesSetEnabled, (_event, id: string, enabled: boolean) => service.sourcesSetEnabled(id, enabled));
	ipcMain.handle(SETTINGS_CHANNELS.assistantSend, (_event, surface, text: string) => service.assistantSend(surface, text));
}
