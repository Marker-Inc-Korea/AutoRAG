import type { IpcRenderer } from "electron";
import { SETTINGS_CHANNELS, type AssistantEvent, type SettingsBridge } from "../shared/settings-contract";

export function createSettingsBridge(ipcRenderer: IpcRenderer): SettingsBridge {
	return {
		get: () => ipcRenderer.invoke(SETTINGS_CHANNELS.get),
		set: (patch) => ipcRenderer.invoke(SETTINGS_CHANNELS.set, patch),
		clearHistory: () => ipcRenderer.invoke(SETTINGS_CHANNELS.clearHistory),
		permGet: (path) => ipcRenderer.invoke(SETTINGS_CHANNELS.permGet, path),
		permSet: (path, value, cascade) => ipcRenderer.invoke(SETTINGS_CHANNELS.permSet, path, value, cascade),
		permClear: (path) => ipcRenderer.invoke(SETTINGS_CHANNELS.permClear, path),
		permExplicitUnder: (path) => ipcRenderer.invoke(SETTINGS_CHANNELS.permExplicitUnder, path),
		contactsList: () => ipcRenderer.invoke(SETTINGS_CHANNELS.contactsList),
		contactsAdd: (contact) => ipcRenderer.invoke(SETTINGS_CHANNELS.contactsAdd, contact),
		contactsUpdate: (id, patch) => ipcRenderer.invoke(SETTINGS_CHANNELS.contactsUpdate, id, patch),
		contactsRemove: (id) => ipcRenderer.invoke(SETTINGS_CHANNELS.contactsRemove, id),
		contactsTest: (id) => ipcRenderer.invoke(SETTINGS_CHANNELS.contactsTest, id),
		requestsList: (status) => ipcRenderer.invoke(SETTINGS_CHANNELS.requestsList, status),
		requestsRespond: (id, allow) => ipcRenderer.invoke(SETTINGS_CHANNELS.requestsRespond, id, allow),
		sourcesList: () => ipcRenderer.invoke(SETTINGS_CHANNELS.sourcesList),
		sourcesSetEnabled: (id, enabled) => ipcRenderer.invoke(SETTINGS_CHANNELS.sourcesSetEnabled, id, enabled),
		assistantSend: (surface, text) => ipcRenderer.invoke(SETTINGS_CHANNELS.assistantSend, surface, text),
		onEvent: (listener: (event: AssistantEvent) => void) => {
			const handler = (_event: Electron.IpcRendererEvent, payload: AssistantEvent): void => listener(payload);
			ipcRenderer.on(SETTINGS_CHANNELS.assistantEvent, handler);
			return () => ipcRenderer.removeListener(SETTINGS_CHANNELS.assistantEvent, handler);
		},
	} as SettingsBridge & { onEvent: (listener: (event: AssistantEvent) => void) => () => void };
}
