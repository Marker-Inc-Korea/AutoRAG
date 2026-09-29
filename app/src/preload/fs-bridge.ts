import type { IpcRenderer } from "electron";
import { FS_CHANNELS, type FsBridge } from "../shared/fs-contract";

/** Typed window.autorag.fs wrappers over ipcRenderer.invoke on the contract channels. */
export function createFsBridge(ipcRenderer: IpcRenderer): FsBridge {
	return {
		listDir: (path) => ipcRenderer.invoke(FS_CHANNELS.listDir, path),
		stat: (path) => ipcRenderer.invoke(FS_CHANNELS.stat, path),
		search: (query) => ipcRenderer.invoke(FS_CHANNELS.search, query),
		locations: () => ipcRenderer.invoke(FS_CHANNELS.locations),
		copy: (paths, destDir) => ipcRenderer.invoke(FS_CHANNELS.copy, paths, destDir),
		move: (paths, destDir) => ipcRenderer.invoke(FS_CHANNELS.move, paths, destDir),
		duplicate: (paths) => ipcRenderer.invoke(FS_CHANNELS.duplicate, paths),
		rename: (path, newName) => ipcRenderer.invoke(FS_CHANNELS.rename, path, newName),
		trash: (paths) => ipcRenderer.invoke(FS_CHANNELS.trash, paths),
		reveal: (path) => ipcRenderer.invoke(FS_CHANNELS.reveal, path),
		quickLook: (path) => ipcRenderer.invoke(FS_CHANNELS.quickLook, path),
		clipboardSet: (clipboard) => ipcRenderer.invoke(FS_CHANNELS.clipboardSet, clipboard),
		clipboardGet: () => ipcRenderer.invoke(FS_CHANNELS.clipboardGet),
		copyPathsToClipboard: (paths) => ipcRenderer.invoke(FS_CHANNELS.copyPathsToClipboard, paths),
		versionFamilies: () => ipcRenderer.invoke(FS_CHANNELS.versionFamilies),
	};
}
