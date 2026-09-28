import { ipcMain } from "electron";
import { FS_CHANNELS, type FsBridge, type FsClipboard } from "../shared/fs-contract";

/** Register one ipcMain.handle per FS_CHANNELS channel, in contract order. */
export function registerFsIpcHandlers(service: FsBridge): void {
	ipcMain.handle(FS_CHANNELS.listDir, (_event, path: string) => service.listDir(path));
	ipcMain.handle(FS_CHANNELS.stat, (_event, path: string) => service.stat(path));
	ipcMain.handle(FS_CHANNELS.search, (_event, query: string) => service.search(query));
	ipcMain.handle(FS_CHANNELS.locations, () => service.locations());
	ipcMain.handle(FS_CHANNELS.copy, (_event, paths: readonly string[], destDir: string) => service.copy(paths, destDir));
	ipcMain.handle(FS_CHANNELS.move, (_event, paths: readonly string[], destDir: string) => service.move(paths, destDir));
	ipcMain.handle(FS_CHANNELS.duplicate, (_event, paths: readonly string[]) => service.duplicate(paths));
	ipcMain.handle(FS_CHANNELS.rename, (_event, path: string, newName: string) => service.rename(path, newName));
	ipcMain.handle(FS_CHANNELS.trash, (_event, paths: readonly string[]) => service.trash(paths));
	ipcMain.handle(FS_CHANNELS.reveal, (_event, path: string) => service.reveal(path));
	ipcMain.handle(FS_CHANNELS.quickLook, (_event, path: string) => service.quickLook(path));
	ipcMain.handle(FS_CHANNELS.clipboardSet, (_event, clipboard: FsClipboard) => service.clipboardSet(clipboard));
	ipcMain.handle(FS_CHANNELS.clipboardGet, () => service.clipboardGet());
	ipcMain.handle(FS_CHANNELS.copyPathsToClipboard, (_event, paths: readonly string[]) =>
		service.copyPathsToClipboard(paths),
	);
}
