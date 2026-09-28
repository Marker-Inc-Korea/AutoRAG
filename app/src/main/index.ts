import { join } from "node:path";
import { app, BrowserWindow, clipboard, shell } from "electron";
import { createFsService } from "./fs-service";
import { registerFsIpcHandlers } from "./ipc";

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
	registerFsIpcHandlers(createFsService({ shell, clipboard }));
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
