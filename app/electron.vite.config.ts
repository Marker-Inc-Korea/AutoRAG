import { defineConfig } from "electron-vite";

export default defineConfig({
	main: {},
	preload: {},
	renderer: {
		esbuild: {
			jsx: "automatic",
		},
		server: {
			port: Number(process.env.PORT) || 5173,
		},
	},
});
