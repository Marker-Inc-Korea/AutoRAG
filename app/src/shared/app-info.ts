export const APP_NAME = "AutoRAG Finder";
export const APP_VERSION = "0.1.0";

export function formatWindowTitle(appName: string, version: string): string {
	const normalizedVersion = version.startsWith("v") ? version.slice(1) : version;
	return `${appName} v${normalizedVersion}`;
}
