import { DefaultPackageManager, getAgentDir, SettingsManager } from "@earendil-works/pi-coding-agent";

/**
 * The most-installed pi package that drives a session from WhatsApp (QR pairing, allow-listed
 * contacts/groups, inbound messages become prompts, assistant replies go back). Pinned so a
 * published update never changes what runs unattended.
 */
export const WHATSAPP_PI_SOURCE = "npm:whatsapp-pi@1.0.70";

export interface WhatsAppExtension {
	readonly paths: string[];
	readonly flags: Readonly<Record<string, string | boolean>>;
}

export interface WhatsAppExtensionInput {
	readonly flags: Readonly<Record<string, string | boolean | undefined>>;
	/** Installs (if missing) the pi package sources and returns their extension entry files. */
	readonly resolveSources?: (sources: string[]) => Promise<string[]>;
	readonly cwd?: string;
}

async function installWithPi(cwd: string, sources: string[]): Promise<string[]> {
	const agentDir = getAgentDir();
	const packageManager = new DefaultPackageManager({
		cwd,
		agentDir,
		settingsManager: SettingsManager.create(cwd, agentDir),
	});
	const resolved = await packageManager.resolveExtensionSources(sources, { local: false, temporary: false });
	return resolved.extensions.filter((entry) => entry.enabled).map((entry) => entry.path);
}

/**
 * Resolves `autorag tui --whatsapp [--whatsapp-group JID]` into the extension entry files and flag
 * values the interactive session loads, or `undefined` when WhatsApp control was not requested.
 */
export async function resolveWhatsAppExtension(input: WhatsAppExtensionInput): Promise<WhatsAppExtension | undefined> {
	const group = input.flags["whatsapp-group"];
	if (input.flags.whatsapp !== true) {
		if (group !== undefined) throw new Error("--whatsapp-group requires --whatsapp.");
		return undefined;
	}
	const resolve = input.resolveSources ?? ((sources) => installWithPi(input.cwd ?? process.cwd(), sources));
	const paths = await resolve([WHATSAPP_PI_SOURCE]);
	if (paths.length === 0) throw new Error(`${WHATSAPP_PI_SOURCE} exposes no pi extension; cannot enable WhatsApp.`);
	return {
		paths,
		flags: {
			"whatsapp-pi-online": true,
			...(typeof group === "string" && group !== "" ? { "whatsapp-group": group } : {}),
		},
	};
}
