/**
 * Live E2E: the real `whatsapp-pi` package hosted by AutoRAG's interactive session.
 *
 * Gated: it installs the package from npm and opens a real WhatsApp Web socket (no account is
 * paired: WhatsApp answers with a pairing QR, which proves the extension loaded, the flags reached
 * it, and Baileys reached WhatsApp's servers). Run explicitly:
 *
 *   AUTORAG_WHATSAPP_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/whatsapp.test.ts
 *
 * With a paired account, `AUTORAG_WHATSAPP_PAIRED=1` is the manual-QA path (see docs).
 */
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveWhatsAppExtension } from "../../src/cli/whatsapp-extension.ts";

const LIVE = process.env.AUTORAG_WHATSAPP_LIVE === "1";
const root = mkdtempSync(join(tmpdir(), "autorag-wa-live-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

describe.skipIf(!LIVE)("whatsapp-pi hosted by the AutoRAG interactive session", () => {
	it("installs the package, loads it with the online flag, and reaches WhatsApp", async () => {
		const home = join(root, "home");
		mkdirSync(join(home, ".pi", "agent"), { recursive: true });
		process.env.HOME = home;
		process.env.PI_CODING_AGENT_DIR = join(home, ".pi", "agent");
		const whatsapp = await resolveWhatsAppExtension({ flags: { whatsapp: true }, cwd: root });
		expect(whatsapp?.paths[0]).toMatch(/whatsapp-pi\.ts$/);

		const agent = new AutoRAGAgent({
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			piAgentDir: process.env.PI_CODING_AGENT_DIR,
			piSessionDir: join(root, "sessions"),
			piExtensionPaths: whatsapp?.paths,
			piExtensionFlags: whatsapp?.flags,
		});
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			const notices: string[] = [];
			const status: Record<string, string | undefined> = {};
			// pi copies the UI context, so it must be a plain object, not a Proxy.
			const noop = (): void => undefined;
			const ui = {
				notify: (message: string) => notices.push(message),
				setStatus: (key: string, text: string | undefined) => {
					status[key] = text;
				},
				onTerminalInput: () => noop,
				select: async () => undefined,
				confirm: async () => false,
				input: async () => undefined,
				setWorkingMessage: noop,
				setWorkingVisible: noop,
				setWorkingIndicator: noop,
				setWidget: noop,
				setFooter: noop,
				setHeader: noop,
				setTitle: noop,
				getEditorText: () => "",
				setEditorText: noop,
			};
			await hosted.runtime.session.bindExtensions({ uiContext: ui as never });
			// With no stored credentials the online flag cannot connect; whatsapp-pi says so.
			await new Promise((r) => setTimeout(r, 3000));
			expect(notices.join("\n")).toMatch(/no saved WhatsApp credentials/i);
			expect(status.whatsapp).toBeDefined();
			const tools = hosted.runtime.session.getAllTools().map((t) => t.name);
			// Replies relay through message_end; the model gets no tool to message arbitrary numbers.
			expect(tools).not.toContain("send_wa_message");
		} finally {
			await hosted.dispose();
		}
	}, 240_000);
});
