import { describe, expect, it } from "vitest";
import { parseArgs } from "../../src/cli/index.ts";
import { resolveWhatsAppExtension, WHATSAPP_PI_SOURCE } from "../../src/cli/whatsapp-extension.ts";

describe("autorag tui --whatsapp", () => {
	it("parses --whatsapp and --whatsapp-group", () => {
		const parsed = parseArgs(["tui", "--whatsapp", "--whatsapp-group", "120363@g.us"]);
		expect("error" in parsed).toBe(false);
		if ("error" in parsed) return;
		expect(parsed.flags.whatsapp).toBe(true);
		expect(parsed.flags["whatsapp-group"]).toBe("120363@g.us");
	});

	it("installs the pinned whatsapp-pi package and brings it online", async () => {
		const requested: string[][] = [];
		const result = await resolveWhatsAppExtension({
			flags: { whatsapp: true },
			resolveSources: async (sources) => {
				requested.push(sources);
				return ["/agent/npm/node_modules/whatsapp-pi/whatsapp-pi.ts"];
			},
		});
		expect(requested).toEqual([[WHATSAPP_PI_SOURCE]]);
		expect(result?.paths).toEqual(["/agent/npm/node_modules/whatsapp-pi/whatsapp-pi.ts"]);
		expect(result?.flags).toEqual({ "whatsapp-pi-online": true });
	});

	it("binds the agent to one group when --whatsapp-group is given", async () => {
		const result = await resolveWhatsAppExtension({
			flags: { whatsapp: true, "whatsapp-group": "120363@g.us" },
			resolveSources: async () => ["/x/whatsapp-pi.ts"],
		});
		expect(result?.flags).toEqual({ "whatsapp-pi-online": true, "whatsapp-group": "120363@g.us" });
	});

	it("does nothing without --whatsapp, and rejects --whatsapp-group alone", async () => {
		const never = async (): Promise<string[]> => {
			throw new Error("must not install");
		};
		expect(await resolveWhatsAppExtension({ flags: {}, resolveSources: never })).toBeUndefined();
		await expect(
			resolveWhatsAppExtension({ flags: { "whatsapp-group": "1@g.us" }, resolveSources: never }),
		).rejects.toThrow(/--whatsapp/);
	});

	it("fails clearly when the package exposes no extension entry", async () => {
		await expect(
			resolveWhatsAppExtension({ flags: { whatsapp: true }, resolveSources: async () => [] }),
		).rejects.toThrow(/whatsapp-pi/);
	});
});
