import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { describe, expect, it, vi } from "vitest";
import autoragExtension from "../../src/extension.ts";

describe("autoragExtension", () => {
	it("is a function", () => {
		expect(typeof autoragExtension).toBe("function");
	});

	it("registers check_memory tool and Pi event handlers", () => {
		const registerTool = vi.fn();
		const on = vi.fn();
		const pi = {
			registerTool,
			on,
		} as Partial<ExtensionAPI> as ExtensionAPI;

		autoragExtension(pi);

		expect(registerTool).toHaveBeenCalledTimes(1);
		expect(registerTool.mock.calls[0][0].name).toBe("check_memory");
		const events = on.mock.calls.map((call) => call[0]);
		expect(events).toContain("session_start");
		expect(events).toContain("tool_result");
		expect(events).toContain("before_agent_start");
		expect(events).toContain("message_end");
	});
});
