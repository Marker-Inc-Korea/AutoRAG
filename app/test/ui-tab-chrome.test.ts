import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function declarations(selector: string): string {
	const css = readFileSync(new URL("../src/renderer/styles/finder.css", import.meta.url), "utf8");
	const match = css.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`));
	if (match === null) {
		throw new Error(`missing ${selector}`);
	}
	return match[1] ?? "";
}

describe("finder tab chrome", () => {
	it("does not paint a native button face behind the tab title", () => {
		const rule = declarations("\\.tab__title");
		expect(rule).toMatch(/appearance:\s*none/);
		expect(rule).toMatch(/background:\s*transparent/);
		expect(rule).toMatch(/border:\s*none/);
		expect(rule).toMatch(/padding:\s*0/);
		expect(rule).toMatch(/color:\s*inherit/);
	});
});
