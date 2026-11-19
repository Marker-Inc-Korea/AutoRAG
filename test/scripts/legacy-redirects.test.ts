import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const SCRIPT = fileURLToPath(new URL("../../scripts/docs/legacy-redirects.mjs", import.meta.url));

let root: string;
let legacy: string;
let site: string;

function put(base: string, rel: string, body: string): void {
	const path = join(base, rel);
	mkdirSync(dirname(path), { recursive: true });
	writeFileSync(path, body);
}

function run(): string {
	return execFileSync(process.execPath, [SCRIPT, legacy, site], { encoding: "utf8" });
}

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "legacy-redirects-"));
	legacy = join(root, "legacy-build");
	site = join(root, "site");
	mkdirSync(site, { recursive: true });
});

afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("legacy docs redirect stubs", () => {
	it("keeps every old github.io/AutoRAG/<page>.html link alive by redirecting it into /legacy/", () => {
		put(legacy, "tutorial.html", "<html>legacy tutorial</html>");
		put(legacy, "nodes/retrieval/bm25.html", "<html>bm25</html>");
		put(legacy, "optimization/sample_config.html", "<html>sample</html>");
		run();

		for (const rel of ["tutorial.html", "nodes/retrieval/bm25.html", "optimization/sample_config.html"]) {
			const stub = readFileSync(join(site, rel), "utf8");
			expect(stub, rel).toContain(`url=/legacy/${rel}`);
			expect(stub, rel).toContain(`href="https://autorag-docs.nomadamas.org/legacy/${rel}"`);
			// A deep link to a section (`#heading`) and a search query must survive the hop.
			expect(stub, rel).toContain("location.search");
			expect(stub, rel).toContain("location.hash");
		}
	});

	it("never replaces a page the new site already serves", () => {
		put(site, "index.html", "NEW SITE ROOT");
		put(site, "troubleshooting.html", "NEW TROUBLESHOOTING");
		put(legacy, "index.html", "legacy index");
		put(legacy, "troubleshooting.html", "legacy troubleshooting");
		put(legacy, "install.html", "legacy install");
		run();

		expect(readFileSync(join(site, "index.html"), "utf8")).toBe("NEW SITE ROOT");
		expect(readFileSync(join(site, "troubleshooting.html"), "utf8")).toBe("NEW TROUBLESHOOTING");
		expect(existsSync(join(site, "install.html"))).toBe(true);
	});

	it("only stubs pages, not assets or build metadata", () => {
		put(legacy, "tutorial.html", "x");
		put(legacy, "_static/pygments.css", "x");
		put(legacy, "searchindex.js", "x");
		put(legacy, ".buildinfo", "x");
		run();

		expect(existsSync(join(site, "tutorial.html"))).toBe(true);
		expect(existsSync(join(site, "_static"))).toBe(false);
		expect(existsSync(join(site, "searchindex.js"))).toBe(false);
		expect(existsSync(join(site, ".buildinfo"))).toBe(false);
	});

	it("escapes the target so an unusual file name cannot break out of the markup", () => {
		put(legacy, 'weird"><script>alert(1)</script>.html', "x");
		put(legacy, "with space & amp.html", "x");
		run();

		const stub = readFileSync(join(site, "with space & amp.html"), "utf8");
		expect(stub).toContain("url=/legacy/with%20space%20&amp;%20amp.html");
		expect(stub).not.toContain("<script>alert(1)");
	});

	it("fails loudly when the legacy build is missing, so a broken Sphinx build cannot ship silently", () => {
		expect(() => execFileSync(process.execPath, [SCRIPT, join(root, "nope"), site], { stdio: "pipe" })).toThrow();
	});
});
