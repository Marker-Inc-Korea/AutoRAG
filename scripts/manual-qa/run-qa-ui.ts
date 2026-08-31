/**
 * Manual QA for `autorag ui` (#1486).
 *
 * Starts the real loopback server against a temp config, then walks the
 * operator surface: page load, add GitHub/RSS, secret stripping, toggle,
 * browse, folder save, unauthorized request.
 *
 * Run: bun scripts/manual-qa/run-qa-ui.ts
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startUiServer } from "../../src/ui/server.ts";

interface CheckResult {
	name: string;
	pass: boolean;
	note?: string;
}

const results: CheckResult[] = [];
function check(name: string, pass: boolean, note?: string): void {
	results.push({ name, pass, note });
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-ui-qa-"));
const configPath = join(tmpRoot, "config.json");
mkdirSync(join(tmpRoot, "docs", "nested"), { recursive: true });
writeFileSync(
	configPath,
	JSON.stringify({
		searchPaths: [join(tmpRoot, "docs")],
		workspacePath: tmpRoot,
		memoryPath: join(tmpRoot, "memory.json"),
	}),
);

const token = "qa".repeat(16);
const server = await startUiServer({ configPath, host: "127.0.0.1", port: 0, token });

async function api(path: string, init: { method?: string; token?: string | null; body?: unknown } = {}) {
	const headers: Record<string, string> = {};
	if (init.token) headers["x-autorag-token"] = init.token;
	if (init.body !== undefined) headers["content-type"] = "application/json";
	const response = await fetch(new URL(path, server.origin), {
		method: init.method ?? "GET",
		headers,
		body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
	});
	const text = await response.text();
	let json: unknown;
	try {
		json = JSON.parse(text);
	} catch {
		json = undefined;
	}
	return { status: response.status, text, json };
}

try {
	const denied = await api("/api/state", { token: null });
	check("unauthenticated state is 401", denied.status === 401);

	const page = await api(`/?token=${token}`, { token: null });
	check("page renders", page.status === 200 && page.text.includes("Data sources"), `status=${page.status}`);

	const created = await api("/api/connections", {
		method: "POST",
		token,
		body: {
			alias: "work-github",
			type: "github",
			enabled: true,
			connector: { tokenEnv: "GITHUB_TOKEN", token: "ghp_manual-secret", repos: ["Marker-Inc-Korea/AutoRAG"] },
		},
	});
	const configText = readFileSync(configPath, "utf8");
	check("GitHub connection saves", created.status === 200 && configText.includes("work-github"));
	check("token value is not persisted", !configText.includes("ghp_manual-secret") && !created.text.includes("ghp_manual-secret"));
	check("datasourceAccess is written", configText.includes("/work-github/**") && configText.includes("github"));

	const rss = await api("/api/connections", {
		method: "POST",
		token,
		body: { alias: "news", type: "rss", enabled: true, connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] } },
	});
	check("RSS connection saves", rss.status === 200);

	const probed = await api("/api/connections/news/test", { method: "POST", token });
	check("RSS probe is ready", probed.status === 200 && (probed.json as { ok?: boolean }).ok === true);

	const toggled = await api("/api/connections/news/toggle", { method: "POST", token, body: { enabled: false } });
	check("disable connection", toggled.status === 200);

	const browsed = await api(`/api/browse?path=${encodeURIComponent(join(tmpRoot, "docs"))}`, { token });
	check("browse lists nested folder", browsed.status === 200 && browsed.text.includes("nested"));

	const extra = join(tmpRoot, "extra");
	mkdirSync(extra);
	const folders = await api("/api/folders", {
		method: "POST",
		token,
		body: { searchPaths: [join(tmpRoot, "docs"), extra] },
	});
	check("save folders", folders.status === 200 && readFileSync(configPath, "utf8").includes("extra"));

	const removed = await api("/api/connections/work-github", { method: "DELETE", token });
	check("remove connection", removed.status === 200 && !readFileSync(configPath, "utf8").includes("work-github"));

	let rejected = false;
	try {
		await startUiServer({ configPath, host: "0.0.0.0", port: 0, token });
	} catch (error) {
		rejected = error instanceof Error && /loopback/i.test(error.message);
	}
	check("non-loopback bind is rejected", rejected);
} finally {
	await server.close();
	rmSync(tmpRoot, { recursive: true, force: true });
}

const failed = results.filter((item) => !item.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed.`);
if (failed.length > 0) process.exitCode = 1;
