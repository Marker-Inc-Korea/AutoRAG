import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { startUiServer, type UiServer } from "../../src/ui/server.ts";

let root: string;
let configPath: string;
let server: UiServer | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ui-server-"));
	configPath = join(root, "config.json");
	mkdirSync(join(root, "docs"));
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [join(root, "docs")],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
		}),
	);
});

afterEach(async () => {
	await server?.close();
	server = undefined;
	rmSync(root, { recursive: true, force: true });
});

async function request(
	target: UiServer,
	path: string,
	init: { method?: string; token?: string | null; body?: unknown } = {},
): Promise<{ status: number; text: string; json: unknown; headers: Headers }> {
	const url = new URL(path, target.origin);
	const headers: Record<string, string> = {};
	if (init.token) headers["x-autorag-token"] = init.token;
	if (init.body !== undefined) headers["content-type"] = "application/json";
	const response = await fetch(url, {
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
	return { status: response.status, text, json, headers: response.headers };
}

describe("datasource UI server", () => {
	it("refuses a non-loopback bind", async () => {
		await expect(startUiServer({ configPath, host: "0.0.0.0", port: 0, token: "t".repeat(32) })).rejects.toThrow(
			/loopback/i,
		);
	});

	it("requires the session token and never echoes connector secrets", async () => {
		server = await startUiServer({ configPath, host: "127.0.0.1", port: 0, token: "k".repeat(32) });
		expect(server.origin).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
		expect(server.url).toContain("token=");

		const denied = await request(server, "/api/state", { token: null });
		expect(denied.status).toBe(401);

		const page = await request(server, `/?token=${server.token}`, { token: null });
		expect(page.status).toBe(200);
		expect(page.text).toContain("Data sources");
		expect(page.headers.get("set-cookie") ?? "").toContain("autorag_ui=");

		const created = await request(server, "/api/connections", {
			method: "POST",
			token: server.token,
			body: {
				alias: "work-github",
				type: "github",
				enabled: true,
				connector: { tokenEnv: "GITHUB_TOKEN", token: "ghp_live-secret", repos: ["acme/repo"] },
			},
		});
		expect(created.status).toBe(200);
		expect(created.text).not.toContain("ghp_live-secret");

		const state = await request(server, "/api/state", { token: server.token });
		expect(state.status).toBe(200);
		expect(state.text).not.toContain("ghp_live-secret");
		expect(JSON.stringify(state.json)).toContain("work-github");

		const saved = readFileSync(configPath, "utf8");
		expect(saved).not.toContain("ghp_live-secret");
		expect(saved).toContain("GITHUB_TOKEN");
	});

	it("toggles, tests, browses, and removes through the local API", async () => {
		const nested = join(root, "docs", "nested");
		mkdirSync(nested);
		server = await startUiServer({ configPath, host: "127.0.0.1", port: 0, token: "k".repeat(32) });

		await request(server, "/api/connections", {
			method: "POST",
			token: server.token,
			body: {
				alias: "news",
				type: "rss",
				enabled: true,
				connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] },
			},
		});

		const probed = await request(server, "/api/connections/news/test", { method: "POST", token: server.token });
		expect(probed.status).toBe(200);
		expect((probed.json as { ok: boolean }).ok).toBe(true);

		const toggled = await request(server, "/api/connections/news/toggle", {
			method: "POST",
			token: server.token,
			body: { enabled: false },
		});
		expect(toggled.status).toBe(200);

		const browsed = await request(server, `/api/browse?path=${encodeURIComponent(join(root, "docs"))}`, {
			token: server.token,
		});
		expect(browsed.status).toBe(200);
		expect(JSON.stringify(browsed.json)).toContain("nested");

		const removed = await request(server, "/api/connections/news", { method: "DELETE", token: server.token });
		expect(removed.status).toBe(200);
		const state = await request(server, "/api/state", { token: server.token });
		const aliases = ((state.json as { connections?: Array<{ alias: string }> }).connections ?? []).map(
			(item) => item.alias,
		);
		expect(aliases).not.toContain("news");
	});
});
