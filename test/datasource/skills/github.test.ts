import { describe, expect, it } from "vitest";
import { GitHubConnector } from "../../../src/datasource/skills/github/connector.ts";
import { GitHubSkill } from "../../../src/datasource/skills/github/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const ISSUES = [
	{
		number: 12,
		title: "Crash on startup",
		body: "The app crashes when the config file is missing.",
		state: "open",
		updated_at: "2024-03-01T00:00:00.000Z",
		labels: [{ name: "bug" }],
	},
	{
		number: 13,
		title: "Add dark mode",
		body: "",
		state: "open",
		updated_at: "2024-03-02T00:00:00.000Z",
		labels: [],
		pull_request: { url: "pr" },
	},
];

describe("GitHubConnector", () => {
	it("returns not-configured without repos", async () => {
		expect(await new GitHubConnector({}).fetch()).toMatchObject({ ok: false, reason: "not-configured" });
	});

	it("works unauthenticated for public repos", async () => {
		const mock = createMockFetch([{ match: "/repos/acme/app/issues", json: ISSUES }]);
		const result = await new GitHubConnector({
			repos: ["acme/app"],
			tokenEnv: "GITHUB_TEST_UNSET",
			fetchImpl: mock.fetchImpl,
		}).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2);
			const issue = result.documents.find((d) => d.docId === "acme-app-12");
			expect(issue).toMatchObject({
				hierarchy: ["acme", "app", "issues"],
				title: "#12 Crash on startup",
				metadata: { kind: "issue", labels: ["bug"] },
			});
			const pull = result.documents.find((d) => d.docId === "acme-app-13");
			expect(pull).toMatchObject({ hierarchy: ["acme", "app", "pulls"], metadata: { kind: "pull" } });
		}
	});

	it("degrades a missing repo to a warning and keeps other repos", async () => {
		const mock = createMockFetch([
			{ match: "/repos/acme/missing/issues", status: 404 },
			{ match: "/repos/acme/app/issues", json: ISSUES },
		]);
		const result = await new GitHubConnector({
			repos: ["acme/missing", "acme/app"],
			fetchImpl: mock.fetchImpl,
		}).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2);
			expect(result.warnings?.some((w) => w.includes("acme-missing"))).toBe(true);
		}
	});

	it("maps 401 to auth and 403 to rate-limited", async () => {
		const authMock = createMockFetch([{ match: "/issues", status: 401 }]);
		expect(await new GitHubConnector({ repos: ["a/b"], fetchImpl: authMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "auth",
		});
		const rateMock = createMockFetch([{ match: "/issues", status: 403 }]);
		expect(await new GitHubConnector({ repos: ["a/b"], fetchImpl: rateMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "rate-limited",
		});
	});

	it("warns on malformed repo entries without failing", async () => {
		const mock = createMockFetch([{ match: "/repos/acme/app/issues", json: [] }]);
		const result = await new GitHubConnector({
			repos: ["not-a-repo", "acme/app"],
			fetchImpl: mock.fetchImpl,
		}).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.warnings?.length).toBeGreaterThan(0);
	});
});

describe("GitHubSkill", () => {
	it("indexes and searches with opaque /github sources", async () => {
		const mock = createMockFetch([{ match: "/repos/acme/app/issues", json: ISSUES }]);
		const skill = new GitHubSkill({
			instanceId: "acme",
			connectorOptions: { repos: ["acme/app"], token: "ghp_secret", fetchImpl: mock.fetchImpl },
		});
		expect(skill.describe()).toMatchObject({ name: "github", datasourceId: "github" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("crash startup config", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/github\/acme\/chunks\//);
		expect(JSON.stringify(hits)).not.toContain("ghp_secret");
	});
});
