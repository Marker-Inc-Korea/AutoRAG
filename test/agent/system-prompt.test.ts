import { describe, expect, it } from "vitest";
import { toDatasourceAgentSkill } from "../../src/agent/datasource-skill.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { CloudDriveSkill } from "../../src/datasource/skills/cloud-drive/skill.ts";

function prompt(
	toolNames = [
		"bash",
		"search_all_documents",
		"semantic_search_local_docs",
		"semantic_search_local_docs",
		"check_memory",
		"emit_autorag_results",
	],
) {
	return buildSystemPrompt({ toolNames, manifests: [], jikjiIndexingEnabled: true, modelId: "test-model" });
}

describe("buildSystemPrompt single-agent contract", () => {
	it("assigns retrieval, reading, judgment, and curation to one agent", () => {
		const text = prompt();
		expect(text).toContain("retrieve candidates");
		expect(text).toContain("read the relevant source material directly");
		expect(text).toContain("judge the evidence");
		expect(text).toContain("emit_autorag_results");
		expect(text).toContain("generic, stable question");
		expect(text).toContain("baseline retrieval is already running in parallel");
		expect(text).toContain("Avoid spinning repeated near-identical queries against the same datasource");
		expect(text).toContain("Never repeat a generic status message");
		expect(text).toContain("generic, stable question");
		expect(text).toContain("baseline retrieval is already running in parallel");
		expect(text).toContain("Avoid spinning repeated near-identical queries against the same datasource");
		expect(text).toContain("Never repeat a generic status message");
		expect(text).not.toMatch(/subagent|explorer|delegat|Assignment V1|pi-subagents/i);
	});

	it("keeps retrieval, memory, datasource trust, and Jikji guidance", () => {
		const text = prompt();
		expect(text).toContain("search_all_documents");
		expect(text).toContain("semantic_search_local_docs");
		expect(text).toContain("semantic_search_local_docs");
		expect(text).toContain("check_memory");
		expect(text).toContain("default-deny");
		expect(text).toContain("Jikji Local Discovery");
	});

	it("fails closed when no tools are provided", () => {
		expect(prompt([])).toMatch(/blocked\/degraded state/i);
	});

	it("teaches the agent to load and search configured cloud-drive skills", () => {
		const skill = new CloudDriveSkill({
			instanceId: "icloud-docs",
			provider: "icloud",
			connector: { fetch: async () => ({ ok: true, documents: [] }) },
		});
		const prompt = buildSystemPrompt({
			toolNames: ["search_datasource_cloud_drive", "load_datasource_skill"],
			manifests: [],
			datasourceSkills: [toDatasourceAgentSkill(skill.skillManifest())],
		});

		expect(prompt).toContain("datasource-cloud-drive");
		expect(prompt).toContain("load_datasource_skill");
		expect(prompt).toContain("search_datasource_cloud_drive");
		const manifest = skill.skillManifest().content;
		expect(manifest).toContain("Google Drive");
		expect(manifest).toContain("OneDrive");
		expect(manifest).toMatch(/iCloud.*experimental/i);
		expect(manifest).toContain("/cloud-drive/icloud-docs");
	});

	it("lists multiple drive connections as independently loadable skills", () => {
		const personal = new CloudDriveSkill({
			skillName: "personal-google-drive",
			instanceId: "personal",
			provider: "google-drive",
			connector: { fetch: async () => ({ ok: true, documents: [] }) },
		});
		const work = new CloudDriveSkill({
			skillName: "company-onedrive",
			instanceId: "work",
			provider: "onedrive",
			connector: { fetch: async () => ({ ok: true, documents: [] }) },
		});
		const prompt = buildSystemPrompt({
			toolNames: [
				"search_datasource_personal_google_drive",
				"search_datasource_company_onedrive",
				"load_datasource_skill",
			],
			manifests: [],
			datasourceSkills: [personal, work].map((skill) => toDatasourceAgentSkill(skill.skillManifest())),
		});

		expect(prompt).toContain("datasource-personal-google-drive");
		expect(prompt).toContain("datasource-company-onedrive");
		expect(personal.skillManifest().content).toContain("/personal-google-drive/personal");
		expect(work.skillManifest().content).toContain("/company-onedrive/work");
	});

	it("frames AutoRAG as a librarian for document collections, cloud drives, images, and messenger history without codebases", () => {
		const text = prompt();
		expect(text).toContain("document collections, cloud drives, images, and messenger history");
		expect(text).not.toContain("librarian agent for codebases");
	});

	it("includes bullet-point answer guidelines, honest uncertainty, and omits per-source negative reports", () => {
		const text = prompt();
		expect(text).toContain("Answer Guidelines");
		expect(text).toContain("5 bullet points");
		expect(text).toContain("No per-source negative reports");
		expect(text).toContain("Honest and concise uncertainty");
		expect(text).not.toContain("Read before curating");
	});

	it("instructs resolving conflicting information in favor of the freshest data", () => {
		const text = prompt();
		expect(text).toContain("treat the freshest (most recent) information as authoritative");
		expect(text).toContain("Conflict resolution (recency preference)");
		expect(text).toContain("Prefer recent truth");
	});

	it("explains the fast answer first into deeper exploration two-phase workflow", () => {
		const text = prompt();
		expect(text).toContain("two-phase loop");
		expect(text).toContain("PLAN & FAST ANSWER");
		expect(text).toContain("EXPLORE & RETRIEVE");
	});

	it("instructs actively using Jikji when exploring local files and folders", () => {
		const text = prompt();
		expect(text).toContain("primary and preferred tool for exploring local files, folders, and documents");
		expect(text).toContain("actively use `jikji_find` rather than running exploratory `bash` commands");
	});
});
