import { describe, expect, it } from "vitest";
import { toDatasourceAgentSkill } from "../../src/agent/datasource-skill.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { CloudDriveSkill } from "../../src/datasource/skills/cloud-drive/skill.ts";

function prompt(
	toolNames = [
		"bash",
		"search_all_documents",
		"lexical_search_local_docs",
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
		expect(text).toContain("Do not query the same datasource more than three times");
		expect(text).toContain("Never repeat a generic status message");
		expect(text).toContain("generic, stable question");
		expect(text).toContain("baseline retrieval is already running in parallel");
		expect(text).toContain("Do not query the same datasource more than three times");
		expect(text).toContain("Never repeat a generic status message");
		expect(text).not.toMatch(/subagent|explorer|delegat|Assignment V1|pi-subagents/i);
	});

	it("keeps retrieval, memory, datasource trust, and Jikji guidance", () => {
		const text = prompt();
		expect(text).toContain("search_all_documents");
		expect(text).toContain("lexical_search_local_docs");
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
			toolNames: ["search_datasource_documents", "load_datasource_skill"],
			manifests: [],
			datasourceSkills: [toDatasourceAgentSkill(skill.skillManifest())],
		});

		expect(prompt).toContain("datasource-cloud-drive");
		expect(prompt).toContain("load_datasource_skill");
		expect(prompt).toContain("search_datasource_documents");
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
			toolNames: ["search_datasource_documents", "load_datasource_skill"],
			manifests: [],
			datasourceSkills: [personal, work].map((skill) => toDatasourceAgentSkill(skill.skillManifest())),
		});

		expect(prompt).toContain("datasource-personal-google-drive");
		expect(prompt).toContain("datasource-company-onedrive");
		expect(personal.skillManifest().content).toContain("/personal-google-drive/personal");
		expect(work.skillManifest().content).toContain("/company-onedrive/work");
	});
});
