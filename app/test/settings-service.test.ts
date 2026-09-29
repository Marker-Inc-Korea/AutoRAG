import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createSettingsService } from "../src/main/settings-service";
import type { AssistantSurface, DataSourceRow } from "../src/shared/settings-contract";

const directories: string[] = [];

afterEach(async () => {
	await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

function createFakeSession(surface: AssistantSurface, events: unknown[]): {
	readonly agent: { subscribe(listener: (event: unknown) => void): () => void };
	prompt(text: string): Promise<void>;
	dispose(): void;
} {
	const listeners = new Set<(event: unknown) => void>();
	return {
		agent: {
			subscribe(listener) {
				listeners.add(listener);
				return () => listeners.delete(listener);
			},
		},
		async prompt(text) {
			expect(surface).toBe("models");
			expect(text).toBe("Which model is active?");
			for (const event of events) for (const listener of listeners) listener(event);
		},
		dispose() {
			listeners.clear();
		},
	};
}

describe("createSettingsService", () => {
	it("persists settings, permissions, and contacts", async () => {
		const directory = await mkdtemp(join(tmpdir(), "autorag-settings-"));
		directories.push(directory);
		const service = createSettingsService({
			directory,
			createChatSession: () => createFakeSession("models", []),
			send: () => undefined,
		});

		await service.set({ telemetry: true });
		await service.permSet("/tmp/docs", "allow", "apply-all");
		expect((await service.permGet("/tmp/docs/readme.md")).value).toBe("allow");
		expect((await service.contactsAdd({ id: "alice", name: "Alice", role: "Reviewer", description: "Reviews docs" })).ok).toBe(true);
		expect((await service.contactsList()).map((contact) => contact.id)).toEqual(["alice"]);

		const reloaded = createSettingsService({
			directory,
			createChatSession: () => createFakeSession("models", []),
			send: () => undefined,
		});
		expect((await reloaded.get()).telemetry).toBe(true);
		expect((await reloaded.contactsList()).map((contact) => contact.id)).toEqual(["alice"]);
	});

	it("drives the settings assistant through the supplied chat session", async () => {
		const directory = await mkdtemp(join(tmpdir(), "autorag-settings-chat-"));
		directories.push(directory);
		const events: unknown[] = [];
		const service = createSettingsService({
			directory,
			createChatSession: (surface) =>
				createFakeSession(surface, [
					{ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: "Use the active model." } },
				]),
			send: (_channel, event) => events.push(event),
		});

		await service.assistantSend("models", "Which model is active?");

		expect(events).toEqual([
			{ type: "text", surface: "models", delta: "Use the active model." },
			{ type: "done", surface: "models" },
		]);
	});

	it("defaults showHiddenFiles off and persists the toggle", async () => {
		const directory = await mkdtemp(join(tmpdir(), "autorag-settings-hidden-"));
		directories.push(directory);
		const service = createSettingsService({
			directory,
			createChatSession: () => createFakeSession("models", []),
			send: () => undefined,
		});

		expect((await service.get()).showHiddenFiles).toBe(false);

		await service.set({ showHiddenFiles: true });
		const reloaded = createSettingsService({
			directory,
			createChatSession: () => createFakeSession("models", []),
			send: () => undefined,
		});
		expect((await reloaded.get()).showHiddenFiles).toBe(true);
	});

	it("hydrates configured datasource rows and persists toggles through AutoRAG config", async () => {
		const directory = await mkdtemp(join(tmpdir(), "autorag-settings-sources-"));
		directories.push(directory);
		const sources: readonly DataSourceRow[] = [{
			id: "team-slack",
			name: "team-slack",
			kind: "slack",
			detail: "Team messages",
			status: "indexed",
			progress: null,
			enabled: true,
			description: "Team Slack",
		}];
		const toggles: Array<{ id: string; enabled: boolean }> = [];
		const service = createSettingsService({
			directory,
			initialSources: sources,
			setSourceEnabled: async (id, enabled) => {
				toggles.push({ id, enabled });
			},
			createChatSession: () => createFakeSession("models", []),
			send: () => undefined,
		});

		expect(await service.sourcesList()).toEqual(sources);
		await service.sourcesSetEnabled("team-slack", false);

		expect(toggles).toEqual([{ id: "team-slack", enabled: false }]);
		expect((await service.sourcesList())[0]?.enabled).toBe(false);
	});
});
