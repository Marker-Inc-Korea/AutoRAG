import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";

let root: string;
let identityHome: string;
let identityPath: string;
let originalCwd: string;

beforeEach(() => {
	originalCwd = process.cwd();
	root = mkdtempSync(join(tmpdir(), "autorag-identity-test-"));
	identityHome = join(root, "autorag-home");
	identityPath = join(identityHome, "IDENTITY.md");
	for (const directory of [identityHome, join(root, "user-home"), join(root, "workspace"), join(root, "docs")]) {
		mkdirSync(directory, { recursive: true });
	}
	vi.stubEnv("HOME", join(root, "user-home"));
	vi.stubEnv("USERPROFILE", join(root, "user-home"));
	vi.stubEnv("AUTORAG_HOME", identityHome);
	vi.stubEnv("AUTORAG_CONFIG", join(identityHome, "config.json"));
	vi.stubEnv("PI_CODING_AGENT_DIR", join(root, "pi-agent"));
});

afterEach(() => {
	process.chdir(originalCwd);
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true });
});

function makeAgent(overrides: Partial<AutoRAGAgentOptions> = {}): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [join(root, "docs")],
		workspacePath: join(root, "workspace"),
		memoryPath: join(root, "memory.json"),
		piAgentDir: join(root, "pi-agent"),
		minSync: { autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		dupey: false,
		filee: false,
		memoryEmbedder: false,
		persistPiSessions: false,
		...overrides,
	});
}

describe("AutoRAGAgent identity file", () => {
	it("leaves the existing prompt unchanged when the file is absent and does not create a file", () => {
		const disabledPrompt = makeAgent({ loadIdentity: false }).getSystemPrompt();

		expect(makeAgent().getSystemPrompt()).toBe(disabledPrompt);
		expect(existsSync(identityPath)).toBe(false);
	});

	it.each(["", " \t\r\n ", "\u2003\u00a0\n"])(
		"leaves the existing prompt unchanged for blank content %j",
		(content) => {
			const baseline = makeAgent().getSystemPrompt();
			writeFileSync(identityPath, content);

			expect(makeAgent().getSystemPrompt()).toBe(baseline);
			expect(readFileSync(identityPath, "utf8")).toBe(content);
		},
	);

	it("includes nonblank UTF-8 content exactly once without normalizing whitespace or line endings", () => {
		const identity = "  \r\n# IDENTITY-RAW-943b\r\n称呼：小林 🌿\r\nLanguage: 日本語\r\n\t  ";
		writeFileSync(identityPath, identity);

		const prompt = makeAgent().getSystemPrompt();

		expect(prompt).toContain(identity);
		expect(prompt.split("IDENTITY-RAW-943b")).toHaveLength(2);
		expect(readFileSync(identityPath, "utf8")).toBe(identity);
	});

	it("snapshots identity at construction and observes edits or deletion only in new agents", () => {
		const baseline = makeAgent().getSystemPrompt();
		const firstIdentity = "IDENTITY-SNAPSHOT-A: research librarian";
		const secondIdentity = "IDENTITY-SNAPSHOT-B: documentation editor";
		writeFileSync(identityPath, firstIdentity);
		const firstAgent = makeAgent();
		const firstPrompt = firstAgent.getSystemPrompt();
		expect(firstPrompt).toContain(firstIdentity);

		writeFileSync(identityPath, secondIdentity);
		const secondAgent = makeAgent();
		const secondPrompt = secondAgent.getSystemPrompt();
		expect(firstAgent.getSystemPrompt()).toBe(firstPrompt);
		expect(secondPrompt).toContain(secondIdentity);
		expect(secondPrompt).not.toContain(firstIdentity);

		unlinkSync(identityPath);
		expect(firstAgent.getSystemPrompt()).toBe(firstPrompt);
		expect(secondAgent.getSystemPrompt()).toBe(secondPrompt);
		expect(makeAgent().getSystemPrompt()).toBe(baseline);
		expect(existsSync(identityPath)).toBe(false);
	});

	it("keeps agents constructed under different AutoRAG homes independent", () => {
		const firstIdentity = "IDENTITY-HOME-A: technical writer";
		const secondIdentity = "IDENTITY-HOME-B: materials scientist";
		writeFileSync(identityPath, firstIdentity);
		const firstAgent = makeAgent();
		const secondHome = join(root, "second-autorag-home");
		mkdirSync(secondHome);
		writeFileSync(join(secondHome, "IDENTITY.md"), secondIdentity);
		vi.stubEnv("AUTORAG_HOME", secondHome);
		vi.stubEnv("AUTORAG_CONFIG", join(secondHome, "config.json"));
		const secondAgent = makeAgent();

		expect(firstAgent.getSystemPrompt()).toContain(firstIdentity);
		expect(firstAgent.getSystemPrompt()).not.toContain(secondIdentity);
		expect(secondAgent.getSystemPrompt()).toContain(secondIdentity);
		expect(secondAgent.getSystemPrompt()).not.toContain(firstIdentity);
	});

	it.each([
		{ name: "explicit runtime opt-out", options: { loadIdentity: false } },
		{ name: "Discord bridge", options: { discordRemote: true } },
		{ name: "Telegram bridge", options: { telegram: true } },
		{ name: "Discord bridge with explicit loadIdentity", options: { discordRemote: true, loadIdentity: true } },
		{ name: "Telegram bridge with explicit loadIdentity", options: { telegram: true, loadIdentity: true } },
	])("does not read an invalid identity path for $name", ({ options }) => {
		const baseline = makeAgent(options).getSystemPrompt();
		mkdirSync(identityPath);

		expect(makeAgent(options).getSystemPrompt()).toBe(baseline);
	});

	it("omits a valid identity file when loading is explicitly disabled", () => {
		const baseline = makeAgent().getSystemPrompt();
		const identity = "IDENTITY-DISABLED-14c2: private user background";
		writeFileSync(identityPath, identity);

		expect(makeAgent({ loadIdentity: false }).getSystemPrompt()).toBe(baseline);
		expect(readFileSync(identityPath, "utf8")).toBe(identity);
	});

	it("does not misclassify an ordinary Pi extension as a remote bridge", () => {
		const identity = "IDENTITY-LOCAL-EXTENSION-782e: local owner";
		writeFileSync(identityPath, identity);
		const extensionPath = join(root, "local-extension.mjs");
		writeFileSync(extensionPath, "export default function () {}\n");

		expect(makeAgent({ piExtensionPaths: [extensionPath] }).getSystemPrompt()).toContain(identity);
	});

	it("preserves native filesystem errors other than a missing file", () => {
		mkdirSync(identityPath);
		let nativeError: NodeJS.ErrnoException | undefined;
		try {
			readFileSync(identityPath, "utf8");
		} catch (caught) {
			nativeError = caught as NodeJS.ErrnoException;
		}
		expect(nativeError).toMatchObject({ code: "EISDIR" });
		let error: unknown;
		try {
			makeAgent();
		} catch (caught) {
			error = caught;
		}

		expect(error).toMatchObject({ code: "EISDIR", message: nativeError?.message });
		expect((error as NodeJS.ErrnoException).path).toBe(nativeError?.path);
	});

	it("uses canonical AutoRAG home regardless of cwd, workspace, memory, config, or search-root locations", () => {
		const identity = "IDENTITY-CANONICAL-HOME-401f: home identity";
		writeFileSync(identityPath, identity);
		const locations = ["alternate-cwd", "workspace", "memory", "configuration", "docs"];
		for (const location of locations) {
			const directory = join(root, location);
			mkdirSync(directory, { recursive: true });
			writeFileSync(join(directory, "IDENTITY.md"), `IDENTITY-DECOY-${location}`);
		}
		vi.stubEnv("AUTORAG_CONFIG", join(root, "configuration", "config.json"));
		process.chdir(join(root, "alternate-cwd"));
		const options = { memoryPath: join(root, "memory", "custom-memory.json") };
		const prompt = makeAgent(options).getSystemPrompt();

		expect(prompt).toContain(identity);
		expect(prompt).not.toContain("IDENTITY-DECOY-");
		unlinkSync(identityPath);
		expect(makeAgent(options).getSystemPrompt()).toBe(
			makeAgent({ ...options, loadIdentity: false }).getSystemPrompt(),
		);
		expect(existsSync(identityPath)).toBe(false);
	});

	it("uses the isolated user's .autorag home when AUTORAG_HOME is unset", () => {
		const defaultHome = join(root, "user-home", ".autorag");
		mkdirSync(defaultHome);
		const identity = "IDENTITY-DEFAULT-HOME-e0bc: default home owner";
		writeFileSync(join(defaultHome, "IDENTITY.md"), identity);
		vi.stubEnv("AUTORAG_HOME", undefined);
		vi.stubEnv("AUTORAG_CONFIG", join(defaultHome, "config.json"));

		expect(makeAgent().getSystemPrompt()).toContain(identity);
	});
});
