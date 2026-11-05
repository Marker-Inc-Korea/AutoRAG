import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	type CloudReleaseExtensionOptions,
	createCloudReleaseExtension,
} from "../../src/agent/cloud-release-extension.ts";
import type {
	CloudEvictResult,
	CloudFileProbe,
	CloudProvider,
	CloudRoot,
} from "../../src/filesystem/cloud-placeholder.ts";

let root: string;
let cloud: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-cloud-release-")));
	cloud = join(root, "cloud");
	mkdirSync(cloud, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** A probe over temp files: every path in `placeholders` is dataless, and eviction records into `released`. */
class FakeProbe implements CloudFileProbe {
	readonly placeholders = new Set<string>();
	readonly released: string[] = [];
	readonly datalessCalls: string[][] = [];
	canReleaseAny = true;
	evictResult: CloudEvictResult = { ok: true };
	async cloudRoots(): Promise<readonly CloudRoot[]> {
		return [];
	}
	async datalessAmong(paths: readonly string[]): Promise<ReadonlySet<string>> {
		this.datalessCalls.push([...paths]);
		return new Set(paths.filter((path) => this.placeholders.has(path)));
	}
	async providerOf(): Promise<CloudProvider | undefined> {
		return "icloud";
	}
	canRelease(): boolean {
		return this.canReleaseAny;
	}
	async evict(path: string): Promise<CloudEvictResult> {
		if (this.evictResult.ok) this.released.push(path);
		return this.evictResult;
	}
}

interface Harness {
	fire(event: string, payload?: unknown): Promise<void>;
	readonly handlerCount: number;
}

/** Installs the extension against a fake pi, exposing the handlers it registered. */
function harness(
	probe: CloudFileProbe | undefined,
	roots: readonly CloudRoot[],
	options: Partial<CloudReleaseExtensionOptions> = {},
): Harness {
	const handlers = new Map<string, ((event: unknown, ctx: ExtensionContext) => unknown)[]>();
	const pi = {
		on: (event: string, handler: (event: unknown, ctx: ExtensionContext) => unknown) => {
			handlers.set(event, [...(handlers.get(event) ?? []), handler]);
			return () => undefined;
		},
	} as unknown as ExtensionAPI;
	const ctx = {} as unknown as ExtensionContext;
	createCloudReleaseExtension({ probe, cloudRoots: async () => roots, ...options })(pi);
	return {
		handlerCount: handlers.size,
		fire: async (event, payload = {}) => {
			for (const handler of handlers.get(event) ?? []) await handler(payload, ctx);
		},
	};
}

/** A file in the cloud root; the extension never reads its bytes. */
function cloudFile(name: string): string {
	const path = join(cloud, name);
	writeFileSync(path, "x");
	return path;
}

const roots = (): CloudRoot[] => [{ path: cloud, kind: "file-provider" }];

describe("createCloudReleaseExtension", () => {
	it("releases a placeholder a bash command named, and never a file that was already local", async () => {
		const placeholder = cloudFile("a.pdf");
		const local = cloudFile("b.pdf");
		const probe = new FakeProbe();
		probe.placeholders.add(placeholder);
		const { fire } = harness(probe, roots());
		await fire("tool_call", {
			toolName: "bash",
			input: { command: `pdftotext "${placeholder}" "${local}" -` },
		});
		// The run opens the placeholder, which hydrates it.
		probe.placeholders.delete(placeholder);
		await fire("agent_end");
		expect(probe.released).toEqual([placeholder]);
	});

	it("covers a file reached by cd into a cloud directory and a relative name", async () => {
		const dir = join(cloud, "Downloads");
		mkdirSync(dir, { recursive: true });
		const file = join(dir, "report.pdf");
		writeFileSync(file, "x");
		const probe = new FakeProbe();
		probe.placeholders.add(file);
		const { fire } = harness(probe, roots());
		await fire("tool_call", { toolName: "bash", input: { command: `cd "${dir}" && pdftotext "report.pdf" -` } });
		probe.placeholders.delete(file);
		await fire("agent_end");
		expect(probe.released).toEqual([file]);
	});

	it("covers a source seen only in a tool result", async () => {
		const file = cloudFile("source.pdf");
		const probe = new FakeProbe();
		probe.placeholders.add(file);
		const { fire } = harness(probe, roots());
		await fire("tool_result", {
			toolName: "search_all_documents",
			input: {},
			isError: false,
			content: [{ type: "text", text: `Found: ${file}` }],
		});
		probe.placeholders.delete(file);
		await fire("agent_end");
		expect(probe.released).toEqual([file]);
	});

	it("honors the directory child cap", async () => {
		const dir = join(cloud, "many");
		mkdirSync(dir, { recursive: true });
		const probe = new FakeProbe();
		const files: string[] = [];
		for (let index = 0; index < 5; index += 1) {
			const file = join(dir, `f${index}.pdf`);
			writeFileSync(file, "x");
			files.push(file);
			probe.placeholders.add(file);
		}
		const { fire } = harness(probe, roots(), { childCap: 2 });
		await fire("tool_call", { toolName: "read", input: { path: dir } });
		for (const file of files) probe.placeholders.delete(file);
		await fire("agent_end");
		expect(probe.released).toHaveLength(2);
		expect(files).toEqual(expect.arrayContaining(probe.released));
	});

	it("does not release a placeholder the run never hydrated", async () => {
		const file = cloudFile("untouched.pdf");
		const probe = new FakeProbe();
		probe.placeholders.add(file);
		const { fire } = harness(probe, roots());
		await fire("tool_call", { toolName: "read", input: { path: file } });
		await fire("agent_end");
		expect(probe.released).toEqual([]);
	});

	it("ignores paths outside a releasable cloud root", async () => {
		const outside = join(root, "plain.pdf");
		writeFileSync(outside, "x");
		const probe = new FakeProbe();
		const { fire } = harness(probe, roots());
		await fire("tool_call", { toolName: "read", input: { path: outside } });
		expect(probe.datalessCalls).toEqual([]);
	});

	it("registers nothing without a probe", async () => {
		const { fire, handlerCount } = harness(undefined, roots());
		expect(handlerCount).toBe(0);
		await expect(fire("agent_end")).resolves.toBeUndefined();
	});

	it("does not throw when a release fails, and logs one line", async () => {
		const file = cloudFile("busy.pdf");
		const probe = new FakeProbe();
		probe.placeholders.add(file);
		probe.evictResult = { ok: false, message: "busy" };
		const logs: string[] = [];
		const { fire } = harness(probe, roots(), { log: (message) => logs.push(message), evictBackoffMs: 0 });
		await fire("tool_call", { toolName: "read", input: { path: file } });
		probe.placeholders.delete(file);
		await expect(fire("agent_end")).resolves.toBeUndefined();
		expect(logs).toHaveLength(1);
		expect(logs[0]).toContain(file);
		expect(probe.released).toEqual([]);
	});
});
