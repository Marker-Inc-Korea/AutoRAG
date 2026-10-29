import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	type DiscordAttachmentRecord,
	DiscordAttachmentStore,
	discordAttachmentsRoot,
} from "../../../../src/datasource/skills/discrawl/attachments.ts";
import { DiscrawlClient } from "../../../../src/datasource/skills/discrawl/client.ts";
import { buildDatasourceSkills, discordAttachmentConnections } from "../../../../src/datasource/skills/factory.ts";

/**
 * A stand-in for the real `discrawl` binary: it keeps attachment rows in a JSON file, prints them for
 * `attachments --all`, and for `attachments fetch --missing --message <id>` writes the served bytes into
 * `<cache_dir>/media/attachments/<hh>/<sha256>-<name>` (discrawl's own layout) or records `HTTP 404`.
 */
const FAKE_DISCRAWL = `#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const statePath = process.env.DISCRAWL_FAKE_STATE;
const state = JSON.parse(fs.readFileSync(statePath, "utf8"));
const args = process.argv.slice(2);
state.calls.push(args);
const save = () => fs.writeFileSync(statePath, JSON.stringify(state));
const configIndex = args.indexOf("--config");
const config = configIndex === -1 ? "" : fs.readFileSync(args[configIndex + 1], "utf8");
const cacheDir = (config.match(/^cache_dir = (.*)$/m) || [])[1];
const rest = args.filter((arg, index) => !(arg === "--config" || (configIndex !== -1 && index === configIndex + 1)) && arg !== "--json");
if (rest[0] === "doctor") {
	save();
	process.stdout.write("config=ok\\ndatabase=ok\\nfts=ok\\nembeddings=ok\\n");
} else if (rest[0] === "wiretap") {
	save();
	process.stdout.write(JSON.stringify({ messages: 1 }));
} else if (rest[0] === "embed") {
	save();
	process.stdout.write("processed=1\\nsucceeded=1\\nfailed=0\\nremaining_backlog=0\\n");
} else if (rest[0] === "status") {
	save();
	process.stdout.write(JSON.stringify({ database_path: "/fake/discrawl.db", counts: [{ id: "messages", value: 1 }] }));
} else if (rest[0] === "attachments" && rest[1] === "fetch") {
	const messageId = rest[rest.indexOf("--message") + 1];
	const cache = JSON.parse(cacheDir);
	const stats = { attachments: 0, fetched: 0, reused: 0, skipped: 0, failed: 0, bytes: 0 };
	for (const row of state.rows.filter((r) => r.message_id === messageId)) {
		const cached = row.media_path ? path.join(cache, "media", row.media_path) : undefined;
		if (cached && fs.existsSync(cached)) { stats.reused++; continue; }
		stats.attachments++;
		const body = state.bodies[row.attachment_id];
		row.fetched_at = new Date().toISOString();
		if (typeof body !== "string") {
			row.fetch_status = "failed";
			row.fetch_error = "attachment fetch returned HTTP 404";
			stats.failed++;
			continue;
		}
		const hash = crypto.createHash("sha256").update(body).digest("hex");
		row.media_path = "attachments/" + hash.slice(0, 2) + "/" + hash + "-" + row.filename;
		row.content_sha256 = hash;
		row.content_size = body.length;
		row.fetch_status = "fetched";
		row.fetch_error = "";
		const target = path.join(cache, "media", row.media_path);
		fs.mkdirSync(path.dirname(target), { recursive: true });
		fs.writeFileSync(target, body);
		stats.fetched++;
		stats.bytes += body.length;
	}
	save();
	process.stdout.write(JSON.stringify(stats));
} else if (rest[0] === "attachments") {
	save();
	process.stdout.write(JSON.stringify(state.rows));
} else {
	save();
	process.exit(2);
}
`;

interface FakeRow {
	readonly attachment_id: string;
	readonly message_id: string;
	readonly guild_id: string;
	readonly guild_name: string;
	readonly channel_id: string;
	readonly channel_name: string;
	readonly filename: string;
	readonly size: number;
	readonly url: string;
	readonly created_at: string;
}

interface FakeState {
	rows: FakeRow[];
	bodies: Record<string, string | null>;
	calls: string[][];
}

let dir: string;
let statePath: string;
let binaryPath: string;
let root: string;

function signedUrl(id: string, name: string, expiresAtMs: number): string {
	const ex = Math.floor(expiresAtMs / 1000).toString(16);
	const is = Math.floor((expiresAtMs - 86_400_000) / 1000).toString(16);
	return `https://cdn.discordapp.com/attachments/1/${id}/${name}?ex=${ex}&is=${is}&hm=abc&`;
}

function row(id: string, filename: string, expiresAtMs = Date.now() + 12 * 3_600_000, messageId = `m-${id}`): FakeRow {
	return {
		attachment_id: id,
		message_id: messageId,
		guild_id: "g1",
		guild_name: "Team",
		channel_id: "c1",
		channel_name: "general",
		filename,
		size: 10,
		url: signedUrl(id, filename, expiresAtMs),
		created_at: new Date().toISOString(),
	};
}

function writeState(rows: FakeRow[], bodies: Record<string, string | null>): void {
	writeFileSync(statePath, JSON.stringify({ rows, bodies, calls: [] } satisfies FakeState));
}

function readState(): FakeState {
	return JSON.parse(readFileSync(statePath, "utf8")) as FakeState;
}

function fetchCalls(): string[][] {
	return readState().calls.filter((args) => args.includes("fetch"));
}

function store(): DiscordAttachmentStore {
	const client = new DiscrawlClient({ binaryPath, env: { DISCRAWL_FAKE_STATE: statePath } });
	return new DiscordAttachmentStore({ client, root });
}

function recordOf(records: readonly DiscordAttachmentRecord[], id: string): DiscordAttachmentRecord {
	const found = records.find((record) => record.attachmentId === id);
	if (found === undefined) throw new Error(`no record for ${id}`);
	return found;
}

const sha = (text: string): string => createHash("sha256").update(text).digest("hex");

beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "discord-attachments-"));
	statePath = join(dir, "state.json");
	binaryPath = join(dir, "discrawl");
	root = join(dir, "attachments");
	writeFileSync(binaryPath, FAKE_DISCRAWL);
	chmodSync(binaryPath, 0o755);
});

afterEach(() => {
	rmSync(dir, { recursive: true, force: true });
});

describe("DiscordAttachmentStore", () => {
	it("saves a live attachment under <server>/<channel>/<name> and records it in download-status.json", async () => {
		writeState([row("a1", "report.pdf")], { a1: "pdf-bytes" });

		const result = await store().sync({ download: true });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		const saved = join(root, "Team", "general", "report.pdf");
		expect(readFileSync(saved, "utf8")).toBe("pdf-bytes");
		expect(result.run).toMatchObject({ attempted: 1, fetched: 1, failed: 0, bytes: 9 });
		const status = JSON.parse(readFileSync(join(root, "download-status.json"), "utf8"));
		expect(status.summary).toMatchObject({ total: 1, downloaded: 1, downloadable: 0 });
		expect(status.attachments[0]).toMatchObject({
			attachmentId: "a1",
			state: "downloaded",
			path: "Team/general/report.pdf",
			sha256: sha("pdf-bytes"),
		});
	});

	it("does not download again while the saved file is intact", async () => {
		writeState([row("a1", "report.pdf")], { a1: "pdf-bytes" });
		await store().sync({ download: true });

		const second = await store().sync({ download: true });

		expect(fetchCalls()).toHaveLength(1);
		expect(second.ok && recordOf(second.records, "a1").state).toBe("downloaded");
	});

	it("treats a renamed file as missing and downloads the original again", async () => {
		writeState([row("a1", "report.pdf")], { a1: "pdf-bytes" });
		await store().sync({ download: true });
		renameSync(join(root, "Team", "general", "report.pdf"), join(root, "Team", "general", "renamed.pdf"));

		const checked = await store().sync({ download: false });
		expect(checked.ok && recordOf(checked.records, "a1")).toMatchObject({ state: "missing", downloadable: true });

		const redownloaded = await store().sync({ download: true });
		expect(redownloaded.ok && recordOf(redownloaded.records, "a1").state).toBe("downloaded");
		expect(readFileSync(join(root, "Team", "general", "report.pdf"), "utf8")).toBe("pdf-bytes");
		expect(readFileSync(join(root, "Team", "general", "renamed.pdf"), "utf8")).toBe("pdf-bytes");
	});

	it("keeps an edited file as the user's and saves the original beside it", async () => {
		writeState([row("a1", "notes.txt")], { a1: "original" });
		await store().sync({ download: true });
		const edited = join(root, "Team", "general", "notes.txt");
		writeFileSync(edited, "my edits");

		const result = await store().sync({ download: true });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(readFileSync(edited, "utf8")).toBe("my edits");
		expect(readFileSync(join(root, "Team", "general", "notes (2).txt"), "utf8")).toBe("original");
		expect(recordOf(result.records, "a1")).toMatchObject({ state: "downloaded", path: "Team/general/notes (2).txt" });
	});

	it("never tries an expired link and reports it as expired", async () => {
		writeState([row("a1", "old.pdf", Date.now() - 3_600_000)], { a1: "never-served" });

		const result = await store().sync({ download: true });

		expect(fetchCalls()).toHaveLength(0);
		expect(result.ok && result.summary).toMatchObject({ notDownloaded: 1, downloadable: 0, expired: 1 });
	});

	it("records an HTTP 404 as failed without failing the sync", async () => {
		writeState([row("a1", "gone.png")], { a1: null });

		const result = await store().sync({ download: true });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.run).toMatchObject({ attempted: 1, failed: 1, fetched: 0 });
		expect(recordOf(result.records, "a1")).toMatchObject({
			state: "failed",
			error: "attachment fetch returned HTTP 404",
			downloadable: true,
		});
	});

	it("saves two attachments with the same name in one message under distinct names", async () => {
		writeState([row("a1", "image.png", undefined, "m1"), row("a2", "image.png", undefined, "m1")], {
			a1: "first",
			a2: "second",
		});

		const result = await store().sync({ download: true });

		expect(fetchCalls()).toHaveLength(1);
		expect(readFileSync(join(root, "Team", "general", "image.png"), "utf8")).toBe("first");
		expect(readFileSync(join(root, "Team", "general", "image (2).png"), "utf8")).toBe("second");
		expect(result.ok && result.summary.downloaded).toBe(2);
	});

	it("status never spawns a download", async () => {
		writeState([row("a1", "report.pdf")], { a1: "pdf-bytes" });

		const result = await store().sync({ download: false });

		expect(fetchCalls()).toHaveLength(0);
		expect(result.ok && recordOf(result.records, "a1")).toMatchObject({
			state: "not-downloaded",
			downloadable: true,
		});
		expect(existsSync(join(root, "download-status.json"))).toBe(true);
	});
});

describe("discordAttachmentsRoot", () => {
	it("lives under the AutoRAG home, independent of the OS user's other folders", () => {
		expect(discordAttachmentsRoot("nomadamas", { AUTORAG_HOME: "/data/autorag" })).toBe(
			join("/data/autorag", "attachments", "discord", "nomadamas"),
		);
		expect(discordAttachmentsRoot("discord", { HOME: "/home/alice" })).toBe(
			join("/home/alice", ".autorag", "attachments", "discord", "discord"),
		);
	});
});

describe("Discord datasource factory attachment defaults", () => {
	let previousHome: string | undefined;

	beforeEach(() => {
		previousHome = process.env.AUTORAG_HOME;
		process.env.AUTORAG_HOME = join(dir, "autorag-home");
	});

	afterEach(() => {
		if (previousHome === undefined) delete process.env.AUTORAG_HOME;
		else process.env.AUTORAG_HOME = previousHome;
	});

	it("downloads attachments on refresh by default without setting downloadAttachments", async () => {
		writeState([row("a1", "spec.pdf")], { a1: "pdf-bytes" });
		const datasources = {
			nomadamas: { type: "discord", connector: { binaryPath, env: { DISCRAWL_FAKE_STATE: statePath } } },
		};

		expect(discordAttachmentConnections(datasources)[0]?.autoDownload).toBe(true);
		const { skills } = buildDatasourceSkills(datasources, join(dir, "ws"));
		const indexed = await skills[0]?.index();

		expect(indexed?.ok).toBe(true);
		expect(fetchCalls()).toHaveLength(1);
		const saved = join(dir, "autorag-home", "attachments", "discord", "nomadamas", "Team", "general", "spec.pdf");
		expect(readFileSync(saved, "utf8")).toBe("pdf-bytes");
	});

	it("skips attachment downloads on refresh when downloadAttachments is false", async () => {
		writeState([row("a1", "spec.pdf")], { a1: "pdf-bytes" });
		const datasources = {
			nomadamas: {
				type: "discord",
				connector: { binaryPath, downloadAttachments: false, env: { DISCRAWL_FAKE_STATE: statePath } },
			},
		};

		expect(discordAttachmentConnections(datasources)[0]?.autoDownload).toBe(false);
		const { skills } = buildDatasourceSkills(datasources, join(dir, "ws"));
		const indexed = await skills[0]?.index();

		expect(indexed?.ok).toBe(true);
		expect(fetchCalls()).toHaveLength(0);
	});
});
