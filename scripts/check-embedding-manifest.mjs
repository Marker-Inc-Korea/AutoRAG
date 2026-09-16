#!/usr/bin/env node
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = join(repoRoot, "licenses", "embedding-assets.json");
const docsPath = join(repoRoot, "docs", "embedding-runtime.md");
const declared = JSON.parse(readFileSync(manifestPath, "utf8"));
const docs = readFileSync(docsPath, "utf8");
const source = await import(join(repoRoot, "src", "embedding-runtime", "manifest.ts"));

const modelNotices = {
	qwen3: "licenses/qwen3-embedding-notice.txt",
	embeddinggemma: "licenses/gemma-notice.txt",
};
if (source.MODEL_ASSET_URLS.qwen3 !== source.MODEL_ASSETS.qwen3.url) {
	throw new Error("MODEL_ASSET_URLS.qwen3 differs from MODEL_ASSETS.qwen3.url");
}
if (source.MODEL_ASSET_URLS.embeddinggemma !== source.MODEL_ASSETS.embeddinggemma.url) {
	throw new Error("MODEL_ASSET_URLS.embeddinggemma differs from MODEL_ASSETS.embeddinggemma.url");
}
const profileDefinitions = [
	{
		profile: source.resolveProfile("qwen3-embedding-0.6b"),
		model: source.MODEL_ASSETS.qwen3,
		notice: modelNotices.qwen3,
		assetId: "qwen3-embedding-0.6b",
	},
	{
		profile: source.resolveProfile("embeddinggemma-300m"),
		model: source.MODEL_ASSETS.embeddinggemma,
		notice: modelNotices.embeddinggemma,
		assetId: "embeddinggemma-300m",
	},
];
for (const { profile, model } of profileDefinitions) {
	if (profile.modelRevision !== model.revision || profile.artifactSha256 !== model.sha256) {
		throw new Error(`${profile.profileId} profile identity differs from its model asset`);
	}
}
const expectedProfiles = profileDefinitions.map(({ profile, model, notice, assetId }) => ({
	profileId: profile.profileId,
	provider: profile.provider,
	model: profile.model,
	dimension: profile.dimension,
	queryPrefix: profile.queryPrefix,
	passagePrefix: profile.passagePrefix,
	runtimeBuild: profile.runtimeBuild,
	modelRevision: profile.modelRevision,
	artifactSha256: profile.artifactSha256,
	backend: profile.backend,
	modelAssetId: assetId,
	licenseId: model.license,
	noticeFile: notice,
	noticeReference: model.noticeReference,
}));
const runtimeDefinitions = [
	"darwin-arm64-metal",
	"win-x64-cpu",
	"win-x64-vulkan",
].map((platform) => source.platformRuntimeAsset(platform));
const expectedAssets = [
	...profileDefinitions.map(({ model, profile, notice, assetId }) => ({
		kind: "model",
		id: assetId,
		filename: profile.model,
		url: model.url,
		revision: model.revision,
		sha256: model.sha256,
		licenseId: model.license,
		noticeFile: notice,
		noticeReference: model.noticeReference,
	})),

	...runtimeDefinitions.map((asset) => ({
		kind: "runtime",
		id: asset.id,
		platform: asset.platform,
		filename: asset.filename,
		url: asset.url,
		revision: asset.version,
		sha256: asset.sha256,
		licenseId: "MIT",
		noticeFile: "licenses/llama.cpp-MIT.txt",
		archiveMembers: [...asset.archiveMembers],
	})),
];

if (declared.schemaVersion !== 1 || declared.generatedFrom !== "src/embedding-runtime/manifest.ts") {
	throw new Error("embedding-assets.json has an unexpected schema or source declaration");
}
assertEqual(declared.profiles, expectedProfiles, "profiles");
assertEqual(declared.assets, expectedAssets, "assets");
const jsonBlock = extractManifestBlock(docs);
assertEqual(jsonBlock, declared, "docs/embedding-runtime.md manifest block");
for (const entry of [...expectedProfiles, ...expectedAssets]) {
	if (!docs.includes(entry.noticeFile)) throw new Error(`docs/embedding-runtime.md does not reference ${entry.noticeFile}`);
	if (!existsSync(join(repoRoot, entry.noticeFile))) throw new Error(`missing notice file ${entry.noticeFile}`);
}
process.stdout.write(`embedding manifest OK: ${expectedProfiles.length} profiles, ${expectedAssets.length} pinned assets\n`);

function extractManifestBlock(text) {
	const match = text.match(/<!-- EMBEDDING-ASSETS-MANIFEST:BEGIN -->\s*```json\n([\s\S]*?)\n```\s*<!-- EMBEDDING-ASSETS-MANIFEST:END -->/);
	if (!match) throw new Error("docs/embedding-runtime.md is missing the embedded embedding-assets manifest");
	return JSON.parse(match[1]);
}

function assertEqual(actual, expected, label) {
	const actualText = JSON.stringify(actual, null, 2);
	const expectedText = JSON.stringify(expected, null, 2);
	if (actualText !== expectedText) {
		throw new Error(`${label} differs from src/embedding-runtime/manifest.ts\nexpected:\n${expectedText}\nactual:\n${actualText}`);
	}
}
