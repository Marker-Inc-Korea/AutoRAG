import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../../src/agent/agent.ts";

const missingBinaryMode = process.argv.includes("--missing-binary");
const root = mkdtempSync(join(tmpdir(), "autorag-jikji-qa-corpus-"));
const docs = join(root, "docs");
mkdirSync(docs, { recursive: true });
writeFileSync(join(docs, "q3-report.txt"), "Q3 revenue report evidence from Jikji semantic local discovery.\n");

try {
  if (missingBinaryMode) {
    const agent = new AutoRAGAgent({ searchPaths: [docs], workspacePath: root, memoryPath: join(root, "memory.json"), jikji: { binaryPath: join(root, "missing-jikji") } });
    const results = await agent.retrieve("semantic-only", { topK: 2 });
    if (results.some((result) => result.metadata.method === "jikji")) throw new Error("missing Jikji returned a Jikji result");
    console.log("PASS missing jikji returns empty");
    process.exit(0);
  }

  const binaryPath = process.env.JIKJI_BINARY || "/tmp/autorag-jikji-venv/bin/jikji";
  if (!existsSync(binaryPath)) throw new Error(`real Jikji binary missing: ${binaryPath}`);
  const prepare = spawnSync(binaryPath, ["prepare", docs, "--json"], { encoding: "utf8" });
  if (prepare.status !== 0) throw new Error(`Jikji prepare failed: ${prepare.stderr || prepare.stdout}`);

  const agent = new AutoRAGAgent({ searchPaths: [docs], workspacePath: root, memoryPath: join(root, "memory.json"), jikji: { binaryPath } });
  const results = await agent.retrieve("semantic-only", { topK: 2 });
  const jikji = results.find((result) => result.metadata.method === "jikji");
  if (!jikji) throw new Error("missing Jikji retrieval result");
  if (JSON.stringify(results).includes(root)) throw new Error("absolute root leaked in retrieve results");
  console.log(`PASS jikji retrieval source=${jikji.source} method=${jikji.metadata.method}`);

  const response = await agent.searchDocuments("semantic-only", { topK: 2 });
  if (!response.answer.includes("[1]")) throw new Error("missing numbered answer");
  if (response.answer.includes(root)) throw new Error("absolute root leaked in searchDocuments answer");
  console.log("PASS searchDocuments path hidden");
} finally {
  rmSync(root, { recursive: true, force: true });
}
