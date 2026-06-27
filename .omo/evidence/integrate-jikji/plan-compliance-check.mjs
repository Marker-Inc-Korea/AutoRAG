import { existsSync, readFileSync } from "node:fs";

const required = [
  "task-1-red.txt",
  "task-1-green.txt",
  "task-2-red.txt",
  "task-2-green.txt",
  "task-3-red.txt",
  "task-3-green.txt",
  "task-4-red.txt",
  "task-4-green.txt",
  "task-5-red.txt",
  "task-5-green.txt",
  "task-6-red.txt",
  "task-6-green.txt",
  "manual-qa-transcript.txt",
  "final-typecheck.txt",
  "final-test.txt",
  "final-biome.txt",
];

for (const file of required) {
  const path = `.omo/evidence/integrate-jikji/${file}`;
  if (!existsSync(path)) throw new Error(`missing evidence ${path}`);
  if (readFileSync(path, "utf8").trim().length === 0) throw new Error(`empty evidence ${path}`);
}

const diff = existsSync(".omo/evidence/integrate-jikji/final-diff.patch")
  ? readFileSync(".omo/evidence/integrate-jikji/final-diff.patch", "utf8")
  : "";
for (const forbidden of ["args.push(\"--enable-media-index\")", "pip install", "site-packages"]) {
  if (diff.includes(forbidden)) throw new Error(`forbidden guardrail pattern in diff: ${forbidden}`);
}
console.log("PASS plan compliance");
