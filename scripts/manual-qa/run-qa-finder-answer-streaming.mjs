import { _electron } from "playwright";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { refreshQaIndex } from "./warm-qa-index.mjs";

const EVIDENCE = new URL("../../.omo/evidence/ai-finder-app/qa-answer-streaming/", import.meta.url).pathname;
mkdirSync(EVIDENCE, { recursive: true });
const DOCS = `${EVIDENCE}docs`;
mkdirSync(DOCS, { recursive: true });
const LOG = `${EVIDENCE}/run.log`;
const log = (line) => {
	console.log(line);
	appendFileSync(LOG, `${new Date().toISOString()} ${line}\n`);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

writeFileSync(
	`${DOCS}/refund-policy.txt`,
	[
		"환불 정책 (v3, 2026-07-15 승인)",
		"1. 일반 환불은 5영업일 내 처리한다.",
		"2. 환불 예외(exception)는 지급 전 반드시 임원(director) 승인이 필요하다.",
		"3. 재무팀은 요청에 원래의 승인 체인을 첨부해야 한다.",
		"4. 7월 리뷰에서 승인된 예외는 4분기까지 유예된다.",
		"5. 승인 없이 지급된 예외는 회계 감사에서 미승인 지급으로 분류된다.",
	].join("\n"),
);
writeFileSync(
	`${DOCS}/refund-exception-process.txt`,
	[
		"환불 예외 처리 절차 (재무팀 내규, 2026-07-20 개정)",
		"1단계: 요청자가 예외 사유와 금액을 작성해 재무팀에 제출한다.",
		"2단계: 재무팀 담당자가 사유를 검토하고 승인 체인 원본을 첨부한다.",
		"3단계: 임원(director)이 승인하면 지급 요청이 생성된다.",
		"4단계: 승인되지 않은 요청은 자동으로 반려되고 요청자에게 통보된다.",
		"비고: 긴급 예외도 임원 승인을 생략할 수 없다.",
	].join("\n"),
);
writeFileSync(
	`${DOCS}/finance-approval-matrix.txt`,
	[
		"재무 승인 매트릭스 (2026년 상반기)",
		"- 100만원 이하 일반 지급: 팀장 승인",
		"- 100만원 초과 지급: 재무 담당 임원 승인",
		"- 환불 예외 전 구간: 지급 전 임원(director) 승인 필수",
		"- 임원 부재 시: 부임원 대결 가능하되 사후 보고 필수",
	].join("\n"),
);
const CONFIG = `${EVIDENCE}qa-config.json`;
writeFileSync(
	CONFIG,
	JSON.stringify(
		{
			searchPaths: [DOCS],
			workspacePath: EVIDENCE,
			memoryPath: `${EVIDENCE}/memory.json`,
			minSync: {
				enabled: true,
				autoInstall: true,
				embedder: { id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 },
			},
			jikji: false,
			model: {
				provider: "openrouter",
				id: "deepseek/deepseek-v4.1-flash",
				baseUrl: "https://openrouter.ai/api/v1",
				api: "openai-completions",
				apiKeyEnv: "OPENROUTER_API_KEY",
			},
		},
		null,
		2,
	),
);
log(`qa config: ${CONFIG}`);
log("warming the QA parsed mirror and MinSync index...");
await refreshQaIndex(CONFIG);
log("warm-up done");

const app = await _electron.launch({ args: ["app"], env: { ...process.env, AUTORAG_CONFIG: CONFIG } });
const page = await app.firstWindow();
page.setDefaultTimeout(5000);
page.on("console", (message) => log(`[console:${message.type()}] ${message.text().slice(0, 300)}`));
page.on("pageerror", (error) => log(`[pageerror] ${String(error).slice(0, 300)}`));
app.process().stdout?.on("data", (chunk) => log(`[main] ${String(chunk).trim().slice(0, 300)}`));
app.process().stderr?.on("data", (chunk) => log(`[main:err] ${String(chunk).trim().slice(0, 300)}`));

await page.waitForSelector('[aria-label="AI Search"]');
log("app ready");

const QUERY = "환불 예외 승인 규칙을 문서에서 찾아 요약해줘";
await page.fill('input[aria-label="Ask AutoRAG"]', QUERY);
await page.keyboard.press("Enter");
await sleep(500);
await page.screenshot({ path: `${EVIDENCE}/submitted.png` });

const snapshots = [];
let lastProgress = "";
let lastLog = "";
const deadline = Date.now() + 180_000;
while (Date.now() < deadline) {
	await sleep(300);
	const progress = (await page.locator(".ai__progress").textContent().catch(() => "")) ?? "";
	if (progress !== "" && progress !== lastProgress) {
		lastProgress = progress;
		log(`progress: ${progress.slice(0, 120)}`);
	}
	const error = (await page.locator(".ai__error").textContent().catch(() => "")) ?? "";
	if (error !== "") {
		log(`ERROR EVENT: ${error}`);
		await page.screenshot({ path: `${EVIDENCE}/error.png` });
		break;
	}
	const stopped = await page.locator(".ai__stopped").count();
	if (stopped > 0) {
		log("stopped marker present");
		break;
	}
	const quickText = (await page.locator(".ai__answer--quick .ai__answer-content p").textContent().catch(() => "")) ?? "";
	const deepText = (await page.locator(".ai__answer--deep .ai__answer-content p").textContent().catch(() => "")) ?? "";
	const state = `quick=${quickText.length} deep=${deepText.length}`;
	if (state !== lastLog) {
		lastLog = state;
		log(`state: ${state}`);
	}
	if (quickText !== "" && snapshots.at(-1)?.quick !== quickText) {
		snapshots.push({ quick: quickText, deep: deepText });
		await page.screenshot({ path: `${EVIDENCE}/quick-${snapshots.length}.png` });
	}
	if (deepText !== "" && snapshots.at(-1)?.deep !== deepText) {
		snapshots.push({ quick: quickText, deep: deepText });
		await page.screenshot({ path: `${EVIDENCE}/deep-${snapshots.length}.png` });
	}
	const deepMeta = (await page
		.locator('.ai__answer--deep:not(.ai__answer--pending):not(.ai__answer--streaming) .ai__answer-heading span')
		.last()
		.textContent()
		.catch(() => "")) ?? "";
	if (/[0-9]+.[0-9]s/.test(deepMeta)) {
		log(`deep complete: ${deepMeta}`);
		await page.screenshot({ path: `${EVIDENCE}/complete.png` });
		break;
	}
}

log(`snapshots: ${snapshots.length}`);
for (const [index, snap] of snapshots.entries()) {
	log(`[${index + 1}] quick(${snap.quick.length}) ${JSON.stringify(snap.quick.slice(0, 60))}`);
	log(`    deep(${snap.deep.length}) ${JSON.stringify(snap.deep.slice(0, 60))}`);
}
const quickLengths = snapshots.map((snap) => snap.quick.length).filter((length) => length > 0);
const deepLengths = snapshots.map((snap) => snap.deep.length).filter((length) => length > 0);
log(`quick grew across snapshots: ${new Set(quickLengths).size > 1} ${JSON.stringify(quickLengths)}`);
log(`deep grew across snapshots: ${new Set(deepLengths).size > 1} ${JSON.stringify(deepLengths)}`);

await app.close();
log("app closed");
