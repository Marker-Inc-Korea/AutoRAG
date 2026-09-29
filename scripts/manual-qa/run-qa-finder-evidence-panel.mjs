/**
 * Manual QA — AI Search markdown answers + the Evidence panel of the real
 * AI Finder app, per the v6 design reference.
 *
 * Deterministic lane: seeds the chat store (<userData>/chat-history/chats.json)
 * with one chat whose Quick/Deep answers carry markdown (bold, headings,
 * lists, quote, inline code) and numbered citations, and whose evidence items
 * point at REAL fixture files on disk. Loading the chat exercises only the
 * real render path (loadHistory -> publish -> EvidencePanel), then drives:
 * citation chips, evidence tabs, thumbs feedback (toast), native Quick Look
 * (qlmanage process grep), crumb -> new-tab reveal with row flash, ⌘E toggle,
 * and the datasource virtual-source reveal guard (verbatim error toast).
 *
 * Run from the repo root:  bun scripts/manual-qa/run-qa-finder-evidence-panel.mjs
 * Prereq: cd app && bunx electron-vite build
 */
import { mkdirSync, rmSync, writeFileSync, appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, basename } from "node:path";
import { execSync } from "node:child_process";
import { _electron } from "playwright";

const ROOT = new URL("../..", import.meta.url).pathname;
const EVIDENCE = join(ROOT, ".omo/evidence/ai-finder-app/qa-evidence-panel");
mkdirSync(EVIDENCE, { recursive: true });
const LOG = join(EVIDENCE, "run.log");
const log = (line) => { console.log(line); appendFileSync(LOG, `${line}\n`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sh = (cmd) => { try { return execSync(cmd, { encoding: "utf8" }); } catch { return ""; } };
const failures = [];
const check = (label, ok, detail) => { log(`${ok ? "PASS" : "FAIL"} ${label}${detail ? ` :: ${detail}` : ""}`); if (!ok) failures.push(label); };

const RUN_ID = `ulw-evidence-qa-${Date.now().toString(36)}`;
const USER_DATA = join(tmpdir(), RUN_ID);
const DOCS = join(process.env.HOME, "Documents", RUN_ID);
const POLICY = join(DOCS, "qa-evidence-policy.md");
const MEMO = join(DOCS, "qa-evidence-memo.md");
const FOLDER = basename(DOCS);

let app;
let swept = false;
const sweep = () => {
	if (swept) return;
	swept = true;
	app?.close().catch(() => undefined);
	sh(`pkill -f "qlmanage -p ${POLICY}"`);
	rmSync(USER_DATA, { recursive: true, force: true });
	rmSync(DOCS, { recursive: true, force: true });
	log(`cleanup: removed ${USER_DATA} and ${DOCS}; killed fixture qlmanage; app closed`);
};
process.on("exit", sweep);

mkdirSync(join(USER_DATA, "chat-history"), { recursive: true });
mkdirSync(DOCS, { recursive: true });
writeFileSync(POLICY, "## 승인 규칙\n1. 일반 환불은 5영업일 내 처리합니다.\n2. 예외는 임원 승인이 필요합니다.\n비고: 긴급 예외도 승인 생략 불가.\n");
writeFileSync(MEMO, "## 메모\n승인 체인 원본 첨부 필수.\n");
log(`fixtures: ${DOCS}`);

const evidenceAll = [
	{
		number: 1,
		title: "qa-evidence-policy.md",
		summary: "환불 정책에서 승인 규칙을 발췌했습니다.",
		source: POLICY,
		excerpts: [
			"## 승인 규칙\n1. 일반 환불은 5영업일 내 처리합니다.\n2. 예외는 **임원 승인**이 필요합니다.\n> 비고 인용: 긴급 예외도 승인 생략 불가\n```\nif (!director) reject()\n```",
		],
		confidence: 0.92,
		feedbackId: "fb-1",
	},
	{
		number: 2,
		title: "qa-evidence-memo.md",
		summary: "승인 체인 첨부 메모입니다.",
		source: MEMO,
		excerpts: ["승인 체인 원본 첨부 필수."],
		confidence: 0.81,
		feedbackId: "fb-2",
	},
	{
		number: 3,
		title: "채널 스레드 정리",
		summary: "외부 채널에서 온 데이터소스 근거입니다.",
		source: "/kakao/ulw-qa/chunks/7",
		excerpts: ["스레드에서 합의된 내용 요약."],
		confidence: 0.44,
		feedbackId: "fb-3",
	},
];
const record = {
	id: "c-ev-1",
	title: "환불 예외 승인 규칙",
	snippet: "최종 승인액은 ₩3.8억이며, 승인 체인 첨부가 필요합니다.",
	updatedAt: new Date().toISOString(),
	messages: [
		{ role: "user", text: "환불 예외 승인 규칙 요약해줘", attachments: [], at: new Date().toISOString() },
		{
			role: "assistant",
			sessionId: "qa-evidence-session",
			quick: {
				answer: "최종 승인액은 **₩3.8억**이며, 승인 체인 첨부가 필요합니다. [1][2]",
				evidence: evidenceAll.slice(0, 2),
				meta: "0.4s · 2 sources",
			},
			deep: {
				answer: "## 승인 규칙 요약\n환불 예외는 지급 전 **임원 승인**이 필요합니다. [1][2]\n\n- 예외는 자동 반려되지 않습니다 [2]\n- 긴급 예외도 승인 생략 불가 [1]\n\n> 내규 인용 구문이 함께 표시됩니다.\n\n관련 명령은 `bun run qa` 입니다.",
				evidence: evidenceAll,
				meta: "3 sources read · 2s",
			},
			stopped: false,
			at: new Date().toISOString(),
		},
	],
};
writeFileSync(join(USER_DATA, "chat-history", "chats.json"), `${JSON.stringify([record], null, 2)}\n`);

// The feedback lane builds the real main-process agent; point it at a throwaway
// config so the construction succeeds and no workspace state leaks into the repo.
mkdirSync(join(USER_DATA, "workspace"), { recursive: true });
const QA_CONFIG = join(USER_DATA, "qa-config.json");
writeFileSync(QA_CONFIG, JSON.stringify({
	searchPaths: [DOCS],
	workspacePath: join(USER_DATA, "workspace"),
	memoryPath: join(USER_DATA, "workspace", "memory.json"),
	minSync: { enabled: false },
	jikji: false,
	model: {
		provider: "openrouter",
		id: "deepseek/deepseek-v4.1-flash",
		baseUrl: "https://openrouter.ai/api/v1",
		api: "openai-completions",
		apiKeyEnv: "OPENROUTER_API_KEY",
	},
}, null, 2));

app = await _electron.launch({ args: ["app", `--user-data-dir=${USER_DATA}`], cwd: ROOT, env: { ...process.env, AUTORAG_CONFIG: QA_CONFIG } });
const page = await app.firstWindow();
page.setDefaultTimeout(8000);
page.on("pageerror", (error) => log(`[pageerror] ${String(error).slice(0, 300)}`));
await page.waitForSelector('[aria-label="Ask AutoRAG"]');
log("app ready");

// 1. Open the seeded chat through the real history surface.
await page.locator('[aria-label="Chat history"]').click();
await page.locator(".ai__history-item").first().waitFor({ state: "visible" });
await page.locator(".ai__history-item").first().click();
await page.waitForSelector(".ai__answer--deep .ai__block--heading");
await page.screenshot({ path: join(EVIDENCE, "01-answers.png") });

const boldCount = await page.locator(".ai__answer--quick strong").count();
check("quick answer renders **bold** as <strong>", boldCount >= 1, `strong=${boldCount}`);
const quickCites = await page.locator(".ai__answer--quick .ai__citation").allTextContents();
check("quick answer citation chips show bare numbers", quickCites.join(",") === "1,2", JSON.stringify(quickCites));
await sleep(300);
const chipOf = (n) => page.locator(".ai__answer--quick .ai__citation", { hasText: new RegExp(`^${n}$`) }).first();
const chipStyle = async (locator) => locator.evaluate((el) => {
	const s = getComputedStyle(el);
	return { bg: s.backgroundColor, color: s.color, radius: s.borderRadius, fontSize: s.fontSize, fontWeight: s.fontWeight };
});
const inactiveStyle = await chipStyle(chipOf(2));
check(
	"an inactive citation chip matches the reference (accent-chip wash / accent-text / radius 5 / 11 / 600)",
	inactiveStyle.bg === "rgba(255, 99, 99, 0.16)" && inactiveStyle.color === "rgb(214, 69, 69)" && inactiveStyle.radius === "5px" && inactiveStyle.fontSize === "11px" && inactiveStyle.fontWeight === "600",
	JSON.stringify(inactiveStyle),
);
const activeStyle = await chipStyle(chipOf(1));
check(
	"the default-selected evidence lights chip 1 in solid accent (rgb(255,99,99) on dark text)",
	activeStyle.bg === "rgb(255, 99, 99)" && activeStyle.color === "rgb(26, 26, 46)",
	JSON.stringify(activeStyle),
);
const deepHeading = (await page.locator(".ai__answer--deep .ai__block--heading").first().textContent()) ?? "";
check("deep answer renders the ## heading block", deepHeading.includes("승인 규칙 요약"), deepHeading.slice(0, 40));
check("deep answer renders two list blocks", (await page.locator(".ai__answer--deep .ai__block--list").count()) === 2);
check("deep answer renders the quote block", (await page.locator(".ai__answer--deep .ai__block--quote").count()) === 1);
const inlineCode = await page.locator(".ai__answer--deep .ai__inline-code").textContent();
check("deep answer renders inline code", (inlineCode ?? "").includes("bun run qa"), inlineCode);

// 2. Citation click -> evidence panel, selected tab, crumb text.
await page.locator(".ai__answer--quick .ai__citation", { hasText: "2" }).first().click();
await page.waitForSelector(".evidence__tab--active");
const activeCite = await page.locator(".ai__answer--quick .ai__citation--active").allTextContents();
const activeTab = (await page.locator(".evidence__tab--active").textContent())?.trim();
const evTitle = (await page.locator(".evidence__title").textContent()) ?? "";
const crumbText = (await page.locator(".evidence__crumb").textContent()) ?? "";
check("the cited chip lights up with the active accent", JSON.stringify(activeCite) === JSON.stringify(["2"]), JSON.stringify(activeCite));
check("the evidence panel opens on evidence 2", activeTab === "2" && evTitle.startsWith("2. "), `tab=${activeTab} title=${evTitle.slice(0, 32)}`);
check("the breadcrumb shows folder and file segments", crumbText.includes("Document") && crumbText.includes("qa-evidence-memo.md"), crumbText.replace(/\s+/g, " ").slice(0, 120));
const detail = (await page.locator(".evidence__detail").textContent()) ?? "";
check("the detail shows the confidence percentage", detail === "관련도 81%", detail);
await page.screenshot({ path: join(EVIDENCE, "02-evidence-panel.png") });

// 2b. Layout geometry per the reference (strip 40px, tabs 40×30, votes 32×32,
// crumb chip 24px, panel = 44% of the finder column, body padded 52px left).
const geometry = await page.evaluate(() => {
	const box = (sel) => { const e = document.querySelector(sel); return e ? JSON.parse(JSON.stringify(e.getBoundingClientRect())) : null; };
	const style = (sel, props) => {
		const e = document.querySelector(sel);
		if (!e) return null;
		const s = getComputedStyle(e);
		return Object.fromEntries(props.map((p) => [p, s[p]]));
	};
	return {
		panel: box("section.evidence"),
		finder: box('section[aria-label="Finder"]'),
		strip: box(".evidence__strip"),
		tab: box(".evidence__tab"),
		vote: box(".evidence__vote"),
		quicklook: box(".evidence__quicklook"),
		crumb: box(".evidence__crumb"),
		panelBg: style("section.evidence", ["backgroundColor"]),
		stripBg: style(".evidence__strip", ["backgroundColor"]),
		body: style(".evidence__body", ["paddingLeft"]),
		header: style(".evidence__header", ["paddingTop", "paddingLeft", "paddingBottom", "paddingRight"]),
	};
});
const ratio = geometry.panel.height / geometry.finder.height;
check("the open panel takes the reference 44% of the finder column", Math.abs(ratio - 0.44) < 0.02, `ratio=${ratio.toFixed(3)}`);
check("strip is 40px tall on the chrome surface", Math.round(geometry.strip.height) === 40 && geometry.stripBg.backgroundColor === "rgb(246, 246, 248)", JSON.stringify({ h: geometry.strip.height, bg: geometry.stripBg }));
check("evidence tabs are 40x30", Math.round(geometry.tab.width) === 40 && Math.round(geometry.tab.height) === 30, JSON.stringify(geometry.tab));
check("feedback votes are 32x32", Math.round(geometry.vote.width) === 32 && Math.round(geometry.vote.height) === 32, JSON.stringify(geometry.vote));
check("the crumb chip is 24px tall", Math.round(geometry.crumb.height) === 24, JSON.stringify(geometry.crumb));
check("the panel sits on #FAFAFB", geometry.panelBg.backgroundColor === "rgb(250, 250, 251)", JSON.stringify(geometry.panelBg));
check("header/body paddings follow the reference", geometry.header.paddingTop === "14px" && geometry.header.paddingLeft === "20px" && geometry.header.paddingBottom === "10px" && geometry.body.paddingLeft === "52px", JSON.stringify({ header: geometry.header, body: geometry.body }));

// 3. Tabs switch; chunk body renders the reference block types.
await page.locator(".evidence__tab", { hasText: /^1$/ }).first().click();
await page.waitForFunction(() => document.querySelector(".evidence__title")?.textContent?.startsWith("1. "));
const kindCount = async (kind) => page.locator(`.evidence__body .evidence__block--${kind}`).count();
check(
	"evidence 1 chunk renders heading/list ×2/quote/code blocks plus the summary paragraph",
	(await kindCount("heading")) === 1 && (await kindCount("list")) === 2 && (await kindCount("quote")) === 1 && (await kindCount("code")) === 1 && (await kindCount("paragraph")) === 1,
	JSON.stringify({ heading: await kindCount("heading"), list: await kindCount("list"), quote: await kindCount("quote"), code: await kindCount("code"), paragraph: await kindCount("paragraph") }),
);
const evBold = await page.locator(".evidence__body strong").count();
check("chunk text bold renders as emphasized strong", evBold >= 1, `strong=${evBold}`);

// 4. Thumbs-up feedback records through the real bridge and toasts.
await page.locator('button[title="이 근거가 도움이 됨"]').click();
await page.waitForSelector(".toast", { timeout: 8000 });
const toastText = (await page.locator(".toast").textContent()) ?? "";
check("thumbs-up toast follows the reference copy", toastText.includes("근거 1 — 도움이 됨으로 기록했습니다"), toastText.trim());
await sleep(300);
const voteStyle = await page.locator('button[title="이 근거가 도움이 됨"]').evaluate((el) => {
	const s = getComputedStyle(el);
	return { bg: s.backgroundColor, border: s.borderColor, color: s.color };
});
check("the active vote uses the green ok set", voteStyle.bg === "rgba(34, 197, 94, 0.12)", JSON.stringify(voteStyle));
await page.screenshot({ path: join(EVIDENCE, "03-feedback.png") });

// 5. Quick Look spawns the native previewer for the fixture source.
await page.locator(".evidence__quicklook").click();
await sleep(4000);
const ql = sh(`ps -axo command | grep 'qlmanage -p ${POLICY}' | grep -v grep`);
check("Quick Look spawns qlmanage -p for the evidence source", ql.includes(POLICY), (ql.trim().split("\n")[0] ?? "none").trim());

// 6. Crumb click opens the enclosing folder in a NEW tab and reveals the row.
const tabsBefore = await page.locator(".tabs .tab").count();
await page.locator(".evidence__crumb").click();
await page.waitForFunction((count) => document.querySelectorAll(".tabs .tab").length === count + 1, tabsBefore);
await sleep(400);
const newTabTitle = (await page.locator(".tabs .tab--active .tab__title").textContent()) ?? "";
check("the crumb opens the source folder in a new tab", newTabTitle === FOLDER, `tab="${newTabTitle}" want="${FOLDER}"`);
check("the source row is revealed with the flash marker", (await page.locator(".row--flash").count()) >= 1);
await page.screenshot({ path: join(EVIDENCE, "04-new-tab-reveal.png") });

// 7. ⌘E collapses and reopens the strip.
await page.locator(".evidence__title-row").click({ position: { x: 4, y: 4 }, modifiers: [] });
await page.keyboard.press("Meta+e");
await page.waitForFunction(() => document.querySelector("section.evidence")?.className.includes("evidence--closed"));
check("⌘E collapses the panel to the tab strip", true);
await page.screenshot({ path: join(EVIDENCE, "05-collapsed.png") });
await page.locator(".evidence__toggle").waitFor({ state: "visible" });
await page.keyboard.press("Meta+e");
await page.waitForFunction(() => !document.querySelector("section.evidence")?.className.includes("evidence--closed"));
check("⌘E reopens the panel", true);

// 8. A datasource virtual source cannot be revealed; the error toasts verbatim.
await page.locator(".evidence__tab", { hasText: /^3$/ }).first().click();
await page.waitForFunction(() => document.querySelector(".evidence__title")?.textContent?.startsWith("3. "));
await page.locator(".evidence__crumb").click();
await page.waitForSelector(".toast", { timeout: 8000 });
const guardToast = (await page.locator(".toast").textContent()) ?? "";
check("the datasource source fails with a verbatim error toast", guardToast.includes("/kakao/ulw-qa"), guardToast.trim().slice(0, 160));
await page.screenshot({ path: join(EVIDENCE, "06-datasource-guard.png") });

log(`RESULT ${failures.length === 0 ? "ALL PASS" : `${failures.length} FAILURES: ${failures.join(" | ")}`}`);
sweep();
process.exitCode = failures.length === 0 ? 0 : 1;
