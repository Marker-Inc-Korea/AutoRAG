/**
 * Manual QA — the AI Search chat-history popover of the real AI Finder app.
 *
 * Proves the user-visible behavior against the v6 design reference: a seeded
 * chat store renders the History popover with its day groups (오늘 / 어제 /
 * 지난 7일 / 지난 30일 / 그 이전), each row carries title, snippet, and time,
 * the current chat carries the accent dot on rgba(255,99,99,0.08), the search
 * input filters through the store, Enter opens the first hit, Esc closes the
 * popover without clearing the Finder query, and the click-catcher closes it.
 *
 * Run from the repo root:  bun scripts/manual-qa/run-qa-finder-chat-history.mjs
 * Prereq: cd app && bunx electron-vite build
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { _electron, chromium } from "playwright";

const ROOT = new URL("../..", import.meta.url).pathname;
const EVIDENCE = join(ROOT, ".omo/evidence/ai-finder-app/qa-chat-history");
mkdirSync(EVIDENCE, { recursive: true });

const RUN_ID = `ulw-chat-history-qa-${Date.now().toString(36)}`;
const USER_DATA = join(tmpdir(), RUN_ID);
const STORE = join(USER_DATA, "chat-history", "chats.json");
const REFERENCE = join(ROOT, "docs/design-reference/ai-finder/design_handoff_ai_finder_v6/AI Finder v6.dc.html");

let app;
let refBrowser;
let swept = false;
const sweep = () => {
	if (swept) return;
	swept = true;
	refBrowser?.close().catch(() => undefined);
	app?.close().catch(() => undefined);
	rmSync(USER_DATA, { recursive: true, force: true });
	console.log(`cleanup: removed ${USER_DATA}; closed the app and the reference browser`);
};
process.on("exit", sweep);

const failures = [];
const check = (label, ok, detail) => {
	console.log(`${ok ? "PASS" : "FAIL"} ${label}${detail === undefined ? "" : ` :: ${detail}`}`);
	if (!ok) failures.push(label);
};

const pad = (value) => String(value).padStart(2, "0");
const at = (daysAgo, hour, minute) => {
	const date = new Date();
	date.setDate(date.getDate() - daysAgo);
	date.setHours(hour, minute, 0, 0);
	return date;
};
const monthDay = (date) => `${date.getMonth() + 1}월 ${date.getDate()}일`;
const clock = (date) => `${pad(date.getHours())}:${pad(date.getMinutes())}`;

/** The store's own record shape; the list and the search both read these summaries back verbatim. */
const record = (id, title, snippet, when, question, answer) => ({
	id,
	title,
	snippet,
	updatedAt: when.toISOString(),
	messages: [
		{ role: "user", text: question, attachments: [], at: when.toISOString() },
		{
			role: "assistant",
			sessionId: `${id}-session`,
			quick: { answer, evidence: [], meta: "0.7s · 1 source" },
			deep: null,
			stopped: false,
			at: when.toISOString(),
		},
	],
});

const today1 = at(0, 0, 42);
const today2 = at(0, 0, 15);
const yesterday = at(1, 17, 30);
const withinWeek = at(4, 11, 2);
const withinMonth = at(9, 11, 2);
const older = at(40, 11, 2);

const CHATS = [
	record("c1", "Q3 마케팅 예산 승인", "9월 12일 Slack #mkt-budget에서 CPA 악화를 이유로 감액이 합의됐습니다.", today1, "Q3 마케팅 예산 승인 근거 찾아줘", "예산은 3.8억으로 확정됐습니다."),
	record("c2", "퍼포먼스 광고 감액 사유", "8월 CPA가 목표 대비 31% 높았습니다.", today2, "퍼포먼스 광고 예산이 왜 4천만 원 줄었어?", "CPA 악화가 감액 사유입니다."),
	record("c3", "CFO 승인 메일 찾기", "9월 15일 오후 4:12에 보낸 메일입니다.", yesterday, "박지훈 CFO가 보낸 예산 승인 메일 찾아줘", "승인 메일을 찾았습니다."),
	record("c4", "v1 vs v3 예산안 차이", "총액이 4.2억에서 3.8억으로 줄었습니다.", withinWeek, "예산안 v1이랑 v3 차이 정리해줘", "퍼포먼스 광고가 가장 크게 줄었습니다."),
	record("c5", "해외법인 분리 여부", "포함되지 않으며 별도 품의로 분리됩니다.", withinMonth, "해외법인 마케팅 예산은 이번 Q3에 포함돼?", "Q4 별도 품의로 분리됩니다."),
	record("c6", "Q3 캠페인 브리프 초안 위치", "Documents › Marketing › Briefs 폴더에 있습니다.", older, "Q3 캠페인 브리프 초안 어디 있어?", "Documents › Marketing › Briefs에 있습니다."),
];

mkdirSync(join(USER_DATA, "chat-history"), { recursive: true });
writeFileSync(STORE, `${JSON.stringify(CHATS, null, 2)}\n`);

/** Bounded, event-driven wait for the rendered texts to match — never a fixed sleep. */
const waitForTexts = async (page, selector, expected, timeout = 8000) => {
	try {
		await page.waitForFunction(
			([sel, want]) => {
				const got = [...document.querySelectorAll(sel)].map((element) => element.textContent ?? "");
				return got.length === want.length && got.every((value, index) => value === want[index]);
			},
			[selector, expected],
			{ timeout },
		);
		return true;
	} catch {
		return false;
	}
};

const texts = (page, selector) => page.locator(selector).allInnerTexts();
const openHistory = async (page) => {
	await page.locator('[aria-label="Chat history"]').click();
	await page.locator(".ai__history-popover").waitFor({ state: "visible", timeout: 5000 });
	await page.locator(".ai__history-header input").waitFor({ state: "visible", timeout: 5000 });
};
const popoverClosed = async (page) => {
	await page.locator(".ai__history-popover").waitFor({ state: "detached", timeout: 5000 });
	return (await page.locator(".ai__history-popover").count()) === 0;
};

app = await _electron.launch({ args: ["app", `--user-data-dir=${USER_DATA}`], cwd: ROOT });
const page = await app.firstWindow();
await page.waitForSelector('[role="row"]');

// 1. The popover renders the seeded history in the reference's day groups.
await openHistory(page);
const groups = await texts(page, ".ai__history-group");
check(
	"the popover groups history as 오늘 / 어제 / 지난 7일 / 지난 30일 / 그 이전",
	JSON.stringify(groups) === JSON.stringify(["오늘", "어제", "지난 7일", "지난 30일", "그 이전"]),
	JSON.stringify(groups),
);
const expectedTitles = CHATS.map((chat) => chat.title);
const expectedSnippets = CHATS.map((chat) => chat.snippet);
const expectedTimes = [clock(today1), clock(today2), clock(yesterday), monthDay(withinWeek), monthDay(withinMonth), monthDay(older)];
check("every seeded chat is listed, newest first", await waitForTexts(page, ".ai__history-text strong", expectedTitles), JSON.stringify(await texts(page, ".ai__history-text strong")));
check("each row carries its first Quick answer as the snippet", JSON.stringify(await texts(page, ".ai__history-text small")) === JSON.stringify(expectedSnippets));
check("each row carries the reference time label", JSON.stringify(await texts(page, ".ai__history-time")) === JSON.stringify(expectedTimes), JSON.stringify(await texts(page, ".ai__history-time")));

const box = await page.locator(".ai__history-popover").boundingBox();
const popoverStyle = await page.locator(".ai__history-popover").evaluate((element) => {
	const style = getComputedStyle(element);
	return { width: style.width, radius: style.borderRadius, shadow: style.boxShadow };
});
check("the popover is the reference 360px / radius 12 / window shadow surface", Math.round(box.width) === 360 && popoverStyle.radius === "12px" && popoverStyle.shadow.includes("20px 60px"), JSON.stringify({ box, popoverStyle }));
check("no row is marked current before any chat is opened", (await page.locator(".ai__history-item--current").count()) === 0);

const layout = await page.evaluate(() => {
	const panel = document.querySelector(".ai").getBoundingClientRect();
	const popover = document.querySelector(".ai__history-popover").getBoundingClientRect();
	const row = document.querySelector(".ai__history-item").getBoundingClientRect();
	const dot = document.querySelector(".ai__history-dot").getBoundingClientRect();
	const text = document.querySelector(".ai__history-text").getBoundingClientRect();
	const title = document.querySelector(".ai__history-text strong").getBoundingClientRect();
	const snippet = document.querySelector(".ai__history-text small").getBoundingClientRect();
	const time = document.querySelector(".ai__history-time").getBoundingClientRect();
	return { panel, popover, row, dot, text, title, snippet, time };
});
const inside = (inner, outer) => inner.left >= outer.left - 1 && inner.right <= outer.right + 1 && inner.top >= outer.top - 1 && inner.bottom <= outer.bottom + 1;
check("the popover sits inside the AI column, clear of its header", inside(layout.popover, layout.panel) && layout.popover.top >= layout.panel.top + 48, JSON.stringify({ popover: layout.popover.top, panel: layout.panel.top }));
check("the row reads dot → title/snippet → time with no overlap", layout.dot.right <= layout.text.left && layout.text.right <= layout.time.left && inside(layout.dot, layout.row) && inside(layout.time, layout.row));
check("the title sits above its snippet", layout.title.bottom <= layout.snippet.top + 1 && layout.title.left === layout.snippet.left);
await page.screenshot({ path: join(EVIDENCE, "history-open.png") });
await page.locator(".ai__history-popover").screenshot({ path: join(EVIDENCE, "app-history-popover.png") });

// 2. Opening a row loads that chat and marks it current with the accent dot.
await page.locator(".ai__history-item").first().click();
check("clicking a row closes the popover", await popoverClosed(page));
await page.locator(".ai__answer").first().waitFor({ state: "visible", timeout: 5000 });
const loadedAnswer = await page.locator(".ai__answer-content").first().innerText();
check("the clicked chat's answer is loaded into the conversation", loadedAnswer.includes("예산은 3.8억으로 확정됐습니다."), loadedAnswer);
const loadedQuestion = await page.locator(".ai__user-message p").innerText();
check("the clicked chat's question is loaded", loadedQuestion === "Q3 마케팅 예산 승인 근거 찾아줘", loadedQuestion);
await page.screenshot({ path: join(EVIDENCE, "history-item-loaded.png") });

await openHistory(page);
const current = page.locator(".ai__history-item--current");
check("the opened chat is the single current row", (await current.count()) === 1);
const currentStyle = await current.evaluate((element) => ({
	background: getComputedStyle(element).backgroundColor,
	dot: getComputedStyle(element.querySelector(".ai__history-dot")).backgroundColor,
	title: element.querySelector("strong").textContent,
}));
check("the current row uses the reference accent wash and dot", currentStyle.background === "rgba(255, 99, 99, 0.08)" && currentStyle.dot === "rgb(255, 99, 99)", JSON.stringify(currentStyle));
check("the current row is the chat that was opened", currentStyle.title === "Q3 마케팅 예산 승인", currentStyle.title);
await page.screenshot({ path: join(EVIDENCE, "history-current-chat.png") });

// 3. The search input filters through the store.
await page.locator(".ai__history-header input").fill("예산");
check(
	'searching "예산" keeps only matching chats, regrouped',
	await waitForTexts(page, ".ai__history-text strong", ["Q3 마케팅 예산 승인", "v1 vs v3 예산안 차이"]),
	JSON.stringify(await texts(page, ".ai__history-text strong")),
);
check(
	"the regrouped hits keep the day order",
	JSON.stringify(await texts(page, ".ai__history-group")) === JSON.stringify(["오늘", "지난 7일"]),
	JSON.stringify(await texts(page, ".ai__history-group")),
);
await page.screenshot({ path: join(EVIDENCE, "history-search.png") });

await page.locator(".ai__history-header input").fill("zzz-no-hit");
await page.locator(".ai__history-empty").waitFor({ state: "visible", timeout: 5000 });
const emptyText = await page.locator(".ai__history-empty").innerText();
check('the empty state names the query in the reference copy', emptyText === '"zzz-no-hit"에 해당하는 대화가 없습니다', emptyText);
await page.screenshot({ path: join(EVIDENCE, "history-empty.png") });

// 4. Enter opens the first hit of the current query.
await page.locator(".ai__history-header input").fill("CFO");
await waitForTexts(page, ".ai__history-text strong", ["CFO 승인 메일 찾기"]);
await page.keyboard.press("Enter");
check("Enter closes the popover", await popoverClosed(page));
await page.waitForFunction(() => document.querySelector(".ai__user-message p")?.textContent === "박지훈 CFO가 보낸 예산 승인 메일 찾아줘", null, { timeout: 5000 });
check("Enter opened the first hit of the query", (await page.locator(".ai__user-message p").innerText()) === "박지훈 CFO가 보낸 예산 승인 메일 찾아줘");
await page.screenshot({ path: join(EVIDENCE, "history-enter.png") });

// 5. Esc closes the popover and leaves the Finder query alone (it must not bubble to the Finder keymap).
await page.locator(".search__input").fill("qa-marker");
await openHistory(page);
await page.locator(".ai__history-header input").press("Escape");
check("Esc closes the popover", await popoverClosed(page));
const finderQuery = await page.locator(".search__input").inputValue();
check("Esc in the history search leaves the Finder query untouched", finderQuery === "qa-marker", finderQuery);
await page.screenshot({ path: join(EVIDENCE, "history-esc.png") });
await page.locator(".search__input").fill("");
await page.waitForFunction(() => document.querySelector(".search__input")?.value === "");

// 6. The transparent click-catcher closes the popover without reaching the composer.
await openHistory(page);
const body = await page.locator(".ai__body").boundingBox();
await page.mouse.click(body.x + 30, body.y + body.height - 40);
check("clicking behind the popover closes it", await popoverClosed(page));
check("the click-catcher swallowed the click", (await page.locator(".ai__composer input").inputValue()) === "");
await page.screenshot({ path: join(EVIDENCE, "history-backdrop.png") });

// 7. The advertised ⌘⇧H shortcut toggles the popover.
await page.keyboard.press("Meta+Shift+H");
await page.locator(".ai__history-popover").waitFor({ state: "visible", timeout: 5000 });
check("⌘⇧H opens the popover", (await page.locator(".ai__history-popover").count()) === 1);
await page.keyboard.press("Meta+Shift+H");
check("⌘⇧H closes it again", await popoverClosed(page));

// 8. Adjacent Finder surfaces still work after the popover keyboard work.
await page.locator(".sidebar__item", { hasText: "Recents" }).first().click();
await page.waitForFunction(() => document.querySelector(".list__empty")?.textContent === "최근에 연 파일이 없습니다", null, { timeout: 5000 });
check("the Recents view still renders its empty state", (await page.locator(".list__empty").innerText()) === "최근에 연 파일이 없습니다");
await page.screenshot({ path: join(EVIDENCE, "finder-regression.png") });

// 9. Fidelity read: the same metrics/colors taken from the reference prototype and from the app.
refBrowser = await chromium.launch();
const refPage = await refBrowser.newPage({ viewport: { width: 1360, height: 820 } });
await refPage.goto(`file://${REFERENCE}`);
await refPage.locator('[title="Chat history ⌘⇧H"]').click();
const refPopover = refPage.locator('[data-screen-label="Chat History"]');
await refPopover.waitFor({ state: "visible", timeout: 5000 });
await refPage.screenshot({ path: join(EVIDENCE, "reference-window.png") });
await refPopover.screenshot({ path: join(EVIDENCE, "reference-history-popover.png") });

const FIDELITY_PROPS = {
	popover: ["width", "borderRadius", "backgroundColor", "boxShadow", "borderTopWidth"],
	header: ["height", "paddingLeft", "gap", "borderBottomWidth"],
	list: ["padding", "gap"],
	group: ["fontSize", "fontWeight", "color", "padding"],
	row: ["padding", "borderRadius", "gap", "backgroundColor"],
	dot: ["width", "height", "borderRadius", "marginTop", "backgroundColor"],
	title: ["fontSize", "fontWeight"],
	snippet: ["fontSize", "color"],
	time: ["fontSize", "color"],
};

await openHistory(page);
const appMetrics = await page.evaluate((props) => {
	const pick = {
		popover: () => document.querySelector(".ai__history-popover"),
		header: () => document.querySelector(".ai__history-header"),
		list: () => document.querySelector(".ai__history-list"),
		group: () => document.querySelector(".ai__history-group"),
		row: () => document.querySelector(".ai__history-item--current"),
		dot: () => document.querySelector(".ai__history-item--current .ai__history-dot"),
		title: () => document.querySelector(".ai__history-text strong"),
		snippet: () => document.querySelector(".ai__history-text small"),
		time: () => document.querySelector(".ai__history-time"),
	};
	const out = {};
	for (const [name, keys] of Object.entries(props)) {
		const style = getComputedStyle(pick[name]());
		out[name] = Object.fromEntries(keys.map((key) => [key, style[key]]));
	}
	return out;
}, FIDELITY_PROPS);
const refMetrics = await refPage.evaluate((props) => {
	const popover = document.querySelector('[data-screen-label="Chat History"]');
	const header = popover.children[0];
	const list = popover.children[1];
	const row = list.children[1];
	const text = row.children[1];
	const pick = {
		popover: () => popover,
		header: () => header,
		list: () => list,
		group: () => list.children[0],
		row: () => row,
		dot: () => row.children[0],
		title: () => text.children[0],
		snippet: () => text.children[1],
		time: () => row.children[2],
	};
	const out = {};
	for (const [name, keys] of Object.entries(props)) {
		const style = getComputedStyle(pick[name]());
		out[name] = Object.fromEntries(keys.map((key) => [key, style[key]]));
	}
	return out;
}, FIDELITY_PROPS);

const diffs = [];
for (const [name, keys] of Object.entries(FIDELITY_PROPS)) {
	for (const key of keys) {
		const appValue = appMetrics[name][key];
		const refValue = refMetrics[name][key];
		if (appValue !== refValue) diffs.push(`${name}.${key}: app=${appValue} reference=${refValue}`);
	}
}
check("the popover matches the reference metrics and colors", diffs.length === 0, diffs.join(" | ") || "all equal");
console.log(`app metrics: ${JSON.stringify(appMetrics)}`);
console.log(`reference popover: ${JSON.stringify(await refPopover.boundingBox())}`);

sweep();
console.log(failures.length === 0 ? "RESULT: PASS" : `RESULT: FAIL (${failures.join(", ")})`);
console.log(`evidence: ${EVIDENCE}`);
process.exit(failures.length === 0 ? 0 : 1);
