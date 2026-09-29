/**
 * Manual QA — the Recents view of the real AI Finder app.
 *
 * Proves the user-visible behavior: before anything is opened the view says so,
 * a double-clicked file and a Space-previewed file are both collected, the list
 * is most-recent-first, and the history survives a restart (persisted under the
 * Electron userData path, which is what makes it work on Windows and macOS).
 *
 * Run from the repo root:  bun scripts/manual-qa/run-qa-finder-recents.mjs
 * Prereq: cd app && bunx electron-vite build
 */
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { _electron } from "playwright";

const ROOT = new URL("../..", import.meta.url).pathname;
const EVIDENCE = join(ROOT, ".omo/evidence/ai-finder-app/qa-recents");
mkdirSync(EVIDENCE, { recursive: true });

const sh = (cmd) => {
	try {
		return execSync(cmd, { encoding: "utf8" }).trim();
	} catch {
		return "";
	}
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** GUI application processes, so a launch from a double-click is observable. */
const guiProcesses = () =>
	new Set(
		sh(`ps -axo pid=,command= | grep -F '.app/Contents/MacOS/' | grep -v -F '/Contents/Frameworks/' | grep -v grep`)
			.split("\n")
			.filter(Boolean)
			.map((line) => line.trim()),
	);
const appNameOf = (line) => line.match(/([^/]+)\.app\/Contents\/MacOS\//)?.[1] ?? line;

/** Applications that can hold a document open, so the double-click handler is findable. */
const DOCUMENT_HANDLERS = [
	"TextEdit",
	"Preview",
	"Numbers",
	"Pages",
	"Keynote",
	"Microsoft Word",
	"Visual Studio Code",
	"Safari",
];
const frontmostApp = () =>
	sh(`osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`);
/** The handler that actually took the file — marker-matched on the fixture name. */
const handlerHolding = (needle) => {
	for (const app of DOCUMENT_HANDLERS) {
		if (sh(`pgrep -x "${app}"`) === "") continue;
		const documents = sh(`osascript -e 'tell application "${app}" to get name of every document'`);
		if (documents.includes(needle)) return `${app}: ${documents}`;
	}
	return "";
};

const RUN_ID = `ulw-recents-qa-${Date.now().toString(36)}`;
const textEditWasRunning = sh('pgrep -x "TextEdit"') !== ""; const FIXTURE_OPEN = join(process.env.HOME, "Documents", `${RUN_ID}-opened.txt`);
const FIXTURE_PREVIEW = join(process.env.HOME, "Documents", `${RUN_ID}-previewed.txt`);
const USER_DATA = join(tmpdir(), `${RUN_ID}-userdata`);
const STORE = join(USER_DATA, "recents", "recents.json");

await Bun.write(FIXTURE_OPEN, "AutoRAG Recents QA fixture (opened with the default app)\n");
await Bun.write(FIXTURE_PREVIEW, "AutoRAG Recents QA fixture (previewed with Quick Look)\n");
mkdirSync(USER_DATA, { recursive: true });

const failures = [];
const check = (label, ok, detail) => {
	console.log(`${ok ? "PASS" : "FAIL"} ${label}${detail === undefined ? "" : ` :: ${detail}`}`);
	if (!ok) failures.push(label);
};

const openApp = async () => {
	const app = await _electron.launch({ args: ["app", `--user-data-dir=${USER_DATA}`], cwd: ROOT });
	const page = await app.firstWindow();
	await page.waitForSelector('[role="row"]');
	return { app, page };
};

const clickNav = async (page, label) => {
	await page.bringToFront();
	await page.locator(".sidebar__item", { hasText: label }).first().click();
	await sleep(400);
};

const rowNames = (page) => page.locator('[role="row"] .row__label').allInnerTexts();

let { app, page } = await openApp();

// The store is the app's own userData path, so the history is OS-native and survives restarts.
const userData = await app.evaluate(({ app: electronApp }) => electronApp.getPath("userData"));
check(
	"userData path is the injected directory",
	realpathSync(userData) === realpathSync(USER_DATA),
	`${userData} vs ${USER_DATA}`,
);

// 1. Empty state before anything was opened — an honest message, not "빈 폴더".
await clickNav(page, "Recents");
const emptyText = await page.locator(".list__empty").innerText();
check("Recents starts empty", (await rowNames(page)).length === 0, JSON.stringify(await rowNames(page)));
check("empty state names the Recents view", emptyText === "최근에 연 파일이 없습니다", emptyText);
const emptyNav = await page.locator('.sidebar__item[aria-current="true"] .sidebar__item-label').innerText();
const emptyTab = await page.locator(".tab--active .tab__title").innerText();
const emptyStatus = await page.locator(".status-bar span").first().innerText();
check("the Recents place is the active sidebar item", emptyNav === "Recents", emptyNav);
check("the tab is titled Recents", emptyTab === "Recents", emptyTab);
check("the status bar counts zero items", emptyStatus === "0 items", emptyStatus);
await page.screenshot({ path: join(EVIDENCE, "recents-empty.png") });
check("no store file before anything was opened", !existsSync(STORE), STORE);

// 2. Double-click a real file in Documents → the OS default app opens it and it becomes recent.
await clickNav(page, "Documents");
const openedRow = page.locator('[role="row"]', { hasText: `${RUN_ID}-opened.txt` });
await openedRow.waitFor({ timeout: 10_000 });
const guiBefore = guiProcesses();
await openedRow.dblclick();
await sleep(4000);
const launchedByDoubleClick = [...guiProcesses()].filter((line) => !guiBefore.has(line));
await page.screenshot({ path: join(EVIDENCE, "after-open-dblclick.png") });
console.log(`apps newly running: ${launchedByDoubleClick.map(appNameOf).join(", ") || "none"}`);
console.log(`frontmost after double-click: ${frontmostApp()}`);
check(
	"double-click handed the file to an OS application",
	handlerHolding(RUN_ID).length > 0,
	handlerHolding(RUN_ID) || "no running handler lists the fixture document",
);

// 3. Space on a second file → Quick Look previews it and it becomes recent too.
await page.bringToFront();
const previewRow = page.locator('[role="row"]', { hasText: `${RUN_ID}-previewed.txt` });
await previewRow.click();
await page.keyboard.press(" ");
await sleep(4000);
check(
	"Space launched the native Quick Look preview",
	sh(`ps -axo command | grep -F 'qlmanage -p ${FIXTURE_PREVIEW}' | grep -v grep`).length > 0,
);
await page.screenshot({ path: join(EVIDENCE, "after-space-preview.png") });

const storeAfter = existsSync(STORE) ? JSON.parse(readFileSync(STORE, "utf8")) : null;
check(
	"store records both files, most recent first",
	JSON.stringify(storeAfter) === JSON.stringify([FIXTURE_PREVIEW, FIXTURE_OPEN]),
	JSON.stringify(storeAfter),
);
copyFileSync(STORE, join(EVIDENCE, "store.json"));
// 4. Recents lists both files, newest first, with their real names.
await clickNav(page, "Recents");
const recentNames = await rowNames(page);
check(
	"Recents lists the previewed file then the opened one",
	JSON.stringify(recentNames) === JSON.stringify([`${RUN_ID}-previewed.txt`, `${RUN_ID}-opened.txt`]),
	JSON.stringify(recentNames),
);
const populatedStatus = await page.locator(".status-bar span").first().innerText();
const populatedColumns = await page.locator(".col-header__label").allInnerTexts();
check("the status bar counts the two recents", populatedStatus === "2 items", populatedStatus);
check(
	"the file columns are shown for the Recents list",
	populatedColumns.join("|") === "Name|Date Modified|Size|Kind",
	populatedColumns.join("|"),
);
await page.screenshot({ path: join(EVIDENCE, "recents-populated.png") });

// 5. Restart → the history is still there (persisted, not in-memory).
await app.close();
({ app, page } = await openApp());
await clickNav(page, "Recents");
const afterRestart = await rowNames(page);
check(
	"Recents survives an app restart",
	JSON.stringify(afterRestart) === JSON.stringify([`${RUN_ID}-previewed.txt`, `${RUN_ID}-opened.txt`]),
	JSON.stringify(afterRestart),
);
await page.screenshot({ path: join(EVIDENCE, "recents-after-restart.png") });

// 6. Cleanup — only what this script created.
await app.close();
sh(`osascript -e 'tell application "TextEdit" to close (every document whose name contains "${RUN_ID}")'`);
sh(`pkill -f 'qlmanage -p ${FIXTURE_PREVIEW}'`);
for (const line of launchedByDoubleClick) {
	const name = appNameOf(line);
	if (["TextEdit", "Preview", "Numbers", "Pages", "Keynote", "QuickTime Player"].includes(name)) {
		sh(`osascript -e 'tell application "${name}" to quit'`);
	}
}
if (!textEditWasRunning && sh('osascript -e \'tell application "TextEdit" to get name of every document\'') === "") {
	sh('osascript -e \'tell application "TextEdit" to quit\'');
}
rmSync(FIXTURE_OPEN, { force: true });
rmSync(FIXTURE_PREVIEW, { force: true });
rmSync(USER_DATA, { recursive: true, force: true });
console.log(`cleanup: removed ${FIXTURE_OPEN} ${FIXTURE_PREVIEW} ${USER_DATA}; killed qlmanage for the preview fixture`);
console.log(`cleanup: qlmanage alive = ${sh(`pgrep -fl 'qlmanage -p ${FIXTURE_PREVIEW}'`) || "none"}`);
console.log(`cleanup: fixtures alive = ${existsSync(FIXTURE_OPEN) || existsSync(FIXTURE_PREVIEW)}`);
console.log(`cleanup: document handler still holding the fixture = ${handlerHolding(RUN_ID) || "none"}`);
console.log(`cleanup: unrelated apps that appeared during the run = ${[...guiProcesses()].filter((line) => !guiBefore.has(line)).map(appNameOf).join(", ") || "none"}`);

console.log(failures.length === 0 ? "RESULT: PASS" : `RESULT: FAIL (${failures.join(", ")})`);
console.log(`evidence: ${EVIDENCE}`);
process.exit(failures.length === 0 ? 0 : 1);
