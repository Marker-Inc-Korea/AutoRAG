# AutoRAG Finder App — Agent Notes

The Electron desktop app (`@autorag/app`). The root `AGENTS.md` stays
binding (guardrails, git workflow, DCO). This file adds the app-specific
map, the run commands, and the manual-QA recipe.

## Layout

- `src/main/` — Electron main process. `fs-service.ts` implements the
  filesystem contract with Electron `shell`/`clipboard` injected for
  testability; `ipc.ts` registers one `ipcMain.handle` per channel, in
  contract order.
- `src/preload/` — typed `window.autorag.fs` wrappers over
  `ipcRenderer.invoke`.
- `src/renderer/` — React UI. `data/source.ts` is the single data port:
  the real bridge when `window.autorag.fs` exists, an in-memory fixture
  source otherwise (vitest, plain browser, preload-not-yet-loaded).
  Components never touch `window` directly.
- `src/shared/` — contracts crossing the process boundary.
  `fs-contract.ts` channel names must stay in 1:1 sync with `ipc.ts` and
  `preload/fs-bridge.ts`: adding a bridge method means touching all
  three plus both `FinderSource` adapters in `data/source.ts`.
- `test/` — vitest, one file per state module. No DOM environment:
  UI logic lives in pure functions under `renderer/src/state/`, so hook
  behavior is proven by manual QA, not unit tests.

## UI library direction

For app UI work, actively use **shadcn/ui**: pull the component source from
the shadcn registry (e.g. `https://ui.shadcn.com/r/styles/new-york-v4/<name>.json`),
keep the official Tailwind classes where possible, and when wiring plain CSS,
translate them 1:1. Brand/experience colors come from our design tokens
(`styles/tokens.css`) — map only colors to tokens, never hand-tune shadcn
geometry, spacing, or motion. When a component looks off to a human eye,
revert to the literal registry incarnation first, confirm it visually, and
then re-apply the token colors. Tailwind CSS is approval-cleared for the app;
if/when it lands, prefer classNames over bespoke CSS.

## Commands

Run from the repo root unless noted. Required before any commit:
`cd app && bun run typecheck && bun run test`.

| Command | What it does |
| --- | --- |
| `make electron` | Dev mode: vite dev server + Electron with HMR. `PORT=9234` picks the vite port, `ELECTRON_ARGS=...` forwards args to Electron. |
| `cd app && bun run typecheck` | `tsc --noEmit` for main/preload (node) and renderer (web) projects. |
| `cd app && bun run test` | The app vitest suite. |
| `cd app && bunx electron-vite build && ../node_modules/.bin/electron .` | Production-mode run from `out/`. Use this when QA must exercise packaged behavior (no vite, no HMR). |

`make e2e-live` / `make e2e-live-cold` run the librarian live-E2E
(`scripts/live-e2e/`), **not** this app — never use them for app QA.

## Manual QA — drive the real app with Playwright

Unit tests alone never prove an app change; run the changed behavior
through the real window. Playwright is already in the root
`node_modules` (no install, no new dependency) and its `_electron` API
launches the real app and drives it with trusted input events
(`dblclick`, `keyboard.press`) plus screenshots.

Save a script under `scripts/manual-qa/` (repo convention:
`run-qa-finder-<topic>.mjs`) and run it with `bun`. Evidence goes to
`.omo/evidence/ai-finder-app/<topic>/` — screenshots and logs stay
untracked.

Worked example — the double-click → OS default application behavior
(`.key` → the machine's registered default handler) with the Space →
Quick Look regression check:

```js
import { _electron } from "playwright"; // resolved from the root node_modules
import { execSync } from "node:child_process";
import { mkdirSync } from "node:fs";

const EVIDENCE = new URL("../../.omo/evidence/ai-finder-app/qa-open-default-app/", import.meta.url).pathname;
mkdirSync(EVIDENCE, { recursive: true });
const sh = (cmd) => { try { return execSync(cmd, { encoding: "utf8" }).trim(); } catch { return ""; } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const FIXTURE = `${process.env.HOME}/Documents/ulw-open-qa.key`;

// 1. Fixture with a UNIQUE name — the process-table checks below grep for
//    this exact path so an unrelated user preview can never false-positive.
await Bun.write(FIXTURE, "QA fixture\n");

// 2. Launch the real app (production build; run electron-vite build first).
const app = await _electron.launch({ args: ["app"] }); // run from the repo root
const page = await app.firstWindow();
await page.waitForSelector('[role="row"]');
const row = page.locator('[role="row"]', { hasText: "ulw-open-qa.key" });

// 3. Double-click must launch the OS default application for the extension.
//    (grep for whatever the machine's registered handler is — here a .key
//    opens "Keynote Creator Studio", not Apple Keynote; defer to the OS.)
await row.dblclick();
await sleep(5000); // give the OS handler time to spawn
console.log("default app:", sh("pgrep -fl Keynote") || "NONE — FAIL");
await page.screenshot({ path: `${EVIDENCE}/after-dblclick.png` });

// 4. Space must still open the native Quick Look (qlmanage on macOS).
await row.click();
await page.keyboard.press(" ");
await sleep(5000);
console.log("quick look:", sh(`ps -axo command | grep 'qlmanage -p .*ulw-open-qa.key' | grep -v grep`) || "NONE — FAIL");
await page.screenshot({ path: `${EVIDENCE}/after-space.png` });

// 5. Cleanup is part of the script: close the app, kill ONLY what this
//    script spawned (marker-matched), remove the fixture. Never pkill a
//    broad pattern — this machine runs other AutoRAG clones and previews.
await app.close();
execSync(`pkill -f "qlmanage -p ${FIXTURE}"`);
execSync(`rm -f "${FIXTURE}"`);
```

Rules that make the QA real:

- **Assert the OS effect, not just the DOM.** After a double-click,
  check the process table for the default application actually
  launching (match the spawned pid + start time, or the handler's
  process name). After Space, check `qlmanage -p <your fixture path>`.
- **Unique fixture names.** Grep the process table for the exact
  fixture path; an unrelated preview of a different file must never
  satisfy the check.
- **Screenshot after every step** (`page.screenshot`), before moving on.
- **Marker-matched cleanup only.** `pkill -f` with the exact fixture
  path or the exact app you launched; broad patterns (`pkill electron`)
  kill other sessions' instances on this shared machine.
- **Windows:** the same script body works ( `_electron.launch`,
  `dblclick`, screenshots); the OS-effect checks differ — the default
  app opens via ShellExecute, and Quick Look does not exist there.
