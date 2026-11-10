/**
 * Live E2E for AutoRAG citation cards: real Electron, real `hermes serve`, the real plugin package installed from
 * disk, and the real librarian over a synthetic corpus. Only the chat model is the harness's mock inference server,
 * scripted to answer with `::autorag-cite{...}` directives (what the `search` skill tells a real model to write).
 *
 * Copy this file into apps/desktop/e2e/ of a hermes-agent checkout and run
 *   AUTORAG_E2E_ROOT=/tmp/autorag-finder-e2e AUTORAG_LIBRARIAN_CLI=<librarian>/dist/cli/index.js \
 *   AUTORAG_PLUGIN_SRC=<AutoRAG>/plugins/hermes playwright test e2e/autorag-cite.spec.ts
 * (plugins/hermes-tests/e2e/run.sh does exactly that).
 */
import fs from 'node:fs'
import path from 'node:path'

import type { ElectronApplication, Page } from '@playwright/test'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { type MockServer, startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady, type Sandbox } from './fixtures'
import { expect, test } from './test'

const ROOT = process.env.AUTORAG_E2E_ROOT ?? ''
const CLI = process.env.AUTORAG_LIBRARIAN_CLI ?? ''
const SRC = process.env.AUTORAG_PLUGIN_SRC ?? ''

test.skip(!ROOT || !CLI || !SRC, 'AUTORAG_E2E_ROOT, AUTORAG_LIBRARIAN_CLI and AUTORAG_PLUGIN_SRC are required')

const TRIGGER = 'E2E_CITE_TRIGGER'
const PDF_ID = '/docs/Hanbit_Logistics_Contract_Renewal_2026.pdf'
const SLACK_ID = '/slack/default/chunks/C01GEN-1793000000-000100'
const MADE_UP_ID = '/slack/default/chunks/invented-by-the-model'
const REPLY = [
  'The renewal sets the price in two places.',
  `::autorag-cite{n="1" id="${PDF_ID}" q="unit price"}`,
  `::autorag-cite{n="2" id="${SLACK_ID}" q="unit price"}`,
  `::autorag-cite{n="3" id="${MADE_UP_ID}" q="unit price"}`
].join('\n\n')

interface Fixture {
  app: ElectronApplication
  page: Page
  sandbox: Sandbox
  mock: MockServer
}

let fixture: Fixture
const rendererLog: string[] = []

function copyDir(from: string, to: string) {
  fs.mkdirSync(to, { recursive: true })
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    if (entry.name === '__pycache__') continue
    const a = path.join(from, entry.name)
    const b = path.join(to, entry.name)
    if (entry.isDirectory()) copyDir(a, b)
    else fs.copyFileSync(a, b)
  }
}

test.beforeAll(async () => {
  test.setTimeout(300_000)
  const mock = await startMockServer({ replyForPrompt: prompt => (prompt.includes(TRIGGER) ? REPLY : 'ok') })
  const sandbox = createSandbox('autorag-cite')
  // The app runs with HOME=<sandbox>. The librarian finds the crawler archives (Slack, Telegram, Discord) under HOME
  // and would otherwise start downloading ~1.5 GB of embedding models, so link both in from the demo root
  // (setup-demo.sh builds them there). Without the archives a chat citation is correctly reported as "not in the index".
  const link = (from: string, to: string) => {
    if (!fs.existsSync(from) || fs.existsSync(to)) return
    fs.mkdirSync(path.dirname(to), { recursive: true })
    fs.symlinkSync(from, to)
  }
  const demoHome = path.join(ROOT, 'home')
  for (const name of fs.existsSync(demoHome) ? fs.readdirSync(demoHome) : []) {
    if (['.hermes', '.npm', '.pi', '.local', '.cache'].includes(name)) continue
    link(path.join(demoHome, name), path.join(sandbox.root, name))
  }
  link(path.join(process.env.HOME ?? '', '.cache', 'minsync'), path.join(sandbox.root, '.cache', 'minsync'))
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, 'plugins:\n  enabled:\n    - autorag\n  disabled: []\n')
  writeEnvFile(sandbox.hermesHome)
  copyDir(SRC, path.join(sandbox.hermesHome, 'plugins', 'autorag'))
  const env = buildAppEnv(sandbox, {
    AUTORAG_HOME: path.join(ROOT, 'autorag-home'),
    AUTORAG_CONFIG: path.join(ROOT, 'autorag-home', 'config.json'),
    AUTORAG_WORKSPACE: path.join(ROOT, 'workspace'),
    AUTORAG_LIBRARIAN_CLI: CLI,
    // A checkout without its own `.venv` (Hermes keeps its interpreters under ~/.hermes/installs) is otherwise
    // rejected as a dev root and Desktop waits for first-run setup. Any interpreter that can import hermes works.
    ...(process.env.HERMES_DESKTOP_PYTHON ? { HERMES_DESKTOP_PYTHON: process.env.HERMES_DESKTOP_PYTHON } : {}),
    // The librarian child needs `minsync` for file hits; the sandbox HOME hides the cargo bin dir it lives in.
    PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH ?? ''}`
  })
  const { app, page } = await launchDesktop(env)
  // Plugin load failures are reported on the renderer console, which nothing else records.
  page.on('console', msg => {
    if (msg.type() === 'error' || msg.type() === 'warning') rendererLog.push(`${msg.type()}: ${msg.text().slice(0, 500)}`)
  })
  page.on('pageerror', err => rendererLog.push(`pageerror: ${String(err).slice(0, 500)}`))
  fixture = { app, page, sandbox, mock }
  await waitForAppReady({ app, page } as Parameters<typeof waitForAppReady>[0], 240_000)
  // Record what the OS would open instead of launching a real app during the test.
  await app.evaluate(({ shell }) => {
    const g = globalThis as unknown as { __opened: string[] }
    g.__opened = []
    shell.openPath = async (p: string) => {
      g.__opened.push(p)
      return ''
    }
    shell.showItemInFolder = (p: string) => {
      g.__opened.push(`reveal:${p}`)
    }
  })
  // A package's desktop half ships opt-in (installed but inert until the user turns it on). Record that choice the way
  // the Plugins page does, then reload so the loader registers it.
  await page.evaluate(() => {
    const key = 'hermes.desktop.pluginDecisions.v2'
    const decisions = JSON.parse(localStorage.getItem(key) || '{}')
    localStorage.setItem(key, JSON.stringify({ ...decisions, autorag: true }))
  })
  await page.reload()
  await waitForAppReady({ app, page } as Parameters<typeof waitForAppReady>[0], 240_000)
  // The desktop half is copied into the app's own plugin folder after boot. If it is missing, every test below would
  // fail as "0 cards"; say so once, with what Desktop did list.
  const chip = page.getByRole('button', { name: /AutoRAG search/i }).first()
  try {
    await chip.waitFor({ state: 'visible', timeout: 90_000 })
  } catch {
    const body = (await page.locator('body').innerText()).slice(0, 300)
    const ls = (dir: string) => (fs.existsSync(dir) ? fs.readdirSync(dir).join(',') : '(missing)')
    const home = fixture.sandbox.hermesHome
    const log = path.join(home, 'logs', 'desktop.log')
    const lines = fs.existsSync(log)
      ? fs.readFileSync(log, 'utf8').split('\n').filter(l => /plugin|desktop-plugins|reconcile/i.test(l)).slice(-8).join('\n')
      : '(no desktop.log)'
    throw new Error(
      `the AutoRAG desktop plugin did not load (no status-bar chip).\n` +
        `plugins/autorag: ${ls(path.join(home, 'plugins', 'autorag'))}\n` +
        `plugins/autorag/desktop: ${ls(path.join(home, 'plugins', 'autorag', 'desktop'))}\n` +
        `desktop-plugins: ${ls(path.join(home, 'desktop-plugins'))}\n` +
        `desktop.log:\n${lines}\n` +
        `renderer console (plugin-related, last 12):\n${rendererLog.filter(l => /plugin|autorag/i.test(l)).slice(-12).join('\n') || '(none)'}\n` +
        `Page text: ${body}`
    )
  }
})

test('the evidence route answers inside the live backend', async () => {
  test.setTimeout(600_000)
  const { page } = fixture
  const outcomes: Record<string, string> = {}
  for (const id of [PDF_ID, SLACK_ID, MADE_UP_ID]) {
    const started = Date.now()
    const reply = await page.evaluate(async cited => {
      try {
        const api = (window as unknown as { hermesDesktop: { api: (o: Record<string, unknown>) => Promise<unknown> } }).hermesDesktop.api
        await api({ path: `/api/plugins/autorag/evidence?id=${encodeURIComponent(cited)}&q=unit%20price`, timeoutMs: 150_000 })
        return 'ok'
      } catch (error) {
        return String((error as Error)?.message ?? error).slice(0, 700)
      }
    }, id)
    outcomes[id] = `${Date.now() - started} ms: ${reply}`
  }
  test.info().annotations.push({ type: 'evidence-route', description: JSON.stringify(outcomes) })
  console.log('EVIDENCE_ROUTE', JSON.stringify(outcomes))
  expect(outcomes[PDF_ID]).toMatch(/: ok$/)
  expect(outcomes[SLACK_ID]).toMatch(/: ok$/)
  expect(outcomes[MADE_UP_ID]).toMatch(/not-in-index/)
})

test.afterAll(async () => {
  // The sandbox is deleted below; keep its logs when asked, so a boot failure can be read afterwards.
  const keep = process.env.AUTORAG_E2E_LOG_DIR
  if (keep && fixture) {
    for (const dir of [path.join(fixture.sandbox.hermesHome, 'logs'), path.join(fixture.sandbox.userDataDir, 'logs')]) {
      if (fs.existsSync(dir)) copyDir(dir, path.join(keep, path.basename(path.dirname(dir)), 'logs'))
    }
  }
  await fixture?.app.close().catch(() => undefined)
  await fixture?.mock.close()
  fixture?.sandbox.cleanup()
})

test('an assistant message with citation directives renders one card per source', async () => {
  // Every card looks its source up as soon as the message settles; two lookups run at a time and each starts a
  // librarian process, so on a loaded machine the last card can take minutes.
  test.setTimeout(300_000)
  const { page } = fixture
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').first()
  await composer.click()
  await composer.type(`${TRIGGER} what is the unit price?`)
  await page.keyboard.press('Enter')

  const chips = page.locator('.autorag-chip')
  await expect(chips).toHaveCount(3, { timeout: 120_000 })
  // The directive source text itself must not be shown to the user.
  await expect(page.getByText('::autorag-cite')).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('01-cards.png') })
})

test('a card for a real file shows the quoted passage and opens the file', async () => {
  test.setTimeout(300_000)
  const { page, app } = fixture
  const card = page.locator('.autorag-cite').nth(0)
  await card.locator('.autorag-chip').click()
  const quote = card.locator('.autorag-cite-quote')
  await expect(quote).toContainText('unit price', { ignoreCase: true, timeout: 180_000 })
  await page.screenshot({ path: test.info().outputPath('02-file-evidence.png') })

  await card.getByRole('button', { name: /Open file|파일 열기/ }).click()
  await expect
    .poll(async () => app.evaluate(() => (globalThis as unknown as { __opened: string[] }).__opened.length), { timeout: 15_000 })
    .toBeGreaterThan(0)
  const opened = await app.evaluate(() => (globalThis as unknown as { __opened: string[] }).__opened)
  expect(opened.at(-1)).toMatch(/Hanbit_Logistics_Contract_Renewal_2026\.pdf$/)
})

test('a card for a chat message shows sender and text and offers no file actions', async () => {
  test.setTimeout(300_000)
  const { page } = fixture
  const card = page.locator('.autorag-cite').nth(1)
  await card.locator('.autorag-chip').click()
  await expect(card.locator('.autorag-cite-quote')).toContainText('11,500', { timeout: 180_000 })
  await expect(card).toContainText('Mina Park')
  await expect(card.getByRole('button', { name: /Open file|파일 열기/ })).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('03-chat-evidence.png') })
})

test('a citation the index does not contain is marked unverified', async () => {
  test.setTimeout(300_000)
  const { page } = fixture
  const card = page.locator('.autorag-cite').nth(2)
  await expect(card).toHaveAttribute('data-state', 'unknown', { timeout: 180_000 })
  await card.locator('.autorag-chip').click()
  await expect(card.locator('.autorag-cite-note')).toContainText(/not in the AutoRAG index|인덱스에서 이 출처를 찾을 수 없습니다/)
  await expect(card.getByRole('button', { name: /Open file|파일 열기/ })).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('04-unverified.png') })
})
