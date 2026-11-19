/**
 * Live E2E for the AutoRAG finder pane: real Electron, real `hermes serve`, the real plugin package installed
 * from disk, and the real `autorag lite find` CLI over a synthetic corpus. Only the chat model is the harness's
 * mock inference server.
 *
 * Copy this file into apps/desktop/e2e/ of a hermes-agent checkout and run
 *   AUTORAG_E2E_ROOT=/tmp/autorag-finder-e2e AUTORAG_LIBRARIAN_CLI=<librarian>/dist/cli/index.js \
 *   AUTORAG_PLUGIN_SRC=<AutoRAG>/plugins/hermes playwright test e2e/autorag-finder.spec.ts
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

interface Fixture {
  app: ElectronApplication
  page: Page
  sandbox: Sandbox
  mock: MockServer
}

let fixture: Fixture

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
  // One launch: the sandbox is seeded (mock provider, plugin copied in, plugin allow-listed) BEFORE the app
  // starts, so Hermes loads the package on first boot. The backend router only mounts for a plugin listed in
  // plugins.enabled (a security gate); it reads the librarian CLI path and the synthetic AutoRAG home from env.
  const mock = await startMockServer()
  const sandbox = createSandbox('autorag-finder')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, 'plugins:\n  enabled:\n    - autorag\n  disabled: []\n')
  writeEnvFile(sandbox.hermesHome)
  copyDir(SRC, path.join(sandbox.hermesHome, 'plugins', 'autorag'))
  const env = buildAppEnv(sandbox, {
    AUTORAG_HOME: path.join(ROOT, 'autorag-home'),
    AUTORAG_CONFIG: path.join(ROOT, 'autorag-home', 'config.json'),
    AUTORAG_WORKSPACE: path.join(ROOT, 'workspace'),
    AUTORAG_LIBRARIAN_CLI: CLI
  })
  const { app, page } = await launchDesktop(env)
  fixture = { app, page, sandbox, mock }
  await waitForAppReady({ app, page } as Parameters<typeof waitForAppReady>[0], 240_000)
})

test.afterAll(async () => {
  await fixture?.app.close().catch(() => undefined)
  await fixture?.mock.close()
  fixture?.sandbox.cleanup()
})

test('the plugin shows up and loads its desktop half', async () => {
  const { page } = fixture
  await page.getByRole('button', { name: /AutoRAG search/i }).first().waitFor({ state: 'visible', timeout: 60_000 })
  await page.screenshot({ path: test.info().outputPath('01-statusbar-chip.png') })
})

test('Cmd/Ctrl+Alt+F opens the finder pane and a search streams tabs then results', async () => {
  const { page } = fixture
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Alt+f' : 'Control+Alt+f')
  const input = page.getByRole('textbox', { name: /AutoRAG search/i })
  await input.waitFor({ state: 'visible', timeout: 20_000 })
  await input.fill('납품 계약서')
  await input.press('Enter')

  const tabs = page.getByRole('tab')
  await expect(tabs.first()).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('tab', { name: /Files|파일/ })).toBeVisible()
  await expect(page.getByRole('tab', { name: /Documents|문서/ })).toBeVisible()
  await page.screenshot({ path: test.info().outputPath('02-tabs-streaming.png') })

  await page.getByRole('tab', { name: /Documents|문서/ }).click()
  const rows = page.getByRole('option')
  await expect(rows.first()).toBeVisible({ timeout: 120_000 })
  await page.screenshot({ path: test.info().outputPath('03-document-results.png') })
})

test('selecting a document and pressing Add to chat puts an @file ref in the composer', async () => {
  const { page } = fixture
  await page.getByRole('option').first().click()
  const add = page.getByRole('button', { name: /Add 1 to chat|1개 대화에 추가/ })
  await expect(add).toBeEnabled()
  await add.click()
  await expect(page.getByRole('status').filter({ hasText: /Added 1|1개를 대화에 추가/ })).toBeVisible({ timeout: 30_000 })
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').first()
  await expect(composer).toContainText('@file', { timeout: 15_000 })
  await page.screenshot({ path: test.info().outputPath('04-attached-to-chat.png') })
})
