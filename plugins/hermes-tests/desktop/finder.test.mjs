// Tests for plugins/hermes/desktop/plugin.js (the AutoRAG finder pane).
// Run: node --test plugins/hermes-tests/desktop/
//
// plugin.js may import only @hermes/plugin-sdk and react (the app's loader refuses anything else), so it is
// one file. These tests load it the way the loader does: specifiers rewritten to stub modules.
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { describe, it } from 'node:test'

const PLUGIN = resolve(import.meta.dirname, '../../hermes/desktop/plugin.js')

async function loadPlugin() {
  const source = readFileSync(PLUGIN, 'utf8')
  const named = new Set()
  for (const m of source.matchAll(/import\s*\{([^}]*)\}\s*from\s*'@hermes\/plugin-sdk'/g)) {
    for (const part of m[1].split(',')) {
      const name = part.trim().split(/\s+as\s+/)[0]
      if (name) named.add(name)
    }
  }
  const dir = mkdtempSync(join(tmpdir(), 'finder-plugin-'))
  const sdk = join(dir, 'sdk.mjs')
  writeFileSync(
    sdk,
    [...named].map(n => `export const ${n} = globalThis.__sdk?.${n} ?? (() => null)`).join('\n') + '\n'
  )
  const react = join(dir, 'react.mjs')
  writeFileSync(
    react,
    `const id = x => x
export const useState = v => [typeof v === 'function' ? v() : v, () => {}]
export const useEffect = () => {}
export const useMemo = f => f()
export const useRef = v => ({ current: v })
export const useCallback = id
export const useSyncExternalStore = (_s, g) => g()
export default {}
`
  )
  const jsx = join(dir, 'jsx.mjs')
  writeFileSync(jsx, `export const jsx = (t, p, k) => ({ t, p, k })\nexport const jsxs = jsx\nexport const Fragment = 'fragment'\n`)
  const rewritten = source
    .replaceAll("'@hermes/plugin-sdk'", JSON.stringify(pathToFileURL(sdk).href))
    .replaceAll("'react/jsx-runtime'", JSON.stringify(pathToFileURL(jsx).href))
    .replaceAll("'react'", JSON.stringify(pathToFileURL(react).href))
  const out = join(dir, 'plugin.mjs')
  writeFileSync(out, rewritten)
  return import(pathToFileURL(out).href)
}

// The stub SDK module reads these when it is first imported, so they exist before the plugin loads.
globalThis.__sdk = {
  host: { state: { focusedSessionId: { get: () => 's' } }, composer: { insertText: async () => true }, request: async () => ({}), revealPane: () => {} },
  PANES_AREA: 'panes',
  PALETTE_AREA: 'palette',
  KEYBINDS_AREA: 'keybinds',
  STATUSBAR_AREAS: { right: 'statusBar.right', left: 'statusBar.left' }
}
const mod = await loadPlugin()
const { createFinder, applyLine, initialState, attachText, formatStamp } = mod

const GROUPS = {
  type: 'groups',
  groups: [
    { key: 'files' },
    { key: 'content' },
    { key: 'source:kakao', datasourceId: 'kakao', datasourceType: 'kakao' }
  ]
}
const file = (id, path, extra = {}) => ({
  id, kind: 'file', title: path.split('/').pop(), author: null, snippet: '', content: '', path,
  line: null, timestamp: null, url: null, score: null, ...extra
})
const message = (id, extra = {}) => ({
  id, kind: 'message', title: '루킹 야구부', author: '브리타', snippet: '다음 주 화요일에 보자',
  content: '다음 주 화요일에 보자\n장소는 잠실', path: null, line: null,
  timestamp: Date.UTC(2026, 10, 9, 5, 30), url: null, score: null, ...extra
})

describe('applyLine', () => {
  it('a groups line creates pending groups labelled by datasource, files tab active', () => {
    const s = applyLine(initialState('q'), GROUPS)
    assert.deepEqual(s.groups.map(g => [g.key, g.status]), [['files', 'pending'], ['content', 'pending'], ['source:kakao', 'pending']])
    assert.equal(s.groups[2].datasourceId, 'kakao')
    assert.equal(s.activeTab, 'files')
  })

  it('a group line fills hits, status and timing, and keys every hit by group', () => {
    let s = applyLine(initialState('q'), GROUPS)
    s = applyLine(s, { type: 'group', key: 'files', status: 'ready', hits: [file('a', '/x/a.pdf')], error: null, elapsedMs: 12 })
    assert.equal(s.groups[0].status, 'ready')
    assert.equal(s.groups[0].hits[0].key, 'files:a')
    assert.equal(s.groups[0].elapsedMs, 12)
  })

  it('keeps the datasource error text verbatim and off/unavailable statuses', () => {
    let s = applyLine(initialState('q'), GROUPS)
    s = applyLine(s, { type: 'group', key: 'source:kakao', status: 'error', hits: [], error: 'katok: not signed in', elapsedMs: 1 })
    s = applyLine(s, { type: 'group', key: 'content', status: 'off', hits: [], error: null, elapsedMs: 0 })
    assert.equal(s.groups[2].error, 'katok: not signed in')
    assert.equal(s.groups[1].status, 'off')
  })

  it('done ends the run; unknown lines and groups are ignored', () => {
    let s = applyLine(initialState('q'), GROUPS)
    const before = s
    assert.equal(applyLine(s, { type: 'group', key: 'source:nope', status: 'ready', hits: [], error: null }), before)
    assert.equal(applyLine(s, { type: 'weird' }), before)
    s = applyLine(s, { type: 'done', elapsedMs: 5 })
    assert.equal(s.phase, 'done')
  })

  it('tolerates hits with missing fields', () => {
    let s = applyLine(initialState('q'), GROUPS)
    s = applyLine(s, { type: 'group', key: 'files', status: 'ready', hits: [{ id: 'z', path: '/p/z.txt' }, null, { title: 'no id' }], error: null })
    assert.equal(s.groups[0].hits.length, 2)
    assert.equal(s.groups[0].hits[0].title, 'z.txt')
    assert.equal(s.groups[0].hits[0].snippet, '')
  })
})

// ---- controller ---------------------------------------------------------------------------------

function harness(over = {}) {
  const calls = []
  const jobs = over.jobs ?? [[
    { state: 'running', lines: [GROUPS], next: 1 },
    { state: 'running', lines: [{ type: 'group', key: 'files', status: 'ready', hits: [file('a', '/d/a.pdf')], error: null, elapsedMs: 3 }], next: 2 },
    { state: 'done', lines: [{ type: 'done', elapsedMs: 9 }], next: 3 }
  ]]
  let jobNo = 0
  let polls = {}
  const rest = async (path, opts = {}) => {
    calls.push(['rest', path, opts.method ?? 'GET', opts.body])
    if (over.rest) {
      const r = await over.rest(path, opts)
      if (r !== undefined) return r
    }
    if (path === '/find' && opts.method === 'POST') return { id: `job${++jobNo}` }
    const m = /^\/find\/(job\d+)/.exec(path)
    if (m && (opts.method ?? 'GET') === 'GET') {
      const script = jobs[Number(m[1].slice(3)) - 1] ?? []
      const i = polls[m[1]] = (polls[m[1]] ?? 0)
      polls[m[1]] += 1
      return script[Math.min(i, script.length - 1)] ?? { state: 'done', lines: [], next: 0 }
    }
    if (m && opts.method === 'DELETE') return { cancelled: true }
    throw new Error(`unexpected ${path}`)
  }
  const request = async (method, params) => {
    calls.push(['request', method, params])
    if (over.request) return over.request(method, params)
    return { attached: true, ref_text: `@file:${params.path}` }
  }
  const composer = {
    insertText: async (sid, text, opts) => {
      calls.push(['insert', sid, text, opts])
      return over.insertOk ?? true
    }
  }
  const finder = createFinder({
    rest, request, composer,
    getSessionId: () => (over.sessionId === undefined ? 'sess-1' : over.sessionId),
    // A real macrotask: a microtask-only sleep lets a wait loop starve every setImmediate in the test.
    sleep: () => new Promise(r => setImmediate(r)),
    pollMs: 0,
    ...(over.prepareWaitMs === undefined ? {} : { prepareWaitMs: over.prepareWaitMs })
  })
  return { finder, calls, composer }
}

const settled = async finder => {
  for (let i = 0; i < 200 && finder.getState().phase === 'running'; i++) await new Promise(r => setImmediate(r))
}

describe('controller.search', () => {
  it('posts the query once, polls with the cursor and applies each line exactly once', async () => {
    const { finder, calls } = harness()
    await finder.search('계약서')
    await settled(finder)
    const s = finder.getState()
    assert.equal(s.phase, 'done')
    assert.equal(s.groups[0].hits.length, 1)
    assert.deepEqual(calls[0], ['rest', '/find', 'POST', { query: '계약서' }])
    const afters = calls.filter(c => c[0] === 'rest' && c[2] === 'GET').map(c => c[1])
    assert.ok(afters[0].includes('after=0'))
    assert.ok(afters.some(p => p.includes('after=1')))
    assert.ok(afters.some(p => p.includes('after=2')))
  })

  it('does nothing for an empty or whitespace query', async () => {
    const { finder, calls } = harness()
    await finder.search('   ')
    assert.equal(calls.length, 0)
    assert.equal(finder.getState().phase, 'idle')
  })

  it('a new search cancels the previous job and drops its late lines', async () => {
    const slow = [{ state: 'running', lines: [], next: 0 }]
    const { finder, calls } = harness({ jobs: [slow, [{ state: 'done', lines: [GROUPS, { type: 'done' }], next: 2 }]] })
    await finder.search('first')
    await finder.search('second')
    await settled(finder)
    assert.ok(calls.some(c => c[0] === 'rest' && c[1] === '/find/job1' && c[2] === 'DELETE'))
    const s = finder.getState()
    assert.equal(s.query, 'second')
    assert.equal(s.phase, 'done')
  })

  it('a start failure shows the backend detail, not a stack', async () => {
    const { finder } = harness({ rest: async (_p, o) => { if (o.method === 'POST') { const e = new Error('503'); e.detail = 'Node.js 24+ was not found on PATH'; throw e } } })
    await finder.search('x')
    const s = finder.getState()
    assert.equal(s.phase, 'error')
    assert.match(s.error, /Node\.js 24\+/)
  })

  it('a failed job marks groups still pending as errors with the job error', async () => {
    const { finder } = harness({ jobs: [[{ state: 'error', error: 'finder exited with code 2', lines: [GROUPS], next: 1 }]] })
    await finder.search('x')
    await settled(finder)
    const s = finder.getState()
    assert.equal(s.phase, 'error')
    assert.ok(s.groups.every(g => g.status === 'error' && /code 2/.test(g.error)))
  })

  it('dispose cancels the running job and stops polling', async () => {
    const { finder, calls } = harness({ jobs: [[{ state: 'running', lines: [], next: 0 }]] })
    await finder.search('x')
    finder.dispose()
    await new Promise(r => setTimeout(r, 5))
    assert.ok(calls.some(c => c[2] === 'DELETE'))
    const n = calls.length
    await new Promise(r => setTimeout(r, 10))
    assert.equal(calls.length, n)
  })

  it('notifies subscribers on change and stops after unsubscribe', async () => {
    const { finder } = harness()
    let n = 0
    const off = finder.subscribe(() => { n++ })
    await finder.search('x')
    await settled(finder)
    assert.ok(n >= 3)
    off()
    const before = n
    finder.setActiveTab('content')
    assert.equal(n, before)
  })
})

// ---- the librarian is installed in the background; a search never waits on npm ---------------------------

function notReady(state) {
  const e = new Error('503')
  e.detail = { state, message: `the AutoRAG finder is ${state}` }
  return e
}

const until = async (predicate, ms = 2000) => {
  const end = Date.now() + ms
  while (!predicate() && Date.now() < end) await new Promise(r => setImmediate(r))
  assert.ok(predicate(), 'condition was not reached in time')
}

describe('preparing', () => {
  it('shows a preparing phase while the install runs, then searches by itself', async () => {
    let ready = false
    const { finder, calls } = harness({
      rest: async (path, opts) => {
        if (path === '/find' && opts.method === 'POST' && !ready) throw notReady('installing')
        if (path === '/status') return { state: ready ? 'ready' : 'installing' }
      }
    })
    const searching = finder.search('계약서') // stays pending until the install finishes
    await until(() => finder.getState().phase === 'preparing')
    ready = true
    await searching
    await settled(finder)
    assert.equal(finder.getState().phase, 'done')
    assert.equal(calls.filter(c => c[1] === '/find' && c[2] === 'POST').length, 2)
  })

  it('a failed install is an error with the reason, not an endless wait', async () => {
    const { finder } = harness({
      rest: async (path, opts) => {
        if (path === '/find' && opts.method === 'POST') throw notReady('installing')
        if (path === '/status') return { state: 'failed', error: 'npm ERR! network' }
      }
    })
    await finder.search('x')
    for (let i = 0; i < 50 && finder.getState().phase === 'preparing'; i++) await new Promise(r => setImmediate(r))
    const s = finder.getState()
    assert.equal(s.phase, 'error')
    assert.match(s.error, /network/)
  })

  it('a newer search replaces one that is still waiting for the install', async () => {
    let ready = false
    const { finder, calls } = harness({
      rest: async (path, opts) => {
        if (path === '/find' && opts.method === 'POST' && !ready) throw notReady('installing')
        if (path === '/status') return { state: ready ? 'ready' : 'installing' }
      }
    })
    const first = finder.search('first')
    await until(() => finder.getState().phase === 'preparing')
    const second = finder.search('second')
    await first // the replaced search ends without retrying
    ready = true
    await second
    await settled(finder)
    assert.equal(finder.getState().query, 'second')
    assert.equal(finder.getState().phase, 'done')
    const retried = calls.filter(c => c[1] === '/find' && c[2] === 'POST' && c[3].query === 'first').length
    assert.equal(retried, 1) // 'first' was tried once and never again after being replaced
  })

  it('prepare() asks the backend to start installing once, without waiting for it', async () => {
    const { finder, calls } = harness({ rest: async path => (path === '/prepare' ? { state: 'installing' } : undefined) })
    await finder.prepare()
    assert.deepEqual(calls[0].slice(0, 3), ['rest', '/prepare', 'POST'])
    assert.equal(finder.getState().phase, 'idle')
  })

  it('gives up waiting for the install after the cap', async () => {
    const { finder } = harness({
      prepareWaitMs: 5,
      rest: async (path, opts) => {
        if (path === '/find' && opts.method === 'POST') throw notReady('installing')
        if (path === '/status') return { state: 'installing' }
      }
    })
    await finder.search('x')
    await new Promise(r => setTimeout(r, 60))
    const s = finder.getState()
    assert.equal(s.phase, 'error')
    assert.match(s.error, /still being prepared|timed out/i)
  })
})

// ---- selection + attach -------------------------------------------------------------------------

async function withResults(over = {}) {
  const h = harness({
    jobs: [[{
      state: 'done',
      next: 4,
      lines: [
        GROUPS,
        { type: 'group', key: 'files', status: 'ready', hits: [file('a', '/d/a.pdf'), file('b', '/d/b.hwp')], error: null },
        { type: 'group', key: 'content', status: 'ready', hits: [file('a2', '/d/a.pdf', { kind: 'content', snippet: '계약 기간', line: 12 })], error: null },
        { type: 'group', key: 'source:kakao', status: 'ready', hits: [message('m1')], error: null },
        { type: 'done' }
      ]
    }]],
    ...over
  })
  await h.finder.search('계약')
  await settled(h.finder)
  return h
}

describe('selection', () => {
  it('toggle, select all in the active tab, clear', async () => {
    const { finder } = await withResults()
    finder.toggle('files:a')
    assert.deepEqual(Object.keys(finder.getState().selected), ['files:a'])
    finder.toggle('files:a')
    assert.deepEqual(Object.keys(finder.getState().selected), [])
    finder.selectAll('files')
    assert.equal(Object.keys(finder.getState().selected).length, 2)
    finder.clearSelection()
    assert.equal(Object.keys(finder.getState().selected).length, 0)
  })

  it('a new search clears the selection', async () => {
    const { finder } = await withResults()
    finder.toggle('files:a')
    await finder.search('다른 검색')
    assert.equal(Object.keys(finder.getState().selected).length, 0)
  })
})

describe('attach', () => {
  it('stages each file through file.attach in the focused session and inserts the refs once', async () => {
    const { finder, calls } = await withResults()
    await finder.attach(['files:a', 'files:b'])
    const attaches = calls.filter(c => c[0] === 'request')
    assert.deepEqual(attaches.map(c => [c[1], c[2].session_id, c[2].path]), [
      ['file.attach', 'sess-1', '/d/a.pdf'],
      ['file.attach', 'sess-1', '/d/b.hwp']
    ])
    const inserts = calls.filter(c => c[0] === 'insert')
    assert.equal(inserts.length, 1)
    assert.equal(inserts[0][1], 'sess-1')
    assert.equal(inserts[0][2].trim(), '@file:/d/a.pdf @file:/d/b.hwp')
    assert.equal(finder.getState().notice.kind, 'attached')
    assert.equal(finder.getState().notice.count, 2)
  })

  it('uses the gateway ref_text, so quoting of odd paths is the gateway\'s', async () => {
    const { finder, calls } = await withResults({ request: async (_m, p) => ({ attached: true, ref_text: '@file:`' + p.path + '`' }) })
    await finder.attach(['files:a'])
    assert.match(calls.find(c => c[0] === 'insert')[2], /@file:`\/d\/a\.pdf`/)
  })

  it('attaches the same file once even when it is selected in two tabs', async () => {
    const { finder, calls } = await withResults()
    await finder.attach(['files:a', 'content:a2'])
    assert.equal(calls.filter(c => c[0] === 'request').length, 1)
  })

  it('a message hit becomes a quoted evidence block with its source, no file.attach', async () => {
    const { finder, calls } = await withResults()
    await finder.attach(['source:kakao:m1'])
    assert.equal(calls.filter(c => c[0] === 'request').length, 0)
    const text = calls.find(c => c[0] === 'insert')[2]
    assert.match(text, /^> kakao · 루킹 야구부 · 브리타 · 2026-11-09 /m)
    assert.match(text, /^> 다음 주 화요일에 보자$/m)
    assert.match(text, /^> 장소는 잠실$/m)
  })

  it('mixed selection puts file refs first, then quotes, in one insert', async () => {
    const { finder, calls } = await withResults()
    await finder.attach(['source:kakao:m1', 'files:a'])
    const inserts = calls.filter(c => c[0] === 'insert')
    assert.equal(inserts.length, 1)
    const text = inserts[0][2]
    assert.ok(text.indexOf('@file:/d/a.pdf') < text.indexOf('> kakao'))
  })

  it('refuses without a chat session and touches nothing', async () => {
    const { finder, calls } = await withResults({ sessionId: null })
    const before = calls.length
    await finder.attach(['files:a'])
    assert.equal(calls.length, before)
    assert.equal(finder.getState().notice.kind, 'no-session')
  })

  it('one failing file does not lose the others; the notice counts both', async () => {
    const { finder, calls } = await withResults({
      request: async (_m, p) => { if (p.path.endsWith('.hwp')) throw new Error('boom'); return { attached: true, ref_text: `@file:${p.path}` } }
    })
    await finder.attach(['files:a', 'files:b'])
    assert.equal(calls.find(c => c[0] === 'insert')[2].trim(), '@file:/d/a.pdf')
    const n = finder.getState().notice
    assert.equal(n.kind, 'partial')
    assert.deepEqual([n.count, n.failed], [1, 1])
  })

  it('says WHY a file could not be staged, with the gateway\'s own words', async () => {
    const { finder } = await withResults({
      request: async () => { const e = new Error('Error: file.attach 5028'); e.message = 'path is outside the allowed folders'; throw e }
    })
    await finder.attach(['files:a'])
    const n = finder.getState().notice
    assert.equal(n.kind, 'failed')
    assert.match(n.reason, /outside the allowed folders/)
  })

  it('a refused attach (attached:false + message) reports its message', async () => {
    const { finder } = await withResults({ request: async () => ({ attached: false, message: 'file too large' }) })
    await finder.attach(['files:a'])
    assert.match(finder.getState().notice.reason, /file too large/)
  })

  it('all files failing inserts nothing', async () => {
    const { finder, calls } = await withResults({ request: async () => { throw new Error('nope') } })
    await finder.attach(['files:a'])
    assert.equal(calls.filter(c => c[0] === 'insert').length, 0)
    assert.equal(finder.getState().notice.kind, 'failed')
  })

  it('reports a composer that did not accept the text', async () => {
    const { finder } = await withResults({ insertOk: false })
    await finder.attach(['files:a'])
    assert.equal(finder.getState().notice.kind, 'composer')
  })

  it('caps one attach at 20 items', async () => {
    const many = Array.from({ length: 25 }, (_, i) => file(`f${i}`, `/d/f${i}.txt`))
    const { finder, calls } = harness({
      jobs: [[{ state: 'done', next: 3, lines: [GROUPS, { type: 'group', key: 'files', status: 'ready', hits: many, error: null }, { type: 'done' }] }]]
    })
    await finder.search('x')
    await settled(finder)
    await finder.attach(many.map(h => `files:${h.id}`))
    assert.equal(calls.filter(c => c[0] === 'request').length, 20)
    assert.equal(finder.getState().notice.capped, true)
  })

  it('clears the selection after a successful attach', async () => {
    const { finder } = await withResults()
    finder.toggle('files:a')
    await finder.attach(Object.keys(finder.getState().selected))
    assert.equal(Object.keys(finder.getState().selected).length, 0)
  })
})

describe('attachText / formatStamp', () => {
  it('flattens, bounds and prefixes every line of a long message', () => {
    const text = attachText({ ...message('m'), content: 'a'.repeat(900) + '\nb' }, 'kakao')
    for (const line of text.split('\n')) assert.ok(line.startsWith('> '))
    assert.ok(text.length < 800)
  })

  it('formatStamp renders UTC-independent local date, null-safe', () => {
    assert.equal(formatStamp(null), '')
    assert.match(formatStamp(Date.UTC(2026, 10, 9, 5, 30)), /^2026-11-0[89] \d\d:\d\d$/)
  })
})

describe('plugin contract', () => {
  it('default-exports a HermesPlugin named autorag that is opt-in on', () => {
    const p = mod.default
    assert.equal(p.id, 'autorag')
    assert.equal(typeof p.register, 'function')
  })

  it('register adds a pane, a palette command, a keybind and a status chip, and disposes the finder', () => {
    const regs = []
    const disposers = []
    const ctx = {
      register: c => { regs.push(c); return () => {} },
      registerMany: cs => { regs.push(...cs); return () => {} },
      rest: async () => ({}),
      onDispose: f => disposers.push(f),
      i18n: { register: () => {} },
      os: { revealPath: async () => true, openExternal: async () => true },
      storage: { get: (_k, f) => f, set: () => {}, remove: () => {} }
    }
    globalThis.document = { createElement: () => ({ remove: () => {} }), head: { append: () => {} } }
    mod.default.register(ctx)
    const areas = regs.map(r => r.area)
    assert.ok(areas.includes('panes') && areas.includes('palette') && areas.includes('keybinds') && areas.includes('statusBar.right'))
    const key = regs.find(r => r.area === 'keybinds')
    assert.deepEqual(key.data.defaults, ['mod+alt+f'])
    assert.ok(disposers.length >= 1)

    // A Cmd+F-style bar lives in the right sidebar next to the file tree, never in the middle of the chat.
    const pane = regs.find(r => r.area === 'panes')
    assert.equal(pane.data.placement, 'right')
    // `enforce` re-homes a pane an earlier version already adopted into the chat zone.
    assert.deepEqual(pane.data.dock, { pane: 'files', pos: 'center', enforce: true })
    // Docked in the files zone, so it must not out-size it: a wider tenant widens the whole fixed zone (max() of
    // its tabs) and, with no cap of its own, uncaps it, which squeezed the chat to a sliver in the real app.
    assert.deepEqual(
      { width: pane.data.width, minWidth: pane.data.minWidth, maxWidth: pane.data.maxWidth },
      { width: '20rem', minWidth: '10rem', maxWidth: '20rem' }
    )
    assert.ok(!('defaultCollapsed' in pane.data) || pane.data.defaultCollapsed === false)
  })
})
