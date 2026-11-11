// AutoRAG finder for Hermes Desktop: a Cmd/Ctrl+Alt+F search pane over file names, document content and every
// connected datasource, filtered for relevance by the pinned `autorag lite find`. Selected results are handed to
// the current chat: files are staged with the gateway's own `file.attach` and their `@file:` refs are put in the
// composer; chat/mail hits become quoted evidence blocks with their source.
//
// Only @hermes/plugin-sdk and react are imported (catalog rule 8). The controller below is plain functions so
// plugins/hermes-tests/desktop/finder.test.mjs can drive it without the app.
import {
  Button, GlyphSpinner, icons, KEYBINDS_AREA, PALETTE_AREA, PANES_AREA, ScrollArea, STATUSBAR_AREAS, Tip, host,
  usePluginI18n
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'autorag'
const PANE = 'finder'
const PANE_ID = `${ID}:${PANE}`
const MAX_ATTACH = 20
const QUOTE_CHARS = 700
const POLL_MS = 250
const MAX_POLL_ERRORS = 8
// How long a search waits for the librarian install before giving up: the first install downloads the package once.
const PREPARE_WAIT_MS = 5 * 60 * 1000

// ---- state ---------------------------------------------------------------------------------------

export function initialState(query = '') {
  return { query, phase: query ? 'running' : 'idle', error: null, groups: [], activeTab: 'files', selected: {}, notice: null, elapsedMs: null }
}

function normalizeHit(groupKey, raw) {
  if (!raw || typeof raw !== 'object') return null
  const path = typeof raw.path === 'string' && raw.path ? raw.path : null
  const id = String(raw.id ?? path ?? raw.title ?? '')
  if (!id) return null
  const title = (typeof raw.title === 'string' && raw.title) || (path ? path.split(/[\\/]/).pop() : id)
  return {
    key: `${groupKey}:${id}`,
    id,
    kind: raw.kind === 'message' ? 'message' : raw.kind === 'content' ? 'content' : 'file',
    title,
    author: typeof raw.author === 'string' && raw.author ? raw.author : null,
    snippet: typeof raw.snippet === 'string' ? raw.snippet : '',
    content: typeof raw.content === 'string' ? raw.content : '',
    path,
    line: Number.isInteger(raw.line) ? raw.line : null,
    timestamp: Number.isFinite(raw.timestamp) ? raw.timestamp : null,
    url: typeof raw.url === 'string' && /^https?:\/\//i.test(raw.url) ? raw.url : null
  }
}

export function applyLine(state, line) {
  if (!line || typeof line !== 'object') return state
  if (line.type === 'groups' && Array.isArray(line.groups)) {
    const groups = line.groups
      .filter(g => g && typeof g.key === 'string')
      .map(g => ({
        key: g.key,
        datasourceId: g.datasourceId ?? null,
        datasourceType: g.datasourceType ?? null,
        status: 'pending',
        hits: [],
        error: null,
        elapsedMs: null
      }))
    return { ...state, groups, activeTab: groups[0]?.key ?? 'files' }
  }
  if (line.type === 'group') {
    const at = state.groups.findIndex(g => g.key === line.key)
    if (at < 0) return state
    const hits = (Array.isArray(line.hits) ? line.hits : []).map(h => normalizeHit(line.key, h)).filter(Boolean)
    const next = { ...state.groups[at], status: line.status ?? 'ready', hits, error: line.error ?? null, elapsedMs: line.elapsedMs ?? null }
    return { ...state, groups: state.groups.map((g, i) => (i === at ? next : g)) }
  }
  if (line.type === 'done') return { ...state, phase: 'done', elapsedMs: line.elapsedMs ?? null }
  return state
}

function failPending(state, message) {
  return {
    ...state,
    phase: 'error',
    error: message,
    groups: state.groups.map(g => (g.status === 'pending' ? { ...g, status: 'error', error: message } : g))
  }
}

export function formatStamp(ms) {
  if (!Number.isFinite(ms)) return ''
  const d = new Date(ms)
  const p = n => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

// What a source is called in the pane. The config alias (`mailcrawl`, `nomadamas`) names an install, not a kind of
// source; the type says what it is. Types with no friendlier name keep their own, and an unknown source falls back
// to its alias.
const SOURCE_NAMES = { mailcrawl: 'mail', hister: 'browser' }

export function sourceName(group) {
  const id = group?.datasourceId ?? String(group?.key ?? 'search').replace(/^source:/, '')
  const type = group?.datasourceType
  if (typeof type !== 'string' || !type) return SOURCE_NAMES[id] ?? id
  return SOURCE_NAMES[type] ?? type
}

/** Tab names by group key. Two sources of the same kind (two Discord servers) are told apart by their alias. */
export function tabLabels(groups) {
  const sources = groups.filter(g => String(g.key).startsWith('source:'))
  const kinds = new Map()
  for (const g of sources) kinds.set(sourceName(g), (kinds.get(sourceName(g)) ?? 0) + 1)
  const out = {}
  for (const g of sources) {
    const name = sourceName(g)
    out[g.key] = (kinds.get(name) ?? 0) > 1 ? `${name} · ${g.datasourceId ?? g.key.replace(/^source:/, '')}` : name
  }
  return out
}

function sourceLabel(group) {
  if (!group) return 'search'
  return sourceName(group)
}

/** A chat or mail hit as a quote: where it came from, who wrote it, when, then the passage. */
export function attachText(hit, label) {
  const head = [label, hit.title, hit.author, formatStamp(hit.timestamp)].filter(Boolean).join(' · ')
  const body = (hit.content || hit.snippet || '').replace(/\r/g, '').trim()
  const clipped = body.length > QUOTE_CHARS ? `${body.slice(0, QUOTE_CHARS)}…` : body
  const lines = [head, ...clipped.split('\n')].map(l => `> ${l}`.trimEnd())
  if (hit.url) lines.push(`> ${hit.url}`)
  return lines.join('\n')
}

function errorText(error) {
  if (!error) return 'unknown error'
  const detail = error.detail ?? error.message ?? error
  return typeof detail === 'string' ? detail : String(detail?.message ?? JSON.stringify(detail))
}

/** Electron IPC preserves only the Error message, including the HTTP status and JSON body. */
function notReadyState(error) {
  let detail = error?.detail
  if (!detail) {
    const message = typeof error === 'string' ? error : error?.message
    const body = typeof message === 'string' ? /(?:^|\s)503:\s*(\{[\s\S]*\})$/.exec(message) : null
    if (!body) return null
    try {
      detail = JSON.parse(body[1]).detail
    } catch {
      return null
    }
  }
  const state = detail?.state
  return state === 'installing' || state === 'idle' || state === 'failed' ? state : null
}

// ---- controller ----------------------------------------------------------------------------------

export function createFinder({ rest, request, composer, getSessionId, sleep = ms => new Promise(r => setTimeout(r, ms)), pollMs = POLL_MS, prepareWaitMs = PREPARE_WAIT_MS }) {
  let state = initialState()
  let run = 0
  let jobId = null
  let disposed = false
  const listeners = new Set()

  const emit = () => listeners.forEach(fn => fn(state))
  const set = next => {
    state = typeof next === 'function' ? next(state) : next
    emit()
  }
  const cancelJob = id => {
    if (id) void rest(`/find/${id}`, { method: 'DELETE' }).catch(() => {})
  }

  async function startJob(mine, text) {
    const deadline = Date.now() + prepareWaitMs
    for (;;) {
      try {
        return (await rest('/find', { method: 'POST', body: { query: text } })).id
      } catch (error) {
        if (!notReadyState(error)) throw error
        if (mine !== run || disposed) return null
        set(s => (s.phase === 'preparing' ? s : { ...s, phase: 'preparing' }))
        // Wait for the background install by polling its status, never by holding the search open.
        for (;;) {
          await sleep(Math.max(pollMs, 1))
          if (mine !== run || disposed) return null
          let status
          try {
            status = await rest('/status')
          } catch {
            status = { state: 'installing' }
          }
          if (status.state === 'failed') throw new Error(status.error || 'the AutoRAG finder could not be installed')
          if (status.state === 'ready') break
          if (Date.now() > deadline) throw new Error('The AutoRAG finder is still being prepared (timed out). Try again in a minute.')
        }
        set(s => ({ ...s, phase: 'running' }))
      }
    }
  }

  async function search(query) {
    const text = String(query ?? '').trim()
    if (!text || disposed) return
    const mine = ++run
    cancelJob(jobId)
    jobId = null
    set({ ...initialState(text) })
    let id
    try {
      id = await startJob(mine, text)
    } catch (error) {
      if (mine === run) set(failPending({ ...state, phase: 'running' }, errorText(error)))
      return
    }
    if (id === null) return
    if (mine !== run || disposed) {
      cancelJob(id)
      return
    }
    jobId = id
    void poll(mine, id)
  }

  /** Start the background install when the pane opens, so the first search finds it ready. */
  async function prepare() {
    if (disposed) return
    try {
      await rest('/prepare', { method: 'POST' })
    } catch {
      // The search itself reports an unavailable backend, with its reason.
    }
  }

  async function poll(mine, id) {
    let cursor = 0
    let failures = 0
    while (mine === run && !disposed) {
      let reply
      try {
        reply = await rest(`/find/${id}?after=${cursor}`)
        failures = 0
      } catch (error) {
        if (++failures >= MAX_POLL_ERRORS && mine === run) return set(failPending(state, errorText(error)))
        await sleep(pollMs)
        continue
      }
      if (mine !== run || disposed) return
      let next = state
      for (const line of reply.lines ?? []) next = applyLine(next, line)
      cursor = reply.next ?? cursor + (reply.lines?.length ?? 0)
      if (next !== state) set(next)
      if (reply.state === 'error') return set(failPending(state, reply.error || 'the finder stopped'))
      if (reply.state === 'cancelled') return
      if (reply.state === 'done') return set(s => (s.phase === 'running' ? { ...s, phase: 'done' } : s))
      await sleep(pollMs)
    }
  }

  const hitByKey = key => {
    for (const g of state.groups) {
      const hit = g.hits.find(h => h.key === key)
      if (hit) return { hit, group: g }
    }
    return null
  }

  async function attach(keys) {
    const sessionId = getSessionId()
    if (!sessionId) return set({ ...state, notice: { kind: 'no-session' } })
    const picked = []
    const seenPaths = new Set()
    for (const key of keys) {
      const found = hitByKey(key)
      if (!found) continue
      if (found.hit.path) {
        if (seenPaths.has(found.hit.path)) continue
        seenPaths.add(found.hit.path)
      }
      picked.push(found)
    }
    const capped = picked.length > MAX_ATTACH
    const items = picked.slice(0, MAX_ATTACH)

    const refs = []
    const quotes = []
    let failed = 0
    let reason = ''
    for (const { hit, group } of items) {
      if (hit.path) {
        try {
          const res = await request('file.attach', { session_id: sessionId, path: hit.path, name: hit.title })
          if (res && res.attached !== false && typeof res.ref_text === 'string' && res.ref_text) refs.push(res.ref_text)
          else {
            failed += 1
            reason ||= String(res?.message || 'the chat did not return a file reference')
          }
        } catch (error) {
          failed += 1
          reason ||= errorText(error)
        }
      } else {
        quotes.push(attachText(hit, sourceLabel(group)))
      }
    }
    const count = refs.length + quotes.length
    if (count === 0) return set({ ...state, notice: { kind: 'failed', failed, reason } })
    const text = [refs.join(' '), ...quotes].filter(Boolean).join('\n\n')
    const accepted = await composer.insertText(sessionId, `${text}\n`, { mode: 'block' })
    if (!accepted) return set({ ...state, notice: { kind: 'composer' } })
    set({
      ...state,
      selected: {},
      notice: failed ? { kind: 'partial', count, failed, capped, reason } : { kind: 'attached', count, capped }
    })
  }

  return {
    getState: () => state,
    subscribe(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
    search,
    prepare,
    attach,
    setActiveTab: key => set({ ...state, activeTab: key }),
    toggle(key) {
      const selected = { ...state.selected }
      if (selected[key]) delete selected[key]
      else selected[key] = true
      set({ ...state, selected })
    },
    selectAll(groupKey) {
      const group = state.groups.find(g => g.key === groupKey)
      set({ ...state, selected: Object.fromEntries((group?.hits ?? []).map(h => [h.key, true])) })
    },
    clearSelection: () => set({ ...state, selected: {} }),
    dismissNotice: () => set({ ...state, notice: null }),
    dispose() {
      disposed = true
      run += 1
      cancelJob(jobId)
      jobId = null
      listeners.clear()
    }
  }
}

// ---- copy ----------------------------------------------------------------------------------------

const EN = {
  title: 'AutoRAG search',
  placeholder: 'Search files, documents and chats…',
  hint: 'Enter searches file names, document contents and every connected source. Only relevant results are shown.',
  files: 'Files',
  content: 'Documents',
  searching: 'Searching',
  noResults: 'Nothing relevant found',
  off: 'Document search is off in the AutoRAG config',
  unavailable: 'The document index is not built yet. Run an AutoRAG refresh.',
  attach: 'Add to chat',
  attachN: 'Add {n} to chat',
  selectAll: 'Select all',
  clear: 'Clear',
  reveal: 'Show in Finder',
  open: 'Open link',
  noSession: 'Open a chat first, then add results to it.',
  attached: 'Added {n} to the chat.',
  partial: 'Added {n}; {failed} could not be staged.',
  failed: 'Could not stage the selected files.',
  composer: 'The chat input did not accept the text. Click into the chat and try again.',
  capped: ' Only the first 20 were added.',
  line: 'line',
  error: 'Search failed',
  preparing: 'Getting AutoRAG ready (first run only, about a minute)…'
}
const KO = {
  title: 'AutoRAG 검색',
  placeholder: '파일, 문서, 대화 검색…',
  hint: 'Enter로 파일 이름, 문서 내용, 연결된 모든 소스를 검색합니다. 관련 있는 결과만 보여줍니다.',
  files: '파일',
  content: '문서 내용',
  searching: '검색 중',
  noResults: '관련 결과가 없습니다',
  off: 'AutoRAG 설정에서 문서 검색이 꺼져 있습니다',
  unavailable: '문서 인덱스가 아직 없습니다. AutoRAG refresh를 실행하세요.',
  attach: '대화에 추가',
  attachN: '{n}개 대화에 추가',
  selectAll: '모두 선택',
  clear: '선택 해제',
  reveal: 'Finder에서 보기',
  open: '링크 열기',
  noSession: '먼저 대화를 열고 결과를 추가하세요.',
  attached: '{n}개를 대화에 추가했습니다.',
  partial: '{n}개 추가, {failed}개는 첨부하지 못했습니다.',
  failed: '선택한 파일을 첨부하지 못했습니다.',
  composer: '대화 입력창이 텍스트를 받지 못했습니다. 입력창을 클릭한 뒤 다시 시도하세요.',
  capped: ' 처음 20개만 추가했습니다.',
  line: '줄',
  error: '검색 실패',
  preparing: 'AutoRAG를 준비하는 중입니다 (처음 한 번만, 1분 정도 걸립니다)…'
}
const LOCALES = { en: EN, ko: KO }
const fill = (text, vars) => String(text).replace(/\{(\w+)\}/g, (_, k) => vars?.[k] ?? '')

// ---- style (disk plugins are not scanned by Tailwind; theme variables only) ------------------------

const CSS = `
.autorag-finder{display:flex;flex-direction:column;height:100%;min-height:0;font-size:12px;color:var(--ui-text-primary)}
.autorag-bar{display:flex;gap:6px;align-items:center;padding:8px 10px 4px}
.autorag-input{flex:1;min-width:0;background:transparent;border:0;border-bottom:1px solid var(--ui-stroke-secondary);color:inherit;font:inherit;padding:4px 2px;outline:none}
.autorag-input:focus{border-bottom-color:var(--ui-accent)}
.autorag-hint{padding:2px 12px 6px;color:var(--ui-text-tertiary);font-size:11px}
.autorag-tabs{display:flex;gap:2px;padding:0 8px;overflow-x:auto;border-bottom:1px solid var(--ui-stroke-secondary)}
.autorag-tab{display:flex;gap:5px;align-items:center;padding:5px 8px;border:0;background:transparent;color:var(--ui-text-tertiary);cursor:pointer;font:inherit;white-space:nowrap;border-bottom:2px solid transparent}
.autorag-tab[data-active=true]{color:var(--ui-text-primary);border-bottom-color:var(--ui-accent)}
.autorag-count{color:var(--ui-text-quaternary);font-variant-numeric:tabular-nums}
.autorag-rows{flex:1;min-height:0}
.autorag-row{display:flex;gap:8px;align-items:flex-start;padding:6px 12px;border:0;background:transparent;width:100%;text-align:left;color:inherit;font:inherit;cursor:pointer}
.autorag-row:hover,.autorag-row:focus-visible{background:var(--chrome-action-hover)}
.autorag-row[data-selected=true]{background:color-mix(in srgb,var(--ui-accent) 14%,transparent)}
.autorag-check{margin-top:2px;accent-color:var(--ui-accent)}
.autorag-main{min-width:0;flex:1}
.autorag-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}
.autorag-meta{color:var(--ui-text-tertiary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.autorag-snippet{color:var(--ui-text-secondary);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.autorag-snippet mark{background:color-mix(in srgb,var(--ui-accent) 30%,transparent);color:inherit;border-radius:2px}
.autorag-empty{padding:18px 12px;color:var(--ui-text-tertiary)}
.autorag-err{padding:10px 12px;color:var(--ui-text-secondary);white-space:pre-wrap;word-break:break-word}
.autorag-foot{display:flex;gap:8px;align-items:center;padding:6px 10px;border-top:1px solid var(--ui-stroke-secondary)}
.autorag-notice{flex:1;color:var(--ui-text-secondary)}
`

// ---- view ----------------------------------------------------------------------------------------

function highlight(text, query) {
  const terms = query.split(/\s+/).filter(t => t.length > 1)
  if (!text || terms.length === 0) return text
  const escaped = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) => (i % 2 === 1 ? jsx('mark', { children: part }, i) : part))
}

function tabLabel(group, t, labels) {
  if (group.key === 'files') return t('files')
  if (group.key === 'content') return t('content')
  return labels[group.key] ?? sourceName(group)
}

function Row({ hit, selected, query, onToggle, ctx }) {
  const meta = [hit.author, formatStamp(hit.timestamp), hit.line ? `L${hit.line}` : null, hit.path && hit.kind !== 'file' ? hit.path : null]
    .filter(Boolean)
    .join(' · ')
  const where = hit.kind === 'file' ? hit.path : null
  return jsxs('div', {
    className: 'autorag-row',
    role: 'option',
    'aria-selected': selected,
    'data-selected': selected,
    tabIndex: 0,
    onClick: () => onToggle(hit.key),
    onKeyDown: e => {
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault()
        onToggle(hit.key)
      }
    },
    children: [
      jsx('input', { className: 'autorag-check', type: 'checkbox', checked: selected, readOnly: true, tabIndex: -1, 'aria-label': hit.title }),
      jsxs('div', {
        className: 'autorag-main',
        children: [
          jsx('div', { className: 'autorag-title', children: hit.title }),
          where ? jsx('div', { className: 'autorag-meta', children: where }) : meta ? jsx('div', { className: 'autorag-meta', children: meta }) : null,
          hit.snippet && hit.kind !== 'file' ? jsx('div', { className: 'autorag-snippet', children: highlight(hit.snippet, query) }) : null
        ]
      }),
      hit.path
        ? jsx(Tip, {
            label: 'Reveal',
            children: jsx(Button, {
              variant: 'ghost', size: 'icon-xs', 'aria-label': 'Reveal',
              onClick: e => { e.stopPropagation(); void ctx.os.revealPath(hit.path) },
              children: jsx(icons.FolderOpen, {})
            })
          })
        : hit.url
          ? jsx(Button, {
              variant: 'ghost', size: 'icon-xs', 'aria-label': 'Open',
              onClick: e => { e.stopPropagation(); void ctx.os.openExternal(hit.url) },
              children: jsx(icons.ExternalLink, {})
            })
          : null
    ]
  }, hit.key)
}

function FinderPane({ finder, ctx }) {
  const t0 = usePluginI18n(ID)
  const t = (k, v) => fill(t0(k), v)
  const [, tick] = useState(0)
  const [text, setText] = useState('')
  const inputRef = useRef(null)
  useEffect(() => finder.subscribe(() => tick(n => n + 1)), [finder])
  useEffect(() => {
    inputRef.current?.focus()
    void finder.prepare()
  }, [finder])
  const s = finder.getState()
  const group = s.groups.find(g => g.key === s.activeTab) ?? s.groups[0]
  const picked = Object.keys(s.selected)
  const tabNames = tabLabels(s.groups)

  const body = () => {
    if (s.phase === 'idle') return jsx('div', { className: 'autorag-empty', children: t('hint') })
    if (s.phase === 'preparing') return jsx('div', { className: 'autorag-empty', role: 'status', children: [jsx(GlyphSpinner, { ariaLabel: t('preparing') }, 's'), ` ${t('preparing')}`] })
    if (!group) return s.phase === 'error' ? jsx('div', { className: 'autorag-err', children: s.error }) : jsx('div', { className: 'autorag-empty', children: jsx(GlyphSpinner, { ariaLabel: t('searching') }) })
    if (group.status === 'pending') return jsx('div', { className: 'autorag-empty', children: [jsx(GlyphSpinner, { ariaLabel: t('searching') }, 's'), ` ${t('searching')}…`] })
    if (group.status === 'error') return jsx('div', { className: 'autorag-err', children: group.error || t('error') })
    if (group.status === 'off') return jsx('div', { className: 'autorag-empty', children: t('off') })
    if (group.status === 'unavailable') return jsx('div', { className: 'autorag-empty', children: t('unavailable') })
    if (group.hits.length === 0) return jsx('div', { className: 'autorag-empty', children: t('noResults') })
    return jsx('div', { role: 'listbox', 'aria-multiselectable': true, children: group.hits.map(h => jsx(Row, { hit: h, selected: Boolean(s.selected[h.key]), query: s.query, onToggle: finder.toggle, ctx }, h.key)) })
  }

  const notice = s.notice && ({
    'no-session': t('noSession'),
    attached: t('attached', { n: s.notice.count }) + (s.notice.capped ? t('capped') : ''),
    partial: t('partial', { n: s.notice.count, failed: s.notice.failed }) + (s.notice.capped ? t('capped') : '') + (s.notice.reason ? ` (${s.notice.reason})` : ''),
    failed: t('failed') + (s.notice.reason ? ` (${s.notice.reason})` : ''),
    composer: t('composer')
  })[s.notice.kind]

  return jsxs('div', {
    className: 'autorag-finder',
    children: [
      jsxs('div', {
        className: 'autorag-bar',
        children: [
          jsx(icons.Search, {}),
          jsx('input', {
            ref: inputRef, className: 'autorag-input', value: text, placeholder: t('placeholder'), 'aria-label': t('title'),
            onChange: e => setText(e.target.value),
            onKeyDown: e => {
              if (e.key === 'Enter' && !e.nativeEvent?.isComposing) void finder.search(text)
              if (e.key === 'Escape') setText('')
            }
          }),
          s.phase === 'running' ? jsx(GlyphSpinner, { ariaLabel: t('searching') }) : null
        ]
      }),
      s.groups.length
        ? jsx('div', {
            className: 'autorag-tabs', role: 'tablist',
            children: s.groups.map(g => jsxs('button', {
              type: 'button', role: 'tab', className: 'autorag-tab', 'data-active': g.key === s.activeTab, 'aria-selected': g.key === s.activeTab,
              onClick: () => finder.setActiveTab(g.key),
              children: [
                tabLabel(g, t0, tabNames),
                g.status === 'pending' ? jsx(GlyphSpinner, { ariaLabel: t('searching') }) : jsx('span', { className: 'autorag-count', children: g.status === 'ready' ? g.hits.length : '–' })
              ]
            }, g.key))
          })
        : null,
      jsx(ScrollArea, { className: 'autorag-rows', children: body() }),
      jsxs('div', {
        className: 'autorag-foot',
        children: [
          jsx('span', { className: 'autorag-notice', role: 'status', children: notice || '' }),
          group && group.status === 'ready' && group.hits.length
            ? jsx(Button, { variant: 'ghost', size: 'xs', onClick: () => finder.selectAll(group.key), children: t('selectAll') })
            : null,
          picked.length ? jsx(Button, { variant: 'ghost', size: 'xs', onClick: finder.clearSelection, children: t('clear') }) : null,
          jsx(Button, { size: 'xs', disabled: picked.length === 0, onClick: () => void finder.attach(picked), children: picked.length ? t('attachN', { n: picked.length }) : t('attach') })
        ]
      })
    ]
  })
}

// ---- plugin --------------------------------------------------------------------------------------

export default {
  id: ID,
  name: 'AutoRAG search',
  description: 'Search files, documents and chats with AutoRAG and add the results to the current chat.',
  defaultEnabled: true,
  register(ctx) {
    ctx.i18n.register(LOCALES)
    const style = document.createElement('style')
    style.textContent = CSS
    document.head.append(style)
    ctx.onDispose(() => style.remove())

    const finder = createFinder({
      rest: (path, opts) => ctx.rest(path, opts),
      request: (method, params) => host.request(method, params),
      composer: host.composer,
      getSessionId: () => host.state.focusedSessionId.get()
    })
    ctx.onDispose(() => finder.dispose())
    // Install the librarian in the background now, so the first search does not wait on npm.
    void finder.prepare()

    const open = () => host.revealPane(PANE_ID)
    const title = LOCALES.en.title
    ctx.registerMany([
      // A tab in the right sidebar, stacked with the file tree (its `files` pane): the same place as the folder
      // browser the finder replaces, never a tile taking width from the chat. A zone is as wide as its widest tab
      // and is capped only if EVERY tab declares a cap, so this declares the files pane's own width, min and max
      // (apps/desktop FILE_BROWSER_*): a wider or uncapped tab turned the zone into an uncapped track that took
      // the chat's width.
      {
        id: PANE, area: PANES_AREA, title,
        data: {
          placement: 'right', width: '20rem', minWidth: '10rem', maxWidth: '20rem',
          dock: { pane: 'files', pos: 'center', enforce: true }
        },
        render: () => jsx(FinderPane, { finder, ctx })
      },
      { id: 'open', area: PALETTE_AREA, data: { id: 'autorag.find', label: title, keywords: ['autorag', 'search', 'find', '검색'], run: open } },
      { id: 'key', area: KEYBINDS_AREA, data: { id: 'autorag.find', label: title, category: 'AutoRAG', defaults: ['mod+alt+f'], run: open } },
      {
        id: 'chip', area: STATUSBAR_AREAS.right, order: 60,
        render: () => jsx(Tip, { label: title, children: jsx(Button, { variant: 'ghost', size: 'xs', 'aria-label': title, onClick: open, children: jsx(icons.Search, {}) }) })
      }
    ])
  }
}
