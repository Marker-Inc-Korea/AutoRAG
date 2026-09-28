# AI Finder v6 — Component & Behavior Inventory

> Derived from `design_handoff_ai_finder_v6/AI Finder v6.dc.html` and `README.md`.
> Each section is factual; no design opinions.

---

## 1. Mock-Data Fixtures

All constants live at the module scope of the `<script data-dc-script>` block.

| Name | Type | Shape | Example entries (1–2) |
|------|------|-------|-----------------------|
| `KD` | `Record<Kind, {l:string, bg:string, fg?:string, kind:string}>` | Kind → tile letter, bg oklch, optional white fg, human kind | `pdf:{l:'P',bg:'oklch(0.6 0.17 28)',kind:'PDF Document'}`, `slack:{l:'S',bg:'oklch(0.52 0.16 340)',kind:'Slack Thread'}` |
| `PERM` | `Record<Perm, {label,title,sub,dot,fg,bg}>` | Three access-level descriptors | `ask:{label:'Ask',title:'허용 시 공유',sub:'요청이 오면 Requests에서 확인',dot:'#F59E0B',fg:'#B45309',bg:'rgba(245,158,11,0.14)'}` |
| `CONTACTS0` | `Contact[]` | `{id,cid,name,role,desc}` | `{id:'p1',cid:'jihoon.park@markr.ai',name:'박지훈',role:'CFO · Finance',desc:'재무 총괄...'}` |
| `REQ0` | `Request[]` | `{id,from,time,q,files:[[name,kind,loc]],status,pending?}` | `{id:'r1',from:'p2',time:'10분 전',q:'Q3 퍼포먼스 광고...',files:[['집행내역_9월_중간.csv','csv','Documents/Finance/2026 Q3']],status:'pending'}` |
| `AG` | `(name)=>Answer` | Factory: returns `{quick[],deep[],quickMeta,deepMeta}` | `AG('박지훈')` → Quick reads `[name]의 에이전트에게 물어봤습니다...` |
| `F` | `(name,kind,date,size?) => FileItem` | `{n:string,k:string,d:string,s:string}` | `F('Q3_마케팅예산_v3.xlsx','xlsx','Sep 13, 16:48','84 KB')` |
| `D` | `(name,date) => DirItem` | `{n:string,k:'folder',d:string,s:'—'}` | `D('Desktop','Sep 22, 11:40')` |
| `FS` | `Record<path, (FileItem\|DirItem)[]>` | Flat dict of directory listings; paths are the key | `'Documents/Finance/2026 Q3':[D('벤더 견적','Sep 14, 09:12'),F('Q3_마케팅예산_v3.xlsx','xlsx','Sep 13, 16:48','84 KB'),...]` |
| `FAMS` | `Family[]` | `{head:string, m:[where,name,rel][]}` | `{head:'Documents/Finance/2026 Q3/Q3_마케팅예산_v3.xlsx', m:[['Documents/Finance/2026 Q3','Q3_마케팅예산_v2.xlsx','near'],['Downloads','Q3_마케팅예산_v3 (1).xlsx','exact']]}` |
| `REL` | `Record<rel, string>` | Three labels for stack members | `{exact:'동일본', near:'유사본', contains:'포함본'}` |
| `IDX_FAIL` | `Record<path, reason>` | Files that failed to index | `'Documents/Finance/2026 Q3/집행내역_9월_중간.csv':'인코딩 오류로 CSV를 읽지 못했습니다'` |
| `MEM` | `Record<path, {f,rel}>` | Reverse-lookup built from FAMS | Derived at load: `MEM['Downloads/Blue_Agency_견적서_v2.pdf'] = {f: FAMS[1], rel:'exact'}` |
| `HEAD` | `Record<path, Family>` | Head-lookup built from FAMS | `HEAD[FAMS[0].head] = FAMS[0]` |
| `PLACES` | `SidebarGroup[]` | `{label,items:[name,iconOrKind][]}` | `{label:'Favorites',items:[['Recents',svgPath],['Desktop',svgPath],...]}` |
| `EV` | `Evidence[]` | `{n,k,title,loc,file,meta,md:Block[],pv:Preview}` | 9 items: xlsx budget sheet, mail approval, Slack thread, pdf, notion page, discord, telegram, docx, pptx |
| `A1` | `Answer` | Answer fixture for first chat message | `{quick:['최종 승인액은 **₩3.8억**...'], deep:[...], quickMeta:'0.9s · 2 sources', deepMeta:'14 sources read · 41s'}` |
| `A2` | `Answer` | Answer fixture for user's follow-up send | `{quick:['근거 자료 9건을 찾았습니다...'], deep:[...], quickMeta:'0.8s · 2 sources', deepMeta:'22 sources read · 38s'}` |
| `HIST` | `ChatHistoryItem[]` | `{id,group,time,title,q?,a:Answer?}` | 7 items (오늘×2, 어제×2, 지난 7일×2, 지난 30일×1) |
| `SRC` | `Source[]` | `{id,name,tile,bg,detail,status,pct,sub,fg?,off?}` | 9 sources: local, icloud, gdrive, dropbox, slack, gmail, notion, discord, telegram |
| `SRC_DESC` | `Record<id, string>` | Agent-facing description per source | `local:'이 Mac의 Desktop, Documents, Downloads 폴더...'` |
| `SRC_AVAIL` | `AvailableSource[]` | Sources not yet connected | 6 items: outlook, onedrive, teams, kakao, confluence, github |
| `MODELS` | `Record<id, Model>` | `{name,provider,tile,bg,note}` | 6 models: claude-opus, claude-sonnet, claude-haiku, gpt, gemini, local |
| `B` | `(t,x)=>Block` | Block factory: `{t:string,x:string}` | `B('h','Title')` → `{t:'h',x:'Title'}` |
| `AV` | `string[]` | 4 avatar bg colors | `['oklch(0.55 0.14 20)','oklch(0.55 0.13 150)','oklch(0.55 0.14 260)','oklch(0.58 0.14 60)']` |

**Helper functions (live, not constants):** `parse`, `tileOf`, `flags`, `cfgIntent`.

---

## 2. Component Tree

```
AppShell (window, 14px radius, shadow)
├── Sidebar (212px, bg #F6F6F8)
│   ├── TrafficLights (48px bar)
│   ├── NavGroups (scrollable)
│   │   ├── Favorites (Recents, Desktop, Downloads, Documents)
│   │   ├── Cloud (iCloud Drive, Google Drive, Dropbox)
│   │   ├── Sources (Slack, Gmail, Notion, Discord, Telegram)
│   │   └── People (→ Contacts)
│   └── Footer
│       ├── IndexingProgress (bar + label)
│       ├── RequestsButton (bell, count badge)
│       └── SettingsButton (gear, ⌘, keycap)
│
├── FinderColumn (grid col 2)
│   ├── TabStrip (48px)
│   │   ├── Tab[] (min 84px, max 190px, close ×)
│   │   └── NewTabButton (+)
│   ├── Toolbar (52px)
│   │   ├── BackButton / ForwardButton
│   │   ├── Breadcrumbs
│   │   └── SearchField (236×32, ⌘F keycap)
│   ├── [isFiles] ColumnHeader (30px, 6-column grid sortable)
│   ├── [isFiles] FileList (scrollable)
│   │   ├── Row[] (32px, draggable, selectable, context menu)
│   │   │   ├── NameCell (glyph/tile, name, stack badge)
│   │   │   ├── DateCell, SizeCell, KindCell
│   │   │   ├── IndexBadge (pill, toggleable)
│   │   │   └── AccessBadge (pill, menu on click)
│   │   │       └── AccessMenu (248px popover, 3 options)
│   │   └── EmptyState
│   ├── [isContacts] ContactsView (scrollable card list)
│   │   ├── Header (+ Add contact)
│   │   └── ContactCard[] (drag source → chat)
│   ├── StatusBar (32px)
│   └── EvidencePanel (flex 0 0 44, collapsible)
│       ├── EvidenceTabStrip (40px)
│       ├── [evOpen] EvidenceContent
│       │   ├── EvidenceHeader (tile, title, 👍/👎, Quick Look)
│       │   ├── EvidenceBreadcrumb (click → reveal in Finder)
│       │   └── EvidenceBody (scrollable, MD-like blocks)
│       └── [evOpen] ToggleButton (chevron, ⌘E)
│
├── AiSearchColumn (grid col 3, left border)
│   ├── AiHeader (48px)
│   │   ├── Title ("AI Search")
│   │   ├── HistoryButton (⌘⇧H, popover toggle)
│   │   └── NewChatButton (+)
│   ├── [histOpen] HistoryPopover (absolute, search + list)
│   ├── MessageList (scrollable)
│   │   ├── EmptyChat (centered welcome)
│   │   ├── UserMessage (right-aligned, bubble + attach chips)
│   │   ├── AssistantMessage (two stacked cards: Quick, Deep)
│   │   │   ├── QuickCard (dot, meta, paragraphs with cites, or skeleton, or stopped)
│   │   │   └── DeepCard (red dot, meta, paragraphs with cites, or skeleton/progress, or stopped)
│   │   └── CiteChip (inline [n], click → evidence)
│   ├── [isDragging] DropOverlay (dashed border, blur)
│   └── Composer (input + send/stop button + draft attach chips)
│
├── Overlays (positioned absolutely inside window)
│   ├── QuickLook (modal, 760px, scrim + blur 3)
│   │   ├── sheet: SpreadsheetGrid
│   │   ├── doc: DocumentPaper
│   │   ├── mail: MailPreview
│   │   ├── chat: ChatPreview
│   │   └── generic: GenericPreview (large tile / folder glyph)
│   ├── RequestsPopover (420px, bottom-left anchored)
│   │   ├── Header (segmented tab: pending/done)
│   │   └── RequestCard[] (question + files + actions)
│   ├── ContactEditor (modal, 540px, scrim + blur 8)
│   │   ├── ContactIdField (mono, new: editable, edit: readonly)
│   │   ├── NameField / RoleField (side by side)
│   │   ├── DescTextarea
│   │   └── Footer (Test connection, Delete, Cancel, Save)
│   ├── PermissionConfirm (modal, 460px)
│   │   ├── ChangeRow (from → to pill + count)
│   │   ├── BodyText
│   │   ├── [pcHasMix] MixNote (amber warning)
│   │   └── Footer (Cancel, Keep, Apply)
│   ├── Settings (modal, 760×580, scrim + blur 8)
│   │   ├── GeneralTab
│   │   │   ├── Language select, Global shortcut, toggles
│   │   │   ├── Update section (version, check, auto/beta toggles)
│   │   │   └── Privacy (Clear chat history)
│   │   ├── ModelsTab
│   │   │   ├── CurrentModelCard
│   │   │   └── ConfigChat (message bubbles + suggestion chips)
│   │   └── SourcesTab
│   │       ├── SourceList (rows with status, progress, toggle)
│   │       └── SourceChat (bubbles + suggestion chips)
│   └── Toast (bottom center, auto-hide 2.4s)
│
└── ContextMenu (fixed position at cursor, right-click)
```

---

## 3. Interaction List per Region

### Sidebar
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click nav item | `onClick: () => this.navigate(label)` | `tabs[active].path`, `tabs[active].sel`, `tabs[active].back`, `query` |
| Click Requests | `openInbox` | `inboxOpen` toggled |
| Click Settings | `openSettings` | `settings = true` |

### TabStrip
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click tab | `onClick: () => setState({active: tb.id})` | `active` |
| Click close × | `onClose: e => {e.stopPropagation(); remove tab}` | `tabs`, `active` (reposition if needed) |
| Click + button | `newTab` | `tabs` (append), `active`, `nextId` |

### Toolbar
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click back | `goBack` | `path`, `back`, `fwd` |
| Click fwd | `goFwd` | `path`, `back`, `fwd` |
| Click breadcrumb | `onClick: () => this.navigate(path)` | `path`, `sel`, `back`, `query` |
| Search input change | `onQuery` | `query` |
| Search focus/blur | `onSearchFocus` / `onSearchBlur` | `searchFocus` |
| Search Enter | `onSearchKey: if Enter → openItem(list[0])` | (navigate or open QL) |
| Search Escape | `onSearchKey: if Escape → {query:'', blur}` | `query` |

### ColumnHeader
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click sort col | `onClick: cycle sort` | `sort` (key, dir, flipped; null on 3rd click) |

### FileList Rows
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click | `clickRow` | `sel`, `selWhere`, `selSet`, `anchor`, `zone: 'finder'` |
| ⌘-click | `clickRow` (metaKey branch) | `selSet` toggled |
| Shift-click | `clickRow` (shiftKey branch) | `selSet` range |
| Double-click | `onOpen` | `path` (folders navigate) or `ql: 'finder'` |
| Drag start | `onDragStart` | `dragging` (after 0ms timeout) |
| Drag end | `onDragEnd` | `dragging: null` |
| Right-click | `onContextMenu` | `ctx`, `sel`/`selSet` |
| Click Index badge | `onIdx` (toggle or retry) | `idx`, `idxRetry`, `idxFixed` |
| Click Access badge | `onPerm` | `permMenu` (path or null) |
| Click Access menu option | `reqPerm` | `permConfirm` (folder) or direct `perms` (file) |
| Delete/Backspace key | `trashMany` / `trash` | `trashed`, `sel`, `selWhere`, `selSet` |

### ContextMenu
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click Quick Look/Open | `onClick` | `ql: 'finder'` or `path` (folder) |
| Click retry/toggle-index | `onClick` | `idxRetry`, `idxFixed` or `idx` |
| Click trash | `onClick` | `trashed` |
| Outside mousedown | `onDown` | `ctx: null` |

### ContactsView
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click + Add contact | `addContact` | `ce` initialised |
| Click Edit | `onEdit` | `ce` initialised |
| Drag contact card | `onDragStart` | `dragging` (k='contact') |
| Search filter | (via `onQuery`) | `query` |

### EvidencePanel
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click tab | `onClick` | `ev`, `evOpen: true`, `zone: 'evidence'` |
| Click 👍 | `evUp` | `evFb` (toggle up/null) |
| Click 👎 | `evDown` | `evFb` (toggle down/null) |
| Click Quick Look | `openEvQL` | `ql: 'ev'` |
| Click breadcrumb | `revealEv` | new tab, `flash`, `ql: null` |
| Toggle button | `toggleEv` | `evOpen` toggled |
| Focus evidence content | `focusEvidence` | `zone: 'evidence'` |

### AiSearch (MessageList)
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click citation | `s.onClick` | `ev: n`, `evOpen: true`, `zone: 'evidence'` |
| Click history button | `toggleHist` | `histOpen` toggled |
| Click new chat | `newChat` | `messages: []`, `draft: ''`, `curChat: null` |

### Composer
| Event | Handler | State mutated |
|-------|---------|---------------|
| Input change | `onDraft` | `draft` |
| KeyDown Enter | `onComposerKey` → `send()` | `messages` appended (user + asst skeletons), `draft: ''`, `attach: []` |
| KeyDown Escape (while generating) | `onComposerKey` → `stopGen()` | `messages[last].stopped: true` |
| Click send/stop | `sendOrStop` | `messages` or `stop` |
| DragOver/DragLeave | `onDragOver` / `onDragLeave` | `dropHover` |
| Drop | `onDrop` | `attach` (de-duped by name), `dragging: null` |
| Click attach × | `onRemove` | `attach` filtered |

### HistoryPopover
| Event | Handler | State mutated |
|-------|---------|---------------|
| Input change | `onHistQuery` | `histQuery` |
| KeyDown Escape | `onHistKey` | `histOpen: false` |
| KeyDown Enter | `onHistKey` | `loadChat(first hit)`, `histOpen: false` |
| Click item | `onClick` | `loadChat(c)`, `histOpen: false` |
| Click backdrop | `closeHist` | `histOpen: false` |

### QuickLook
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click backdrop | `closeQL` | `ql: null` |
| Click Ask AI | `qlAsk` | `ql: null`, `attach` appended |
| Click Show in Finder | `revealEv` | new tab, `flash`, `ql: null` |

### RequestsPopover
| Event | Handler | State mutated |
|-------|---------|---------------|
| Click Allow | `resolveReq('allowed')` | `requests[x].status`, toast |
| Click Deny | `resolveReq('denied')` | `requests[x].status`, toast |
| Click Show in Finder | `revealPath` | new tab, `flash` |
| Tab click | `setState({inboxTab})` | `inboxTab` |
| Click backdrop | `closeInbox` | `inboxOpen: false` |

### ContactEditor
| Event | Handler | State mutated |
|-------|---------|---------------|
| Cid change | `ce_cid` | `ce.cid` |
| Name/Role/Desc change | `ceSet` | `ce.name/role/desc` |
| Test connection | `ceTestConn` | `ceTest`, `ceTestMs` |
| Delete | `ceDelete` | `contacts` filtered, `ce: null` |
| Save | `ceSave` | `contacts` updated/added, `ce: null` |
| Cancel / backdrop | `closeCE` | `ce: null` |
| Focus out of cid with dup | (live in renderVals) | `ceCidBorder` red |

### PermissionConfirm
| Event | Handler | State mutated |
|-------|---------|---------------|
| Apply | `pcApply` | `perms` (recursive), `permConfirm: null` |
| Keep | `pcKeep` | `perms` (shallow), `permConfirm: null` |
| Cancel / backdrop | `closePC` | `permConfirm: null` |

### Settings
| Event | Handler | State mutated |
|-------|---------|---------------|
| Tab click | `setState({settingsTab})` | `settingsTab` |
| Language select | `onLang` | `lang` |
| Toggle switches | `toggleLogin/Menu/AutoUp/Beta` | respective bool |
| Check update | `checkUpdate` | `upd: 'checking' → 'done'` |
| Clear history | `clearHistory` | toast |
| Model config send | `cfgSend` / `cfgAsk` | `cfgMsgs`, `model` |
| Source chat send | `srcSend` / `srcAsk` | `srcMsgs`, `srcExtra`, `srcDesc`, `srcDescEdited`, `srcOpen` |
| Source row expand | `onDetail` | `srcOpen` toggled |
| Source toggle | `onToggle` | `srcOff` toggled |
| Source edit desc | `onEditDesc` | `srcDraft` prefilled |
| Close | `closeSettings` / Esc | `settings: false` |

---

## 4. Keyboard Shortcuts

| Key | Scope | Action | Notes |
|-----|-------|--------|-------|
| `Esc` | global | Close topmost: **permConfirm → ce → ctx → inbox/permMenu → hist → ql → settings → clear query** | Exact priority order from `onKey`. Stops generation when composer is focused during gen. |
| `⌘,` | global | Toggle Settings | |
| `⌘K` | global | Focus composer | |
| `⌘F` | global | Focus finder search | |
| `⌘T` | global | New tab | |
| `⌘W` | global | Close current tab | Shown in tab tooltip |
| `⌘E` | global | Toggle Evidence panel | |
| `⌘⇧H` | global | Toggle chat history | |
| `⌘N` | global | New chat | Clears messages |
| `⌘⇧R` / `⌘O` | global | Show current evidence in Finder | Opens new tab, flashes row |
| `⌥1`–`⌥9` | global | Jump to evidence N + open panel | |
| `space` | Finder/Ev zone | Toggle Quick Look | Source: Finder selection when zone=finder, current evidence when zone=evidence |
| `↑`/`↓` | Finder zone | Move row selection | Scrolls into view |
| `↑`/`↓` | Evidence zone | Change evidence number | |
| `←`/`→` | Evidence zone | Change evidence number | |
| `Enter` | Finder zone | Open selected row | Folder→navigate, file→Quick Look |
| `Delete`/`Backspace` | Finder zone, not typing | Move selection to trash | |

**Esc priority order (first match wins):**
1. `permConfirm` → close
2. `ce` → close
3. `ctx` → close
4. `inboxOpen` or `permMenu` → close
5. `histOpen` → close (only if `ce` is also null)
6. `ql` → close
7. `settings` → close (only after ql cleared)
8. `query` → clear

---

## 5. State Model → Component Mapping

Fields from `AppState` (TypeScript sketch in README).

| Field | Type | Written by | Read by |
|-------|------|------------|---------|
| `tabs` | `Tab[]` | navigate, goBack/goFwd, newTab, closeTab, reveal, revealPath | TabStrip, Toolbar (breadcrumbs), FileList, ColumnHeader |
| `active` | `number` | tab click, newTab, closeTab | TabStrip, curRows, renderVals routing |
| `query` | `string` | search field onChange, Esc, breadcrumb nav, reveal | SearchField, FileList filter, StatusBar |
| `sort` | `{key,dir}` or null | ColumnHeader click (3-cycle) | curRows sort |
| `zone` | `'finder'\|'evidence'` | row click, evidence focus, citation click, reveal | Row bg color, arrow/space routing |
| `messages` | `(Msg\|AsstMsg)[]` | send, timers, stopGen, loadChat, newChat | MessageList (chat area) |
| `draft` | `string` | composer onChange, send, sendOrStop, qlAsk | Composer input, send button color |
| `attach` | `{name,k}[]` | onDrop, qlAsk, remove × | Composer draftAttach row, user msg attachments |
| `curChat` | `string\|null` | loadChat, newChat | History filter, evFb key |
| `ev` | `number` | tab click, citation click, ←/→/↑/↓/⌥N, reveal | Evidence tabs, Evidence header, Quick Look |
| `evOpen` | `boolean` | toggleEv, citation click, ⌘E | Evidence panel flex, chevron |
| `evFb` | `Record` | 👍/👎 click | Button bg/fg colors |
| `perms` | `Record<path,Perm>` | reqPerm (folder), setPerm (apply/keep), perm menu option (file) | AccessBadge rendering, permOf, permMix |
| `idx` | `Record<path,bool>` | toggleIdx | IndexBadge dot/ring |
| `idxRetry` | `Record<path,bool>` | retryIdx timer start | IndexBadge spin, label |
| `idxFixed` | `Record<path,bool>` | retryIdx timer end | IDX_FAIL suppression |
| `trashed` | `Record<path,true>` | trash, trashMany | curRows filter |
| `contacts` | `Contact[]` | ceSave (add/edit), ceDelete | ContactsView, cById, ceDup check |
| `requests` | `Request[]` | resolveReq | RequestsPopover, pendingCount |
| `inboxOpen` | `boolean` | openInbox, closeInbox | Requests popover visibility |
| `inboxTab` | `'pending'\|'done'` | tab click | Request filter |
| `ql` | `null\|'finder'\|'ev'` | space, openEvQL, closeQL, qlAsk, reveal | Quick Look visibility, content source |
| `ctx` | `{x,y,p,r}` or null | onContextMenu, close handlers | ContextMenu position/items |
| `permMenu` | `path` or null | onPerm, close handlers | AccessMenu per-row visibility |
| `permConfirm` | `{...}` or null | reqPerm (folder), closePC, pcApply/Keep | PermissionConfirm modal |
| `ce` | `Contact` or null | addContact, onEdit, closeCE, ceSave, ceDelete | ContactEditor modal |
| `settings` | `boolean` | openSettings, closeSettings, Esc | Settings modal |
| `settingsTab` | `string` | tab click | Settings tab body |
| `model` | `string` (ModelId) | cfgAsk (by intent) | Model card, render |
| `srcOff` | `Record<id,bool>` | onToggle | Source opacity, toggle knob |
| `histOpen` | `boolean` | toggleHist, closeHist, Esc | History popover |
| `histQuery` | `string` | onHistQuery | History filter |
| `toast` | `string\|null` | showToast, timer | Toast overlay |
| `dragging` | `{name,k}` or null | onDragStart, onDrop, onDragEnd | DropOverlay visibility/content |
| `dropHover` | `boolean` | onDragOver/Leave/End/Drop | DropOverlay border/bg/scale, composer border |
| `searchFocus` | `boolean` | onSearchFocus/Blur | Search border color |
| `srcOpen` | `string` or null | onDetail | Source row expanded/collapsed |
| `srcDesc` | `Record<id,string>` | srcAsk (edit) | Source detail card |
| `srcDraft` | `string` | onSrcDraft, onEditDesc | Source chat input |
| `srcMsgs` | `Msg[]` | srcAsk | Source chat bubbles |
| `cfgDraft` | `string` | onCfgDraft | Model config chat input |
| `cfgMsgs` | `Msg[]` | cfgAsk | Model config chat bubbles |
| `deepStep` | `number` | stepTimer interval | Deep progress step text |
| `flash` | `number` | reveal, revealPath; cleared by timer | Tab flash dot, row bg/ring |
| `lang` | `string` | onLang | Language select value |
| `login`/`menu`/`autoUp`/`beta` | `boolean` | toggle handlers | Settings toggle knobs |
| `upd` | `'idle'\|'checking'\|'done'` | checkUpdate timer | Settings update status |