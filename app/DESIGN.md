# AI Finder v6 Design System

Implementation contract for the **AutoRAG Agent — AI Finder** desktop app (Electron + React + TypeScript, hand-written CSS, no Tailwind).

## Reference contract

This is a **concrete-reference implementation**. The design is final and is recreated pixel-accurately; nothing here is reinterpreted.

| | |
|---|---|
| Spec | `docs/design-reference/ai-finder/design_handoff_ai_finder_v6/README.md` |
| Prototype | `docs/design-reference/ai-finder/design_handoff_ai_finder_v6/AI Finder v6.dc.html` (cited below as `proto:<line>`) |
| Machine copy | `app/src/renderer/styles/tokens.css` — `:root` custom properties, names and values 1:1 with §2–§7 |

**Token source policy.** Every value traces to the handoff README's token tables first. Where the README documents a component but omits a literal, the value is taken from the prototype and carries a `proto:<line>` citation in the table. No third source exists: a value that is in neither is not a token and must not be invented.

**Binding rules.**

1. No hard-coded color, radius, shadow, duration, or font size in component CSS. Reference a token from `tokens.css`.
2. Pixel sizes that belong to a single component (a 236px search field, a 44% evidence panel) live in §4/§5 and are tokenized when they repeat. Sizes that appear once are still written from the token names in §4 where one exists.
3. A value the reference does not contain requires a reference change first, then a token, then usage. Never a one-off override.
4. `tokens.css` and this document change in the same commit, always.
5. UI copy is Korean with some English labels. Every string is reproduced **exactly** as written in this document or the prototype — no re-translation, no rewording, no punctuation drift (`·`, `›`, `…`, `⌘`, `⇧`, `⌥`, `↩`, `⌫`).

---

## 1. Atmosphere & Identity

A **macOS-native, paper-quiet workspace**: a soft gray desk, one floating white window, near-black ink, and exactly one warm accent. The product is a Finder that thinks, so the surface reads as a system tool, not as a chat app — restraint is the identity.

- **Material:** flat white surfaces separated by 1px hairlines, not by shadow. Only four things float: the window, popovers/menus, modals, and the composer/toast. Everything else is flush.
- **Ink:** a single near-black (`#1A1A2E`) stepped down through body → muted → faint. Gray is information hierarchy, never decoration.
- **Accent:** coral `#FF6363`, used only for primary action, selection, focus, and the active citation. **Text on accent is `#1A1A2E`, not white** — the accent is a highlighter, not a button fill with white type.
- **Color as data:** status color (green/amber/blue/red/orange) is reserved for index and permission state. It never decorates.
- **Motion:** one 6px rise on enter, 150–200ms color fades. Nothing bounces, nothing parallaxes.
- **Density:** 32px rows, 22px pills, 11px badges. Finder density, not marketing-page air.

**Anti-patterns for this product.** No colored accent borders to mark selection (the reference marks selection with an accent *wash* + zone-dependent tint). No emoji as icons — inline SVG strokes only (Lucide-style, `24 × 24` viewBox, stroke-width 1.8–2.6, round caps). No gradients except the two the reference defines (the evidence marker and the skeleton shimmer). No white-on-accent text.

---

## 2. Color

All values are literal from the reference. Token names below are the exact custom-property names in `tokens.css`.

### 2.1 Surfaces

| Token | Value | Use |
|---|---|---|
| `--bg-desk` | `#E9E9EE` | area behind the window |
| `--bg-window` | `#FFFFFF` | main surfaces |
| `--bg-chrome` | `#F6F6F8` | sidebar, tab strips, panel headers, Quick Look body |
| `--bg-subtle` | `#FAFAFB` | evidence panel |
| `--bg-subtle-alt` | `#FAFAFC` | answer cards, draft attachment chips |
| `--bg-input` | `#F0F0F4` | search field, composer, user bubble, segmented-control track, progress track, toast |
| `--bg-selected-nav` | `#EAEAF0` | active sidebar item, toggled icon buttons |
| `--bg-code` | `#F4F4F7` | evidence `code` block |
| `--bg-chip-sent` | `#F3F3F7` | attachment chip under a sent user message (`proto:281`) |
| `--bg-tile-fallback` | `#8A8A9C` | md / zip tiles and the unknown-kind fallback |

### 2.2 Hover washes

One ink at four alphas. Never a border color change to signal hover on a row.

| Token | Value | Use |
|---|---|---|
| `--bg-hover` | `rgba(26,26,46,0.05)` | default hover |
| `--bg-hover-row` | `rgba(26,26,46,0.04)` | list rows, keycap background |
| `--bg-hover-icon` | `rgba(26,26,46,0.06)` | icon buttons |
| `--bg-hover-strong` | `rgba(26,26,46,0.08)` | tab close button (`proto:81`) |

### 2.3 Borders

| Token | Value | Use |
|---|---|---|
| `--border` | `#E3E3EA` | all 1px borders, scrollbar thumb, quote left rule |
| `--border-divider` | `#EDEDF2` | list row separators |
| `--border-strong` | `#C9C9D4` | hover borders, disabled icons, tree connectors, breadcrumb separator, disabled send button, drop-overlay idle ring |

### 2.4 Text

| Token | Value | Use |
|---|---|---|
| `--text-primary` | `#1A1A2E` | titles, active row text, bold runs |
| `--text-body` | `#3A3A4E` | body, answer text, stack-child names |
| `--text-muted` | `#6E6E80` | meta text, labels, placeholders, inactive tabs |
| `--text-faint` | `#A0A0B0` | separators (`›`), lock icons, excluded-index hollow ring |
| `--text-on-accent` | `#1A1A2E` | text/icons on `--accent` — never white |
| `--text-on-tile` | `#FFFFFF` | file-tile letter (`proto:888`) |

### 2.5 Accent

| Token | Value | Use |
|---|---|---|
| `--accent` | `#FF6363` | primary buttons, selection, focus border, active citation, active sidebar icon, tab flash dot |
| `--accent-hover` | `#FF7A7A` | primary button hover |
| `--accent-text` | `#D64545` | accent-colored text, citation chip fg, destructive menu item, Requests count badge bg |
| `--accent-text-strong` | `#C73A3A` | strongest accent text |
| `--accent-drop-hover` | `rgba(255,99,99,0.07)` | drop overlay, hover state |
| `--accent-history-current` | `rgba(255,99,99,0.08)` | current chat row in History |
| `--accent-soft` | `rgba(255,99,99,0.12)` | selected row while the Finder zone is focused |
| `--accent-chat-mark` | `rgba(255,99,99,0.14)` | highlighted message in a `chat` preview |
| `--accent-chip` | `rgba(255,99,99,0.16)` | inactive citation chip background |
| `--accent-flash` | `rgba(255,99,99,0.18)` | revealed row background, 1.8s |
| `--accent-preview-mark` | `rgba(255,99,99,0.2)` | highlighted `sheet` rows, `mail` marks |
| `--accent-doc-mark` | `rgba(255,99,99,0.22)` | `doc` preview mark blocks |
| `--accent-pending-border` | `rgba(255,99,99,0.35)` | Deep card border while pending |
| `--accent-flash-ring` | `rgba(255,99,99,0.6)` | revealed-row inset ring |

### 2.6 Object colors

| Token | Value | Use |
|---|---|---|
| `--folder` | `#7C86E8` | folder glyph fill |
| `--switch-off` | `#CFCFD9` | toggle track when off; empty-state send button |
| `--highlight-marker` | `linear-gradient(transparent 55%, rgba(255,214,10,0.55) 55%)` | highlighted passage in Evidence (with `box-decoration-break: clone`, `padding: 0 1px`) |
| `--traffic-close` | `#FF5F57` | traffic light |
| `--traffic-min` | `#FEBC2E` | traffic light |
| `--traffic-max` | `#28C840` | traffic light |

### 2.7 Status

Each status is a triad: dot / text / tinted background.

| Token | Value | Use |
|---|---|---|
| `--ok-dot` | `#22C55E` | Indexed, success, toast dot |
| `--ok-text` | `#15803D` | Indexed label, 👍 active fg |
| `--ok-bg` | `rgba(34,197,94,0.14)` | Indexed badge |
| `--ok-bg-soft` | `rgba(34,197,94,0.12)` | 👍 active background |
| `--ok-border` | `rgba(34,197,94,0.45)` | 👍 active border |
| `--warn-dot` | `#F59E0B` | Syncing, Ask permission, sidebar syncing dot, indexing bar fill |
| `--warn-text` | `#B45309` | Syncing / Ask label |
| `--warn-bg` | `rgba(245,158,11,0.14)` | Ask badge |
| `--warn-bg-strong` | `rgba(245,158,11,0.16)` | Permission Confirm amber note |
| `--info-dot` | `#3B82F6` | Allow permission |
| `--info-text` | `#1D4ED8` | Allow label |
| `--info-bg` | `rgba(59,130,246,0.12)` | Allow badge |
| `--danger-dot` | `#EF4444` | Deny permission, destructive, invalid input border |
| `--danger-text` | `#C62828` | Deny label, 👎 active fg |
| `--danger-bg` | `rgba(239,68,68,0.12)` | Deny badge |
| `--danger-bg-soft` | `rgba(239,68,68,0.10)` | 👎 active background |
| `--danger-border` | `rgba(239,68,68,0.4)` | 👎 active border |
| `--error-orange-dot` | `#FF8C00` | index failure, source error, sidebar error dot |
| `--error-orange-text` | `#C2410C` | retry badge fg, error label |
| `--error-orange-bg` | `rgba(255,140,0,0.12)` | retry badge bg |
| `--error-orange-bg-strong` | `rgba(255,140,0,0.16)` | source error badge bg |
| `--index-dot-included` | `oklch(0.62 0.14 150)` | green dot on the "포함" badge |
| `--index-dot-excluded-ring` | `inset 0 0 0 1.5px #A0A0B0` | hollow dot on the "제외" badge |

### 2.8 Scrims

| Token | Value | Use |
|---|---|---|
| `--scrim-quicklook` | `rgba(20,20,40,0.16)` | Quick Look scrim (blur 3px) |
| `--scrim-modal` | `rgba(20,20,40,0.22)` | Contact Editor / Permission Confirm / Settings scrim (blur 8px) |
| `--drop-overlay-bg` | `rgba(250,250,252,0.86)` | drop overlay, idle (blur 4px) |

### 2.9 File-type tiles

Small rounded squares with a white letter, 10px/700. The letter and background come straight from the prototype's `KD` map (`proto:684–700`).

| Type | Letter | Token | Background |
|---|---|---|---|
| xlsx | X | `--tile-xlsx` | `oklch(0.62 0.14 150)` |
| csv | C | `--tile-csv` | `oklch(0.58 0.1 175)` |
| pdf | P | `--tile-pdf` | `oklch(0.6 0.17 28)` |
| docx | W | `--tile-docx` | `oklch(0.58 0.15 258)` |
| pptx | P | `--tile-pptx` | `oklch(0.66 0.15 55)` |
| md | M | `--tile-md` | `#8A8A9C` |
| zip | Z | `--tile-zip` | `#8A8A9C` |
| png | I | `--tile-png` | `oklch(0.58 0.13 305)` |
| slack | S | `--tile-slack` | `oklch(0.52 0.16 340)` |
| mail | M | `--tile-mail` | `oklch(0.58 0.15 12)` |
| notion | N | `--tile-notion` | `#1A1A2E` (fg `--tile-notion-fg` `#FFFFFF`) |
| discord | D | `--tile-discord` | `oklch(0.56 0.17 278)` |
| telegram | T | `--tile-telegram` | `oklch(0.64 0.13 232)` |
| contact | @ | `--tile-contact` | `oklch(0.52 0.13 285)` |

`folder` has no tile — it renders the 18px folder glyph in `--folder`. Unknown kinds fall back to the `md` entry, i.e. `--bg-tile-fallback`. Tile foreground is `--text-on-tile` unless the map overrides it.

### 2.10 Rules

- The accent is **never** a background for white text. `--text-on-accent` is `#1A1A2E`.
- Selection is a wash, not an edge: `--accent-soft` when the Finder zone is focused, `--bg-input` when another zone is. No `border-left` accent stripe anywhere.
- Status color pairs a 6px dot with its text color. A status label without its dot is not a valid state.
- Inherited (non-explicit) permission badges are colorless: transparent bg, `--border`, `--text-muted`. Color means "set here".
- The only permitted colored edges are: the focused search field / input (`--accent`), the pending Deep card (`--accent-pending-border`), the drop-overlay hover ring (`--accent`), the duplicate-ID input (`--danger-dot`), and the feedback toggles.

---

## 3. Typography

### Font stack

| Token | Value |
|---|---|
| `--font-sans` | `Inter, Pretendard, -apple-system, system-ui, sans-serif` |
| `--font-mono` | `ui-monospace, SFMono-Regular, Menlo, monospace` |

Inter covers Latin, Pretendard covers Korean; the stack order makes that automatic. Mono is for IDs and code only (Contact ID field, evidence `code` blocks). `-webkit-font-smoothing: antialiased` on `body`.

### Scale

Size tokens are namespaced `--font-size-*` so they never collide with the `--text-*` **color** tokens in §2.4.

| Token | Size | Weights | Use |
|---|---|---|---|
| `--font-size-title` | `18px` | 600 | modal titles, empty-chat title, `doc` preview title (700) |
| `--font-size-body` | `14px` | 400–600 | body, list rows, buttons in panels, inputs, bubbles |
| `--font-size-sm` | `13px` | 400–500 | context menu, source descriptions |
| `--font-size-meta` | `12px` | 400–600 | meta, small buttons, column headers, evidence body |
| `--font-size-badge` | `11px` | 600–700 | badges, keycaps, group labels, citation chips, evidence tile letter |
| `--font-size-tile` | `10px` | 700 | file-tile letters |

Weight tokens: `--weight-regular` `400`, `--weight-medium` `500`, `--weight-semibold` `600`, `--weight-bold` `700`.

Line-height tokens: `--leading-body` `1.5` (body, bubbles, answer text), `--leading-read` `1.6` (evidence blocks and document text).

Letter-spacing: `--tracking-label` `0.02em`, on 11px/600 sidebar group labels only.

### Rules

- Paragraph-like text carries `text-wrap: pretty` (answer paragraphs, evidence `p`/`mark`, bubbles, config messages).
- 9px is used once, for the 14px attachment-chip tile letter (`proto:281`) — `--font-size-tile-xs` `9px`. Do not introduce it elsewhere.
- Korean bubbles use `word-break: keep-all` (`proto:612`).
- Names and paths ellipsize (`white-space: nowrap; overflow: hidden; text-overflow: ellipsis`); they never wrap.
- `**bold**` in answer text renders as `--text-primary` at `--weight-semibold`, not as a heavier size.

---

## 4. Spacing & Layout

### Base unit

The reference is pixel-exact and uses 1/2/3/5px steps in real components (a 3px segmented-control inset, a 1px sidebar group gap). A synthetic 4px or 8px grid would break fidelity, so the scale is **literal-valued, 2px-rhythm with a 1px minimum**, and the allowed steps are closed:

| Token | Value | Token | Value |
|---|---|---|---|
| `--space-1` | `1px` | `--space-12` | `12px` |
| `--space-2` | `2px` | `--space-14` | `14px` |
| `--space-3` | `3px` | `--space-16` | `16px` |
| `--space-4` | `4px` | `--space-18` | `18px` |
| `--space-5` | `5px` | `--space-20` | `20px` |
| `--space-6` | `6px` | `--space-22` | `22px` |
| `--space-8` | `8px` | `--space-24` | `24px` |
| `--space-10` | `10px` | `--space-40` | `40px` |

Value-named because the contract is pixel fidelity: `--space-14` cannot be mistranslated the way `--space-md` can.

### Control sizes

| Token | Value | Use |
|---|---|---|
| `--size-row` | `32px` | sidebar item, file row, primary/ghost buttons, search field, icon buttons, send button, select |
| `--size-pill` | `22px` | index badge, access badge, toggle track height |
| `--size-icon-sm` | `28px` | small icon buttons, context-menu items, breadcrumb crumb, expand chevron |
| `--size-chip` | `24px` | sent attachment chip, evidence breadcrumb chip, segmented item |
| `--size-chip-draft` | `26px` | composer draft attachment chip |
| `--size-tab` | `36px` | finder tab height, toast height, toggle track width |
| `--size-bar` | `48px` | sidebar top bar, tab strip, AI Search header |
| `--size-toolbar` | `52px` | Finder toolbar, Quick Look header |
| `--size-col-header` | `30px` | file list column header |
| `--size-status-bar` | `32px` | Finder status bar |
| `--size-ev-strip` | `40px` | evidence tab strip (also the collapsed panel height) |
| `--size-settings-row` | `56px` | Settings list rows (min-height) |
| `--size-dot` | `6px` | every StatusDot, the tab flash dot, the sidebar sync/error dot |
| `--size-traffic` | `12px` | traffic-light circle |
| `--size-knob` | `16px` | toggle knob diameter |
| `--knob-on` | `17px` | toggle knob `left` when on (`proto:1164`) |
| `--knob-off` | `3px` | toggle knob `left` when off (`proto:1164`) |

### App shell

| Token | Value |
|---|---|
| `--desk-padding` | `18px` |
| `--app-min-width` | `1360px` |
| `--app-min-height` | `820px` |
| `--grid-main` | `212px minmax(640px, 1.4fr) minmax(440px, 1fr)` |
| `--col-sidebar` | `212px` |
| `--evidence-open-height` | `44%` |

- The desk is `--bg-desk` with `--desk-padding` on all sides. The window inside is `--bg-window`, 1px `--border`, `--radius-14`, `--shadow-window`, `overflow: hidden`, `position: relative`.
- The prototype scales the app with `zoom = min(1, vw/1360, vh/820)` (`proto:1189`). **The Electron app does not zoom** — it sets `minWidth: 1360`, `minHeight: 820` on the BrowserWindow instead.
- Main grid columns: sidebar `--col-sidebar`, Finder `minmax(640px, 1.4fr)`, AI Search `minmax(440px, 1fr)` with a 1px left border.
- All overlays (Quick Look, modals, toast, Requests, History, drop overlay) are absolutely positioned inside the window.

### Component grids

| Token | Value | Use |
|---|---|---|
| `--grid-file-row` | `minmax(0,1fr) 100px 60px 96px 66px 76px` | column header and every file row; gap `--space-12`, padding `0 16px 0 20px` |
| `--grid-source-row` | `28px 1fr 150px 28px 36px` | Settings → Data Sources row |
| `--grid-sheet` | `24px 1.1fr 1fr 1fr 1fr` | Quick Look `sheet` preview |

Columns, in order: **Name | Date Modified | Size (right-aligned) | Kind (or "Where" while searching) | Index | Access**.

### Overlay widths

| Token | Value | Token | Value |
|---|---|---|---|
| `--w-search-field` | `236px` | `--w-quicklook` | `760px` |
| `--w-access-menu` | `248px` | `--w-contact-editor` | `540px` |
| `--w-context-menu` | `220px` | `--w-permission-confirm` | `460px` |
| `--w-history` | `360px` | `--w-settings` | `760px` |
| `--w-requests` | `420px` | `--h-settings` | `580px` |
| `--w-tab-min` | `84px` | `--w-tab-max` | `190px` |
| `--w-crumb-max` | `220px` | `--w-bubble-max` | `78%` |

### Rules

- Every padding, gap, and margin resolves to a `--space-*` token. A value outside the closed scale is a reference violation, not a new step.
- Sidebar scroll area: padding `4px 10px 12px`, `14px` gap between groups, `1px` gap inside a group. Group label padding `4px 8px 6px`.
- File list padding `4px 8px`; rows `32px` tall, padding `0 8px 0 12px`.
- Message list padding `20px 20px 8px`, gap `20px`. Composer padding `8px 20px 14px`; composer box padding `10px 10px 10px 14px`, gap `8px`.
- Evidence header padding `14px 16px 10px 20px`; body padding `4px 20px 20px 52px`, gap `8px`.
- The layout owns its scroll: the sidebar group area, the file list, the evidence body, the message list, and each popover body are the scroll containers. The window itself never scrolls (`overflow: hidden`).

---

## 5. Components

Primitives first. Every composed surface (sidebar, tab strip, toolbar, file row, evidence panel, answer stream, Quick Look, Requests, modals, Settings) is assembled from these and adds no new tokens.

### Keycap

Hint glyph for a shortcut. Rendered **only** when the `showHotkeys` prop is true.

- 11px `--text-muted`, padding `1px 5px`, 1px `--border`, `--radius-5`, bg `--bg-hover-row` (`proto:69`).
- Content is the literal glyph string: `⌘,` `⌘F` `⌘T` `⌘E` `⌘N` `⌘⇧H` `⌘⇧R` `space` `↩` `⌫` `⌥` `Space`.
- **States:** static. No hover, no focus, never interactive — it labels a shortcut, it does not invoke one.
- Placement: trailing inside a button (Settings, search field, Quick Look header), or standalone in the status bar.

### Toggle

36 × 22 switch (`--size-tab` × `--size-pill`).

- Track: `--radius-11`, bg `--accent` when on, `--switch-off` when off.
- Knob: 16px white circle, `left: 17px` (on) / `left: 3px` (off), transition `left var(--dur-knob)` (`proto:1164`).
- **States:** on / off / disabled-by-parent (the whole row drops to `opacity: 0.6` when a source is disabled — the toggle itself keeps full opacity semantics).
- Used by: Launch at login, menu bar, auto-update, beta updates, telemetry, per-source enable.

### PillBadge

22px pill, `--radius-11`, 11px/600, padding `0 8px`, gap `5px`, `white-space: nowrap`, hover `border-color: --border-strong`. Two variants.

**Index variant** (files only):

| State | Background | Border | Text | Dot | Label |
|---|---|---|---|---|---|
| Included | `transparent` | `--border` | `--text-body` | 6px `--index-dot-included` | `포함` |
| Excluded | `--bg-input` | `transparent` | `--text-muted` | hollow: `--index-dot-excluded-ring` | `제외` |
| Failed | `--error-orange-bg` | `transparent` | `--error-orange-text` | retry icon (10px) | `재시도` |
| Retrying | `--error-orange-bg` | `transparent` | `--error-orange-text` | icon rotating 360° over `--dur-spin` | `재시도 중` |

Tooltips: failed → `인덱싱 실패 · <reason> — 클릭하면 다시 시도`. Toggling shows `<name> — 인덱싱에서 제외했습니다` / `… 포함했습니다`; a completed retry shows `<name> — 인덱싱을 완료했습니다` after 1500ms. Stack members default to excluded; everything else defaults to included.

**Access variant** (files and folders): 6px dot + label.

| State | Background | Border | Text | Dot |
|---|---|---|---|---|
| Explicit Ask | `--warn-bg` | `transparent` | `--warn-text` | `--warn-dot` |
| Explicit Allow | `--info-bg` | `transparent` | `--info-text` | `--info-dot` |
| Explicit Deny | `--danger-bg` | `transparent` | `--danger-text` | `--danger-dot` |
| Inherited | `transparent` | `--border` | `--text-muted` | resolved permission dot |
| Mixed (folders) | `--bg-input` | `transparent` | `--text-muted` | conic-gradient of the distinct permission dots |

Inherited tooltips append ` · 상위 폴더 설정`. Labels are `Ask` / `Allow` / `Deny` / `Mixed`. Clicking opens the Access menu (see ContextMenu → Access variant).

### FileTile

Rounded square with a centered letter, `--weight-bold`, color `--text-on-tile` unless the `KD` entry overrides it. Background from §2.9.

| Size | Radius | Font | Where |
|---|---|---|---|
| 14px | `--radius-4` | `--font-size-tile-xs` | attachment chips (sent + draft) |
| 16px | `--radius-4` | `--font-size-tile` | sidebar Sources items |
| 18px | `--radius-5` | `--font-size-tile` | file rows, drag image |
| 20px | `--radius-5` | `--font-size-tile` | Quick Look header |
| 22px | `--radius-6` | `--font-size-badge` | evidence header, Requests file list |
| 36px | `--radius-8` | — | Settings model card |
| 64px | `--radius-14` | 20px | drop-overlay stacked tile (back layer rotated `-8deg`, `--shadow-drag`) |
| 88 × 108 | `--radius-12` | 32px | Quick Look `generic` preview, `--shadow-raised` |

The `contact` kind renders as a **circle** (`--radius-full`) at drag-image size (`proto:995`). Folders render the folder glyph in `--folder` instead of a tile — 18px in rows, 96px in the `generic` preview.

- **States:** static. The tile never has hover or focus of its own; its row owns interaction.

### TrafficLights

Three 12px circles, 8px gap, 16px left padding, in the 48px sidebar top bar: `--traffic-close`, `--traffic-min`, `--traffic-max`, left to right.

- **States:** decorative in this build (the prototype renders them inert). If wired to real window controls later, they gain hover glyphs — that is a reference change, not a local decision.

### AnswerCard

The Quick/Deep answer pair. Card: padding `14px 16px`, `--radius-10`, bg `--bg-subtle-alt`, 1px `--border`.

- **Header:** 6px dot + name (600) + meta. Quick uses a `--text-body`-colored dot and meta like `0.9s · 2 sources`; Deep uses an `--accent` dot.
- **Pending:** shimmer skeleton bars, height 10, `--radius-5`, gradient `--shimmer-gradient`, `background-size: 200% 100%`, `shimmer var(--dur-shimmer) linear infinite`. Quick widths `90% / 70%`; Deep widths `96% / 100% / 82% / 94% / 58%`.
- **Deep pending:** border `--accent-pending-border`; meta cycles `Searching 9 sources…` → `Reading 22 documents…` → `Cross-checking figures…` → `Writing answer…` at `latency/4` intervals.
- **Ready:** answer text 14px/`--leading-body` `--text-body`; `**bold**` → `--text-primary`/600; `[n]` → CitationChip.
- **Stopped:** body `응답 생성을 중단했습니다.` in `--text-muted`, meta `Stopped`.
- Enter animation: `rise var(--dur-modal) var(--ease-out)`.

### CitationChip

Inline `[n]` marker inside answer text.

- 18px tall, `min-width: 18px`, padding `0 4px`, margin `0 2px`, `--radius-5`, 11px/600, `vertical-align: 1px`, `transition: background var(--dur-color-slow)`.
- **Inactive:** bg `--accent-chip`, fg `--accent-text`.
- **Active** (matches the selected evidence): bg `--accent`, fg `--text-on-accent`.
- **Click:** selects that evidence, opens the Evidence panel, sets the focus zone to `evidence`.

### ContextMenu

One floating list primitive, three configurations. Shared: bg `--bg-window`, 1px `--border`, `--shadow-window`, `rise` on enter, closes on outside mousedown or Esc.

| Variant | Width | Radius | Padding | Item height | Enter duration |
|---|---|---|---|---|---|
| Row context menu | `--w-context-menu` | `--radius-10` | `--space-5` | `--size-icon-sm` | `--dur-menu` |
| Access menu | `--w-access-menu` | `--radius-10` | `--space-6` | auto (dot + title + subtitle) | `--dur-popover` |
| History / Requests popover | `--w-history` / `--w-requests` | `--radius-12` | `0` | — | `--dur-popover` |

**Row context menu** items, in order: `Quick Look` + `space` keycap (folders: `열기` + `↩`), separator, `인덱싱에서 제외` / `인덱싱에 포함` (or `인덱싱 다시 시도` in `--error-orange-text` when the file failed), separator, `휴지통으로 이동` + `⌫` keycap in `--accent-text` (multi-selection: `N개 항목 휴지통으로 이동`). Position `fixed` at the cursor.

**Access menu** header `에이전트 요청 시 공유`; three options, each a dot + title + subtitle with an `--accent` ✓ on the current one:

- `허용 시 공유` / `요청이 오면 Requests에서 확인` (Ask)
- `항상 허용` / `에이전트 요청에 자동으로 공유` (Allow)
- `항상 거부` / `요청을 자동으로 거절` (Deny)

Footer note: folders → `하위 폴더와 파일에 모두 적용됩니다`; files → `이 파일에만 적용됨`, `<folder> 폴더 설정을 따르는 중`, or `기본값`; mixed folders → `하위 항목 N개가 다르게 설정되어 있습니다…`. The menu opens **upward** when the row is among the last 4 and the list has more than 6 rows.

- **States:** item hover `--bg-hover-row`; destructive items keep `--accent-text` at every state; the checked access option shows the accent ✓.

### Toast

Bottom center, `bottom: 24px`, `transform: translateX(-50%)`, height `--size-tab`, padding `0 14px`, gap `10px`, `--radius-10`, bg `--bg-input`, 1px `--border`, `--shadow-raised`, 12px `--text-primary`, leading 6px `--ok-dot`, `rise var(--dur-modal) var(--ease-out)`.

- **States:** visible / hidden. Auto-hides after `--time-toast`; a new toast replaces the current one (never stacks).

### SegmentedControl

Track: bg `--bg-input`, padding `--space-3`, gap `--space-2`, `--radius-8`.

| Variant | Item height | Item radius | Font |
|---|---|---|---|
| Requests (`대기 중 N` / `처리됨`) | `--size-chip` | `--radius-6` | 12px/500 |
| Settings (`General` / `Models` / `Data Sources`) | `--size-icon-sm` | `--radius-7` | 12px/500 |

- **Active:** bg `--bg-window`, fg `--text-primary`. **Inactive:** `transparent`, fg `--text-muted`.
- Also used for the Light/Dark/System theme choice in Settings → General.

### StatusDot

6px circle, `--radius-full`, `flex: none`. The single carrier of state color.

| State | Color | Where |
|---|---|---|
| Indexed / success | `--ok-dot` | index badge, toast, Data Sources |
| Syncing / Ask | `--warn-dot` | sidebar (Gmail, Google Drive), access badge, Data Sources |
| Allow | `--info-dot` | access badge |
| Deny / destructive | `--danger-dot` | access badge |
| Error | `--error-orange-dot` | sidebar (Discord), Data Sources |
| Quick | `--text-body` | Quick answer card |
| Deep / current chat / revealed tab | `--accent` | Deep card, History current row, tab flash dot |
| Mixed | conic-gradient of the distinct permission dots | folder access badge |

Sidebar and tab-flash dots are also 6px; the Requests count badge is a different primitive (min `18 × 18`, `--radius-9`, bg `--accent-text`, `#FFFFFF` 11/700).

- **States:** static; the dot never animates. The tab flash dot is shown for `--time-flash` and then removed.

### Composed surfaces (assembled from the primitives above)

Sidebar (`--bg-chrome`, right border) · Tab strip (48px, bottom-aligned 36px tabs, active tab white with `margin-bottom: -1px`) · Toolbar (52px, back/forward, breadcrumbs, search field) · Column header + file rows on `--grid-file-row` · Version stack (18px pill, bg `--bg-input`, `--border` when open, chevron rotates 90°, tooltip `다른 버전 N개 · 클릭하면 펼쳐집니다`; children indented 18px with an L connector in 1px `--border-strong`, 4px corner radius) · Status bar · Evidence panel (`--bg-subtle`, `flex: 0 0 44%`, 40 × 30 tabs) · AI Search header/message list/composer · Drop overlay (`inset 60px 16px 16px`, 1.5px dashed, `--radius-14`, `backdrop-filter: blur(4px)`) · Quick Look · Requests · Contact Editor · Permission Confirm · Settings.

Each is specified literally in the handoff README §§1–4; this document does not restate that text, and the implementation follows it verbatim including every Korean string.

---

## 6. Motion & Interaction

### Timing

| Token | Value | Use |
|---|---|---|
| `--dur-menu` | `120ms` | context menu enter |
| `--dur-popover` | `150ms` | menus, popovers, drop overlay enter; color/border transitions |
| `--dur-modal` | `200ms` | modals, messages, toast enter; background/transform transitions |
| `--dur-step` | `250ms` | step changes |
| `--dur-color` | `150ms` | color transition, fast |
| `--dur-color-slow` | `200ms` | color transition, slow (citation chip, send button) |
| `--dur-knob` | `200ms` | toggle knob `left` |
| `--dur-flash-fade` | `600ms` | revealed-row `background`/`box-shadow` fade |
| `--dur-shimmer` | `1.4s` | skeleton shimmer, linear infinite |
| `--dur-spin` | `1.4s` | retry spinner, 360° |
| `--time-retry` | `1500ms` | retry-in-progress hold before success |
| `--time-flash` | `1800ms` | revealed row + tab dot flash hold |
| `--time-toast` | `2400ms` | toast auto-hide |
| `--ease-out` | `ease-out` | every enter animation |

### Keyframes (app base stylesheet, not `tokens.css`)

```css
@keyframes rise   { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes shimmer{ 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
```

`--shimmer-gradient` is `linear-gradient(90deg,#F0F0F4 0%,#E4E4EA 50%,#F0F0F4 100%)` at `background-size: 200% 100%`. The README gives the stops (`#F0F0F4 → #E4E4EA → #F0F0F4`) and the 200% size; the `90deg` axis and the stop positions come from `proto:1151`.

### Rules

- Animate `transform`, `opacity`, `filter`, `background`, `border-color`, and `left` (the toggle knob only). Never animate width/height/top/margin.
- Every overlay enters with `rise`; nothing exits with an animation (the reference removes overlays immediately).
- Motion is state feedback only: enter, hover/active color, toggle knob, pending shimmer, retry spin, reveal flash. No decorative or idle motion anywhere.
- `prefers-reduced-motion: reduce` → `rise` collapses to an opacity-only fade, shimmer and spin become static (shimmer holds the mid stop `#E4E4EA`, the retry icon stays at 0°), color transitions stay (they carry state, and are under 200ms).

### Behavior timings (product logic, not CSS)

Quick answer resolves after `0.8s`; Deep after `deepLatency` (default `4s`) with progress steps at `latency/4`; Deep progress is streamed and cancellable. Test connection shows `연결 확인 중…` for `1.1s`; Check for Updates shows `Checking…` for `1.2s`; the Settings config chat replies after `500ms`. These live in the app's timing constants, not in `tokens.css`.

### Keyboard

`⌘K` composer · `⌘F` Finder search · `⌘T` new tab · `⌘W` close tab · `⌘E` Evidence panel · `⌘⇧H` history · `⌘N` new chat · `⌘,` Settings · `⌘⇧R` / `⌘O` Show in Finder · `⌥1`–`⌥9` jump to evidence · `space` Quick Look · `↑`/`↓` move selection or change evidence · `←`/`→` change evidence · `Enter` open · `Delete`/`Backspace` trash · `Esc` closes the top-most overlay in the order: permission confirm → contact editor → context menu → Requests/access menu → history → Quick Look → Settings, otherwise clears the search query; in the composer while generating, Esc stops generation.

**Focus zones** are `finder` or `evidence`. The zone decides the selected-row color (`--accent-soft` vs `--bg-input`) and where the arrow keys and `space` act.

---

## 7. Depth & Surface

### Strategy

Depth is hairlines first, shadow last. Four elevations only:

| Level | Token | Value | Members |
|---|---|---|---|
| 0 — flush | — | 1px `--border` / `--border-divider` | sidebar, tab strip, toolbar, rows, evidence panel, status bar |
| 1 — raised | `--shadow-raised` | `0 4px 16px rgba(20,20,40,0.08)` | composer, toast, Quick Look paper previews, `generic` tile |
| 1d — drag | `--shadow-drag` | `0 4px 16px rgba(20,20,40,0.12)` | drop-overlay stacked tile (`proto:325`) |
| 2 — floating | `--shadow-window` | `0 20px 60px rgba(20,20,40,0.18)` | the window, menus, popovers, modals, Quick Look panel |

Scrims sit between 1 and 2: `--scrim-quicklook` with `blur(3px)`, `--scrim-modal` with `blur(8px)`, drop overlay `--drop-overlay-bg` with `blur(4px)`.

### Radii

| Token | Value | Use |
|---|---|---|
| `--radius-2` | `2px` | stop-button square |
| `--radius-3` | `3px` | `doc` preview mark block |
| `--radius-4` | `4px` | 14–16px tiles, progress bar, tree-connector corner |
| `--radius-5` | `5px` | 18–20px tiles, keycaps, citation chips, skeleton bars |
| `--radius-6` | `6px` | buttons, rows, chips, 22px tiles, code block, segmented item |
| `--radius-7` | `7px` | Settings segmented item |
| `--radius-8` | `8px` | small cards, inner boxes, segmented track, send button, 36px tile |
| `--radius-9` | `9px` | Requests count badge (min 18 × 18) |
| `--radius-10` | `10px` | cards, composer, menus, search field, toast, scrollbar thumb, bubbles |
| `--radius-11` | `11px` | pill badges and toggle track (height 22) |
| `--radius-12` | `12px` | popovers, 88 × 108 preview tile |
| `--radius-14` | `14px` | window, modals, drop overlay, 64px stacked tile |
| `--radius-full` | `50%` | dots, traffic lights, avatars, contact tiles |

Tabs use `10px 10px 0 0` — `--radius-10` on the top corners only.

### Scrollbars

| Token | Value |
|---|---|
| `--scrollbar-size` | `10px` |
| `--scrollbar-thumb` | `#E3E3EA` |
| `--scrollbar-radius` | `10px` |
| `--scrollbar-inset` | `3px` |

```css
::-webkit-scrollbar { width: var(--scrollbar-size); height: var(--scrollbar-size); }
::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: var(--scrollbar-radius);
  border: var(--scrollbar-inset) solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-track { background: transparent; }
```

### Rules

- A surface gets a shadow only if it floats above the window's own plane. Cards, rows, and panels get hairlines.
- `overflow: hidden` on the window and every rounded container that clips children (tab strip, popovers, modals, Settings cards) so radii stay honest.
- Borders are always exactly 1px, except the drop-overlay's 1.5px dashed ring and the 2px inset quote rule (`inset 2px 0 0 #E3E3EA`).

---

## 8. Accessibility Constraints & Accepted Debt

### Constraints

1. **Text on accent is `#1A1A2E`.** `#1A1A2E` on `#FF6363` is ~7.0:1 — AA for all sizes. White on `#FF6363` would be ~2.6:1 and is forbidden; this is the reason the rule exists, not a stylistic preference.
2. **`--text-muted` (`#6E6E80`) on white is ~5.5:1** — AA for normal text; it is the floor for meaningful text. `--text-faint` (`#A0A0B0`, ~2.6:1) is **decoration only**: separators, lock glyphs, disabled chevrons, the hollow index ring. It must never carry information that is not also conveyed structurally.
3. **State is never color-only.** Every status pairs a dot with a **text label** (`포함` / `제외` / `재시도` / `Ask` / `Allow` / `Deny` / `Mixed` / `Indexed` / `Syncing` / `Paused` / `Error`). Explicit-vs-inherited permission is carried by the tinted-vs-outlined treatment *and* by the tooltip suffix ` · 상위 폴더 설정`. The mixed conic dot is always accompanied by the literal label `Mixed`.
4. **Focus is visible.** The search field and every text input take an `--accent` border on focus. Buttons and rows that the reference gives no focus ring get `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }` — the only coloured edge permitted beyond the list in §2.10. Hover styling alone is never a focus indicator.
5. **Keyboard parity.** Every action reachable by mouse has a keyboard path: selection and navigation via arrows/Enter, Quick Look via `space`, evidence via `←/→` and `⌥1–9`, overlay dismissal via `Esc`, trash via `Delete`. Focus zones (`finder` / `evidence`) are set by clicks *and* by the shortcuts that act on them.
6. **Semantics over paint.** Rows are `role="row"` inside a `role="grid"` list with `aria-selected`; badges are real `<button>`s with `aria-label` carrying the full tooltip text; menus are `role="menu"` with `role="menuitem"` children and focus trapped until dismissal; modals are `role="dialog" aria-modal="true"` with the title as `aria-labelledby`. Toasts are `role="status" aria-live="polite"` — they auto-hide after 2400ms, so the live region is the only way the message reaches a screen reader.
7. **Reduced motion** is honored per §6 rules; no information is conveyed by motion alone (the reveal flash is always accompanied by selection + scroll-into-view).
8. **Hit targets.** 32px rows and buttons are comfortable; the 22px pills, 22px tab close button, and 18px stack chevron are below the 24px guidance and are compensated by full keyboard equivalents (context menu for index/trash, Access menu opens from the same keyboard-focusable pill).
9. **Korean typography.** Pretendard must load before first paint of Korean text or the fallback shifts metrics; `word-break: keep-all` on bubbles and `text-wrap: pretty` on paragraphs prevent mid-word breaks in Korean copy.

### Accepted Debt

1. **Light theme only.** Settings exposes a Light/Dark/System segmented control, but the reference defines no dark palette. The control ships inert (it stores the preference) until a dark token set is added to §2. Tracked as the largest known gap.
2. **`--text-faint` on `--bg-chrome`** (`#A0A0B0` on `#F6F6F8`, ~2.5:1) fails AA. Kept because the reference uses it only for `›` separators and lock glyphs, which are redundant with adjacent text. Do not extend its use.
3. **22px pill hit targets** are below the 24px minimum (constraint 8). Kept for Finder density, which is the product's identity; mitigated by keyboard paths.
4. **Prototype dead CSS not adopted:** the prototype's global `a:hover { color: #FF8A8A }` (`proto:18`) is not a token — the prototype renders zero `<a>` elements and the README's accent hover token is `#FF7A7A`. If real links appear, they use `--accent` / `--accent-hover`.
5. **`--accent-text-strong` (`#C73A3A`) has no call site** in the prototype; the README lists it as part of `accent.text`, so it is carried as a token for parity and is currently unused.
6. **No zoom scaling.** The prototype's `zoom = min(1, vw/1360, vh/820)` is deliberately dropped in favor of a hard 1360 × 820 minimum window, per the README's own instruction. Below that minimum the app is unusable rather than shrunk — accepted for a desktop-only target.
7. **Brand tiles are letter placeholders.** Slack/Gmail/Notion/Discord/Telegram render as colored letter tiles, not official logos, pending licensing review.
8. **`--tile-md` and `--tile-zip` are the same value** (`#8A8A9C`, also `--bg-tile-fallback`). Kept as three names because the reference treats them as three concepts; consolidating would lose the "unknown kind" meaning.
9. **Evidence/code/paper surfaces use four near-white values** (`#FAFAFB`, `#FAFAFC`, `#F4F4F7`, `#F3F3F7`) that are visually indistinguishable in isolation. They are preserved exactly because the reference is pixel-final; do not "simplify" them into one token.
10. **The dupey-required banner is outside the reference.** The reference assumes duplicate detection always works. The app adds one operational error surface, `.list__error` (handoff README has no counterpart), shown above the file list when the required `dupey` CLI is missing or a version-family scan fails. It is built only from existing tokens (`--danger-text`, `--danger-bg-soft`, `--danger-border`, `--bg-window`, `--font-mono`), carries `role="alert"`, states the failure and the exact install command, and offers 다시 확인 to re-probe. It is an error path, never a normal state.
