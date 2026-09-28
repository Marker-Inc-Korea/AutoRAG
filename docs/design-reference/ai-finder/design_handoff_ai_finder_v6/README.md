# Handoff: AutoRAG Agent — AI Finder (main window, v6)

## Overview
AutoRAG Agent is a **macOS desktop app** that combines a Finder-style file browser with an AI search assistant. It indexes local folders, cloud drives (iCloud, Google Drive, Dropbox), and communication tools (Slack, Gmail, Notion, Discord, Telegram) and shows all of them as one folder tree. The user asks questions in natural language and gets **two answers side by side**:
- **Quick**: about 1 second, few sources.
- **Deep**: cross-checks many sources, takes tens of seconds.

Both answers carry numbered citations `[n]`. Each citation opens its **Evidence** (the source excerpt with the relevant passage highlighted).

The user can also ask a colleague's agent through **Contacts**. Other agents' requests for the user's files arrive in **Requests**. Sharing is governed by per-file and per-folder **Access** rules: Ask, Allow, Deny.

This handoff covers the **main window only**, a single 3-column screen plus its overlays. Onboarding is a separate design and is not part of this bundle.

## About the design files
`AI Finder v6.dc.html` is a **design reference built in HTML**, not production code. It is a working prototype that shows the intended look, copy, and behavior. All data is mocked and all network or AI calls are faked with timers.

Your job is to **recreate this design in the target codebase's environment**, using its patterns and libraries. If no codebase exists yet, a good default for this product is **Tauri or Electron + React + TypeScript**, or native **SwiftUI**. Build a real component structure; don't port the prototype's single-file style.

How the prototype file is organized:
- The markup sits between `<x-dc>` and `</x-dc>`. `{{ name }}` holes are values computed in `renderVals()`.
- `<sc-for list="{{ xs }}" as="x">` is a loop and `<sc-if value="{{ cond }}">` is a conditional.
- `style-hover="…"` is the `:hover` style.
- All logic is in the `class Component` at the bottom: a React-like class with `state`, `setState`, and `renderVals()`.
- The mock data constants (`FS`, `EV`, `A1`, `A2`, `HIST`, `SRC`, `CONTACTS0`, `REQ0`, …) sit at the top of that script. Reuse them as fixtures.
- To view it, open the HTML file in a browser with `support.js` next to it.

## Fidelity
**High fidelity.** Colors, type sizes, spacing, radii, copy, and interactions are final. Recreate them pixel-accurately. UI copy is **Korean, with some English labels**. Keep every string exactly as written in this document or in the prototype.

---

## Design tokens

### Colors
| Token | Value | Use |
|---|---|---|
| `bg.desk` | `#E9E9EE` | area behind the window |
| `bg.window` | `#FFFFFF` | main surfaces |
| `bg.chrome` | `#F6F6F8` | sidebar, tab strips, panel headers |
| `bg.subtle` | `#FAFAFB` / `#FAFAFC` | evidence panel, answer cards |
| `bg.input` | `#F0F0F4` | search field, composer, user bubble, segmented control track |
| `bg.selectedNav` | `#EAEAF0` | active sidebar item, toggled icon buttons |
| `bg.hover` | `rgba(26,26,46,0.05)` (rows `0.04`, icon buttons `0.06`) | hover |
| `border` | `#E3E3EA` | all 1px borders |
| `border.divider` | `#EDEDF2` | list row separators |
| `border.strong` | `#C9C9D4` | hover borders, disabled icons, tree lines |
| `text.primary` | `#1A1A2E` | |
| `text.body` | `#3A3A4E` | |
| `text.muted` | `#6E6E80` | meta text, labels, placeholders |
| `text.faint` | `#A0A0B0` | separators (›), lock icons |
| `accent` | `#FF6363` | primary buttons, selection, focus, active citation. **Text on accent is `#1A1A2E`, not white.** |
| `accent.hover` | `#FF7A7A` | |
| `accent.soft` | `rgba(255,99,99,0.12)` | selected row (Finder focused) |
| `accent.flash` | `rgba(255,99,99,0.18)` + inset ring `rgba(255,99,99,0.6)` | revealed row, 1.8 s |
| `accent.text` | `#D64545` / `#C73A3A` | accent-colored text, citation chip fg |
| `folder` | `#7C86E8` | folder glyph fill |
| `ok` | `#22C55E` dot / `#15803D` text / `rgba(34,197,94,0.14)` bg | Indexed, success |
| `warn` | `#F59E0B` dot / `#B45309` text / `rgba(245,158,11,0.14–0.16)` bg | Syncing, Ask permission |
| `info` | `#3B82F6` dot / `#1D4ED8` text / `rgba(59,130,246,0.12)` bg | Allow permission |
| `danger` | `#EF4444` dot / `#C62828` text / `rgba(239,68,68,0.12)` bg | Deny permission, destructive |
| `error.orange` | `#FF8C00` dot / `#C2410C` text / `rgba(255,140,0,0.12–0.16)` bg | index failure, source error |
| `switch.off` | `#CFCFD9` | toggle track when off |
| `highlight.marker` | `linear-gradient(transparent 55%, rgba(255,214,10,0.55) 55%)` | highlighted passage in Evidence |
| `traffic lights` | `#FF5F57`, `#FEBC2E`, `#28C840` (12px circles, 8px gap) | |

**File-type tiles** are small rounded squares with a white letter. The background colors are:

| Type | Letter | Background |
|---|---|---|
| xlsx | X | `oklch(0.62 0.14 150)` |
| csv | C | `oklch(0.58 0.1 175)` |
| pdf | P | `oklch(0.6 0.17 28)` |
| docx | W | `oklch(0.58 0.15 258)` |
| pptx | P | `oklch(0.66 0.15 55)` |
| md, zip | M, Z | `#8A8A9C` |
| png | I | `oklch(0.58 0.13 305)` |
| slack | S | `oklch(0.52 0.16 340)` |
| mail | M | `oklch(0.58 0.15 12)` |
| notion | N | `#1A1A2E` |
| discord | D | `oklch(0.56 0.17 278)` |
| telegram | T | `oklch(0.64 0.13 232)` |
| contact | @ | `oklch(0.52 0.13 285)` |

The full map is `KD` in the prototype.

### Typography
- **Fonts:** `Inter` for Latin and `Pretendard` for Korean. Stack: `Inter, Pretendard, -apple-system, system-ui, sans-serif`. Monospace (IDs, code): `ui-monospace, SFMono-Regular, Menlo, monospace`.
- **Sizes:**
  - 18px / 600: modal titles, empty-chat title
  - 14px / 400–600: body, list rows, buttons in panels
  - 13px: context menu, source descriptions
  - 12px: meta, small buttons, column headers
  - 11px: badges, keycaps, group labels (600, letter-spacing 0.02em)
  - 10px: tile letters
- **Line-height:** 1.5 for body, 1.6 for evidence and document text.
- Apply `text-wrap: pretty` to paragraphs.

### Radii
- 4–5px: tiles, keycaps
- 6px: buttons, rows
- 8px: small cards, inner boxes
- 10px: cards, composer, menus, search field
- 11px: pill badges, height 22
- 12px: popovers
- 14px: window and modals

### Shadows
- Window and popovers: `0 20px 60px rgba(20,20,40,0.18)`
- Composer, toast, and paper previews: `0 4px 16px rgba(20,20,40,0.08)`

### Motion
- **Enter animation:** `rise`, from `opacity 0; translateY(6px)` to identity. Durations: 120–150 ms for menus and popovers, 200 ms for modals and messages, 250 ms for step changes. Easing `ease-out`.
- **Color transitions:** 150–200 ms.
- **Toggle knob:** `left` over 200 ms.
- **Skeleton shimmer:** 1.4 s linear infinite, gradient `#F0F0F4 → #E4E4EA → #F0F0F4`, 200% background-size.
- **Retry spinner:** 360° rotation over 1.4 s.

### Scrollbars
10px wide, thumb `#E3E3EA`, radius 10, 3px transparent inset.

---

## Layout: app shell
- The window fills the viewport with **18px padding** on the `#E9E9EE` desk. The inner window has 1px `#E3E3EA` border, radius 14, the window shadow, and `overflow: hidden`.
- The prototype scales the whole app with `zoom = min(1, vw/1360, vh/820)`, so the **minimum design size is 1360 × 820**. In a real app, set that as the minimum window size instead of zooming.
- Main grid: `grid-template-columns: 212px minmax(640px, 1.4fr) minmax(440px, 1fr)`.
  1. **Sidebar**, 212px.
  2. **Finder column**: tabs, toolbar, file list, status bar, and the Evidence panel docked at the bottom.
  3. **AI Search column**, with a 1px left border.

All overlays (Quick Look, modals, toast, Requests) are positioned absolutely inside the window.

---

## Screens / components

### 1. Sidebar (`#F6F6F8`, right border)
- **Top bar**, 48px: traffic lights, 16px left padding.
- **Scrollable groups** (padding `4px 10px 12px`, 14px gap between groups). Each group has an 11px/600 muted label (padding `4px 8px 6px`) and items that are 32px tall, radius 6, padding `0 8px`, gap 10, 14px text.
  - **Favorites:** Recents, Desktop, Downloads, Documents (16px stroke icons, 1.8 stroke).
  - **Cloud:** iCloud Drive, Google Drive, Dropbox.
  - **Sources:** Slack, Gmail, Notion, Discord, Telegram. These show a 16px file-type tile instead of an icon.
  - **People:** Contacts.
  - **Active item:** the item whose name is the first path segment of the current tab. It gets bg `#EAEAF0`, fg `#1A1A2E`, and icon `#FF6363`. Inactive items have fg `#3A3A4E` and icon `#6E6E80`.
  - **Status dots** (6px, trailing): amber `#F59E0B` for syncing (Gmail, Google Drive), orange `#FF8C00` for error (Discord).
- **Footer** (top border, padding 10, gap 8):
  - Indexing progress: label "Indexing · Gmail" with "88%" at 12px, over a 4px bar (track `#F0F0F4`, fill `#F59E0B`).
  - **Requests** button: bell icon. When there are pending requests it shows a count badge (min 18×18, radius 9, `#D64545`, white 11/700). It gets bg `#EAEAF0` while the popover is open.
  - **Settings** button: gear icon and a `⌘,` keycap.
  - **Keycap style:** 11px, `#6E6E80`, padding `1px 5px`, 1px border, radius 5, bg `rgba(26,26,46,0.04)`. Keycaps show only when the `showHotkeys` prop is true.

### 2. Finder column

**Tab strip** (48px, `#F6F6F8`, bottom border; tabs are bottom-aligned)
- Tabs are flexible: min 84px, max 190px, height 36, radius `10px 10px 0 0`, 12px/500 text.
- The **active tab** is white with a 1px border and no bottom border, and uses `margin-bottom: -1px` so it merges with the toolbar. Inactive tabs are transparent with `#6E6E80` text.
- Each tab has a 22px close button (✕). Closing is blocked when only one tab remains.
- A revealed tab shows a 6px `#FF6363` flash dot for 1.8 s.
- While searching, the active tab title becomes `Search "<query>"`.
- A **+** button (32px) opens a new tab on the same path (`⌘T`).

**Toolbar** (52px, white, bottom border, gap 8, padding `0 12px`)
- Back and forward buttons, 32px. Chevrons are `#3A3A4E` when enabled and `#C9C9D4` when disabled.
- **Breadcrumbs:**
  - Show at most the last 2 path segments, prefixed with "…" when the path is truncated.
  - Separator: `/` in `#C9C9D4`.
  - Each crumb is a 28px button with max-width 220 and ellipsis.
  - The last crumb is `#1A1A2E`/600; the others are `#6E6E80`.
  - Clicking a crumb navigates there.
- **Search field:** 236 × 32, radius 10, bg `#F0F0F4`.
  - The border is `#FF6363` when focused and `#E3E3EA` otherwise.
  - It contains a magnifier icon, a 12px input, and a `⌘F` keycap.
  - Placeholder: "Instant search", or "Search contacts" on the Contacts path.
  - Enter opens the first result. Esc clears the query and blurs.

**Column header** (30px, 12px/500 muted, bottom border)
- Grid: `minmax(0,1fr) 100px 60px 96px 66px 76px`, gap 12, padding `0 16px 0 20px`. The same grid is used for rows.
- Columns: **Name | Date Modified | Size (right-aligned) | Kind (or "Where" while searching) | Index | Access**.
- The first four are sort buttons. The active column is `#1A1A2E` and shows a 10px chevron-up, rotated 180° when descending.
- **Sort cycle** per column:
  1. First click sorts ascending. Date and Size start **descending** instead.
  2. Second click flips the direction.
  3. Third click clears sorting.
- Folders always sort before files. Name sort uses Korean locale compare.

**File rows** (list padding `4px 8px`; rows 32px tall, radius 6, padding `0 8px 0 12px`)
- **Name cell:**
  - Folders show an 18px folder glyph (`#7C86E8`). Files show an 18px tile with radius 5.
  - Name is 14px with ellipsis.
  - **Version stack badge:** shown when this file heads a family of duplicates or versions. It is an 18px pill with bg `#F0F0F4` (or `#E3E3EA` when open), a chevron that rotates 90° when open, and the count. Tooltip: "다른 버전 N개 · 클릭하면 펼쳐집니다".
  - **Stack expansion:** the stack expands automatically when the head or any member is selected. Children are indented 18px, have an L-shaped tree connector (1px `#C9C9D4`, 4px corner radius), and use `#3A3A4E` names. A child's Kind column shows the relation (동일본 = exact copy, 유사본 = near copy, 포함본 = contains) followed by ` · <folder>` when the child lives in a different folder.
  - Members of a stack whose head is in the same folder are hidden from the flat list.
- **Date, Size, Kind:** 12px muted. While searching, Kind is replaced by the location path joined with " › ".
- **Index badge:** a 22px pill, radius 11, 11px/600. Files only.
  - Included: 1px `#E3E3EA` border, a green 6px dot `oklch(0.62 0.14 150)`, label "포함".
  - Excluded: bg `#F0F0F4`, a hollow dot (inset ring 1.5px `#A0A0B0`), label "제외".
  - Clicking toggles included/excluded and shows a toast: "<name> — 인덱싱에서 제외했습니다" or "… 포함했습니다".
  - Stack members default to excluded; everything else defaults to included.
  - **Failed index**, for files in `IDX_FAIL`: orange style (`#C2410C` on `rgba(255,140,0,0.12)`) with a retry icon and the label "재시도". The tooltip is "인덱싱 실패 · <reason> — 클릭하면 다시 시도". Clicking changes the label to "재시도 중" and spins the icon for 1.5 s, then the file becomes "포함" and a toast shows "<name> — 인덱싱을 완료했습니다".
- **Access badge:** 22px pill with a 6px dot and a label.
  - **Explicit** setting on this item: tinted bg and fg per permission (Ask = amber, Allow = blue, Deny = red).
  - **Inherited** setting: transparent bg, `#E3E3EA` border, `#6E6E80` text. Tooltip adds " · 상위 폴더 설정".
  - **Mixed** (a folder whose descendants have different explicit values): label "Mixed", bg `#F0F0F4`, and a conic-gradient dot made of the distinct permission colors.
  - Clicking opens the **Access menu**:
    - 248px wide, radius 10, padding 6.
    - Header: "에이전트 요청 시 공유".
    - Three options, each with a dot, a title, and a subtitle, plus an accent ✓ on the current one:
      - "허용 시 공유" / "요청이 오면 Requests에서 확인" (Ask)
      - "항상 허용" / "에이전트 요청에 자동으로 공유" (Allow)
      - "항상 거부" / "요청을 자동으로 거절" (Deny)
    - Footer note: for folders, "하위 폴더와 파일에 모두 적용됩니다". For files, "이 파일에만 적용됨", "<folder> 폴더 설정을 따르는 중", or "기본값". For mixed folders, "하위 항목 N개가 다르게 설정되어 있습니다…".
    - The menu opens **upward** when the row is among the last 4 and the list has more than 6 rows.
    - It closes on outside mousedown or Esc.
  - Picking a value for a **folder** always goes through the **Permission Confirm** modal (see §4.4). For a **file** it applies immediately.
- **Row states:**
  - Hover: `rgba(26,26,46,0.04)`.
  - Selected while the Finder zone is focused: `rgba(255,99,99,0.12)`.
  - Selected while another zone is focused: `#F0F0F4`.
  - Flash after reveal: `accent.flash` for 1.8 s, with a 600 ms transition.
- **Selection:**
  - Click selects one row.
  - **⌘-click** toggles a row in the multi-selection.
  - **Shift-click** selects the range from the anchor.
  - ↑/↓ moves the selection and keeps it scrolled into view (row height 32; target scrollTop is `index*32 - 80`).
  - Double-click or Enter opens a folder (navigate into it) or a file (Quick Look). A search result opens by navigating to its folder with the file selected.
- **Drag:** rows are draggable into the AI panel. Use a custom drag image: a white 32px chip with the tile and name. Dragging part of a multi-selection drags all of it, labeled "N개 항목".
- **Context menu** (right-click, position fixed at the cursor, 220px, radius 10, padding 5, 28px items):
  - "Quick Look" with `space` keycap (or "열기" with `↩` for folders)
  - separator
  - "인덱싱에서 제외" or "인덱싱에 포함" (or "인덱싱 다시 시도" in orange when the file failed)
  - separator
  - "휴지통으로 이동" with `⌫` keycap, in red `#D64545`. When several items are selected: "N개 항목 휴지통으로 이동".
- **Delete/Backspace** (not while typing, Finder zone focused) moves the selection to trash, with toast "<name> — 휴지통으로 이동했습니다" or "N개 항목을 휴지통으로 이동했습니다".
- **Search results:** a summary line "N results across all locations" above the list. Search matches file names by substring across every location except Recents.
- **Empty states** (centered, 14px muted, padding 40): "빈 폴더", or `"<q>"와 일치하는 파일이 없습니다`.

**Contacts view** (shown instead of the file list when the path is `Contacts`)
- Header: "Contacts" (14/600) with the subtitle "내 에이전트가 이 사람들의 에이전트에게 대신 묻고 답을 받아옵니다. 채팅으로 끌어다 놓으면 해당 연락처에게 물어봅니다."
- Accent button "+ Add contact" on the right.
- **Contact cards:** padding 14, radius 10, 1px border (hover `#C9C9D4`), cursor grab. Each card has:
  - name (14/600), role (12 muted), and an **Edit** button (28px, pencil icon) on the first line
  - a description (12px `#3A3A4E`) below
- Cards are draggable into the chat as `@Name`.
- The search field filters contacts by name, role, description, and ID.

**Status bar** (32px, top border, 12px muted)
- Left: "N items" plus " · K selected" when there is a selection (or "N contacts" in the Contacts view).
- Right: a `space` keycap followed by "Quick Look · Drag into chat to ask".

**Evidence panel** (docked at the bottom of the Finder column, `#FAFAFB`, top border)
- **Height:** `flex 0 0 44%` when open. When collapsed only the tab strip (40px) remains. Toggle with `⌘E` or the double-chevron button at the right (44px wide; the chevron rotates 180° when collapsed).
- **Tab strip:** one 40 × 30 tab per citation number `1…N`, where N is the highest citation in the latest assistant answer. The active tab is `#FAFAFB` with a border and merges into the panel below. Clicking a tab (or a citation chip in the chat) selects it and focuses the Evidence zone.
- **Header** (padding `14px 16px 10px 20px`):
  - Row 1: 22px tile, "<n>. <title>" (14/600), 👍 and 👎 buttons (32px each, 1px border), and a **Quick Look** button (32px, eye icon).
  - Row 2 (indented 32px): a breadcrumb chip (24px, bg `#F0F0F4`, " › " separators, last segment bold). Clicking it runs **Show in Finder**. After the chip, a detail string in 12px muted (e.g. "Sheet "Summary" · 수정 9월 13일 김서연").
- **Feedback buttons:** 👍 toggles green (`rgba(34,197,94,0.12)` bg, `#15803D` fg, `rgba(34,197,94,0.45)` border) and 👎 toggles red (`rgba(239,68,68,0.10)`, `#C62828`, `rgba(239,68,68,0.4)`). The value is stored per `(chatId, evidenceNumber)`, and clicking the active value clears it. Toasts: "근거 n — 도움이 됨으로 기록했습니다" or "근거 n — 다음 검색부터 우선순위를 낮춥니다".
- **Body** (scroll, padding `4px 20px 20px 52px`, gap 8) shows a markdown-like excerpt made of these block types:
  - `h`: 14/600, prefixed with a muted "## "
  - `meta`: 12 muted
  - `p`: 12/1.6 `#3A3A4E`
  - `mark`: 12/1.6 `#1A1A2E` with the yellow marker highlight
  - `li`: an en dash, then the text
  - `q`: muted, with an inset 2px left rule `#E3E3EA`
  - `code`: 11px mono, bg `#F4F4F7`, 1px border, radius 6, `white-space: pre`
- **Keyboard** in the Evidence zone: ←/→ or ↑/↓ changes the evidence number. `⌥1…9` jumps directly to that number and opens the panel.

### 3. AI Search column
**Header** (48px, `#F6F6F8`): "AI Search" (14/600), then a **History** button (clock icon; bg `#EAEAF0` when open; `⌘⇧H`) and a **New chat** button (+; `⌘N`; clears the messages).

**History popover** (absolute, top 52, right 12, width 360, max-height 72%, radius 12)
- Search input: "대화 기록 검색". Esc closes the popover and Enter opens the first hit.
- Results are grouped as 오늘, 어제, 지난 7일, 지난 30일.
- Each item shows a title (14/500), a snippet (the first Quick answer with markup stripped, 12 muted), and a time. The current chat gets bg `rgba(255,99,99,0.08)` and a 6px accent dot.
- Empty state: `"<q>"에 해당하는 대화가 없습니다`.
- A transparent click-catcher behind the popover closes it.

**Message list** (scroll, padding `20px 20px 8px`, gap 20; auto-scrolls to the bottom when messages change)
- **Empty chat:** "무엇을 찾아드릴까요?" (18/600) with "파일, 메일, Slack, Notion 전체에서 찾아 빠른 답변과 정확한 답변을 함께 드립니다."
- **User message** (right-aligned): a "You" label, then a bubble (max-width 78%, padding `10px 14px`, radius 10, bg `#F0F0F4`, border). Attachments show below as chips (24px, tile and name).
- **Assistant message:** an "Assistant" label, then two stacked cards (padding `14px 16px`, radius 10, bg `#FAFAFC`, border).
  - **Quick** card: 6px `#3A3A4E` dot, "Quick" (600), and meta (e.g. "0.9s · 2 sources").
  - **Deep** card: 6px `#FF6363` dot, "Deep", and meta. While pending, its border is `rgba(255,99,99,0.35)` and the meta cycles through the progress steps "Searching 9 sources…", "Reading 22 documents…", "Cross-checking figures…", "Writing answer…" at `latency/4` intervals.
  - **Pending:** shimmer skeleton bars (height 10, radius 5). Quick uses widths 90%/70%; Deep uses 96/100/82/94/58%.
  - **Stopped:** "응답 생성을 중단했습니다." in muted text, with meta "Stopped".
  - **Answer text:** 14/1.5 `#3A3A4E`. `**bold**` renders as `#1A1A2E`/600.
  - **Citation chips** `[n]`: inline, 18px tall, min-width 18, radius 5, 11/600. Inactive: bg `rgba(255,99,99,0.16)`, fg `#D64545`. Active (the currently selected evidence): bg `#FF6363`, fg `#1A1A2E`. Clicking one selects that evidence, opens the panel, and focuses the Evidence zone.

**Drop overlay** (shown while any item is being dragged; covers the chat at `inset 60px 16px 16px`)
- 1.5px dashed border, radius 14, `backdrop-filter: blur(4px)`.
- A 64px stacked tile with an accent "+" badge.
- **Idle:** border `#C9C9D4`, bg `rgba(250,250,252,0.86)`. Title "여기로 끌어서 AI에게 물어보기", or for a contact "여기로 끌어서 이 연락처에게 물어보기".
- **Hover:** border `#FF6363`, bg `rgba(255,99,99,0.07)`, scale 1.04. Title "놓으면 대화에 첨부됩니다", or "놓으면 <name>의 에이전트에게 물어봅니다".
- The item name shows below the title.
- **Drop** adds the items as draft attachments (de-duplicated by name) and focuses the composer.

**Composer** (padding `8px 20px 14px`)
- Box: bg `#F0F0F4`, radius 10, border (`#FF6363` while a drag hovers), composer shadow.
- **Draft attachment chips:** 26px, with a × to remove.
- **Input:** 14px. Placeholder "파일, 메일, 메신저 전체에서 물어보세요", or "여기에 놓으면 첨부됩니다" while a drag hovers.
- **Send/Stop button** (32px, radius 8):
  - Idle with content: bg `#FF6363`, up-arrow icon.
  - Empty: bg `#C9C9D4`.
  - **Generating:** bg `#1A1A2E` with a 10px white rounded stop square. Tooltip "Stop generating (Esc)". Clicking, or pressing Esc in the composer, cancels all pending timers and marks the message stopped. Both cards that are still pending switch to the stopped state.
- **Sending:**
  - Enter sends; ignore Enter while an IME is composing (`isComposing`).
  - With no text and a contact attached, the message becomes "Q3 예산 관련해서 확인 부탁해".
  - With no text and files attached, it becomes "첨부한 파일 기준으로 정리해줘".
  - Quick resolves after 0.8 s. Deep resolves after `deepLatency` seconds (default 4).
  - If a contact was attached, the answer comes "via agent" from that contact (fixture `AG(name)`).
  - Sending is blocked while generating.

### 4. Overlays

**4.1 Quick Look** (`space` toggles it; Esc closes)
- **Scrim:** `rgba(20,20,40,0.16)` with a 3px blur. Clicking the scrim closes.
- **Panel:** 760px wide, 80% tall, radius 14.
- **Header** (52px, chrome bg):
  - Close button, tile, name, and location ("Documents › Finance › …").
  - From Finder: an "Ask AI about this" button (sparkle icon). It attaches the file to the composer, closes Quick Look, and focuses the composer.
  - From Evidence: an accent-outlined "Show in Finder" button with a `⌘⇧R` keycap.
  - Always: a `space` keycap.
- **Body** (`#F6F6F8`, padding 20) renders by kind:
  - `sheet`: a spreadsheet grid with columns 24px, 1.1fr, 1fr, 1fr, 1fr; A–D column headers; row numbers; highlighted rows `rgba(255,99,99,0.2)`; header and total rows in 600.
  - `doc`: a white paper card (padding `22px 24px`) with a subtitle, an 18/700 title, a divider, and blocks. `mark` blocks are 600 on `rgba(255,99,99,0.22)`.
  - `mail`: subject, a 32px avatar, from/to, date, and body blocks (`mark` on `rgba(255,99,99,0.2)`, quoted text with a left rule).
  - `chat`: a channel header, then messages with 28px colored avatars; the highlighted message has bg `rgba(255,99,99,0.14)`.
  - `generic`, for files with no preview: a large tile (88 × 108) or a 96px folder glyph, the name, "Kind · size" or "Folder · N items", and "Modified <date>".

**4.2 Requests popover** (anchored at left 220, bottom 12; 420px wide; radius 12)
- Header: "Requests", a segmented control with "대기 중 N" and "처리됨", and a close button.
- **Each request:**
  - "**<name>**의 에이전트가 요청" and "<role> · <time>"
  - the question in a `#F6F6F8` box (14px)
  - "요청한 자료", listing each file's tile, name, folder, and its current Access label with a dot
- **Pending requests** end with buttons: "Show in Finder" (ghost), then "거부" (outline, red text) and "허용" (accent).
  - Allow or Deny sets the status, sets the time to "방금", and shows a toast: "<name>의 요청을 허용했습니다 · 파일을 공유했어요" or "…거부했습니다".
  - Show in Finder opens a new tab at the file and flashes it.
- **Done requests** show "허용됨" or "거부됨", plus either " · 권한 설정에 따라 자동 처리" (when handled automatically) or " · <time>".
- Empty states: "대기 중인 요청이 없습니다" or "처리된 요청이 없습니다".

**4.3 Contact Editor modal** (scrim `rgba(20,20,40,0.22)` + blur 8; 540px wide; radius 14)
- Title: "New contact" or "Edit contact".
- **New contact:** a **Contact ID \*** field (mono font) with placeholder "예: jihoon.park@markr.ai" and hint "상대 에이전트의 고유 ID. 등록 후에는 변경할 수 없습니다." A duplicate ID turns the border red (`#EF4444`) and the hint to "이미 등록된 Contact ID입니다".
- **Edit contact:** the ID is shown read-only in a gray box with a lock icon.
- **이름** and **역할 / 소속** inputs sit side by side, each 36px, radius 8, with an accent focus border.
- **설명** textarea (3 rows) with the helper "· 사람과 에이전트가 같은 설명을 봅니다".
- **Footer:**
  - Edit only: **Test connection** (bolt icon). It shows a status "연결 확인 중…" for 1.1 s, then "연결됨 · 응답 Nms" (green) or "응답 없음" (red).
  - Edit only: **Delete** (red text).
  - Cancel.
  - Save: accent when valid, `#C9C9D4` otherwise. Valid means a name is present and, for a new contact, a unique ID.
- Esc closes.

**4.4 Permission Confirm modal** (460px wide) — shown whenever a folder's access value changes
- Title: a folder glyph and `"<folder>" 폴더 권한 변경`.
- A row showing the change: a from-pill (current value, or "Mixed" with a gray dot), then →, then a to-pill (tinted), then "하위 항목 N개" right-aligned. N counts all descendants recursively.
- Body: "이 폴더와 하위 항목 N개가 '<title>'으로 바뀝니다. <sub>."
- **When descendants have their own explicit values**, an amber note appears: "하위 항목 M개는 따로 설정되어 있습니다. '모두 적용'은 개별 설정을 덮어쓰고, '개별 설정 유지'는 그 항목들을 그대로 둡니다." In this case the buttons are Cancel, **개별 설정 유지**, and **모두 적용** (accent). Otherwise they are Cancel and **적용**.
- **Apply** sets the folder's value and deletes all explicit descendant values. **Keep** sets the folder's value and leaves the descendants alone. Both show a toast.

**4.5 Settings modal** (`⌘,`; 760 × 580; radius 14)
- Header: "Settings" and a segmented control with **General**, **Models**, **Data Sources**.
- **General:** sections of 56px rows in bordered cards.
  - 일반: Language select (한국어 / English / 日本語 / 简体中文); Global shortcut, shown as `⌥` `Space` keycaps; Launch at login toggle; plus a menu-bar toggle and a Light/Dark/System theme segmented control (see the truncated lines in the source).
  - 소프트웨어 업데이트: "AutoRAG Agent 1.4.2" with status "마지막 확인: 오늘 09:00". The **Check for Updates** button shows "Checking…" for 1.2 s, then the status reads "최신 버전입니다 · 방금 확인". Also the toggles **Automatically install updates** and **Beta updates**, and a telemetry toggle.
  - 개인정보: **Clear chat history** with a red "Clear…" button.
  - **Toggle style:** 36 × 22 track, radius 11, `#FF6363` on / `#CFCFD9` off; 16px white knob at left 17px (on) or 3px (off).
- **Models:** a current-model card (36px tile, name, provider, and a "Quick · Deep 공통" pill). Below it a **config chat**:
  - Message bubbles: user bubbles right-aligned on `#F0F0F4`, assistant bubbles left-aligned on white.
  - Suggestion chips: 가장 정확한 모델로 / 빠른 응답 위주로 / 데이터가 기기 밖으로 안 나가게.
  - Input with a sparkle icon and placeholder "예: 회사 데이터는 밖으로 안 나가게 해줘".
  - The prototype maps intent with the keyword regexes in `cfgIntent` to one of: Claude Opus, Sonnet, Haiku, GPT-5, Gemini 2.5 Pro, or Llama 3.1 8B (on-device), and replies after 500 ms. In production, replace this with a real LLM tool call.
- **Data Sources:**
  - Summary: "9 sources · 226,269 items indexed · 2 syncing · 1 needs attention".
  - A list of sources. Each row uses the grid `28px 1fr 150px 28px 36px`: tile, name and detail, status badge and progress bar, an expand chevron, and an enable toggle.
  - **Status badges:** Indexed (green), Syncing (amber), Paused (gray), Error (orange).
  - A disabled source shows at 0.6 opacity with status Paused.
  - **Expanding** a row shows its agent-facing **Description** card with a "채팅으로 수정" button, which pre-fills the chat input with "<name> 설명 손보기: ".
  - Below the list, a 240px **sources chat** with chips (연결 가능한 데이터소스들 나열 / 데이터소스 설명 손보기 / Discord 다시 연결) and placeholder "예: Outlook 연결해줘, Slack 설명 손봐줘". It supports listing available sources, connecting one (adds a row with Syncing 3% and auto-expands it), editing a description, and reconnect help. The logic is in `srcAsk`; replace it with a real agent.

**4.6 Toast**
- Bottom center, 24px from the bottom, 36px tall, radius 10, bg `#F0F0F4`, border, 12px text, green 6px dot.
- Auto-hides after 2.4 s. A new toast replaces the current one.

---

## Keyboard shortcuts (global unless noted)
| Key | Action |
|---|---|
| `⌘K` | Focus the composer |
| `⌘F` | Focus Finder search |
| `⌘T` | New tab |
| `⌘W` | Close tab (shown in the tab close tooltip) |
| `⌘E` | Toggle the Evidence panel |
| `⌘⇧H` | Toggle chat history |
| `⌘N` | New chat |
| `⌘,` | Open Settings |
| `⌘⇧R` or `⌘O` | Show the current evidence in Finder |
| `⌥1`–`⌥9` | Jump to evidence n |
| `space` | Toggle Quick Look (source: the Finder selection, or the current evidence when the Evidence zone is focused) |
| `↑` / `↓` | Move the selection (Finder zone) or change evidence (Evidence zone) |
| `←` / `→` | Change evidence (Evidence zone) |
| `Enter` | Open the selected item |
| `Delete` / `Backspace` | Move the selection to trash (Finder zone, not typing) |
| `Esc` | Closes the top-most thing, in priority order: permission confirm, contact editor, context menu, Requests or access menu, history, Quick Look, Settings. Otherwise clears the search query. In the composer while generating, Esc stops generation. |

**Focus zones:** `finder` or `evidence`. Clicking a row sets `finder`. Clicking inside the Evidence panel, a citation, or an evidence tab sets `evidence`. The zone changes the selected-row color and decides where arrow keys and space act.

**Show in Finder (reveal):** opens a **new tab** at the item's folder, selects the item, scrolls to it, flashes the row and the tab dot for 1.8 s, clears the search, closes Quick Look, and sets the zone to finder.

---

## State model (TypeScript sketch)
```ts
type Perm = 'ask' | 'allow' | 'deny';
interface Tab { id: number; path: string; sel: string | null; selWhere?: string | null;
  selSet?: string[] | null; anchor?: string | null; back: string[]; fwd: string[]; flash?: boolean }
interface Msg { role: 'user'; text: string; attach: {name: string; k: Kind}[] }
interface AsstMsg { role: 'asst'; a: Answer; quickReady: boolean; deepReady: boolean; stopped?: boolean }
interface Answer { quick: string[]; deep: string[]; quickMeta: string; deepMeta: string } // paragraphs use **bold** and [n] markers

interface AppState {
  tabs: Tab[]; active: number; query: string; sort: {key: 'name'|'date'|'size'|'kind'|'where'; dir: 1|-1; flipped?: boolean} | null;
  zone: 'finder' | 'evidence';
  messages: (Msg | AsstMsg)[]; draft: string; attach: {name: string; k: Kind}[]; curChat: string | null;
  ev: number; evOpen: boolean; evFb: Record<`${chatId}:${n}`, 'up'|'down'|null>;
  perms: Record<path, Perm>;           // explicit values only; everything else inherits
  idx: Record<path, boolean>;           // index include overrides
  idxRetry: Record<path, boolean>; idxFixed: Record<path, boolean>; trashed: Record<path, true>;
  contacts: Contact[]; requests: Request[]; inboxOpen: boolean; inboxTab: 'pending'|'done';
  ql: null | 'finder' | 'ev'; ctx: {x: number; y: number; path: string} | null; permMenu: path | null;
  permConfirm: {path: string; v: Perm; from: Perm|'mixed'; mixN: number; n: number} | null;
  ce: Contact | null; settings: boolean; settingsTab: 'general'|'models'|'sources'; model: ModelId;
  srcOff: Record<id, boolean>; histOpen: boolean; histQuery: string; toast: string | null; dragging: {name: string; k: Kind} | null;
}
```

### Business rules
- **Permission resolution** (`permOf`): walk from the full path up through its ancestors. The first explicit value wins. It counts as `explicit` only if it is set on the item itself. The default is `ask`.
- **Mixed** (`permMix`): a folder is mixed when any explicit descendant value differs from the folder's resolved value.
- **Evidence count:** the highest `[n]` in the latest assistant message, counting Quick always and Deep only once it is ready. Evidence tabs are `1…count`.
- **Generating:** true when the last message is from the assistant, is not stopped, and Quick is not ready (or, with parallel mode on, Deep is not ready). While generating, the composer button is Stop.
- **Version families** (`FAMS`): each family has a head path and members, and each member has a relation (exact, near, or contains). Members in the head's folder are hidden unless the stack is expanded.
- **Paths** are strings like `Documents/Finance/2026 Q3/파일.xlsx`. The top-level roots are the sidebar item names.

### Data and back end (to be built for real)
- File system and connector index (local + cloud + messaging), with per-item metadata: name, kind, modified date, size, index status and failure reason, and version family.
- Search API: instant filename search, plus the AI **Quick** and **Deep** streams. Both return paragraphs with citation indices and an ordered evidence list. Each evidence item has a kind, title, location, file, meta, excerpt blocks, and a preview payload.
- Agent-to-agent protocol for contacts: ask, requests, and approve/deny, with the Access rules enforced server-side.
- Stream Deep progress steps to the UI, and support cancellation (the Stop button).

## Assets
- There are no images. All icons are inline SVG strokes (Lucide-style, 24 × 24 viewBox, stroke-width 1.8–2.6, round caps). Use your icon library's equivalents: clock, plus, x, chevrons, search, bell, settings, eye, thumbs-up, pencil, bolt, lock, sparkle, arrow-up, rotate-cw, folder.
- Fonts: Inter (Google Fonts) and Pretendard (`cdn.jsdelivr.net/gh/orioncactus/pretendard`).
- Brand tiles for sources (Slack, Gmail, and so on) are placeholder letter tiles. Replace them with official logos if licensing allows.

## Files
- `AI Finder v6.dc.html`: the full interactive prototype. Markup is at the top; the mock data and all behavior logic are in the `<script data-dc-script>` block at the bottom.
- `support.js`: the prototype runtime, needed only to open the HTML locally. Do not port it.

Props used by the prototype (for demoing only):
- `showHotkeys` (boolean, default true): show or hide keycaps.
- `deepLatency` (1–12 s, default 4): simulated Deep answer time.
- `evidenceOpen` (boolean, default true): initial state of the Evidence panel.
