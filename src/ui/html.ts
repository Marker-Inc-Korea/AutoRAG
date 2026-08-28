import type { UiState } from "./config-store.ts";

export function renderUiPage(state: UiState): string {
	const folders = state.searchPaths.map((path) => escapeHtml(path)).join("\n");
	const cards = state.connections
		.map((connection) => {
			const status = connection.probe.ok ? "ok" : "warn";
			const enabled = connection.enabled ? "On" : "Off";
			return `<article class="card" data-alias="${escapeHtml(connection.alias)}">
  <div class="row">
    <div>
      <h3>${escapeHtml(connection.alias)}</h3>
      <p class="muted">${escapeHtml(titleFor(state, connection.type))} · ${escapeHtml(connection.type)}</p>
    </div>
    <span class="pill ${status}">${escapeHtml(connection.probe.status)}</span>
  </div>
  <p>${escapeHtml(connection.probe.detail)}</p>
  <div class="row actions">
    <button type="button" data-act="toggle" data-enabled="${connection.enabled ? "false" : "true"}">${enabled === "On" ? "Disable" : "Enable"}</button>
    <button type="button" data-act="test">Test</button>
    <button type="button" class="danger" data-act="remove">Remove</button>
  </div>
</article>`;
		})
		.join("\n");

	const pickerButtons = state.picker
		.map(
			(entry) =>
				`<button type="button" class="pick" data-type="${escapeHtml(entry.type)}" aria-pressed="false"><strong>${escapeHtml(entry.title)}</strong><small>${escapeHtml(entry.summary)}</small></button>`,
		)
		.join("");

	return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoRAG data sources</title>
  <style>
    :root { --ink:#1c1917; --muted:#57534e; --bg:#f6f5f1; --card:#fff; --line:#e7e5e4; --ok:#0f766e; --warn:#b45309; --danger:#b91c1c; }
    * { box-sizing: border-box; }
    body { margin:0; font: 16px/1.45 ui-sans-serif, system-ui, -apple-system, sans-serif; color:var(--ink); background:var(--bg); }
    main { max-width: 720px; margin: 0 auto; padding: 32px 20px 80px; }
    h1 { font-size: 28px; letter-spacing: -0.03em; margin: 0 0 4px; }
    h2 { font-size: 18px; margin: 32px 0 12px; }
    h3 { margin: 0; font-size: 16px; }
    p { margin: 8px 0; }
    .muted { color: var(--muted); }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; margin: 0 0 12px; }
    .row { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .actions { margin-top: 12px; }
    button, select, input, textarea { font: inherit; }
    button { border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 6px 12px; cursor: pointer; }
    button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
    button.primary { background: var(--ink); color: #fff; border-color: var(--ink); }
    button.danger { color: var(--danger); }
    .pill { font-size: 12px; border-radius: 999px; padding: 3px 8px; border: 1px solid var(--line); }
    .pill.ok { color: var(--ok); border-color: #99f6e4; background: #f0fdfa; }
    .pill.warn { color: var(--warn); border-color: #fde68a; background: #fffbeb; }
    label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 4px; }
    input, select, textarea { width: 100%; border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; background: #fff; }
    textarea { min-height: 88px; }
    .help { font-size: 12px; color: var(--muted); margin: 4px 0 0; }
    .flash { display: none; padding: 10px 12px; border-radius: 10px; background: #ecfdf5; color: #065f46; margin-bottom: 12px; }
    .flash.show { display: block; }
    .flash.bad { background: #fef2f2; color: #991b1b; }
    .browse { font-size: 13px; max-height: 180px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; margin-top: 8px; background: #fff; }
    .browse button { display: block; width: 100%; text-align: left; border: 0; border-radius: 0; padding: 6px 10px; }
    .empty { color: var(--muted); padding: 8px 0; }
    .picker { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    button.pick { display: block; width: 100%; text-align: left; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; cursor: pointer; background: #fff; }
    button.pick strong { display: block; font-size: 14px; font-weight: 650; }
    button.pick small { display: block; color: var(--muted); font-size: 12px; margin-top: 4px; font-weight: 400; }
    button.pick.on, button.pick[aria-pressed="true"] { border-color: var(--ink); box-shadow: 0 0 0 1px var(--ink); background: #fafaf9; }
    #handoff[hidden] { display: none; }
    #prompt { min-height: 140px; font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    @media (max-width: 640px) { .picker { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <p class="muted">AutoRAG</p>
  <h1>Data sources</h1>
  <p class="muted">Connect folders and apps on this computer. Tokens stay in environment variables — they are never written to config.</p>
  <div id="flash" class="flash"></div>

  <h2>Local folders</h2>
  <section class="card">
    <label for="folders">Folders to search</label>
    <textarea id="folders">${folders}</textarea>
    <p class="help">BM25, MinSync, and Jikji index these folders by default. One absolute path per line.</p>
    <div class="row actions"><button type="button" class="primary" id="save-folders">Save folders</button></div>
  </section>

  <h2>Connections</h2>
  ${cards || '<p class="empty">Nothing connected yet.</p>'}

  <h2>Add a source</h2>
  <section class="card">
    <p class="muted">Choose the app or source. Then give it a name and say what’s in it.</p>
    <div class="picker" id="picker">${pickerButtons}</div>
    <div id="handoff" hidden>
      <label for="alias">Name</label>
      <input id="alias" placeholder="family-kakao">
      <label for="note">Description</label>
      <input id="note" placeholder="Family group chat — travel plans and reimbursements">
      <div id="extras"></div>
      <label for="prompt">Copy this into your coding agent</label>
      <textarea id="prompt" readonly></textarea>
      <p class="help">Paste it into the coding agent you use for setup. It will register the connection in AutoRAG config.</p>
      <div class="row actions">
        <button type="button" class="primary" id="copy-prompt">Copy for coding agent</button>
      </div>
    </div>
  </section>
</main>
<script>
const token = new URLSearchParams(location.search).get("token") || "";
const picker = ${JSON.stringify(state.picker)};
let selectedType = "";

function flash(message, bad) {
  const el = document.getElementById("flash");
  el.textContent = message;
  el.className = "flash show" + (bad ? " bad" : "");
}

async function api(path, opts) {
  const headers = { "x-autorag-token": token, ...(opts && opts.body ? { "content-type": "application/json" } : {}) };
  const res = await fetch(path, { ...opts, headers, body: opts && opts.body ? JSON.stringify(opts.body) : undefined });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { message: text }; }
  if (!res.ok) throw new Error(data.error || data.message || res.statusText);
  return data;
}

function extraValues() {
  const extras = {};
  document.querySelectorAll("[data-extra]").forEach((el) => {
    extras[el.getAttribute("data-extra")] = el.value.trim();
  });
  return extras;
}

async function refreshPrompt() {
  if (!selectedType) return;
  const params = new URLSearchParams({ type: selectedType });
  const alias = document.getElementById("alias").value.trim();
  const note = document.getElementById("note").value.trim();
  if (alias) params.set("alias", alias);
  if (note) params.set("note", note);
  const extras = extraValues();
  if (Object.keys(extras).length > 0) params.set("extras", JSON.stringify(extras));
  try {
    const data = await api("/api/prompt?" + params.toString());
    document.getElementById("prompt").value = data.prompt || "";
  } catch (error) { flash(error.message, true); }
}

function attr(value) {
  return String(value || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}

async function renderExtras(type) {
  const entry = picker.find((item) => item.type === type);
  const root = document.getElementById('extras');
  if (!entry || !entry.extras || entry.extras.length === 0) {
    root.innerHTML = '';
    return;
  }
  let choices = { rcloneRemotes: [], mailAccounts: [] };
  try { choices = await api('/api/choices?type=' + encodeURIComponent(type)); } catch (error) { /* keep empty */ }
  root.innerHTML = entry.extras.map((extra) => {
    const id = 'extra-' + extra.key;
    const label = '<label for="' + id + '">' + extra.label + '</label>';
    if (extra.kind === 'textarea') {
      return label + '<textarea id="' + id + '" data-extra="' + extra.key + '" placeholder="' + attr(extra.placeholder) + '"></textarea>' + (extra.help ? '<p class="help">' + extra.help + '</p>' : '');
    }
    if (extra.kind === 'select') {
      const list = extra.choices === 'rclone-remotes' ? (choices.rcloneRemotes || []) : extra.choices === 'mail-accounts' ? (choices.mailAccounts || []) : [];
      const opts = ['<option value="">Skip for now</option>'].concat(list.map((item) => '<option value="' + attr(item.value) + '">' + item.label + '</option>'));
      if (extra.allowOther && !list.some((item) => item.value === 'other')) opts.push('<option value="other">Other</option>');
      return label + '<select id="' + id + '" data-extra="' + extra.key + '">' + opts.join('') + '</select><input id="' + id + '-other" data-extra="' + extra.key + 'Other" placeholder="Other" hidden>';
    }
    return label + '<input id="' + id + '" data-extra="' + extra.key + '" placeholder="' + attr(extra.placeholder) + '">';
  }).join('');
  root.querySelectorAll('[data-extra]').forEach((el) => el.addEventListener('input', refreshPrompt));
  root.querySelectorAll('select[data-extra]').forEach((el) => {
    el.addEventListener('change', () => {
      const other = document.getElementById(el.id + '-other');
      if (other) other.hidden = el.value !== 'other';
      refreshPrompt();
    });
  });
}

async function selectType(type, button) {
  if (!type) return;
  selectedType = type;
  document.querySelectorAll("button.pick").forEach((el) => {
    const on = el === button;
    el.classList.toggle("on", on);
    el.setAttribute("aria-pressed", on ? "true" : "false");
  });
  document.getElementById("handoff").hidden = false;
  const alias = document.getElementById("alias");
  if (!alias.value.trim()) alias.placeholder = type;
  await renderExtras(type);
  refreshPrompt();
}

document.getElementById("picker").addEventListener("click", (event) => {
  const pick = event.target.closest("button.pick");
  if (!pick) return;
  selectType(pick.getAttribute("data-type") || "", pick);
});

document.getElementById("alias").addEventListener("input", refreshPrompt);
document.getElementById("note").addEventListener("input", refreshPrompt);

document.getElementById("copy-prompt").addEventListener("click", async () => {
  const text = document.getElementById("prompt").value;
  if (!text) { flash("Pick a source type first.", true); return; }
  try {
    await navigator.clipboard.writeText(text);
    flash("Prompt copied.");
  } catch {
    document.getElementById("prompt").select();
    flash("Select the prompt and copy it.", true);
  }
});

document.getElementById("save-folders").addEventListener("click", async () => {
  try {
    const searchPaths = document.getElementById("folders").value.split(String.fromCharCode(10)).map((line) => line.trim()).filter(Boolean);
    await api("/api/folders", { method: "POST", body: { searchPaths } });
    location.reload();
  } catch (error) { flash(error.message, true); }
});

document.querySelectorAll("article.card").forEach((card) => {
  card.addEventListener("click", async (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const alias = card.getAttribute("data-alias");
    try {
      if (button.dataset.act === "remove") {
        await api("/api/connections/" + encodeURIComponent(alias), { method: "DELETE" });
        location.reload();
      } else if (button.dataset.act === "toggle") {
        await api("/api/connections/" + encodeURIComponent(alias) + "/toggle", { method: "POST", body: { enabled: button.dataset.enabled === "true" } });
        location.reload();
      } else if (button.dataset.act === "test") {
        const result = await api("/api/connections/" + encodeURIComponent(alias) + "/test", { method: "POST" });
        flash(result.detail || "Checked.", !result.ok);
      }
    } catch (error) { flash(error.message, true); }
  });
});
</script>
</body>
</html>`;
}

function titleFor(state: UiState, type: string): string {
	return state.catalog.find((entry) => entry.type === type)?.title ?? type;
}

function escapeHtml(value: string): string {
	return value.replace(/[&<>"']/g, (char) => {
		switch (char) {
			case "&":
				return "&amp;";
			case "<":
				return "&lt;";
			case ">":
				return "&gt;";
			case '"':
				return "&quot;";
			case "'":
				return "&#39;";
			default:
				return char;
		}
	});
}
