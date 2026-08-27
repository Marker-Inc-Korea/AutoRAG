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

	const typeOptions = state.catalog
		.map((entry) => `<option value="${escapeHtml(entry.type)}">${escapeHtml(entry.title)}</option>`)
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
    <p class="help">One absolute path per line.</p>
    <div class="row actions"><button type="button" class="primary" id="save-folders">Save folders</button></div>
  </section>

  <h2>Connections</h2>
  ${cards || '<p class="empty">Nothing connected yet.</p>'}

  <h2>Add a source</h2>
  <form class="card" id="add">
    <label for="alias">Name</label>
    <input id="alias" name="alias" placeholder="work-github" required>
    <label for="type">Type</label>
    <select id="type" name="type">${typeOptions}</select>
    <p id="type-help" class="help"></p>
    <div id="fields"></div>
    <div class="row actions">
      <button type="submit" class="primary">Save connection</button>
      <button type="button" id="test-new">Test</button>
    </div>
  </form>
</main>
<script>
const catalog = ${JSON.stringify(state.catalog)};
const token = new URLSearchParams(location.search).get("token") || "";
function esc(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

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

function fieldValue(field) {
  const el = document.getElementById("f-" + field.key);
  if (!el) return undefined;
  if (field.kind === "checkbox") return el.checked;
  return el.value;
}

function connectorFromForm() {
  const type = document.getElementById("type").value;
  const entry = catalog.find((item) => item.type === type);
  const connector = {};
  let instanceId;
  for (const field of entry.fields) {
    const value = fieldValue(field);
    if (field.key === "instanceId") { instanceId = value; continue; }
    const key = field.key.startsWith("connector.") ? field.key.slice(10) : field.key;
    connector[key] = value;
  }
  return { type, alias: document.getElementById("alias").value, instanceId, connector, enabled: true };
}

function renderFields() {
  const type = document.getElementById("type").value;
  const entry = catalog.find((item) => item.type === type);
  document.getElementById("type-help").textContent = [entry.summary, entry.installHint].filter(Boolean).join(" ");
  document.getElementById("fields").innerHTML = entry.fields.map((field) => {
    const id = "f-" + field.key;
    if (field.kind === "textarea" || field.kind === "path-list") {
      return label(field, '<textarea id="' + id + '" placeholder="' + (field.placeholder || "") + '"></textarea>');
    }
    if (field.kind === "select") {
      const opts = (field.options || []).map((opt) => '<option value="' + opt.value + '">' + opt.label + "</option>").join("");
      return label(field, '<select id="' + id + '">' + opts + "</select>");
    }
    if (field.kind === "checkbox") {
      return '<label><input id="' + id + '" type="checkbox"> ' + field.label + "</label>";
    }
    const extra = field.kind === "path" || field.kind === "path-list"
      ? ' <button type="button" data-browse="' + id + '">Browse</button><div class="browse" id="b-' + id + '" hidden></div>'
      : "";
    return label(field, '<input id="' + id + '" placeholder="' + (field.placeholder || "") + '">' + extra);
  }).join("");
}

function label(field, control) {
  return "<label for=\\"f-" + field.key + "\\">" + field.label + "</label>" + control + (field.help ? '<p class="help">' + field.help + "</p>" : "");
}

document.getElementById("type").addEventListener("change", renderFields);
renderFields();

document.getElementById("save-folders").addEventListener("click", async () => {
  try {
    const searchPaths = document.getElementById("folders").value.split(/\\n/).map((line) => line.trim()).filter(Boolean);
    await api("/api/folders", { method: "POST", body: { searchPaths } });
    location.reload();
  } catch (error) { flash(error.message, true); }
});

document.getElementById("add").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/connections", { method: "POST", body: connectorFromForm() });
    location.reload();
  } catch (error) { flash(error.message, true); }
});

document.getElementById("test-new").addEventListener("click", async () => {
  try {
    const created = await api("/api/connections", { method: "POST", body: connectorFromForm() });
    const alias = created.connection && created.connection.alias || connectorFromForm().alias;
    const result = await api("/api/connections/" + encodeURIComponent(alias) + "/test", { method: "POST" });
    flash(result.detail || "Checked.", !result.ok);
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

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-browse]");
  if (!button) return;
  const id = button.getAttribute("data-browse");
  const input = document.getElementById(id);
  const box = document.getElementById("b-" + id);
  box.hidden = false;
  const path = input.value;
  try {
    const data = await api("/api/browse?path=" + encodeURIComponent(path || ""));
    const parent = data.parent ? '<button type="button" data-path="' + esc(data.parent) + '">..</button>' : "";
    box.innerHTML = parent + data.entries.filter((entry) => entry.directory).map((entry) =>
      '<button type="button" data-path="' + esc(entry.path) + '">' + esc(entry.name) + "</button>"
    ).join("");
    box.onclick = (inside) => {
      const next = inside.target.closest("[data-path]");
      if (!next) return;
      input.value = next.getAttribute("data-path");
      button.click();
    };
  } catch (error) { flash(error.message, true); }
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
