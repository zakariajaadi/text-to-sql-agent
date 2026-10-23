// Studio of the semantic layer, step 1: the tables, profiling and what the profile detects in each column.
// Everything comes from the API of offline/studio/app.py; the page keeps no state of its own but the
// chosen database, filter and selection.
// Two tables that read alike: the tables of the database (a status pill each), and the columns of the
// chosen table (counts, and a pill saying how the agent reads the column).

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("fr-FR");
const pct = (x) => (x == null ? "—" : `${fmt.format(Math.round(x * 100))} %`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const plural = (n, word) => `${fmt.format(n)} ${word}${n > 1 ? "s" : ""}`;

const ui = {
  dbs: [], db: null, data: null, filter: "all", query: "", selected: new Set(),
  current: null, detail: null, job: null, since: 0, polling: null, logOpen: true,
  clockOffset: 0,   // seconds the clock of the page is ahead of the server's
};

// The counters of a running job move every second, between two polls
setInterval(() => { if (ui.job?.running) renderJob(); }, 1000);

function setJob(job) {
  ui.job = job;
  ui.clockOffset = Date.now() / 1000 - job.now;
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* private window: nothing remembered */ }
}
function stored(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw Object.assign(new Error(body?.detail || response.statusText), { status: response.status });
  return body;
}

// ── Databases ────────────────────────────────────────────

async function init() {
  ui.dbs = await api("/api/databases");
  const saved = stored("studio.db");
  ui.db = ui.dbs.some((d) => d.name === saved) ? saved : ui.dbs[0]?.name;
  $("db-select").innerHTML = ui.dbs.map((d) =>
    `<option value="${esc(d.name)}" ${d.name === ui.db ? "selected" : ""}>${esc(d.name)}</option>`).join("");
  $("db-select").onchange = (e) => {
    ui.db = e.target.value; store("studio.db", ui.db);
    ui.selected.clear(); ui.current = null; ui.detail = null;
    renderDetail(); loadTables();
  };
  await loadTables();
  await pollJob();
}

async function loadTables(refresh = false) {
  const db = ui.db;
  $("refresh").classList.toggle("busy", refresh);
  try {
    const data = await api(`/api/db/${encodeURIComponent(db)}/tables${refresh ? "?refresh=true" : ""}`);
    if (db !== ui.db) return;
    ui.data = data;
  } catch (e) {
    showBanner(`Impossible de lire les tables : ${e.message}`, true);
    return;
  } finally {
    $("refresh").classList.remove("busy");
  }
  renderHeader(); renderRows();
}

function renderHeader() {
  const { data } = ui, s = data.summary;
  const when = new Date(data.read_at * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  $("refresh").title = data.source === "database"
    ? `Relire la base (${s.tables} tables lues à ${when})` : "Relire la base (tables lues dans les fichiers)";
  if (data.error) showBanner(`La base ne répond pas : les tables viennent de profile.yaml et tables.yaml. ${data.error}`);
  else $("banner").hidden = true;
  const counts = { all: s.tables, none: s.tables - s.profiled };
  [...$("filters").children].forEach((chip) => {
    const label = chip.textContent.replace(/\s*\d+$/, "");
    chip.innerHTML = `${label}<span class="count">${counts[chip.dataset.filter]}</span>`;
  });
}

function showBanner(text, error = false) {
  $("banner").textContent = text;
  $("banner").className = `banner${error ? " error" : ""}`;
  $("banner").hidden = false;
}

// ── Tables of the database ───────────────────────────────

function statusPill(table) {
  if (ui.job?.running && ui.job.db === ui.db && ui.job.current === table.name) return '<span class="pill running">en cours</span>';
  if (table.status.profile === "none") return '<span class="pill todo">à profiler</span>';
  return '<span class="pill done">profilée</span>';
}

function visibleRows() {
  const q = ui.query.toLowerCase();
  return (ui.data?.tables || []).filter((t) =>
    (!q || t.name.toLowerCase().includes(q)) && (ui.filter === "all" || t.status.profile === ui.filter));
}

function renderRows() {
  const rows = visibleRows();
  $("rows").innerHTML = rows.map((t) => `<div class="row${t.name === ui.current ? " current" : ""}" data-table="${esc(t.name)}">
      <input type="checkbox" ${ui.selected.has(t.name) ? "checked" : ""} aria-label="Sélectionner ${esc(t.name)}">
      <span class="cell" title="${esc(t.name)}">${esc(t.name)}</span>
      ${statusPill(t)}
    </div>`).join("");
  $("empty").hidden = rows.length > 0;
  $("select-all").checked = rows.length > 0 && rows.every((t) => ui.selected.has(t.name));
  renderActions();
  renderPersonal();
}

function renderActions() {
  $("run-selected").textContent = `Profiler la sélection (${ui.selected.size})`;
}

// Personal data: one rule for the whole database, not a correction to review table by table
function renderPersonal() {
  if (!ui.data) return;
  const { hidden, kept, visible } = ui.data.personal, bar = $("personal-bar");
  bar.hidden = !(hidden + kept + visible);
  if (bar.hidden) return;
  bar.innerHTML = visible
    ? `<span><b>${plural(visible, "donnée")} ${visible > 1 ? "personnelles" : "personnelle"}</b> ${visible > 1 ? "visibles" : "visible"} par l'agent</span>
       <button class="btn small primary" id="personal-all" data-hide="1">Cacher partout</button>`
    : `<span>Données personnelles cachées (${fmt.format(hidden)})${kept ? ` · ${fmt.format(kept)} gardée${kept > 1 ? "s" : ""} visible${kept > 1 ? "s" : ""}` : ""}</span>
       <button class="btn small ghost" id="personal-all" data-hide="0">Tout montrer</button>`;
  $("personal-all").onclick = async (e) => {
    try {
      ui.data = await api(`/api/db/${encodeURIComponent(ui.db)}/personal`,
        { method: "POST", body: { hide: e.target.dataset.hide === "1", version: ui.data.version } });
    } catch (error) {
      showBanner(error.status === 409 ? "tables.yaml a changé : relis la base avant de recommencer." : error.message, true);
      return;
    }
    renderHeader(); renderRows();
    if (ui.current) openTable(ui.current);
  };
}

$("rows").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row) return;
  const name = row.dataset.table;
  if (e.target.matches("input[type=checkbox]")) {
    e.target.checked ? ui.selected.add(name) : ui.selected.delete(name);
    $("select-all").checked = visibleRows().every((t) => ui.selected.has(t.name));
    renderActions();
    return;
  }
  openTable(name);
});

$("select-all").addEventListener("change", (e) => {
  visibleRows().forEach((t) => (e.target.checked ? ui.selected.add(t.name) : ui.selected.delete(t.name)));
  renderRows();
});
$("search").addEventListener("input", (e) => { ui.query = e.target.value; renderRows(); });
$("filters").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  ui.filter = chip.dataset.filter;
  [...$("filters").children].forEach((c) => c.classList.toggle("on", c === chip));
  renderRows();
});
$("refresh").addEventListener("click", () => loadTables(true));

// ── Columns of a table ───────────────────────────────────

async function openTable(name) {
  ui.current = name;
  renderRows();
  try {
    ui.detail = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(name)}`);
  } catch (e) {
    ui.detail = { name, error: e.message };
  }
  if (ui.current === name) renderDetail();
}

// "date dd/MM/yyyy HH:mm:ss" → "date jj/mm/aaaa hh:mm:ss", "decimal" → "nombre"
function formatLabel(format) {
  if (format === "decimal") return "nombre";
  return format.replace("yyyy", "aaaa").replace("dd", "jj").replace("MM", "mm").replace("HH", "hh");
}

const LOW_FILL = 0.8;   // a fill bar under it turns amber

// A pill of "Contenu détecté" that a click turns into an exception (POST .../columns)
const pillButton = (c, action, style, label, title) =>
  `<button class="pill ${style}" data-column="${esc(c.column)}" data-action="${action}" title="${esc(title)}">${esc(label)}</button>`;

// A format the profile detects: applied, kept as text on purpose, or not applied (a table reviewed before)
function formatPill(c, p) {
  const label = formatLabel(p.value), share = p.share != null ? `${pct(p.share)} des valeurs` : "la plupart des valeurs";
  if (p.state === "accepted") return pillButton(c, "text", "type", label, `Lue comme ${label} : ${share}. Cliquer pour la garder en texte.`);
  if (p.current) return `<span class="pill type" title="Format revu à la main (le profil détecte ${esc(label)})">${esc(formatLabel(p.current))}</span>`;
  if (p.state === "rejected") return pillButton(c, "format", "refused", label, `Détectée comme ${label} (${share}), gardée en texte. Cliquer pour la lire comme ${label}.`);
  return pillButton(c, "format", "type pending", label, `Détectée comme ${label} (${share}), pas encore appliquée. Cliquer pour l'appliquer.`);
}

// An empty column: hidden, kept visible on purpose, or not hidden yet
function emptyPill(c, p) {
  const label = c.filled ? "quasi vide" : "vide", filled = c.filled ? `remplie à ${pct(c.filled)}` : "jamais remplie";
  if (p.state === "accepted") return pillButton(c, "show", "hide", label, `Cachée à l'agent : ${filled}. Cliquer pour la laisser visible.`);
  if (p.state === "rejected") return pillButton(c, "hide", "refused", label, `${filled}, gardée visible exprès. Cliquer pour la cacher.`);
  return pillButton(c, "hide", "hide pending", label, `${filled}, pas encore cachée. Cliquer pour la cacher.`);
}

// Personal data, hidden in the whole database at once (personal bar): the pill makes the exception
const PERSONAL_PILLS = {
  hidden:  ["show", "donnée personnelle", "Cachée à l'agent. Cliquer pour la laisser visible."],
  kept:    ["hide", "donnée personnelle · visible", "Gardée visible exprès. Cliquer pour la cacher."],
  visible: ["hide", "donnée personnelle", "Visible par l'agent. Cliquer pour la cacher."],
};

// "Contenu détecté": what the data holds when it differs from the declared type, each pill an exception
function detected(c) {
  const parts = [];
  const format = c.proposals.find((p) => p.kind === "format");
  if (format) parts.push(formatPill(c, format));
  else if (c.format) parts.push(`<span class="pill type" title="Format écrit à la main dans tables.yaml">${esc(formatLabel(c.format))}</span>`);
  if (c.ambiguous) parts.push('<span class="muted-text" title="Aucune valeur ne dit si le jour vient avant le mois : lue en jj/mm, la convention des données vues jusqu\'ici.">jour/mois ?</span>');
  const empty = c.proposals.find((p) => p.kind === "exclude");
  if (empty) parts.push(emptyPill(c, empty));
  if (c.personal) {
    const [action, label, title] = PERSONAL_PILLS[c.personal];
    parts.push(pillButton(c, action, "personal", label, title));
  }
  return parts.join(" ") || '<span class="muted-text">—</span>';
}

function fillBar(filled) {
  if (filled == null) return '<span class="muted-text">—</span>';
  return `<span class="bar"><i class="${filled < LOW_FILL ? "low" : ""}" style="width:${Math.round(filled * 100)}%"></i></span>${pct(filled)}`;
}

function renderDetail(message) {
  const d = ui.detail, el = $("detail");
  if (!d) { el.innerHTML = '<p class="placeholder">Choisis une table pour voir ses colonnes.</p>'; return; }
  if (d.error) { el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2></div><p class="note">${esc(d.error)}</p>`; return; }
  if (!d.profiled) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="profile-one">Profiler</button></div>
      <p class="placeholder">Pas encore profilée${d.columns != null ? ` · ${plural(d.columns, "colonne")}` : ""}.</p>`;
    $("profile-one").onclick = () => runProfile({ tables: [d.name] });
    return;
  }

  // Figures of the profile, one line under the name (the key is said by its pill in the table)
  const figures = [plural(d.rows, "ligne"), plural(d.columns ?? d.column_facts.length, "colonne"),
    d.detected ? `${plural(d.detected, "format")} ${d.detected > 1 ? "détectés" : "détecté"}` : "aucun format détecté"];
  const error = message ? `<p class="note error">${esc(message)}</p>` : "";

  // Columns: declared type against what the profile found
  const lines = d.column_facts.map((c) => `<div class="line${c.excluded ? " hidden-col" : ""}">
      <span class="cell">${esc(c.column)}${c.key ? ` <span class="pill key">${c.key === "unique" ? "clé unique" : "clé"}</span>` : ""}${c.excluded ? ' <span class="pill hide">cachée</span>' : ""}</span>
      <span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>
      <span class="cell">${detected(c)}</span>
      <span class="fill">${fillBar(c.filled)}</span>
      <span class="num">${c.distinct == null ? "" : fmt.format(c.distinct)}</span>
    </div>`).join("");

  el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="figures">${figures.join(" · ")}</span>
      <span class="grow"></span><button class="btn ghost small" id="profile-again">Re-profiler</button></div>`
    + error
    + `<div class="columns">
        <div class="line head"><span>Colonne</span><span>Type en base</span><span>Contenu détecté</span><span class="num">Remplissage</span><span class="num">Distinctes</span></div>
        ${lines}
      </div>`;

  $("profile-again").onclick = () => runProfile({ tables: [d.name] });
  el.querySelectorAll("button.pill[data-action]").forEach((pill) => pill.addEventListener("click", () => changeColumn(pill)));
}

// An exception to what the profile detects, written into tables.yaml at once
async function changeColumn(pill) {
  const d = ui.detail;
  const body = { column: pill.dataset.column, action: pill.dataset.action, version: d.version };
  try {
    ui.detail = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(d.name)}/columns`, { method: "POST", body });
  } catch (e) {
    renderDetail(e.status === 409 ? "tables.yaml a changé depuis l'ouverture de la table : rouvre-la." : e.message);
    return;
  }
  renderDetail();
  loadTables();
}

// ── Jobs ─────────────────────────────────────────────────

async function runProfile(request) {
  try {
    setJob(await api(`/api/db/${encodeURIComponent(ui.db)}/profile`, { method: "POST", body: request }));
  } catch (e) {
    showBanner(e.message, true);
    return;
  }
  ui.since = 0;
  $("job-log").innerHTML = "";
  ui.selected.clear();
  ui.logOpen = true;
  appendLines(ui.job.lines); ui.since = ui.job.next_line;
  renderJob(); renderRows();
  schedulePoll();
}

$("run-selected").addEventListener("click", () => {
  if (!ui.selected.size) { showBanner("Coche d'abord les tables à profiler.", true); return; }
  runProfile({ tables: [...ui.selected] });
});
$("run-missing").addEventListener("click", () => runProfile({ missing: true }));
$("job-stop").addEventListener("click", async () => {
  // "Arrêt en cours" at once, the next poll confirms it
  Object.assign(ui.job, { stopped: true, stopped_at: Date.now() / 1000 - ui.clockOffset });
  renderJob();
  await api("/api/job/stop", { method: "POST" });
  schedulePoll();
});
$("job-toggle").addEventListener("click", () => { ui.logOpen = !ui.logOpen; renderJob(); });

async function pollJob() {
  const before = ui.job;
  let job;
  try { job = await api(`/api/job?since=${before ? ui.since : 0}`); } catch (e) { schedulePoll(); return; }
  if (!job) return;
  if (!before || before.id !== job.id) { $("job-log").innerHTML = ""; }
  appendLines(job.lines);
  ui.since = job.next_line;
  const progressed = !before || before.id !== job.id || before.done.length !== job.done.length
    || before.current !== job.current || before.running !== job.running;
  setJob(job);
  renderJob();
  if (progressed && job.db === ui.db) {
    await loadTables();
    // Only a table measured again gets its detail reloaded: an unsaved review of another one stays
    const wasDone = before?.id === job.id && before.done.includes(ui.current);
    if (ui.current && job.done.includes(ui.current) && !wasDone) openTable(ui.current);
  }
  if (job.running) schedulePoll();
}

function schedulePoll() {
  clearTimeout(ui.polling);
  ui.polling = setTimeout(pollJob, 1000);
}

function appendLines(lines) {
  const log = $("job-log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  log.insertAdjacentHTML("beforeend", lines.map((l) =>
    `<div class="${esc(l.level || "")}">${l.time ? `<span class="t">${esc(l.time)}</span> ` : ""}${esc(l.message)}</div>`).join(""));
  if (atBottom) log.scrollTop = log.scrollHeight;
}

// 12 s, 2 min 05 s
function duration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

const SLOW_SECONDS = 60;   // a table waiting longer: say the database is slow to answer

function renderJob() {
  const job = ui.job;
  if (!job) return;
  $("job").hidden = false;
  const now = Date.now() / 1000 - ui.clockOffset;   // the clock of the server
  const total = job.tables?.length;
  const progress = total ? `${job.done.length + job.failed.length}/${total}` : plural(job.done.length, "faite");
  let title, halo = "";
  if (job.running && job.stopped) {
    halo = "running";
    title = `Arrêt en cours : annulation de la requête sur la base… ${duration(now - job.stopped_at)}`;
  } else if (job.running && !job.current) {
    halo = "running";
    title = `Connexion à la base… ${duration(now - job.started)}`;
  } else if (job.running) {
    halo = "running";
    const waited = now - job.current_since;
    title = `Profiling de ${esc(job.db)} · <b>${esc(job.current)}</b> depuis ${duration(waited)} (${progress})`;
    if (waited > SLOW_SECONDS) title += ' · <span class="slow">la base met du temps à répondre</span>';
  } else if (job.stopped) {
    const query = job.cancel === "cancelled" ? "requête annulée sur la base"
      : job.cancel === "failed" || job.returncode === -9 ? '<span class="slow">la requête peut continuer sur la base</span>'
      : "aucune requête en cours";
    title = `Profiling arrêté · ${progress} · ${query}`;
    halo = "failed";
  } else if (job.returncode) {
    title = `Échec du profiling (code ${job.returncode}) · voir les logs`;
    halo = "failed";
  } else {
    title = `Profiling terminé · ${plural(job.done.length, "table")} ${job.done.length > 1 ? "mesurées" : "mesurée"}`;
  }
  if (job.failed.length) title += ` · <span style="color:var(--bad-ink)">échec : ${job.failed.map(esc).join(", ")}</span>`;
  $("job-title").innerHTML = title;
  $("job-halo").className = `halo ${halo}`;
  $("job-stop").hidden = !job.running || job.stopped;
  $("job-log").hidden = !ui.logOpen;
  $("job-toggle").textContent = ui.logOpen ? "Masquer les logs" : "Voir les logs";
}

init();
