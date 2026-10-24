// Studio of the semantic layer: one screen per step of the pipeline (STEPS), the same layout for each:
// the tables of the database on the left (a status pill each), the chosen table on the right.
//   Profiling: what the profile detects in each column (declared type against the data, exceptions)
//   Semantics: the descriptions the LLM writes, read and corrected in place
// Everything comes from the API of offline/studio/app.py; the page keeps no state of its own but the
// chosen database, step, filter and selection.

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("fr-FR");
const pct = (x) => (x == null ? "—" : `${fmt.format(Math.round(x * 100))} %`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const plural = (n, word) => `${fmt.format(n)} ${word}${n > 1 ? "s" : ""}`;

// The steps the studio has a screen for: their title, sentence, filter and actions
const STEPS = {
  profile: {
    number: 1, title: "Profiling", key: "profile", todo: "À profiler",
    text: "On examine le contenu réel de chaque table pour que l'agent la lise correctement : les colonnes vides, les dates ou montants enregistrés comme du texte, les données personnelles.",
    primary: (n) => `Profiler la sélection (${n})`, secondary: () => "Profiler les manquantes",
  },
  semantics: {
    number: 2, title: "Semantics", key: "semantics", todo: "À décrire",
    text: "L'agent choisit ses tables et écrit ses requêtes à partir de ces descriptions. L'IA les rédige à partir de la structure des tables et de leur profil ; tu les relis et tu les corriges ici.",
    primary: () => "Décrire les manquantes", secondary: (n) => `Redécrire la sélection (${n})`,
  },
  relations: {
    number: 3, title: "Relations", key: "relations", todo: "Isolées",
    text: "L'agent passe d'une table à l'autre grâce à ces liens. Chacun est vérifié sur les données : la part des valeurs d'une table qu'on retrouve dans l'autre.",
    primary: () => "Chercher les relations", secondary: () => "Ajouter une relation",
  },
};

const ui = {
  dbs: [], db: null, step: "profile", data: null, filter: "all", query: "", selected: new Set(),
  rel: null, path: { a: null, b: null, result: null }, adding: false,   // the Relations screen
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
    ui.selected.clear(); ui.current = null; ui.detail = null; ui.rel = null; ui.path = { a: null, b: null, result: null };
    renderDetail(); loadTables().then(() => ui.step === "relations" && loadRelations());
  };
  ui.step = STEPS[location.hash.slice(1)] ? location.hash.slice(1) : STEPS[stored("studio.step")] ? stored("studio.step") : "profile";
  renderStep();
  await loadTables();
  if (ui.step === "relations") await loadRelations();
  await pollJob();
}

// The title, sentence, filter and actions of the current step; its dot filled in the stepper
function renderStep() {
  const step = STEPS[ui.step];
  $("step-number").textContent = `Étape ${step.number} sur 5`;
  $("step-title").textContent = step.title;
  $("step-text").textContent = step.text;
  document.title = `${step.title} · Studio sémantique`;
  [...$("stepper").children].forEach((li) => {
    li.classList.toggle("on", li.dataset.step === ui.step);
    li.toggleAttribute("aria-current", li.dataset.step === ui.step);
  });
  document.body.dataset.step = ui.step;
  renderActions();
}

function goTo(step) {
  if (!STEPS[step] || step === ui.step) return;
  ui.step = step; store("studio.step", step); history.replaceState(null, "", `#${step}`);
  ui.filter = "all"; ui.selected.clear(); ui.detail = null; ui.adding = false;
  [...$("filters").children].forEach((c) => c.classList.toggle("on", c.dataset.filter === "all"));
  renderStep();
  if (step === "relations") { loadRelations(); return; }
  if (ui.data) { renderHeader(); renderRows(); }
  if (ui.current) openTable(ui.current); else renderDetail();
}

$("stepper").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-step]");
  if (li) goTo(li.dataset.step);
});

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
  const todo = { profile: s.tables - s.profiled, semantics: s.tables - (s.described - s.partial),
    relations: ui.data.tables.filter((t) => t.status.relations !== "done").length }[ui.step];
  const counts = { all: s.tables, todo };
  const labels = { all: "Toutes", todo: STEPS[ui.step].todo };
  [...$("filters").children].forEach((chip) => {
    chip.innerHTML = `${labels[chip.dataset.filter]}<span class="count">${counts[chip.dataset.filter]}</span>`;
  });
}

function showBanner(text, error = false) {
  $("banner").textContent = text;
  $("banner").className = `banner${error ? " error" : ""}`;
  $("banner").hidden = false;
}

// ── Tables of the database ───────────────────────────────

function statusPill(table) {
  const job = ui.job;
  if (job?.running && job.db === ui.db && job.step === ui.step
      && (job.step === "profile" ? job.current === table.name : job.started_tables.includes(table.name))) {
    return '<span class="pill running">en cours</span>';
  }
  const status = table.status[STEPS[ui.step].key];
  if (ui.step === "profile") return status === "none" ? '<span class="pill todo">à profiler</span>' : '<span class="pill done">profilée</span>';
  if (ui.step === "relations") {
    const n = ui.rel?.counts[table.name] ?? 0;
    return n ? `<span class="pill done">${plural(n, "lien")}</span>`
      : '<span class="pill decide" title="Aucune relation : l\'agent ne peut joindre cette table à aucune autre">isolée</span>';
  }
  if (status === "none") return '<span class="pill todo">à décrire</span>';
  if (status === "partial") return `<span class="pill decide" title="Colonnes ajoutées depuis la dernière description">${table.undescribed} col. à décrire</span>`;
  return '<span class="pill done">décrite</span>';
}

function visibleRows() {
  const q = ui.query.toLowerCase(), key = STEPS[ui.step].key;
  return (ui.data?.tables || []).filter((t) =>
    (!q || t.name.toLowerCase().includes(q)) && (ui.filter === "all" || t.status[key] !== "done"));
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
  $("run-primary").textContent = STEPS[ui.step].primary(ui.selected.size);
  $("run-secondary").textContent = STEPS[ui.step].secondary(ui.selected.size);
}

// Personal data: one rule for the whole database, not a correction to review table by table
function renderPersonal() {
  if (!ui.data) return;
  const { hidden, kept, visible } = ui.data.personal, bar = $("personal-bar");
  bar.hidden = ui.step !== "profile" || !(hidden + kept + visible);
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
  if (ui.step === "relations") { selectOnMap(name); return; }
  const step = ui.step, path = step === "profile" ? "" : "/semantics";
  try {
    ui.detail = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(name)}${path}`);
  } catch (e) {
    ui.detail = { name, error: e.message };
  }
  if (ui.current === name && ui.step === step) renderDetail();
}

function renderDetail(message) {
  const d = ui.detail, el = $("detail");
  if (ui.step === "relations") { renderRelations(el, message); return; }
  if (!d) { el.innerHTML = '<p class="placeholder">Choisis une table pour voir ses colonnes.</p>'; return; }
  if (d.error) { el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2></div><p class="note">${esc(d.error)}</p>`; return; }
  if (ui.step === "profile") renderProfile(d, el, message); else renderSemantics(d, el, message);
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

function renderProfile(d, el, message) {
  if (!d.profiled) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="profile-one">Profiler</button></div>
      <p class="placeholder">Pas encore profilée${d.columns != null ? ` · ${plural(d.columns, "colonne")}` : ""}.</p>`;
    $("profile-one").onclick = () => runJob("profile", { tables: [d.name] });
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

  $("profile-again").onclick = () => runJob("profile", { tables: [d.name] });
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

// ── Descriptions of a table (Semantics) ──────────────────

function semanticsFigures(d) {
  const c = d.counts;
  return `${plural(c.columns, "colonne")} · ${fmt.format(c.described)}/${fmt.format(c.visible)} décrites · `
    + (c.categorical ? `${fmt.format(c.categorical)} catégorielle${c.categorical > 1 ? "s" : ""}` : "aucune catégorielle");
}

function renderSemantics(d, el, message) {
  if (!d.described) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="describe-one">Décrire</button></div>
      <p class="placeholder">Pas encore décrite : l'IA rédigera la description de la table et de ses colonnes.</p>`;
    $("describe-one").onclick = () => runJob("semantics", { tables: [d.name] });
    return;
  }
  const lines = d.columns.map((c) => {
    const name = `${esc(c.column)}${c.key ? ` <span class="pill key">${c.key === "unique" ? "clé unique" : "clé"}</span>` : ""}`;
    if (c.excluded) {
      return `<div class="sline hidden-col"><span class="cell">${name} <span class="pill hide">cachée</span></span>
        <span class="cell mono">${esc((c.type || "—").toLowerCase())}</span>
        <span class="muted-text">cachée à l'agent : pas de description</span><span></span></div>`;
    }
    const values = c.distinct == null ? "" : `<span class="values" title="${c.sampled ? "Valeurs distinctes sur l'échantillon du profil" : "Valeurs distinctes de la table"}">${plural(c.distinct, "valeur")}</span>`;
    return `<div class="sline">
      <span class="cell">${name}</span>
      <span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>
      <textarea class="edit" rows="1" data-field="column" data-column="${esc(c.column)}" placeholder="Ajouter une description…" aria-label="Description de ${esc(c.column)}">${esc(c.description)}</textarea>
      <label class="switch-cell">${values}<input type="checkbox" class="switch" data-field="categorical" data-column="${esc(c.column)}" ${c.categorical ? "checked" : ""} aria-label="${esc(c.column)} catégorielle"></label>
    </div>`;
  }).join("");

  el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="figures" id="sem-figures">${semanticsFigures(d)}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      <button class="btn ghost small" id="describe-again" title="L'IA réécrit toutes les descriptions de cette table">Redécrire</button></div>
    ${message ? `<p class="note error">${esc(message)}</p>` : ""}
    <div class="table-text">
      <label class="field-label" for="sem-description">Description de la table</label>
      <textarea class="edit block" id="sem-description" rows="2" data-field="description">${esc(d.description)}</textarea>
      <div class="as-of">
        <label class="field-label" for="sem-as-of">Données arrêtées au</label>
        <input type="date" class="edit date" id="sem-as-of" data-field="data_as_of" value="${esc(d.data_as_of)}">
        <span class="muted-text">la date de référence de « ce mois-ci », « l'an dernier »…</span>
      </div>
    </div>
    <div class="columns">
      <div class="sline head"><span>Colonne</span><span>Type</span><span>Description</span><span class="num">Catégorielle</span></div>
      ${lines}
    </div>`;

  $("describe-again").onclick = () => runJob("semantics", { tables: [d.name], fresh: true });
  el.querySelectorAll("textarea.edit").forEach((area) => {
    fitHeight(area);
    area.addEventListener("input", () => fitHeight(area));
    area.addEventListener("change", () => saveSemantics(area, area.value));
  });
  $("sem-as-of").addEventListener("change", (e) => saveSemantics(e.target, e.target.value));
  el.querySelectorAll("input.switch").forEach((box) => box.addEventListener("change", () => saveSemantics(box, box.checked)));
}

// A description grows with its text, no scroll inside it
function fitHeight(area) {
  area.style.height = "auto";
  area.style.height = `${area.scrollHeight}px`;
}

// One field written into tables.yaml as soon as it is left: the page is not re-drawn, the focus stays
async function saveSemantics(input, value) {
  const d = ui.detail, state = $("save-state");
  const body = { field: input.dataset.field, column: input.dataset.column || null, value, version: d.version };
  state.textContent = "Enregistrement…";
  state.className = "save-state";
  try {
    const saved = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(d.name)}/semantics`, { method: "POST", body });
    if (ui.detail?.name !== d.name) return;
    Object.assign(ui.detail, { version: saved.version, counts: saved.counts });
    $("sem-figures").textContent = semanticsFigures(saved);
    state.textContent = "Enregistré";
    state.className = "save-state ok";
    loadTables();
  } catch (e) {
    state.textContent = e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message;
    state.className = "save-state error";
  }
}

// ── Relations: the map of the tables, the path of the agent, the evidence ──

const ORIGINS = { profile: ["mesurée", "Clé de même nom dont les valeurs se retrouvent (profil)"],
  llm: ["IA", "Proposée par l'IA à partir des noms et des descriptions"],
  manual: ["à la main", "Ajoutée dans le studio"] };

async function loadRelations() {
  const db = ui.db;
  try {
    const rel = await api(`/api/db/${encodeURIComponent(db)}/relations`);
    if (db !== ui.db) return;
    ui.rel = rel;
  } catch (e) {
    showBanner(`Impossible de lire les relations : ${e.message}`, true);
    return;
  }
  if (ui.data) { renderHeader(); renderRows(); }
  if (ui.step === "relations") renderDetail();
}

function selectOnMap(name) {
  ui.current = name;
  if (!ui.path.a || ui.path.a === name) ui.path = { a: name, b: ui.path.b === name ? null : ui.path.b, result: null };
  renderDetail();
  if (ui.path.a && ui.path.b) findPath();
}

// The SVG of the map: a box per table, a line per relation, from the table pointing (above) to the one it points to
function mapSvg() {
  const { nodes, width, height } = ui.rel.layout, cur = ui.current;
  const onPath = new Set((ui.path.result?.layers || []).flat());
  const layer = new Map((ui.path.result?.layers || []).flatMap((tables, i) => tables.map((t) => [t, i])));
  const near = new Set(ui.rel.relations.filter((r) => r.source === cur || r.target === cur).flatMap((r) => [r.source, r.target]));
  const edges = ui.rel.relations.filter((r) => r.source !== r.target && nodes[r.source] && nodes[r.target]).map((r) => {
    const s = nodes[r.source], t = nodes[r.target];
    const [x1, y1, x2, y2] = [s.x + s.w / 2, s.y + s.h, t.x + t.w / 2, t.y];
    const down = y2 > y1, my = (y1 + y2) / 2;
    const d = down ? `M${x1} ${y1} C${x1} ${my} ${x2} ${my} ${x2} ${y2}` : `M${x1} ${s.y + s.h / 2} L${x2} ${t.y + t.h / 2}`;
    const inPath = layer.has(r.source) && layer.has(r.target) && Math.abs(layer.get(r.source) - layer.get(r.target)) === 1;
    const cls = ["e", r.weak ? "weak" : "", r.found == null ? "unchecked" : "", cur && (r.source === cur || r.target === cur) ? "on" : "",
      cur && !near.has(r.source) && r.source !== cur ? "dim" : "", inPath ? "path" : ""].join(" ");
    return `<path class="${cls}" d="${d}"><title>${esc(r.from)} → ${esc(r.to)}${r.found != null ? ` · ${pct(r.found)} des valeurs trouvées` : ""}</title></path>`;
  }).join("");
  const isolated = new Set(ui.rel.isolated);
  const boxes = Object.entries(nodes).map(([table, n]) => {
    const cls = ["node", table === cur ? "on" : "", near.has(table) && table !== cur ? "near" : "", isolated.has(table) ? "iso" : "",
      onPath.has(table) ? "path" : "", cur && table !== cur && !near.has(table) && !onPath.has(table) ? "dim" : ""].join(" ");
    return `<g class="${cls}" data-table="${esc(table)}" tabindex="0" role="button" aria-label="${esc(table)}">
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="7"/>
      <text x="${n.x + n.w / 2}" y="${n.y + n.h / 2 + 4}" text-anchor="middle">${esc(table)}</text></g>`;
  }).join("");
  return `<svg class="map" viewBox="0 0 ${width} ${height}" style="min-width:${Math.round(width * 0.7)}px" role="img"
      aria-label="Carte des tables et de leurs relations">${edges}${boxes}</svg>`;
}

function pathText() {
  const p = ui.path;
  if (!p.a || !p.b) return '<span class="muted-text">choisis deux tables pour voir le chemin de l\'agent</span>';
  if (!p.result) return '<span class="muted-text">…</span>';
  if (p.result.joins == null) return '<span class="slow">aucun chemin : ces deux tables ne sont reliées par aucune suite de relations</span>';
  const steps = p.result.layers.map((tables) => tables.map(esc).join(" | ")).join(" → ");
  const joins = plural(p.result.joins, "jointure");
  return `<span class="path-steps">${steps}</span> <span class="muted-text">· ${joins}</span>`
    + (p.result.bridged ? "" : ` <span class="slow">· au-delà de 3 : l'agent ne fera pas ce chemin seul</span>`);
}

function relationLine(r) {
  const [origin, title] = ORIGINS[r.origin] || ["—", "Écrite avant que le studio note l'origine"];
  const found = r.found == null ? '<span class="muted-text" title="Pas encore vérifiée sur les données : Revérifier">non vérifiée</span>'
    : `<span class="bar"><i class="${r.weak ? "low" : ""}" style="width:${Math.round(r.found * 100)}%"></i></span>${pct(r.found)}`;
  return `<div class="rline${r.weak ? " weak" : ""}">
    <span class="cell mono" title="${esc(r.from)} → ${esc(r.to)}">${esc(r.from)} <span class="arrow">→</span> ${esc(r.to)}</span>
    <span class="fill">${found}</span>
    <span class="cell mono muted" title="Combien de lignes d'un côté pour une ligne de l'autre">${esc(r.links)}</span>
    <span class="cell muted" title="${title}">${origin}</span>
    <button class="icon-btn small" data-ignore="${esc(r.from)}|${esc(r.to)}" title="Retirer : la relation va dans ignore et ne reviendra pas" aria-label="Retirer ${esc(r.from)} vers ${esc(r.to)}">✕</button>
  </div>`;
}

function renderRelations(el, message) {
  const rel = ui.rel;
  if (!rel) { el.innerHTML = '<p class="placeholder">Lecture des relations…</p>'; return; }
  const cur = ui.current, tables = ui.data?.tables.map((t) => t.name) || [];
  const shown = (cur ? rel.relations.filter((r) => r.source === cur || r.target === cur) : rel.relations)
    .slice().sort((a, b) => (b.weak - a.weak) || ((a.found == null) - (b.found == null)));
  const ignored = cur ? rel.ignored.filter((r) => r.source === cur || r.target === cur) : rel.ignored;
  const figures = [plural(rel.relations.length, "relation"),
    rel.isolated.length ? `${plural(rel.isolated.length, "table")} ${rel.isolated.length > 1 ? "isolées" : "isolée"}` : "aucune table isolée",
    rel.weak ? `${fmt.format(rel.weak)} à vérifier` : null, rel.unchecked ? `${fmt.format(rel.unchecked)} non vérifiée${rel.unchecked > 1 ? "s" : ""}` : null]
    .filter(Boolean).join(" · ");
  const options = (value) => `<option value="">—</option>` + tables.map((t) => `<option ${t === value ? "selected" : ""}>${esc(t)}</option>`).join("");

  el.innerHTML = `<div class="detail-head"><h2>Carte des tables</h2><span class="figures">${figures}</span>
      <span class="grow"></span><span class="save-state" id="rel-state">${message ? esc(message) : ""}</span>
      <button class="btn ghost small" id="check-relations" title="Mesure de nouveau la part des valeurs trouvées de chaque relation, sans l'IA">Revérifier</button></div>
    <div class="map-wrap">${mapSvg()}</div>
    <div class="path-bar">
      <span class="muted-text">Chemin de</span><select id="path-a" aria-label="Table de départ">${options(ui.path.a)}</select>
      <span class="muted-text">à</span><select id="path-b" aria-label="Table d'arrivée">${options(ui.path.b)}</select>
      <span class="path-result">${pathText()}</span>
    </div>
    <form class="add-form" id="add-form" ${ui.adding ? "" : "hidden"}>
      <span class="muted-text">Nouvelle relation</span>
      <select id="add-source-table" aria-label="Table qui pointe">${options(cur)}</select><select id="add-source-column" aria-label="Colonne qui pointe"></select>
      <span class="arrow">→</span>
      <select id="add-target-table" aria-label="Table pointée">${options(null)}</select><select id="add-target-column" aria-label="Colonne pointée"></select>
      <button class="btn small primary" type="submit">Vérifier et ajouter</button>
      <button class="btn small ghost" type="button" id="add-cancel">Annuler</button>
    </form>
    <div class="columns">
      <div class="rline head"><span>${cur ? `Relations de ${esc(cur)}` : "Toutes les relations"}</span><span class="num">Valeurs trouvées</span><span>Liens</span><span>Origine</span><span></span></div>
      ${shown.map(relationLine).join("") || `<p class="placeholder">${cur ? "Aucune relation : l'agent ne peut joindre cette table à aucune autre." : "Aucune relation."}</p>`}
      ${ignored.length ? `<details class="ignored"><summary>${plural(ignored.length, "relation retirée")}</summary>
        ${ignored.map((r) => `<div class="rline"><span class="cell mono muted">${esc(r.from)} → ${esc(r.to)}</span><span></span><span></span><span></span>
          <button class="btn ghost small" data-restore="${esc(r.from)}|${esc(r.to)}">Rétablir</button></div>`).join("")}</details>` : ""}
    </div>`;

  el.querySelectorAll(".map .node").forEach((g) => {
    g.addEventListener("click", () => openTable(g.dataset.table));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter") openTable(g.dataset.table); });
  });
  $("path-a").onchange = (e) => { ui.path.a = e.target.value || null; ui.path.result = null; findPath(); };
  $("path-b").onchange = (e) => { ui.path.b = e.target.value || null; ui.path.result = null; findPath(); };
  $("check-relations").onclick = () => runJob("relations/run", { check: true });
  el.querySelectorAll("[data-ignore]").forEach((b) => b.addEventListener("click", () => moveRelation(b.dataset.ignore, false)));
  el.querySelectorAll("[data-restore]").forEach((b) => b.addEventListener("click", () => moveRelation(b.dataset.restore, true)));
  for (const end of ["source", "target"]) {
    const fill = () => { $(`add-${end}-column`).innerHTML = columnsOf($(`add-${end}-table`).value); };
    $(`add-${end}-table`).onchange = fill;
    fill();
  }
  $("add-cancel").onclick = () => { ui.adding = false; renderDetail(); };
  $("add-form").onsubmit = (e) => { e.preventDefault(); addRelation(); };
}

// The columns of a table, for the add form
function columnsOf(table) {
  return (ui.rel?.columns[table] || []).map((c) => `<option>${esc(c)}</option>`).join("");
}

async function findPath() {
  const { a, b } = ui.path;
  if (!a || !b || a === b) { ui.path.result = null; renderDetail(); return; }
  try {
    ui.path.result = await api(`/api/db/${encodeURIComponent(ui.db)}/relations/path?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  } catch (e) {
    ui.path.result = { joins: null, layers: [] };
  }
  renderDetail();
}

async function moveRelation(pair, back) {
  const [source, target] = pair.split("|");
  try {
    ui.rel = await api(`/api/db/${encodeURIComponent(ui.db)}/relations/ignore${back ? "?back=true" : ""}`,
      { method: "POST", body: { source, target, version: ui.rel.version } });
  } catch (e) {
    renderDetail(e.status === 409 ? "relationships.yaml a changé : recharge la page" : e.message);
    return;
  }
  if (ui.path.a && ui.path.b) await findPath();
  loadTables(); renderDetail();
}

async function addRelation() {
  const source = `${$("add-source-table").value}.${$("add-source-column").value}`;
  const target = `${$("add-target-table").value}.${$("add-target-column").value}`;
  $("rel-state").textContent = "Vérification sur les données…";
  try {
    ui.rel = await api(`/api/db/${encodeURIComponent(ui.db)}/relations`,
      { method: "POST", body: { source, target, version: ui.rel.version } });
  } catch (e) {
    $("rel-state").textContent = e.status === 409 ? "relationships.yaml a changé : recharge la page" : e.message;
    $("rel-state").className = "save-state error";
    return;
  }
  ui.adding = false;
  const added = ui.rel.relations.find((r) => r.from === source && r.to === target);
  ui.current = source.split(".")[0];
  loadTables();
  renderDetail(added?.found == null ? "Ajoutée, mais aucune valeur à vérifier"
    : `Ajoutée : ${pct(added.found)} des valeurs trouvées${added.weak ? " — peu, à vérifier" : ""}`);
}

// ── Jobs ─────────────────────────────────────────────────

async function runJob(step, request) {
  try {
    setJob(await api(`/api/db/${encodeURIComponent(ui.db)}/${step}`, { method: "POST", body: request }));
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

// The two actions of the page: on the selection, or on what is missing (which one is primary depends on the step)
function runSelection() {
  if (!ui.selected.size) { showBanner("Coche d'abord les tables dans la liste.", true); return; }
  if (ui.step === "profile") runJob("profile", { tables: [...ui.selected] });
  else runJob("semantics", { tables: [...ui.selected], fresh: true });
}
function runMissing() {
  runJob(ui.step === "profile" ? "profile" : "semantics", ui.step === "profile" ? { missing: true } : {});
}
$("run-primary").addEventListener("click", () => {
  if (ui.step === "relations") runJob("relations/run", {});
  else if (ui.step === "profile") runSelection();
  else runMissing();
});
$("run-secondary").addEventListener("click", () => {
  if (ui.step === "relations") { ui.adding = true; renderDetail(); $("add-form")?.scrollIntoView({ block: "nearest" }); }
  else if (ui.step === "profile") runMissing();
  else runSelection();
});
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
    // Only the table the job wrote gets its detail reloaded: a description being typed in another one stays
    const wasDone = before?.id === job.id && before.done.includes(ui.current);
    const ended = before?.id === job.id && before.running && !job.running;
    if (ui.current && job.step === ui.step) {
      if (job.step === "profile" && job.done.includes(ui.current) && !wasDone) openTable(ui.current);
      if (job.step === "semantics" && ended && job.started_tables.includes(ui.current)) openTable(ui.current);
    }
    if (job.step === "relations" && ended && ui.step === "relations") await loadRelations();
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
  const titles = { semantics: semanticsJobTitle, relations: relationsJobTitle, profile: profileJobTitle };
  const [title, halo] = (titles[job.step] || profileJobTitle)(job, now);
  $("job-title").innerHTML = title;
  $("job-halo").className = `halo ${halo}`;
  $("job-stop").hidden = !job.running || job.stopped;
  $("job-log").hidden = !ui.logOpen;
  $("job-toggle").textContent = ui.logOpen ? "Masquer les logs" : "Voir les logs";
}

// A description run: the tables sent to the LLM (8 at a time), tables.yaml written once at the end
function semanticsJobTitle(job, now) {
  const sent = job.started_tables.length;
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running && !sent) return [`Préparation de la description… ${duration(now - job.started)}`, "running"];
  if (job.running) {
    return [`Description de ${esc(job.db)} par l'IA · ${plural(sent, "table")} ${sent > 1 ? "envoyées" : "envoyée"} · ${duration(now - job.started)}`, "running"];
  }
  if (job.stopped) return ["Description arrêtée · rien n'est écrit : tables.yaml n'est mis à jour qu'à la fin", "failed"];
  if (job.returncode) return [`Échec de la description (code ${job.returncode}) · voir les logs`, "failed"];
  if (!sent) return ["Rien à décrire : chaque table a déjà ses descriptions", ""];
  return [`Description terminée · ${plural(sent, "table")} ${sent > 1 ? "décrites" : "décrite"} · à relire`, ""];
}

// A relations run: the measured ones, the LLM, then each relation checked on the data (--check: only that)
function relationsJobTitle(job, now) {
  const checkOnly = job.command.includes("--check"), n = job.checked, checked = `${plural(n, "relation")} ${n > 1 ? "vérifiées" : "vérifiée"}`;
  const what = checkOnly ? "Vérification des relations" : "Recherche des relations";
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running) return [`${what} · ${checked} · ${duration(now - job.started)}`, "running"];
  if (job.stopped) return [`${what} arrêtée · rien n'est écrit : relationships.yaml n'est mis à jour qu'à la fin`, "failed"];
  if (job.returncode) return [`Échec de la ${checkOnly ? "vérification" : "recherche"} (code ${job.returncode}) · voir les logs`, "failed"];
  return [`${checkOnly ? "Vérification" : "Recherche"} terminée · ${checked}`, ""];
}

function profileJobTitle(job, now) {
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
  return [title, halo];
}

init();
