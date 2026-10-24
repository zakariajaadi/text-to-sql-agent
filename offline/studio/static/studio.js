// Studio of the semantic layer: one screen per step of the pipeline (STEPS), the same layout for each:
// the tables of the database on the left (a status pill each), the chosen table on the right.
//   Profiling: what the profile detects in each column (declared type against the data, exceptions)
//   Semantics: the descriptions the LLM writes, read and corrected in place
//   Metrics: the metrics on the left instead of the tables, each tested on the data (counts, never a value)
//   Index: what the agent reads of each table (its document) against the files today, and the eval of the agent
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
  metrics: {
    number: 4, title: "Metrics", key: "metrics", todo: "À revoir",
    text: "L'agent copie ces formules au lieu de refaire ses calculs. L'IA les propose ; chacune est testée sur les données (lignes comptées deux fois, mesures vides, dates manquantes) sans jamais en montrer le résultat.",
    primary: () => "Proposer des métriques", secondary: () => (ui.testing ? "Arrêter les tests" : "Tout tester"),
  },
  index: {
    number: 5, title: "Index", key: "index", todo: "À réindexer",
    text: "Ce que l'agent lit vraiment : une fiche par table, faite des fichiers des étapes 1 à 4, et les valeurs des colonnes catégorielles lues dans la base. Une correction n'arrive chez l'agent qu'une fois réindexée.",
    primary: () => "Réindexer", secondary: () => "Évaluer l'agent",
  },
};

const ui = {
  dbs: [], db: null, step: "profile", data: null, filter: "all", query: "", selected: new Set(),
  rel: null, path: { a: null, b: null, result: null }, adding: false,   // the Relations screen
  relCheck: null,                                                      // a relation measured, not added (its target is no key)
  met: null, metric: null, metricKind: "metric", testing: null,        // the Metrics screen (testing: {current, stop})
  idx: null,                                                           // the Index screen
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
    ui.relCheck = null; closeConfirm($("confirm"));
    ui.met = null; ui.metric = null; if (ui.testing) ui.testing.stop = true; ui.idx = null;
    renderDetail(); loadTables().then(() => loadScreen());
  };
  ui.step = STEPS[location.hash.slice(1)] ? location.hash.slice(1) : STEPS[stored("studio.step")] ? stored("studio.step") : "profile";
  renderStep();
  await loadTables();
  await loadScreen();
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
  const metrics = ui.step === "metrics";
  $("search").placeholder = metrics ? "Rechercher une métrique" : "Rechercher une table";
  $("list-name").textContent = metrics ? "Métrique" : "Table";
  $("list-status").textContent = metrics ? "État" : ui.step === "index" ? "Index" : "Statut";
  $("filters").querySelector('[data-filter="review"]').hidden = ui.step !== "semantics";
  closeConfirm($("confirm"));
  renderActions();
}

// The screens whose content is not the tables: read when they open
function loadScreen() {
  if (ui.step === "relations") return loadRelations();
  if (ui.step === "metrics") return loadMetrics();
  if (ui.step === "index") return loadIndex();
}

function goTo(step) {
  if (!STEPS[step] || step === ui.step) return;
  ui.step = step; store("studio.step", step); history.replaceState(null, "", `#${step}`);
  ui.filter = "all"; ui.selected.clear(); ui.detail = null; ui.adding = false; ui.relCheck = null;
  [...$("filters").children].forEach((c) => c.classList.toggle("on", c.dataset.filter === "all"));
  renderStep();
  if (step === "relations" || step === "metrics" || step === "index") {
    if (step === "index") ui.current = null;   // the Index screen opens on the whole database
    if (ui.data) { renderHeader(); renderRows(); }
    renderDetail(); loadScreen();
    return;
  }
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
    relations: ui.data.tables.filter((t) => t.status.relations !== "done").length,
    metrics: ui.met ? ui.met.counts.check + ui.met.counts.error : 0,
    index: ui.idx ? ui.idx.counts.changed + ui.idx.counts.missing : 0 }[ui.step];
  const review = ui.data.tables.filter((t) => t.rewritten).length;
  const counts = { all: ui.step === "metrics" ? ui.met?.metrics.length ?? 0 : s.tables, todo, review };
  const labels = { all: "Toutes", todo: STEPS[ui.step].todo, review: "À relire" };
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
      && (job.step === "profile" || job.step === "index" ? job.current === table.name : job.started_tables.includes(table.name))) {
    return '<span class="pill running">en cours</span>';
  }
  if (ui.step === "index") return INDEX_PILLS[indexStatus(table.name)] || "";
  const status = table.status[STEPS[ui.step].key];
  if (ui.step === "profile") return status === "none" ? '<span class="pill todo">à profiler</span>' : '<span class="pill done">profilée</span>';
  if (ui.step === "relations") {
    const n = ui.rel?.counts[table.name] ?? 0;
    return n ? `<span class="pill done">${plural(n, "lien")}</span>`
      : '<span class="pill decide" title="Aucune relation : l\'agent ne peut joindre cette table à aucune autre">isolée</span>';
  }
  if (status === "none") return '<span class="pill todo">à décrire</span>';
  if (status === "partial") return `<span class="pill decide" title="Colonnes ajoutées depuis la dernière description">${table.undescribed} col. à décrire</span>`;
  if (table.rewritten) return `<span class="pill decide" title="Descriptions réécrites par l'IA, à relire">${fmt.format(table.rewritten)} à relire</span>`;
  return '<span class="pill done">décrite</span>';
}

function visibleRows() {
  const q = ui.query.toLowerCase(), key = STEPS[ui.step].key;
  const todo = ui.step === "index" ? (t) => ["changed", "missing"].includes(indexStatus(t.name)) : (t) => t.status[key] !== "done";
  const shown = ui.filter === "review" ? (t) => t.rewritten > 0 : todo;
  return (ui.data?.tables || []).filter((t) => (!q || t.name.toLowerCase().includes(q)) && (ui.filter === "all" || shown(t)));
}

function renderRows() {
  if (ui.step === "metrics") { renderMetricRows(); return; }
  $("list-extra").innerHTML = "";
  $("empty").textContent = "Aucune table ne correspond.";
  const rows = visibleRows();
  $("rows").innerHTML = rows.map((t) => `<div class="row${t.name === ui.current ? " current" : ""}" data-table="${esc(t.name)}">
      <input type="checkbox" ${ui.selected.has(t.name) ? "checked" : ""} aria-label="Sélectionner ${esc(t.name)}">
      <span class="cell" title="${esc(t.name)}">${esc(t.name)}</span>
      ${statusPill(t)}
    </div>`).join("");
  $("empty").hidden = rows.length > 0;
  $("select-all").checked = rows.length > 0 && rows.every((t) => ui.selected.has(t.name));
  if (ui.step === "index") renderGone();
  renderActions();
  renderPersonal();
}

function renderActions() {
  $("run-primary").textContent = STEPS[ui.step].primary(ui.selected.size);
  $("run-secondary").textContent = STEPS[ui.step].secondary(ui.selected.size);
  $("run-secondary").hidden = ui.step === "index" && ui.idx?.dataset === false;   // no eval questions for the database
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
  if (ui.step === "metrics") { openMetric(row.dataset.metric, "metric"); return; }
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
  const step = ui.step, db = encodeURIComponent(ui.db), table = encodeURIComponent(name);
  try {
    ui.detail = await api(step === "index" ? `/api/db/${db}/index/${table}`
      : `/api/db/${db}/tables/${table}${step === "profile" ? "" : "/semantics"}`);
  } catch (e) {
    ui.detail = { name, error: e.message };
  }
  if (ui.current === name && ui.step === step) renderDetail();
}

function renderDetail(message) {
  const d = ui.detail, el = $("detail");
  if (ui.step === "relations") { renderRelations(el, message); return; }
  if (ui.step === "metrics") { renderMetric(el, message); return; }
  if (ui.step === "index") { renderIndex(el, message); return; }
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
  return [plural(c.columns, "colonne"), `${fmt.format(c.described)}/${fmt.format(c.visible)} décrites`,
    c.rewritten ? `${fmt.format(c.rewritten)} à relire` : null,
    c.categorical ? `${fmt.format(c.categorical)} catégorielle${c.categorical > 1 ? "s" : ""}` : "aucune catégorielle",
    c.edited ? `${fmt.format(c.edited)} corrigée${c.edited > 1 ? "s" : ""} à la main` : null].filter(Boolean).join(" · ");
}

// The marks of a description (column: its name, "" for the table): rewritten by the LLM (an amber dot, to read
// again), corrected by hand (a pencil, kept when the table is described again); hidden when they don't apply
const PENCIL = '<path d="M10.5 2.5l3 3L6 13l-3.5.5L3 10z"/>';
const rewrittenDot = (column, on) => `<span class="todo-dot" data-dot="${esc(column)}" title="Réécrite par l'IA : à relire" ${on ? "" : "hidden"}></span>`;
const handMark = (column, on) => `<svg class="hand" data-hand="${esc(column)}" viewBox="0 0 16 16" role="img" aria-label="Corrigée à la main" ${on ? "" : "hidden"}><title>Corrigée à la main : gardée quand l'IA redécrit</title>${PENCIL}</svg>`;

function renderSemantics(d, el, message) {
  if (!d.described) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="describe-one">Décrire</button></div>
      <p class="placeholder">Pas encore décrite : l'IA rédigera la description de la table et de ses colonnes.</p>`;
    $("describe-one").onclick = () => runJob("semantics", { tables: [d.name] });
    return;
  }
  const lines = d.columns.map((c) => {
    const key = c.key ? ` <span class="pill key">${c.key === "unique" ? "clé unique" : "clé"}</span>` : "";
    if (c.excluded) {
      return `<div class="sline hidden-col"><span class="cell">${esc(c.column)}${key} <span class="pill hide">cachée</span></span>
        <span class="cell mono">${esc((c.type || "—").toLowerCase())}</span>
        <span class="muted-text">cachée à l'agent : pas de description</span><span></span></div>`;
    }
    const values = c.distinct == null ? "" : `<span class="values" title="${c.sampled ? "Valeurs distinctes sur l'échantillon du profil" : "Valeurs distinctes de la table"}">${plural(c.distinct, "valeur")}</span>`;
    return `<div class="sline">
      <span class="cell">${rewrittenDot(c.column, c.rewritten)}${esc(c.column)}${key}${handMark(c.column, c.edited)}</span>
      <span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>
      <textarea class="edit" rows="1" data-field="column" data-column="${esc(c.column)}" placeholder="Ajouter une description…" aria-label="Description de ${esc(c.column)}">${esc(c.description)}</textarea>
      <label class="switch-cell">${values}<input type="checkbox" class="switch" data-field="categorical" data-column="${esc(c.column)}" ${c.categorical ? "checked" : ""} aria-label="${esc(c.column)} catégorielle"></label>
    </div>`;
  }).join("");

  el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="figures" id="sem-figures">${semanticsFigures(d)}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      <button class="btn ghost small" id="mark-read" title="Les descriptions réécrites par l'IA sortent de « à relire »" ${d.counts.rewritten ? "" : "hidden"}>Tout marquer relu</button>
      <button class="btn ghost small" id="describe-again" title="L'IA réécrit les descriptions de cette table ; celles corrigées à la main sont gardées">Redécrire</button></div>
    <div class="confirm" id="sem-confirm" role="group" aria-label="Redécrire ${esc(d.name)}" hidden></div>
    <p class="legend muted-text" id="sem-legend" ${d.counts.rewritten ? "" : "hidden"}><span><span class="todo-dot"></span>réécrite par l'IA, à relire : le point part quand tu la modifies</span><span><svg class="hand" viewBox="0 0 16 16" aria-hidden="true">${PENCIL}</svg>corrigée à la main, gardée</span></p>
    ${message ? `<p class="note error">${esc(message)}</p>` : ""}
    <div class="table-text">
      <label class="field-label" for="sem-description">${rewrittenDot("", d.description_rewritten)}Description de la table${handMark("", d.description_edited)}</label>
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

  $("describe-again").onclick = () => redescribeTable(d);
  $("mark-read").onclick = () => markRead(d);
  el.querySelectorAll("textarea.edit").forEach((area) => {
    fitHeight(area);
    area.addEventListener("input", () => fitHeight(area));
    area.addEventListener("change", () => saveSemantics(area, area.value));
  });
  $("sem-as-of").addEventListener("change", (e) => saveSemantics(e.target, e.target.value));
  el.querySelectorAll("input.switch").forEach((box) => box.addEventListener("change", () => saveSemantics(box, box.checked)));
}

// The marks of the page after a save, without drawing it again (the focus stays where it is)
function updateMarks(d) {
  const edited = new Set(d.columns.filter((c) => c.edited).map((c) => c.column));
  const rewritten = new Set(d.columns.filter((c) => c.rewritten).map((c) => c.column));
  $("detail").querySelectorAll("[data-dot]").forEach((dot) => {
    dot.hidden = dot.dataset.dot ? !rewritten.has(dot.dataset.dot) : !d.description_rewritten;
  });
  $("detail").querySelectorAll("[data-hand]").forEach((hand) => {
    hand.toggleAttribute("hidden", hand.dataset.hand ? !edited.has(hand.dataset.hand) : !d.description_edited);
  });
  $("sem-figures").textContent = semanticsFigures(d);
  $("sem-legend").hidden = $("mark-read").hidden = !d.counts.rewritten;
}

// Describe a table again: straight away, unless it holds corrections made by hand (kept, or replaced on a click)
function redescribeTable(d) {
  const edited = d.counts.edited;
  if (!edited) { runJob("semantics", { tables: [d.name], fresh: true }); return; }
  const written = 1 + d.counts.visible - edited;
  const which = [d.description_edited ? "la description de la table" : null,
    ...d.columns.filter((c) => c.edited && !c.excluded).map((c) => `<span class="mono">${esc(c.column)}</span>`)].filter(Boolean);
  const list = which.length > 1 ? `${which.slice(0, -1).join(", ")} et ${which.at(-1)}` : which[0];
  openConfirm($("sem-confirm"), {
    text: `L'IA réécrit les <b>${plural(written, "description")}</b> qu'elle a écrites. `
      + `${edited > 1 ? `Tes <b>${fmt.format(edited)} corrections</b> restent telles quelles` : "Ta <b>correction</b> reste telle quelle"} : ${list}.`,
    edited,
    button: (replace) => `Redécrire ${plural(replace ? written + edited : written, "description")}`,
    run: (replace) => runJob("semantics", { tables: [d.name], fresh: true, replace_edited: replace }),
  });
}

// The selection described again: the same question, for every table checked that holds corrections
function redescribeSelection() {
  const chosen = (ui.data?.tables || []).filter((t) => ui.selected.has(t.name));
  const edited = chosen.reduce((n, t) => n + (t.edited || 0), 0);
  const run = (replace) => runJob("semantics", { tables: chosen.map((t) => t.name), fresh: true, replace_edited: replace });
  if (!edited) { run(false); return; }
  openConfirm($("confirm"), {
    text: `L'IA réécrit les descriptions de ${plural(chosen.length, "table")}. `
      + `${edited > 1 ? `Tes <b>${fmt.format(edited)} corrections</b> restent telles quelles` : "Ta <b>correction</b> reste telle quelle"}`
      + ` (${names(chosen.filter((t) => t.edited).map((t) => t.name))}).`,
    edited,
    button: () => `Redécrire ${plural(chosen.length, "table")}`,
    run,
  });
}

// The strip before describing again: what the LLM rewrites, what it keeps, the corrections replaced on a click
function openConfirm(strip, { text, edited, button, run }) {
  strip.innerHTML = `<p>${text}</p>
    <span class="confirm-actions">
      <label class="check"><input type="checkbox" data-replace>Remplacer aussi ${edited > 1 ? `mes ${fmt.format(edited)} corrections` : "ma correction"}</label>
      <button class="btn small ghost" type="button" data-cancel>Annuler</button>
      <button class="btn small primary" type="button" data-run>${button(false)}</button>
    </span>`;
  strip.hidden = false;
  const replace = strip.querySelector("[data-replace]"), go = strip.querySelector("[data-run]");
  replace.onchange = () => { go.textContent = button(replace.checked); };
  strip.querySelector("[data-cancel]").onclick = () => closeConfirm(strip);
  go.onclick = () => { closeConfirm(strip); run(replace.checked); };
  go.focus();
}

function closeConfirm(strip) {
  strip.hidden = true;
  strip.innerHTML = "";
}

// The descriptions the LLM rewrote, read: their dots go
async function markRead(d) {
  try {
    ui.detail = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(d.name)}/semantics`,
      { method: "POST", body: { field: "read", version: d.version } });
  } catch (e) {
    renderDetail(e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message);
    return;
  }
  renderDetail();
  loadTables();
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
    Object.assign(ui.detail, saved);
    updateMarks(saved);
    state.textContent = "Enregistré";
    state.className = "save-state ok";
    loadTables();
  } catch (e) {
    state.textContent = e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message;
    state.className = "save-state error";
  }
}

// ── Relations: the map of the tables, the path of the agent, the evidence ──

const KEY_OVERLAP = 0.9;   // values found under it: the relation is in amber (offline/studio/relations.py)

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
  const pairs = new Map();   // two relations between the same tables: side by side, not one over the other
  const edges = ui.rel.relations.filter((r) => r.source !== r.target && nodes[r.source] && nodes[r.target]).map((r) => {
    const s = nodes[r.source], t = nodes[r.target], pair = [r.source, r.target].sort().join("|");
    const dx = 14 * (pairs.get(pair) || 0);
    pairs.set(pair, (pairs.get(pair) || 0) + 1);
    const [x1, y1, x2, y2] = [s.x + s.w / 2 + dx, s.y + s.h, t.x + t.w / 2 + dx, t.y];
    const down = y2 > y1, my = (y1 + y2) / 2;
    const d = down ? `M${x1} ${y1} C${x1} ${my} ${x2} ${my} ${x2} ${y2}` : `M${x1} ${s.y + s.h / 2} L${x2} ${t.y + t.h / 2}`;
    const inPath = layer.has(r.source) && layer.has(r.target) && Math.abs(layer.get(r.source) - layer.get(r.target)) === 1;
    const cls = ["e", r.weak ? "weak" : "", r.found == null ? "unchecked" : "", cur && (r.source === cur || r.target === cur) ? "on" : "",
      cur && !near.has(r.source) && r.source !== cur ? "dim" : "", inPath ? "path" : ""].join(" ");
    const said = r.fanout ? ` · chaque ligne de ${esc(r.source)} comptée ${decimal(r.fanout)} fois`
      : r.found != null ? ` · ${pct(r.found)} des valeurs trouvées` : "";
    return `<path class="${cls}" d="${d}"><title>${esc(r.from)} → ${esc(r.to)}${said}</title></path>`;
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
    : `<span class="bar"><i class="${r.found < KEY_OVERLAP ? "low" : ""}" style="width:${Math.round(r.found * 100)}%"></i></span>${pct(r.found)}`;
  return `<div class="rline${r.weak ? " weak" : ""}">
    <span class="cell mono" title="${esc(r.from)} → ${esc(r.to)}">${esc(r.from)} <span class="arrow">→</span> ${esc(r.to)}</span>
    <span class="fill">${found}</span>
    ${r.fanout ? `<span class="cell mono warn-text" title="${esc(r.to)} n'est pas une clé : chaque ligne de ${esc(r.source)} comptée ${decimal(r.fanout)} fois en moyenne">${esc(r.links)}</span>`
      : `<span class="cell mono muted" title="Combien de lignes d'un côté pour une ligne de l'autre">${esc(r.links)}</span>`}
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
  // The form keeps the relation just checked; a new one starts from the table chosen on the map
  const chosen = ui.relCheck ? { source: ui.relCheck.source.split("."), target: ui.relCheck.target.split(".") }
    : { source: [cur, null], target: [null, null] };

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
      <select id="add-source-table" aria-label="Table qui pointe">${options(chosen.source[0])}</select><select id="add-source-column" aria-label="Colonne qui pointe"></select>
      <span class="arrow">→</span>
      <select id="add-target-table" aria-label="Table pointée">${options(chosen.target[0])}</select><select id="add-target-column" aria-label="Colonne pointée"></select>
      <button class="btn small primary" type="submit">Vérifier et ajouter</button>
      <button class="btn small ghost" type="button" id="add-cancel">Annuler</button>
    </form>
    ${ui.adding && ui.relCheck ? relationCheck(ui.relCheck) : ""}
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
    const fill = (column) => { $(`add-${end}-column`).innerHTML = columnsOf($(`add-${end}-table`).value, column); };
    $(`add-${end}-table`).onchange = () => { forgetCheck(); fill(); };
    $(`add-${end}-column`).onchange = forgetCheck;
    fill(chosen[end][1]);
  }
  $("add-cancel").onclick = () => { ui.adding = false; ui.relCheck = null; renderDetail(); };
  $("add-form").onsubmit = (e) => { e.preventDefault(); addRelation(); };
  if ($("add-anyway")) $("add-anyway").onclick = () => addRelation(true);
}

// The columns of a table, for the add form
function columnsOf(table, chosen = null) {
  return (ui.rel?.columns[table] || []).map((c) => `<option ${c === chosen ? "selected" : ""}>${esc(c)}</option>`).join("");
}

// Another relation chosen in the form: the check of the previous one goes
function forgetCheck() {
  ui.relCheck = null;
  $("rel-check")?.remove();
}

// A relation measured and not added: its target is no key, a join on it would count each row several times.
// What the data says, as the tests of a metric; added on a click only
function relationCheck(c) {
  const [table] = c.source.split("."), [target] = c.target.split("."), [fewest, most] = c.per_value;
  const found = c.found == null ? '<span class="muted-text">—</span>' : share(c.found, c.found < KEY_OVERLAP);
  const foundText = c.found == null ? `<span class="mono">${esc(c.source)}</span> n'a aucune valeur.`
    : c.found >= 1 ? `Toutes les valeurs de <span class="mono">${esc(c.source)}</span> se retrouvent dans <span class="mono">${esc(target)}</span>.`
    : `${pct(1 - c.found)} des valeurs de <span class="mono">${esc(c.source)}</span> ne se retrouvent pas dans <span class="mono">${esc(target)}</span>.`;
  const lines = [
    checkLine("Valeurs trouvées", found, foundText, c.found != null && c.found < KEY_OVERLAP),
    checkLine("Lignes par valeur", `${fmt.format(fewest)} à ${fmt.format(most)}`,
      `<span class="mono">${esc(c.target)}</span> : ${plural(c.target_values, "valeur")} pour ${plural(c.target_rows, "ligne")}, ce n'est pas une clé.`, true),
  ];
  if (c.matched) {
    const times = c.joined / c.matched;
    lines.push(checkLine("Lignes comptées", `<span class="bar"><i class="low" style="width:100%"></i></span>${fmt.format(c.joined)} / ${fmt.format(c.matched)}`,
      `Chaque ligne de ${esc(table)} comptée ${decimal(times)} fois en moyenne : la jointure multiplie ses montants.`, true));
  }
  const other = ui.rel.relations.find((r) => [r.source, r.target].sort().join("|") === [table, target].sort().join("|"));
  const foot = other ? `<span class="muted-text">L'agent va déjà de ${esc(table)} à ${esc(target)} par <span class="mono">${esc(other.from)} → ${esc(other.to)}</span>.</span>` : "";
  return `<div class="result" id="rel-check" role="status">
      <p class="result-title">Pas ajoutée : <span class="mono">${esc(c.target)}</span> n'est pas une clé, la jointure compterait chaque ligne de ${esc(table)} plusieurs fois.</p>
      <div class="mline head"><span>Sur les données</span><span class="num">Mesure</span><span>Ce que ça veut dire</span></div>
      ${lines.join("")}
      <div class="result-foot">${foot}<span class="grow"></span><button class="btn small" type="button" id="add-anyway">Ajouter quand même</button></div>
    </div>`;
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

// A relation measured on the data, then added; not added when its target is no key, unless forced (a click)
async function addRelation(force = false) {
  const source = `${$("add-source-table").value}.${$("add-source-column").value}`;
  const target = `${$("add-target-table").value}.${$("add-target-column").value}`;
  $("rel-state").textContent = "Vérification sur les données…";
  $("rel-state").className = "save-state";
  let response;
  try {
    response = await api(`/api/db/${encodeURIComponent(ui.db)}/relations`,
      { method: "POST", body: { source, target, version: ui.rel.version, force } });
  } catch (e) {
    $("rel-state").textContent = e.status === 409 ? "relationships.yaml a changé : recharge la page" : e.message;
    $("rel-state").className = "save-state error";
    return;
  }
  ui.rel = response;
  if (!response.added) {
    ui.relCheck = response.check;
    renderDetail();
    $("rel-check")?.scrollIntoView({ block: "nearest" });
    return;
  }
  ui.adding = false;
  ui.relCheck = null;
  const added = ui.rel.relations.find((r) => r.from === source && r.to === target);
  ui.current = source.split(".")[0];
  loadTables();
  renderDetail();
  const [state, warn] = added?.fanout ? [`Ajoutée à la main : chaque ligne de ${added.source} comptée ${decimal(added.fanout)} fois en moyenne`, true]
    : added?.found == null ? ["Ajoutée, mais aucune valeur à vérifier", false]
    : [`Ajoutée : ${pct(added.found)} des valeurs trouvées${added.weak ? " — peu, à vérifier" : ""}`, added.weak];
  $("rel-state").textContent = state;
  $("rel-state").className = `save-state${warn ? " warn" : ""}`;
}

// ── Metrics: the formula, the query a test runs, what the data says ──

async function loadMetrics() {
  const db = ui.db;
  try {
    const met = await api(`/api/db/${encodeURIComponent(db)}/metrics`);
    if (db !== ui.db) return;
    ui.met = met;
  } catch (e) {
    showBanner(`Impossible de lire les métriques : ${e.message}`, true);
    return;
  }
  if (ui.step !== "metrics") return;
  if (ui.data) renderHeader();
  renderRows(); renderDetail();
}

const METRIC_PILLS = {
  ok: '<span class="pill done">vérifiée</span>', check: '<span class="pill decide">à revoir</span>',
  error: '<span class="pill hide">erreur</span>', untested: '<span class="pill todo">non testée</span>',
};

function renderMetricRows() {
  const met = ui.met, q = ui.query.toLowerCase();
  const rows = (met?.metrics || []).filter((m) => (!q || m.name.toLowerCase().includes(q))
    && (ui.filter === "all" || m.state === "check" || m.state === "error"));
  const current = (name, kind) => (name === ui.metric && kind === ui.metricKind ? " current" : "");
  $("rows").innerHTML = rows.map((m) => `<div class="row${current(m.name, "metric")}" data-metric="${esc(m.name)}">
      <span></span><span class="cell" title="${esc(m.name)}">${esc(m.name)}</span>
      ${ui.testing?.current === m.name ? '<span class="pill running">en cours</span>' : METRIC_PILLS[m.state]}
    </div>`).join("");
  $("empty").textContent = met && !met.metrics.length
    ? "Pas encore de métrique : « Proposer des métriques » demande à l'IA d'en écrire." : "Aucune métrique ne correspond.";
  $("empty").hidden = !met || rows.length > 0;
  // What the last run left out, and the names in ignore: under the metrics, quieter
  const group = (title, names, kind, pill) => (names.length ? `<div class="list-group">${title} · ${fmt.format(names.length)}</div>`
    + names.map((name) => `<div class="row aside${current(name, kind)}" data-name="${esc(name)}" data-kind="${kind}">
        <span></span><span class="cell" title="${esc(name)}">${esc(name)}</span><span class="pill todo">${pill}</span></div>`).join("") : "");
  $("list-extra").innerHTML = met ? group("Écartées au dernier lancement", met.skipped.map((s) => s.name), "skipped", "écartée")
    + group("Retirées", met.ignored, "ignored", "retirée") : "";
  renderActions();
}

$("list-extra").addEventListener("click", (e) => {
  if (ui.step === "index") {
    const gone = e.target.closest(".row[data-table]");
    if (gone) openTable(gone.dataset.table);
    return;
  }
  const row = e.target.closest(".row[data-kind]");
  if (row) openMetric(row.dataset.name, row.dataset.kind);
});

function openMetric(name, kind) {
  ui.metric = name; ui.metricKind = kind;
  renderRows(); renderDetail();
}

// Why the generator would not propose a metric today, or why it left a proposal out
const RULES = {
  count: (m) => `Un simple comptage d'une colonne : le générateur ne la proposerait plus. L'agent compte seul, et cette métrique ajoute ${esc(m.tables.join(", ") || "sa table")} à chaque question qui en parle.`,
  same: (m) => `Même formule que ${esc(m.rule_with)} : l'agent ne saura pas laquelle prendre.`,
  distinct: () => "SUM(DISTINCT …) ou AVG(DISTINCT …) : deux lignes de même montant ne compteraient qu'une fois.",
  known: () => "Ce nom est aussi dans ignore : le générateur la tient pour retirée.",
};
const SKIPPED = {
  count: "Un simple comptage d'une colonne : l'agent compte seul.",
  same: "Même formule qu'une métrique déjà écrite, sous un autre nom.",
  distinct: "SUM(DISTINCT …) ou AVG(DISTINCT …) : deux lignes de même montant ne compteraient qu'une fois.",
  known: "Même nom qu'une métrique écrite ou retirée.",
  rejected: "Refusée à la vérification : l'agent ne l'aurait pas acceptée, ou la base ne la compile pas.",
  other: "Écartée par le générateur.",
};
const ERRORS = {
  hidden: "Elle lit une colonne cachée à l'agent : il refusera de démarrer tant qu'elle y est.",
  invalid: "L'agent refusera de démarrer avec cette métrique.",
  joins: "Ses tables ne sont reliées par aucune relation : elle ne peut pas être calculée. La relation s'ajoute dans Relations.",
};

const at = (seconds) => new Date(seconds * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
const decimal = (x) => new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 }).format(x);
const share = (x, warn) => (x == null ? '<span class="muted-text">—</span>'
  : `<span class="bar"><i class="${warn ? "low" : ""}" style="width:${Math.round(Math.min(x, 1) * 100)}%"></i></span>${pct(x)}`);

// One line of "Sur les données": the check, its measure, what it means; amber when worth a look
const checkLine = (label, measure, text, warn = false) => `<div class="mline${warn ? " warn" : ""}">
    <span>${label}</span><span class="fill">${measure}</span><span class="muted-text">${text}</span></div>`;

function checkLines(m) {
  const t = m.test;
  if (!t) return checkLine("Pas encore testée", "", "« Tester » compte ses lignes sur la base, sans lire une valeur.");
  if (t.error) return checkLine("Compile", '<span class="bad-text">non</span>', `<span class="mono">${esc(t.error)}</span>`, true);
  const lines = [checkLine("Compile", `oui · ${decimal(t.seconds)} s`, m.joins ? "La base accepte la formule et ses jointures." : "Une seule table : pas de jointure.")];
  for (const c of t.tables) {
    const ratio = c.rows ? c.counted / c.rows : null;
    const text = c.counted > c.rows ? `Chaque ligne de ${esc(c.table)} comptée ${decimal(ratio)} fois en moyenne : la jointure multiplie ses montants.`
      : c.counted < c.rows ? `${plural(c.rows - c.counted, "ligne")} de ${esc(c.table)} sans correspondance, absentes du total.`
      : `Chaque ligne de ${esc(c.table)} comptée une fois : la jointure n'en double ni n'en perd.`;
    const bar = `<span class="bar"><i class="${c.warn ? "low" : ""}" style="width:${Math.round(Math.min(ratio ?? 0, 1) * 100)}%"></i></span>`;
    lines.push(checkLine("Lignes comptées", `${bar}${fmt.format(c.counted)} / ${fmt.format(c.rows)}`, text, c.warn));
  }
  for (const s of t.measures) {
    const text = s.conditional ? "Vide hors de sa condition, exprès."
      : s.share === 1 ? "Aucune ligne ignorée par le calcul."
      : s.share == null ? "Aucune ligne à calculer." : `${pct(1 - s.share)} des lignes ignorées : <span class="mono">${esc(s.sql)}</span> y est vide.`;
    lines.push(checkLine("Mesure remplie", share(s.share, s.warn), text, s.warn));
  }
  if (t.date) {
    const d = t.date, text = d.share === 1 || d.share == null ? `<span class="mono">${esc(d.column)}</span>`
      : `<span class="mono">${esc(d.column)}</span> : ${pct(1 - d.share)} des lignes hors de toute période.`;
    lines.push(checkLine("Date remplie", share(d.share, d.warn), text, d.warn));
  }
  const sign = { positive: "positif", negative: "négatif", zero: "zéro", empty: "vide" }[t.sign];
  const said = { zero: "Zéro sur toute la base : sa condition ne trouve peut-être aucune ligne.",
    empty: "Vide sur toute la base : aucune ligne à calculer." }[t.sign] || "La valeur n'est jamais affichée.";
  lines.push(checkLine("Résultat", sign, said, t.sign === "zero" || t.sign === "empty"));
  return lines.join("");
}

function renderMetric(el, message) {
  const met = ui.met;
  if (!met) { el.innerHTML = '<p class="placeholder">Lecture des métriques…</p>'; return; }
  const note = message ? `<p class="note error">${esc(message)}</p>` : "";
  if (ui.metricKind === "skipped") {
    const s = met.skipped.find((x) => x.name === ui.metric);
    if (s) {
      el.innerHTML = `<div class="detail-head"><h2>${esc(s.name)}</h2><span class="figures">proposée par l'IA · pas ajoutée</span></div>
        <div class="metric-fields"><p class="said">${SKIPPED[s.reason] || SKIPPED.other}</p>
        ${s.reason === "rejected" || s.reason === "other" ? `<pre class="sql">${esc(s.detail)}</pre>` : ""}
        <p class="muted-text">Le générateur l'a écartée avant de l'écrire dans metrics.yaml : rien à faire.</p></div>`;
      return;
    }
  }
  if (ui.metricKind === "ignored" && met.ignored.includes(ui.metric)) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(ui.metric)}</h2><span class="figures">dans ignore</span><span class="grow"></span>
        <button class="btn ghost small" id="metric-allow">Autoriser à nouveau</button></div>${note}
      <div class="metric-fields"><p class="said">Retirée : le générateur ne la proposera plus. Sa définition n'est plus dans metrics.yaml.</p>
      <p class="muted-text">« Autoriser à nouveau » enlève son nom de ignore : l'IA pourra la reproposer.</p></div>`;
    $("metric-allow").onclick = () => moveMetric(ui.metric, true);
    return;
  }
  const m = met.metrics.find((x) => x.name === ui.metric);
  if (!m) { el.innerHTML = '<p class="placeholder">Choisis une métrique pour voir sa formule et ses tests.</p>'; return; }

  const figures = [m.tables.join(", "), m.sql ? (m.joins ? plural(m.joins, "jointure") : "sans jointure") : null,
    m.test && !m.test.error ? `testée à ${at(m.test.at)}` : null].filter(Boolean).join(" · ");
  const testing = ui.testing?.current === m.name;
  const error = m.error ? `<div class="error-note"><span>${ERRORS[m.error.kind]}</span><code>${esc(m.error.detail)}</code></div>` : "";
  const rule = m.rule ? `<div class="rule-note"><span>${RULES[m.rule](m)}</span><button class="btn small" data-retire="1">Retirer</button></div>` : "";
  const dates = [...new Set([m.time_dimension, ...m.dates, ""])];   // the current date first, "aucune" last
  el.innerHTML = `<div class="detail-head"><h2>${esc(m.name)}</h2><span class="figures">${esc(figures)}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      ${m.sql ? `<button class="btn ghost small" id="metric-test" ${testing ? "disabled" : ""}>${testing ? "Test en cours…" : "Tester"}</button>` : ""}
      <button class="btn ghost small" data-retire="1" title="Son nom va dans ignore : le générateur ne la proposera plus">Retirer</button></div>
    ${note}${error}${rule}
    <div class="metric-fields">
      <div><label class="field-label" for="met-description">Description</label>
        <textarea class="edit block" id="met-description" rows="1" data-field="description">${esc(m.description)}</textarea></div>
      <div><label class="field-label" for="met-synonyms">Synonymes</label>
        <textarea class="edit block" id="met-synonyms" rows="1" data-field="synonyms" placeholder="d'autres mots pour la même chose, séparés par des virgules">${esc(m.synonyms.join(", "))}</textarea></div>
      <div class="formula-line">
        <div><label class="field-label" for="met-expr">Formule</label>
          <textarea class="edit block mono" id="met-expr" rows="1" data-field="expr" spellcheck="false">${esc(m.expr)}</textarea></div>
        <div><label class="field-label" for="met-date">Date par défaut</label>
          <select class="edit-select" id="met-date" data-field="time_dimension">${dates.map((d) =>
            `<option value="${esc(d)}" ${d === m.time_dimension ? "selected" : ""}>${d ? esc(d) : "aucune"}</option>`).join("")}</select></div>
      </div>
      <p class="note error refused" id="metric-refused" hidden></p>
      ${m.sql ? `<div><span class="field-label">Requête testée</span><pre class="sql">${esc(m.sql)}</pre></div>` : ""}
    </div>
    ${m.sql ? `<div class="columns"><div class="mline head"><span>Sur les données</span><span class="num">Mesure</span><span>Ce que ça veut dire</span></div>
      ${checkLines(m)}</div>` : ""}`;

  el.querySelectorAll("textarea.edit").forEach((area) => {
    fitHeight(area);
    area.addEventListener("input", () => fitHeight(area));
    area.addEventListener("change", () => saveMetric(m.name, area));
  });
  $("met-date").addEventListener("change", (e) => saveMetric(m.name, e.target));
  if ($("metric-test")) $("metric-test").onclick = () => testMetrics([m.name]);
  el.querySelectorAll("[data-retire]").forEach((b) => b.addEventListener("click", () => moveMetric(m.name, false)));
}

// One field written into metrics.yaml as soon as it is left; a new formula or date is checked by the base first, then tested
async function saveMetric(name, input) {
  const field = input.dataset.field, state = $("save-state"), refused = $("metric-refused");
  state.textContent = field === "expr" || field === "time_dimension" ? "Vérification sur la base…" : "Enregistrement…";
  state.className = "save-state";
  refused.hidden = true;
  try {
    ui.met = await api(`/api/db/${encodeURIComponent(ui.db)}/metrics/${encodeURIComponent(name)}`,
      { method: "POST", body: { field, value: input.value, version: ui.met.version } });
  } catch (e) {
    state.textContent = e.status === 409 ? "metrics.yaml a changé : recharge la page" : "Refusée";
    state.className = "save-state error";
    // Why, under the fields: nothing is written, the text stays to be corrected
    if (e.status !== 409) { refused.innerHTML = `Rien n'est écrit : <span class="mono">${esc(e.message)}</span>`; refused.hidden = false; }
    return;
  }
  if (field === "expr" || field === "time_dimension") {
    renderRows(); renderDetail();
    testMetrics([name]);
    return;
  }
  state.textContent = "Enregistré";
  state.className = "save-state ok";
}

// The metrics tested one after the other (a query of counts each); "Arrêter les tests" stops after the current one
async function testMetrics(names) {
  if (ui.testing) return;
  const db = ui.db;
  ui.testing = { current: null, stop: false };
  renderActions();
  let message;
  for (const name of names) {
    if (ui.testing.stop || db !== ui.db) break;
    ui.testing.current = name;
    if (ui.step === "metrics") { renderRows(); if (ui.metric === name) renderDetail(); }
    try {
      const met = await api(`/api/db/${encodeURIComponent(db)}/metrics/${encodeURIComponent(name)}/test`, { method: "POST" });
      if (db === ui.db) ui.met = met;
    } catch (e) {
      message = e.status === 503 ? "La base ne répond pas : les métriques ne peuvent pas être testées." : e.message;
      break;
    }
  }
  ui.testing = null;
  if (db !== ui.db || ui.step !== "metrics") return;
  if (ui.data) renderHeader();
  renderRows(); renderDetail(message);
}

async function moveMetric(name, back) {
  try {
    ui.met = await api(`/api/db/${encodeURIComponent(ui.db)}/metrics/${encodeURIComponent(name)}/ignore${back ? "?back=true" : ""}`,
      { method: "POST", body: { version: ui.met.version } });
  } catch (e) {
    renderDetail(e.status === 409 ? "metrics.yaml a changé : recharge la page" : e.message);
    return;
  }
  ui.metricKind = back ? "metric" : "ignored";
  if (back) ui.metric = null;
  loadTables(); renderRows(); renderDetail();
}

// ── Index: what the agent reads of each table, against the files today ──

async function loadIndex() {
  const db = ui.db;
  try {
    const idx = await api(`/api/db/${encodeURIComponent(db)}/index`);
    if (db !== ui.db) return;
    ui.idx = idx;
  } catch (e) {
    showBanner(`Impossible de lire l'index : ${e.message}`, true);
    return;
  }
  if (ui.step !== "index") return;
  if (ui.data) { renderHeader(); renderRows(); }
  if (ui.current) openTable(ui.current); else renderDetail();
}

const indexStatus = (name) => ui.idx?.tables.find((t) => t.name === name)?.status;
const INDEX_PILLS = {
  fresh: '<span class="pill done">à jour</span>',
  changed: '<span class="pill decide" title="Les fichiers ont changé depuis l\'index : l\'agent lit encore l\'ancienne fiche">modifiée</span>',
  missing: '<span class="pill todo" title="Pas dans l\'index : l\'agent ne voit pas cette table">à indexer</span>',
  unknown: '<span class="pill todo" title="Index construit avant que le studio en garde la trace">état inconnu</span>',
};

// The tables the index holds that the database no longer has: under the tables, quieter
function renderGone() {
  const gone = ui.idx?.gone || [];
  $("list-extra").innerHTML = gone.length ? `<div class="list-group">Plus dans la base · ${fmt.format(gone.length)}</div>`
    + gone.map((name) => `<div class="row aside${name === ui.current ? " current" : ""}" data-table="${esc(name)}">
        <span></span><span class="cell" title="${esc(name)}">${esc(name)}</span><span class="pill todo">à retirer</span></div>`).join("") : "";
}

const when = (seconds) => `${new Date(seconds * 1000).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" })} à ${at(seconds)}`;

// "a, b et c", or "a, b, c, d et 3 autres"
function names(list, max = 4) {
  const shown = list.slice(0, max).map((name) => `<span class="mono">${esc(name)}</span>`), more = list.length - shown.length;
  if (more) return `${shown.join(", ")} et ${more} autre${more > 1 ? "s" : ""}`;
  return shown.length > 1 ? `${shown.slice(0, -1).join(", ")} et ${shown.at(-1)}` : shown.join("");
}

// One line of the Index screen; tone "warn" (amber: to look at) or "bad" (red: the agent won't start)
const indexLine = (label, measure, text, tone = "") => `<div class="mline${tone ? ` ${tone}` : ""}">
    <span>${label}</span><span class="fill">${measure}</span><span class="muted-text">${text}</span></div>`;

// What the files changed since the index, or how old an index built before its trace is
function indexNote(idx) {
  if (!idx.trace) {
    if (!idx.values_at) return `<div class="error-note"><span>Jamais indexée : l'agent ne voit aucune table de ${esc(ui.db)}. « Réindexer » écrit leurs fiches.</span></div>`;
    const newer = Object.entries(idx.files).filter(([, time]) => time > idx.values_at).map(([file, time]) => `${file} (${when(time)})`);
    return `<div class="rule-note"><span>Index construit avant que le studio en garde la trace${newer.length
      ? ` : ${newer.join(" et ")} ${newer.length > 1 ? "ont" : "a"} changé depuis` : ""}. Réindexe une fois pour voir ce que l'agent lit, table par table.</span></div>`;
  }
  const changed = idx.tables.filter((t) => t.status === "changed").map((t) => t.name);
  const missing = idx.tables.filter((t) => t.status === "missing").map((t) => t.name);
  const parts = [];
  if (changed.length) {
    parts.push(`${changed.length > 1 ? `${fmt.format(changed.length)} tables ont changé` : "Une table a changé"} depuis (${names(changed)}) : `
      + `l'agent lit encore ${changed.length > 1 ? "leurs anciennes fiches" : "son ancienne fiche"}.`);
  }
  if (missing.length) parts.push(`${names(missing)} ${missing.length > 1 ? "ne sont pas" : "n'est pas"} dans l'index : l'agent ne ${missing.length > 1 ? "les" : "la"} voit pas.`);
  if (idx.gone.length) {
    parts.push(`${names(idx.gone)} ${idx.gone.length > 1 ? "ne sont plus" : "n'est plus"} dans la base : `
      + `${idx.gone.length > 1 ? "leurs fiches partiront" : "sa fiche partira"} à la prochaine indexation.`);
  }
  return parts.length ? `<div class="rule-note"><span>${parts.join(" ")}</span></div>` : "";
}

// What the agent receives: the documents, the values (listed or searched), the sample rows, the metrics it checks at startup
function indexLines(idx) {
  const raw = idx.raw.filter((t) => !idx.gone.includes(t)), m = idx.metrics;
  const rows = idx.sample_rows, now = idx.sample_rows_now;
  const lines = [
    indexLine("Fiches", fmt.format(idx.documents), raw.length
      ? `${names(raw)} sans description : la structure brute seulement. Semantics les décrit.` : "Une par table, toutes décrites.", raw.length ? "warn" : ""),
    indexLine("Valeurs listées", `${fmt.format(idx.inline.columns)} col. · ${fmt.format(idx.inline.values)}`,
      "Listées telles quelles : l'agent les recopie dans ses filtres."),
    indexLine("Valeurs cherchées", `${fmt.format(idx.searched.columns)} col. · ${fmt.format(idx.searched.values)}`,
      "Trop nombreuses pour la fiche : retrouvées quand une question les cite."),
  ];
  if (idx.dropped.length) {
    lines.push(indexLine("Écartées", `${fmt.format(idx.dropped.length)} col.`, `${names(idx.dropped)} : catégorielle${idx.dropped.length > 1 ? "s" : ""}, `
      + "mais aucune valeur indexée (vide, ou trop de valeurs distinctes).", "warn"));
  }
  lines.push(rows !== now
    ? indexLine("Exemples de lignes", rows ? `${fmt.format(rows)} par fiche` : "aucune",
      `La configuration en demande ${now ? fmt.format(now) : "aucune"} : réindexer pour l'appliquer.`, "warn")
    : indexLine("Exemples de lignes", rows ? `${fmt.format(rows)} par fiche` : "aucune",
      rows ? "Envoyés au modèle sous la structure, colonnes cachées retirées." : "Aucune ligne de la base dans les fiches."));
  lines.push(m.errors.length
    ? indexLine("Métriques", `${fmt.format(m.total - m.errors.length)} / ${fmt.format(m.total)}`, "L'agent refusera de démarrer : "
      + m.errors.map((e) => `<span class="mono">${esc(e.name)}</span> (${esc(e.detail)})`).join(", ") + ".", "bad")
    : indexLine("Métriques", m.total ? `${fmt.format(m.total)} / ${fmt.format(m.total)}` : "aucune",
      m.total ? "Elles lisent des colonnes de l'index : l'agent démarre." : "Pas de métrique : rien à vérifier."));
  return lines.join("");
}

// The last eval: its score, whether it measured this index, the dashboard of the database
function evalLines(idx) {
  const head = '<div class="mline head"><span>Évaluation</span><span class="num">Résultats exacts</span><span></span></div>';
  if (!idx.dataset) {
    return head + indexLine("Questions", "aucune", `Pas de questions pour cette base : <span class="mono">offline/eval/datasets/${esc(ui.db.toLowerCase())}.yaml</span>.`);
  }
  const e = idx.eval;
  if (!e) return head + indexLine("Dernière mesure", "—", "Jamais évaluée : « Évaluer l'agent » lui pose les questions de la base.");
  const built = idx.built_at ?? idx.values_at, before = built != null && e.at < built;
  const text = `${when(e.at)} · ${fmt.format(e.exact)} réponses exactes sur ${fmt.format(e.scored)} comparées aux lignes attendues`
    + (e.questions > e.scored ? ` (${fmt.format(e.questions)} questions)` : "") + (before ? " · mesurée avant cet index : à refaire" : "");
  return head + `<div class="mline with-link${before ? " warn" : ""}"><span>Dernière mesure</span><span class="fill">${pct(e.scored ? e.exact / e.scored : null)}</span>
    <span class="muted-text">${text}</span><a class="btn ghost small" href="/eval/${encodeURIComponent(ui.db)}" target="_blank" rel="noopener">Ouvrir le tableau ↗</a></div>`;
}

function renderIndex(el, message) {
  const idx = ui.idx;
  if (!idx) { el.innerHTML = '<p class="placeholder">Lecture de l\'index…</p>'; return; }
  if (ui.current) { renderDocument(el, message); return; }
  const figures = idx.trace ? `indexé le ${when(idx.built_at)} · ${plural(idx.documents, "fiche")} · ${plural(idx.inline.values + idx.searched.values, "valeur")}`
    : idx.values_at ? `index du ${when(idx.values_at)}` : "jamais indexé";
  el.innerHTML = `<div class="detail-head"><h2>Index de ${esc(ui.db)}</h2><span class="figures">${figures}</span></div>
    ${message ? `<p class="note error">${esc(message)}</p>` : ""}${indexNote(idx)}
    ${idx.trace ? `<div class="columns"><div class="mline head"><span>Ce que l'agent reçoit</span><span class="num">Mesure</span><span>Ce que ça veut dire</span></div>
      ${indexLines(idx)}</div>` : ""}
    <div class="columns">${evalLines(idx)}</div>`;
}

// What changed in a document since the index, in a few words each
const CHANGES = {
  described: () => "la table est décrite depuis",
  undescribed: () => "sa description a été retirée",
  description: () => "la description de la table",
  data_as_of: () => "la date des données",
  read_mode: () => "le mode de lecture",
  columns: (c) => `la description de ${names(c.columns)}`,
  formats: (c) => `le format de ${names(c.columns)}`,
  hidden: (c) => `${names(c.columns)} ${c.columns.length > 1 ? "cachées" : "cachée"}`,
  shown: (c) => `${names(c.columns)} de nouveau ${c.columns.length > 1 ? "visibles" : "visible"}`,
  categorical: (c) => [c.added.length && `${names(c.added)} ${c.added.length > 1 ? "devenues catégorielles" : "devenue catégorielle"}`,
    c.removed.length && `${names(c.removed)} plus catégorielle${c.removed.length > 1 ? "s" : ""}`].filter(Boolean).join(", "),
  relations: (c) => [...c.added.map(([s, t]) => `la jointure <span class="mono">${esc(s)} → ${esc(t)}</span> ajoutée`),
    ...c.removed.map(([s, t]) => `la jointure <span class="mono">${esc(s)} → ${esc(t)}</span> retirée`)].join(", "),
  db_columns: (c) => [c.added.length && `${names(c.added)} ${c.added.length > 1 ? "ajoutées" : "ajoutée"} dans la base`,
    c.removed.length && `${names(c.removed)} ${c.removed.length > 1 ? "retirées" : "retirée"} de la base`].filter(Boolean).join(", "),
};

// A column of the document: its type, its format, its values (listed, searched, none), or hidden, or new in the database
function documentLine(c) {
  const type = `<span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>`;
  if (c.hidden) return `<div class="xline hidden-col"><span class="cell">${esc(c.column)} <span class="pill hide">cachée</span></span>${type}<span class="xvalues">hors de la fiche</span></div>`;
  if (c.added) return `<div class="xline warn"><span class="cell">${esc(c.column)}</span>${type}<span class="xvalues">ajoutée depuis l'index</span></div>`;
  const format = c.format ? ` <span class="pill type">${esc(formatLabel(c.format))}</span>` : "";
  const values = c.gone ? "plus dans la base" : !c.values ? "—" : c.values.dropped ? "aucune valeur indexée"
    : `${fmt.format(c.values.count)} · ${c.values.inline ? "dans la fiche" : "cherchées"}`;
  return `<div class="xline${c.gone || c.values?.dropped ? " warn" : ""}"><span class="cell">${esc(c.column)}${format}</span>${type}<span class="xvalues">${values}</span></div>`;
}

function renderDocument(el, message) {
  const d = ui.detail;
  if (!d || d.name !== ui.current) { el.innerHTML = '<p class="placeholder">Lecture de la fiche…</p>'; return; }
  const head = (figures) => `<div class="detail-head"><h2>${esc(d.name)}</h2>${figures ? `<span class="figures">${figures}</span>` : ""}
    <span class="grow"></span><button class="btn ghost small" id="index-back">← Index de ${esc(ui.db)}</button></div>`
    + (message ? `<p class="note error">${esc(message)}</p>` : "");
  if (d.error) {
    el.innerHTML = head("") + `<p class="note error">${esc(d.error)}</p>`;
  } else if (d.status === "unknown") {
    el.innerHTML = head("") + '<p class="placeholder">Index construit avant que le studio en garde la trace : « Réindexer » montre la fiche de cette table.</p>';
  } else if (d.status === "missing") {
    el.innerHTML = head("") + '<p class="placeholder">Pas dans l\'index : l\'agent ne voit pas cette table. « Réindexer » l\'ajoute.</p>';
  } else {
    const figures = [`${fmt.format(d.chars)} caractères`, d.described ? (d.joins ? plural(d.joins, "jointure") : "aucune jointure") : null,
      d.described ? "décrite" : "structure brute", d.data_as_of ? `données au ${d.data_as_of}` : null].filter(Boolean).join(" · ");
    const notes = [
      d.gone && "Plus dans la base : sa fiche partira à la prochaine indexation.",
      d.changes.length && `Depuis l'index : ${d.changes.map((c) => CHANGES[c.kind](c)).join(" · ")}. L'agent lit encore l'ancienne fiche.`,
      !d.described && !d.gone && !d.changes.length && "Sans description : l'agent ne lit que la structure brute. Semantics la décrit.",
    ].filter(Boolean);
    el.innerHTML = head(figures) + notes.map((n) => `<div class="rule-note"><span>${n}</span></div>`).join("")
      + `<div class="columns"><div class="xline head"><span>Colonne de la fiche</span><span>Type</span><span class="num">Valeurs</span></div>
        ${d.columns.map(documentLine).join("")}</div>`;
  }
  $("index-back").onclick = () => { ui.current = null; ui.detail = null; renderRows(); renderDetail(); };
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
  else redescribeSelection();
}
function runMissing() {
  runJob(ui.step === "profile" ? "profile" : "semantics", ui.step === "profile" ? { missing: true } : {});
}
$("run-primary").addEventListener("click", () => {
  if (ui.step === "metrics") runJob("metrics/run", {});
  else if (ui.step === "index") runJob("index/run", {});
  else if (ui.step === "relations") runJob("relations/run", {});
  else if (ui.step === "profile") runSelection();
  else runMissing();
});
$("run-secondary").addEventListener("click", () => {
  if (ui.step === "metrics") {
    if (ui.testing) { ui.testing.stop = true; return; }
    testMetrics((ui.met?.metrics || []).filter((m) => m.sql).map((m) => m.name));
  }
  else if (ui.step === "index") runJob("eval/run", {});
  else if (ui.step === "relations") { ui.adding = true; ui.relCheck = null; renderDetail(); $("add-form")?.scrollIntoView({ block: "nearest" }); }
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
    // The metrics a run adds are tested at once: the data says what to look at
    if (job.step === "metrics" && ended) {
      await loadMetrics();
      renderJob();   // its title counts what the run added and skipped
      const added = (ui.met?.added || []).filter((name) => ui.met.metrics.some((m) => m.name === name && m.sql));
      if (ui.step === "metrics" && added.length) testMetrics(added);
    }
    // A new index or a new eval: what the agent reads, its score
    if ((job.step === "index" || job.step === "eval") && ended) { await loadIndex(); renderJob(); }
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
  const titles = { semantics: semanticsJobTitle, relations: relationsJobTitle, metrics: metricsJobTitle, profile: profileJobTitle,
    index: indexJobTitle, eval: evalJobTitle };
  const [title, halo] = (titles[job.step] || profileJobTitle)(job, now);
  $("job-title").innerHTML = title;
  $("job-halo").className = `halo ${halo}`;
  $("job-stop").hidden = !job.running || job.stopped;
  $("job-log").hidden = !ui.logOpen;
  $("job-toggle").textContent = ui.logOpen ? "Masquer les logs" : "Voir les logs";
}

// An index: the values read table after table, then the documents written into Chroma, then the value index
function indexJobTitle(job, now) {
  const total = ui.data?.summary.described;
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running && job.current) {
    const waited = now - job.current_since;
    let title = `Lecture des valeurs · <b>${esc(job.current)}</b> depuis ${duration(waited)}${total ? ` (${job.done.length + 1}/${total})` : ""}`;
    if (waited > SLOW_SECONDS) title += ' · <span class="slow">la base met du temps à répondre</span>';
    return [title, "running"];
  }
  if (job.running && job.phase === "documents") return [`Écriture des fiches dans Chroma… ${duration(now - job.started)}`, "running"];
  if (job.running && job.phase === "values") return ["Écriture de l'index des valeurs…", "running"];
  if (job.running) return [`Connexion à la base… ${duration(now - job.started)}`, "running"];
  // Before the documents, nothing is written: the agent keeps the previous index
  if (job.stopped) {
    return [job.phase ? '<span class="slow">Index arrêté pendant l\'écriture : il peut être incomplet, réindexe</span>'
      : "Index arrêté · rien n'est écrit : l'agent garde l'index précédent", "failed"];
  }
  if (job.returncode) return [`Échec de l'index (code ${job.returncode}) · ${job.phase ? "" : "rien n'est écrit · "}voir les logs`, "failed"];
  const idx = job.db === ui.db ? ui.idx : null;
  const counts = idx?.trace ? ` · ${plural(idx.documents, "fiche")} · ${plural(idx.inline.values + idx.searched.values, "valeur")}` : "";
  return [`Index terminé${counts} · l'agent le lira à son prochain démarrage`, ""];
}

// An eval: the questions of the dataset asked to the agent, the results written once at the end
function evalJobTitle(job, now) {
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running && !job.progress) return [`Préparation de l'évaluation… ${duration(now - job.started)}`, "running"];
  if (job.running) {
    return [`Évaluation de l'agent sur ${esc(job.db)} · question ${job.progress[0]}/${job.progress[1]} · ${duration(now - job.started)}`, "running"];
  }
  if (job.stopped) return ["Évaluation arrêtée · les résultats précédents sont gardés", "failed"];
  if (job.returncode) return [`Échec de l'évaluation (code ${job.returncode}) · voir les logs`, "failed"];
  const e = job.db === ui.db ? ui.idx?.eval : null;
  return [`Évaluation terminée${e ? ` · ${fmt.format(e.exact)} résultats exacts sur ${fmt.format(e.scored)}` : ""}`, ""];
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
  if (job.command.includes("--fresh")) {
    // Described again: what the LLM changed, to read again, and the corrections made by hand it kept
    const which = sent > 2 ? plural(sent, "table") : job.started_tables.map(esc).join(", ");
    const changed = job.rewritten ? `${plural(job.rewritten, "description")} ${job.rewritten > 1 ? "réécrites" : "réécrite"}, à relire`
      : "aucune description changée";
    const kept = job.kept ? ` · ${job.kept > 1 ? `tes ${fmt.format(job.kept)} corrections gardées` : "ta correction gardée"}` : "";
    return [`Description terminée · ${which} · <b>${changed}</b>${kept}`, ""];
  }
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

// A run of the generator: one call of the LLM, then each proposal checked; metrics.yaml written once at the end
function metricsJobTitle(job, now) {
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running) return [`Proposition des métriques de ${esc(job.db)} par l'IA · ${duration(now - job.started)}`, "running"];
  if (job.stopped) return ["Proposition arrêtée · rien n'est écrit : metrics.yaml n'est écrit qu'à la fin", "failed"];
  if (job.returncode) return [`Échec de la proposition (code ${job.returncode}) · voir les logs`, "failed"];
  const run = job.db === ui.db ? ui.met : null, added = run?.added.length ?? 0, skipped = run?.skipped.length ?? 0;
  return [`Proposition terminée · ${plural(added, "métrique")} ${added > 1 ? "ajoutées" : "ajoutée"}`
    + (skipped ? ` · ${fmt.format(skipped)} écartée${skipped > 1 ? "s" : ""}` : ""), ""];
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
