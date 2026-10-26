// Studio of the semantic layer, three levels as a web app: the home (every database), a database (its tabs),
// an element of it (a table, a metric). The tabs of a database (STEPS) share one layout: its tables on the left
// (a status pill each), the chosen one on the right.
//   Vue d'ensemble: what the agent reads of the database, its last eval, and its setup step by step
//   Contenu: what the profile detects in each column (declared type against the data, exceptions)
//   Descriptions: the descriptions the LLM writes, read and corrected in place
//   Relations: the map of the tables and every relation measured on the data
//   Métriques: the metrics on the left instead of the tables, each tested on the data (counts, never a value)
//   Fiches: what the agent reads of each table (its document) against the files today
// The steps in their order, one after the other, only in the guided setup, opened on purpose.
// Everything comes from the API of offline/studio/app.py; the page keeps no state of its own but the
// chosen database, tab, filter and selection.

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("fr-FR");
const pct = (x) => (x == null ? "—" : `${fmt.format(Math.round(x * 100))} %`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
// SQL as the eval dashboard shows it: its keywords in the accent (the list of offline/eval/eval_lib/dashboard_template.html,
// with CREATE TABLE for the documents of the Fiches tab)
const SQL_KW = /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|HAVING|LIMIT|AS|AND|OR|NOT|IN|EXISTS|DISTINCT|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|WITH|UNION|DESC|ASC|IS|NULL|BETWEEN|LIKE|ROUND|CAST|CREATE|TABLE)\b/gi;
const sqlHtml = (sql) => esc(sql).replace(SQL_KW, (m) => `<span class="sql-kw">${m}</span>`);
const plural = (n, word) => `${fmt.format(n)} ${word}${n > 1 ? "s" : ""}`;

// The tabs of a database, in their order: their title, sentence, filter and actions (none: the tab only shows)
const STEPS = {
  overview: { title: "Vue d'ensemble", key: "index", todo: "À réindexer" },
  profile: {
    number: 1, title: "Contenu", key: "profile", todo: "À profiler",
    text: "On examine le contenu réel de chaque table pour que l'agent la lise correctement : les colonnes vides, les dates ou montants enregistrés comme du texte, les données personnelles, et les partitions, qui disent s'il faut lire la dernière photo ou toutes.",
    primary: (n) => `Profiler la sélection (${n})`, secondary: () => "Profiler les manquantes",
  },
  semantics: {
    number: 2, title: "Descriptions", key: "semantics", todo: "À décrire",
    text: "L'agent choisit ses tables et écrit ses requêtes à partir de ces descriptions. L'IA les rédige à partir de la structure des tables et de leur profil ; tu les relis et tu les corriges ici.",
    primary: () => "Décrire les manquantes", secondary: (n) => `Redécrire la sélection (${n})`,
  },
  relations: {
    number: 3, title: "Relations", key: "relations", todo: "Isolées",
    text: "L'agent passe d'une table à l'autre grâce à ces liens. Chacun est vérifié sur les données : la part des valeurs d'une table qu'on retrouve dans l'autre.",
    primary: () => "Chercher les relations", secondary: () => "Ajouter une relation",
  },
  metrics: {
    number: 4, title: "Métriques", key: "metrics", todo: "À revoir",
    text: "L'agent copie ces formules au lieu de refaire ses calculs. L'IA les propose ; chacune est testée sur les données (lignes comptées deux fois, mesures vides, dates manquantes) sans jamais en montrer le résultat. Une métrique modifiée arrive chez l'agent à son prochain démarrage, sans réindexer.",
    primary: () => "Proposer des métriques", secondary: () => (ui.testing ? "Arrêter les tests" : "Tout tester"),
  },
  index: {
    number: 5, title: "Fiches", key: "index", todo: "À réindexer",
    text: "Ce que l'agent lit de chaque table : sa fiche, faite des descriptions et des relations, et les valeurs de ses colonnes catégorielles lues dans la base. Une correction n'arrive chez l'agent qu'une fois réindexée, à son prochain démarrage.",
  },
};
const TABS = ["overview", "profile", "semantics", "relations", "metrics", "index"];
// The two tabs that read the index: the whole database, and each table's document
const indexScreen = (step = ui.step) => step === "overview" || step === "index";

const ui = {
  dbs: [], db: null, step: "profile", data: null, filter: "all", query: "", selected: new Set(),
  rel: null, path: { a: null, b: null, result: null }, adding: false,   // the Relations screen
  relCheck: null,                                                      // a relation measured, not added (its target is no key)
  met: null, metric: null, metricKind: "metric", testing: null,        // the Metrics screen (testing: {current, stop, panel, done, total})
  tested: null,                                                        // what the last tests in the job panel found: {title: [title, halo], job, at}
  idx: null,                                                           // the Index screen
  home: false, overview: null,                                         // the home: every database, read in the files
  wizard: null,                                                        // the guided setup of the database: {step}
  current: null, detail: null, job: null, since: 0, polling: null, logOpen: false,   // logOpen: the detail of the job
  clockOffset: 0,   // seconds the clock of the page is ahead of the server's
};

// The theme: light, the system's or dark, remembered in this browser under the key of the eval dashboard, which the
// studio serves on the same address: one choice for both pages
const THEME_KEY = "sql-agent-eval-theme";
function applyTheme(mode) {
  if (mode === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = mode;
  [...$("theme-switch").children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeMode === mode)));
}
$("theme-switch").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-theme-mode]");
  if (!button) return;
  applyTheme(button.dataset.themeMode);
  store(THEME_KEY, button.dataset.themeMode);
});
applyTheme(["light", "dark"].includes(stored(THEME_KEY)) ? stored(THEME_KEY) : "system");

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
// A list remembered as JSON: empty when nothing, or something unreadable, is stored
function storedList(key) {
  try { const list = JSON.parse(stored(key) || "[]"); return Array.isArray(list) ? list : []; } catch (e) { return []; }
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
  // The studio opens on its home, unless the address names a database (#Chinook/semantics) or a tab of the last one
  const target = readAddress();
  if (target.db) ui.db = target.db;
  $("db-select").innerHTML = ui.dbs.map((d) =>
    `<option value="${esc(d.name)}" ${d.name === ui.db ? "selected" : ""}>${esc(d.name)}</option>`).join("");
  $("db-select").onchange = (e) => { switchDb(e.target.value); goTo(ui.step, true); };
  ui.step = STEPS[target.screen] ? target.screen : "overview";
  ui.home = Boolean(target.home);
  history.replaceState(null, "", address());   // the first entry of the history names its screen: Back returns to it
  renderStep();
  renderDetail();
  if (ui.home) {
    openHome();   // no database is asked until one opens
  } else {
    if (target.screen === WIZARD) openWizard();
    await loadTables();
    await loadScreen();
    if (!ui.idx) await loadIndex();
  }
  await pollJob();
}

// The address of the screen: #accueil, #<database>/<tab>, #<database>/mise-en-place
const WIZARD = "mise-en-place";
function address() {
  return ui.home ? `#${HOME}` : `#${encodeURIComponent(ui.db)}/${ui.wizard ? WIZARD : ui.step}`;
}
// What an address names: the home, or a database and a screen (an address of before, #semantics: the last database)
function readAddress() {
  const raw = decodeURIComponent(location.hash.slice(1));
  if (!raw || raw === HOME) return { home: true };
  const [db, screen] = raw.includes("/") ? raw.split("/") : [null, raw];
  if (db && !ui.dbs.some((d) => d.name === db)) return { home: true };
  if (!db && !STEPS[screen]) return { home: true };
  return { db, screen };
}
// The address follows the screen: a new entry in the history, unless the address already moved (Back, a link)
function remember() {
  if (location.hash !== address()) history.pushState(null, "", address());
}

// Another database: the screens of the previous one are forgotten
function switchDb(db) {
  ui.db = db; store("studio.db", db); $("db-select").value = db;
  ui.selected.clear(); ui.current = null; ui.detail = null; ui.rel = null; ui.path = { a: null, b: null, result: null };
  ui.relCheck = null; closeConfirm($("confirm"));
  ui.met = null; ui.metric = null; if (ui.testing) ui.testing.stop = true; ui.idx = null;
  ui.tested = null;   // the tests of the other database leave the panel
}

// The database and its tabs, the sentence, filter and actions of the current tab
function renderStep() {
  const step = STEPS[ui.step];
  $("step-text").textContent = step.text || "";
  $("origin-legend").innerHTML = ORIGIN_LEGEND;
  $("origin-legend").hidden = !ORIGIN_STEPS.includes(ui.step);
  $("tab-bar").hidden = !step.text && !step.primary;
  $("crumb-tab").textContent = step.title;
  if (!ui.home && !ui.wizard) document.title = `${step.title} · ${ui.db} · Studio sémantique`;
  document.body.dataset.step = ui.step;
  renderBase();
  renderTabs();
  const metrics = ui.step === "metrics";
  $("search").placeholder = metrics ? "Rechercher une métrique" : "Rechercher une table";
  $("list-name").textContent = metrics ? "Métrique" : "Table";
  $("list-status").textContent = metrics ? "État" : ui.step === "index" ? "Index" : "Statut";
  $("filters").querySelector('[data-filter="review"]').hidden = ui.step !== "semantics";
  $("filters").querySelector('[data-filter="noread"]').hidden = ui.step !== "profile" || !ui.data?.tables.some((t) => t.read_warn);
  $("type-filter-wrap").hidden = ui.step !== "profile";
  $("back-to-list").textContent = metrics ? "← Métriques" : "← Tables";
  closeConfirm($("confirm"));
  renderActions();
  renderJob();   // a job of another screen folds into its quiet line
}

// The name of the database, what it is, its size and its index; its actions on all of it (an eval needs questions)
function renderBase() {
  const db = ui.dbs.find((d) => d.name === ui.db), idx = ui.idx, s = ui.data?.db === ui.db ? ui.data.summary : null;
  $("base-name").textContent = ui.db || "";
  $("base-dialect").textContent = DIALECTS[db?.dialect] || db?.dialect || "";
  $("base-tagline").textContent = db?.tagline || "";
  $("base-tagline").hidden = !db?.tagline;
  const indexed = !idx ? null : idx.trace ? `indexée le ${when(idx.built_at)}` : idx.values_at ? `index du ${when(idx.values_at)}` : "jamais indexée";
  $("base-meta").textContent = [s && plural(s.tables, "table"), indexed].filter(Boolean).join(" · ");
  $("base-eval").hidden = idx?.dataset === false;
  $("menu-eval").hidden = !idx?.eval;
  $("menu-eval").href = `/eval/${encodeURIComponent(ui.db)}`;
}

// The tabs of the database, a count on the ones that hold a list
function renderTabs() {
  const s = ui.data?.db === ui.db ? ui.data.summary : null;
  const counts = { relations: s?.relations, metrics: s?.metrics, index: ui.idx?.trace ? ui.idx.documents : null };
  $("tabs").innerHTML = TABS.map((key) => `<button type="button" data-step="${key}"${key === ui.step ? ' aria-current="page"' : ""}>`
    + `${STEPS[key].title}${counts[key] != null ? `<span class="tab-count">${fmt.format(counts[key])}</span>` : ""}</button>`).join("");
}
$("tabs").addEventListener("click", (e) => {
  const tab = e.target.closest("button[data-step]");
  if (tab) goTo(tab.dataset.step);
});

// The actions on all the database, in its head whatever the tab
$("base-reindex").addEventListener("click", () => runJob("index/run", {}));
$("base-eval").addEventListener("click", () => confirmEval());
$("base-more").addEventListener("click", (e) => {
  e.stopPropagation();
  const open = $("base-menu").hidden;
  $("base-menu").hidden = !open;
  $("base-more").setAttribute("aria-expanded", String(open));
});
document.addEventListener("click", () => { $("base-menu").hidden = true; $("base-more").setAttribute("aria-expanded", "false"); });
$("menu-wizard").addEventListener("click", () => openWizard());

// The screens whose content is not the tables: read when they open
function loadScreen() {
  if (ui.step === "relations") return loadRelations();
  if (ui.step === "metrics") return loadMetrics();
  if (indexScreen()) return loadIndex();
}

// A tab of the current database (`force`: drawn again though it is the current one, after a change of database).
// The promise ends once the screen is read: its tables when they are not the database's, and its own content
function goTo(step, force = false) {
  if (!STEPS[step]) return;
  if (step === ui.step && !ui.home && !ui.wizard && !force) return;
  if (ui.home) leaveHome();
  if (ui.wizard) leaveWizard();
  ui.step = step;
  remember();
  ui.filter = "all"; ui.selected.clear(); ui.detail = null; ui.adding = false; ui.relCheck = null;
  [...$("filters").children].forEach((c) => c.classList.toggle("on", c.dataset.filter === "all"));
  renderStep();
  const tables = ui.data?.db !== ui.db ? loadTables() : null;
  const index = !ui.idx && !indexScreen() ? loadIndex() : null;   // the head of the database tells its index
  if (step === "relations" || step === "metrics" || indexScreen()) {
    if (step === "index" && !force) ui.current = null;   // the Fiches tab opens on no table
    if (ui.data) { renderHeader(); renderRows(); }
    renderDetail();
    return Promise.all([tables, index, loadScreen()]);
  }
  if (ui.data) { renderHeader(); renderRows(); }
  if (ui.current) openTable(ui.current); else renderDetail();
  return Promise.all([tables, index]);
}

// The Back button of the browser, and a link to an address: the screen follows it
function followAddress() {
  const target = readAddress();
  if (target.home) { if (!ui.home) openHome(); return; }
  const other = target.db && target.db !== ui.db;
  if (other) switchDb(target.db);
  if (target.screen === WIZARD) { if (ui.home) leaveHome(); openWizard(); if (other) loadTables().then(() => loadIndex()); return; }
  goTo(target.screen, other);
}
window.addEventListener("popstate", followAddress);
window.addEventListener("hashchange", followAddress);

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
  renderHeader(); renderRows(); renderBase(); renderTabs(); renderWizard();
}

function renderHeader() {
  const { data } = ui, s = data.summary;
  const when = new Date(data.read_at * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  $("refresh").title = data.source === "database"
    ? `Relire la base (${s.tables} tables lues à ${when})` : "Relire la base (tables lues dans les fichiers)";
  if (data.error) showBanner(`La base ne répond pas : les tables viennent de profile.yaml et tables.yaml. ${data.error}`);
  else $("banner").hidden = true;
  const todo = { profile: s.tables - s.profiled, semantics: undescribed(s),
    relations: ui.data.tables.filter((t) => t.status.relations !== "done").length,
    metrics: ui.met ? ui.met.counts.check + ui.met.counts.error : 0,
    index: ui.idx ? ui.idx.counts.changed + ui.idx.counts.missing : 0 }[STEPS[ui.step].key];
  const review = ui.data.tables.filter((t) => t.rewritten).length;
  const noread = ui.data.tables.filter((t) => t.read_warn).length;
  const counts = { all: ui.step === "metrics" ? ui.met?.metrics.length ?? 0 : s.tables, todo, review, noread };
  const labels = { all: "Toutes", todo: STEPS[ui.step].todo, review: "À relire", noread: "Sans mode" };
  $("filters").querySelector('[data-filter="noread"]').hidden = ui.step !== "profile" || !noread;
  if (ui.filter === "noread" && !noread) {   // the last one given a mode: back to every table, not an empty list
    ui.filter = "all";
    [...$("filters").children].forEach((c) => c.classList.toggle("on", c.dataset.filter === "all"));
  }
  const byType = (type) => ui.data.tables.filter((t) => (t.type || "none") === type).length;
  [...$("type-filter").options].forEach((o) => {
    o.textContent = o.value ? `${typeName(o.value === "none" ? null : o.value).toLowerCase()} (${byType(o.value)})` : "tous les types";
  });
  [...$("filters").children].forEach((chip) => {
    chip.innerHTML = `${labels[chip.dataset.filter]}<span class="count">${counts[chip.dataset.filter]}</span>`;
  });
}

// The tables "À décrire": never described, or with columns added since
const undescribed = (s) => s.tables - (s.described - s.partial);

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
  if (indexScreen()) return INDEX_PILLS[indexStatus(table.name)] || "";
  const status = table.status[STEPS[ui.step].key];
  if (ui.step === "profile") {
    if (status === "none") return '<span class="pill todo">à profiler</span>';
    return table.read_warn ? '<span class="pill decide" title="Partitionnée, sans mode de lecture : l\'agent lit toutes les partitions ensemble">sans mode</span>'
      : '<span class="pill done">profilée</span>';
  }
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

// The types of table (offline/studio/table_types.py): their name and what a row of each holds
const TABLE_TYPES = {
  events: ["Événements", "une ligne par chose qui s'est passée : commande, facture, opération"],
  entity: ["Entité", "une ligne par client, produit, employé…"],
  reference: ["Référentiel", "une petite liste de codes : catégorie, pays, statut"],
  bridge: ["Liaison", "relie deux entités, plusieurs à plusieurs"],
};
const typeName = (type) => TABLE_TYPES[type]?.[0] || "À classer";

function visibleRows() {
  const q = ui.query.toLowerCase(), key = STEPS[ui.step].key;
  const type = ui.step === "profile" ? $("type-filter").value : "";   // the type chosen in the list of the Contenu tab
  const todo = indexScreen() ? (t) => ["changed", "missing"].includes(indexStatus(t.name)) : (t) => t.status[key] !== "done";
  const shown = ui.filter === "review" ? (t) => t.rewritten > 0 : ui.filter === "noread" ? (t) => t.read_warn : todo;
  return (ui.data?.tables || []).filter((t) => (!q || t.name.toLowerCase().includes(q)) && (ui.filter === "all" || shown(t))
    && (!type || (t.type || "none") === type));
}

function renderRows() {
  renderPersonal();   // the bars of the personal data and of an older profile: on the Contenu tab only
  renderOutdated();
  if (ui.step === "metrics") { renderMetricRows(); return; }
  $("list-extra").innerHTML = "";
  $("empty").textContent = "Aucune table ne correspond.";
  const rows = visibleRows();
  $("rows").innerHTML = rows.map((t) => `<div class="row${t.name === ui.current ? " current" : ""}" data-table="${esc(t.name)}">
      <input type="checkbox" ${ui.selected.has(t.name) ? "checked" : ""} aria-label="Sélectionner ${esc(t.name)}">
      ${ui.step === "profile" ? `<span class="cell named" title="${esc(t.name)}"><span>${esc(t.name)}</span>`
        + `<small>${t.type_edited ? `${originMark("you", "Type corrigé par toi")} ` : ""}${typeName(t.type).toLowerCase()}</small></span>`
        : `<span class="cell" title="${esc(t.name)}">${esc(t.name)}</span>`}
      ${statusPill(t)}
    </div>`).join("");
  $("empty").hidden = rows.length > 0;
  $("select-all").checked = rows.length > 0 && rows.every((t) => ui.selected.has(t.name));
  if (indexScreen()) renderGone();
  renderActions();
}

// The actions of the current tab, in its bar (the Vue d'ensemble and the Fiches have none: the actions on all the
// database are in its head)
function renderActions() {
  const step = STEPS[ui.step];
  $("run-primary").hidden = !step.primary;
  $("run-secondary").hidden = !step.secondary;
  if (step.primary) $("run-primary").textContent = step.primary(ui.selected.size);
  if (step.secondary) $("run-secondary").textContent = step.secondary(ui.selected.size);
  // "Ajouter une relation" and the tests of the metrics start no job
  $("run-secondary").toggleAttribute("data-job", ui.step !== "relations" && ui.step !== "metrics");
  gateButtons();
}

// What a job is, in the title of a button it greys
const JOB_WHAT = { profile: "profilage", semantics: "description", relations: "relations", metrics: "métriques", index: "index", eval: "évaluation" };

// Why one of the two actions of the page has nothing to do now, or null
function idleReason(button) {
  const s = ui.data?.summary, step = ui.step;
  if (button.id === "run-primary") {
    if (step === "profile" && !ui.selected.size) return "Coche des tables dans la liste";
    if (step === "semantics" && s && !undescribed(s)) return "Toutes les tables sont décrites";
    if (step === "metrics" && ui.testing) return "Attends la fin des tests";
  } else if (button.id === "run-secondary") {
    if (step === "profile" && s && s.profiled >= s.tables) return "Toutes les tables sont profilées";
    if (step === "semantics" && !ui.selected.size) return "Coche des tables dans la liste";
    if (step === "metrics" && !ui.testing && ui.job?.running && ui.job.step === "metrics") return "Attends la fin de la proposition : ses métriques seront testées";
    if (step === "metrics" && !ui.testing && !(ui.met?.metrics || []).some((m) => m.sql)) return "Aucune métrique à tester";
  }
  return null;
}

// The buttons that start a job (data-job) greyed while one runs, the actions of the page greyed when they have
// nothing to do: the reason on hover, their own title otherwise (kept in data-title)
function gateButtons() {
  const job = ui.job?.running ? `Attends la fin du job en cours : ${JOB_WHAT[ui.job.step] || ui.job.step} · ${ui.job.db}` : null;
  document.querySelectorAll("button[data-job], #run-primary, #run-secondary").forEach((button) => {
    if (button.dataset.title === undefined) button.dataset.title = button.title;
    const reason = (button.hasAttribute("data-job") && job) || idleReason(button);
    button.disabled = Boolean(reason);
    button.title = reason || button.dataset.title;
  });
}

// Personal data: one rule for the whole database, not a correction to review table by table
function renderPersonal() {
  if (!ui.data) return;
  const { hidden, kept, visible, hidden_tables: tables } = ui.data.personal, bar = $("personal-bar");
  bar.hidden = ui.step !== "profile" || !(hidden + kept + visible);
  if (bar.hidden) return;
  bar.classList.toggle("warn", Boolean(visible));
  bar.innerHTML = visible
    ? `<span><b>${plural(visible, "donnée")} ${visible > 1 ? "personnelles" : "personnelle"}</b> ${visible > 1 ? "visibles" : "visible"} par l'agent</span>
       <button class="btn small primary" id="personal-all" data-hide="1">Cacher partout</button>`
    : `<span>Données personnelles cachées (${fmt.format(hidden)})${kept ? ` · ${fmt.format(kept)} gardée${kept > 1 ? "s" : ""} visible${kept > 1 ? "s" : ""}` : ""}</span>
       <button class="btn small ghost" id="personal-all" data-hide="0">Tout montrer</button>`;
  $("personal-all").onclick = (e) => {
    if (e.target.dataset.hide === "1") { hidePersonal(true); return; }
    openConfirm($("confirm"), {
      text: `<b>Tout montrer</b> : ${plural(hidden, "colonne")} ${hidden > 1 ? "personnelles" : "personnelle"}`
        + `${tables ? ` de ${plural(tables, "table")}` : ""} ${hidden > 1 ? "redeviendront visibles" : "redeviendra visible"} par l'agent, `
        + "dans ses fiches, les exemples de lignes envoyés à l'IA et ses requêtes.",
      button: () => "Tout montrer", run: () => hidePersonal(false), job: false,
    });
  };
}

// Tables profiled by an older profiler: what they lack measured in one job, nothing measured again
function renderOutdated() {
  const bar = $("outdated-bar"), tables = (ui.data?.tables || []).filter((t) => t.outdated?.length);
  bar.hidden = ui.step !== "profile" || !tables.length;
  if (bar.hidden) return;
  const missing = Object.keys(OUTDATED).filter((m) => tables.some((t) => t.outdated.includes(m)));
  bar.innerHTML = `<span title="Profil fait par une version précédente du studio : ${outdatedSaid(missing)}"><b>${plural(tables.length, "table")}</b>`
    + ` au profil incomplet : ${missing.map((m) => OUTDATED[m][2]).join(", ")}</span>
    <button class="btn small" id="outdated-all" data-job title="${COMPLETE_TITLE}">Compléter</button>`;
  $("outdated-all").onclick = () => runJob("profile", { complete: true });
}

async function hidePersonal(hide) {
  try {
    ui.data = await api(`/api/db/${encodeURIComponent(ui.db)}/personal`, { method: "POST", body: { hide, version: ui.data.version } });
  } catch (error) {
    showBanner(error.status === 409 ? "tables.yaml a changé : relis la base avant de recommencer." : error.message, true);
    return;
  }
  renderHeader(); renderRows();
  if (ui.current) openTable(ui.current);
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

// ↓ and ↑ open the next or the previous row of the list as shown (filters and search applied), as a click would: a
// table, its place on the map of Relations, a metric; ↓ opens the first one when none is. The arrows keep their own
// use in a field, while a menu is open, with a modifier, and where there is no list (home, guided setup, overview)
function stepRow(by) {
  const rows = [...$("rows").querySelectorAll(":scope > .row")];
  const at = rows.findIndex((row) => row.classList.contains("current"));
  const next = at < 0 ? (by > 0 ? rows[0] : null) : rows[at + by];
  if (!next) return false;
  if (ui.step === "metrics") openMetric(next.dataset.metric, "metric");
  else openTable(next.dataset.table);
  return true;
}
document.addEventListener("keydown", (e) => {
  if ((e.key !== "ArrowDown" && e.key !== "ArrowUp") || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  if (e.target.closest?.("textarea, select, [contenteditable], input:not([type=checkbox])")) return;
  if (document.querySelector(".type-menu:not([hidden]), .menu:not([hidden])") || !$("rows").offsetParent) return;
  if (stepRow(e.key === "ArrowDown" ? 1 : -1)) e.preventDefault();
});

$("select-all").addEventListener("change", (e) => {
  visibleRows().forEach((t) => (e.target.checked ? ui.selected.add(t.name) : ui.selected.delete(t.name)));
  renderRows();
});
$("search").addEventListener("input", (e) => { ui.query = e.target.value; renderRows(); });
$("type-filter").addEventListener("change", () => renderRows());
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
  const fromList = !document.body.classList.contains("card-open");   // on a phone: the card comes in view, once
  if (ui.step === "relations") { selectOnMap(name); renderRows(); showCurrentRow(); if (fromList) showCard(); return; }
  ui.current = name;
  renderRows(); showCurrentRow();
  const step = ui.step, db = encodeURIComponent(ui.db), table = encodeURIComponent(name);
  try {
    ui.detail = await api(step === "index" ? `/api/db/${db}/index/${table}`
      : `/api/db/${db}/tables/${table}${step === "profile" ? "" : "/semantics"}`);
  } catch (e) {
    ui.detail = { name, error: e.message };
  }
  if (ui.current === name && ui.step === step) { renderDetail(); if (fromList) showCard(); }
}

// The row opened, in view in its list, which scrolls on its own (a table opened from the search, a list scrolled away)
function showCurrentRow() {
  const box = $("list-scroll"), row = box.querySelector(".row.current");
  if (!row) return;
  if (row.offsetTop < box.scrollTop) box.scrollTop = row.offsetTop;
  else if (row.offsetTop + row.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = row.offsetTop + row.offsetHeight - box.clientHeight;
}

// On a phone a card takes the place of its list: in view once opened (studio.css, body.card-open), from its way back
function showCard() {
  if (window.matchMedia("(max-width: 560px)").matches) ($("back-to-list").hidden ? $("detail") : $("back-to-list")).scrollIntoView({ block: "start" });
}

$("back-to-list").addEventListener("click", () => {
  if (ui.step === "metrics") ui.metric = null;
  else { ui.current = null; ui.detail = null; }
  renderRows(); renderDetail();
});

function renderDetail(message) {
  const d = ui.detail, el = $("detail");
  el.classList.toggle("bare", ui.step === "overview");   // the Vue d'ensemble holds two panels of its own
  el.classList.remove("menu-open");   // drawn again, its menus closed
  if (ui.step === "relations") renderRelations(el, message);
  else if (ui.step === "metrics") renderMetric(el, message);
  else if (indexScreen()) renderIndex(el, message);
  else if (!d) el.innerHTML = '<p class="placeholder">Choisis une table pour voir ses colonnes.</p>';
  else if (d.error) el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2></div><p class="note">${esc(d.error)}</p>`;
  else if (ui.step === "profile") renderProfile(d, el, message);
  else renderSemantics(d, el, message);
  // A card open: on a phone it takes the place of the list. On Relations, a table chosen or a path asked, whose way
  // back is in the card ("← Toutes les relations")
  const open = ui.step === "metrics" ? Boolean(ui.metric) : ui.step === "relations" ? Boolean(ui.current || ui.path.a) : Boolean(ui.current);
  document.body.classList.toggle("card-open", open);
  $("back-to-list").hidden = ui.step === "relations";
  gateButtons();
}

// "date dd/MM/yyyy HH:mm:ss" → "date jj/mm/aaaa hh:mm:ss", "decimal" → "nombre"
function formatLabel(format) {
  if (format === "decimal") return "nombre";
  return format.replace("yyyy", "aaaa").replace("dd", "jj").replace("MM", "mm").replace("HH", "hh");
}

const LOW_FILL = 0.01;   // a fill bar under it turns amber: the profile proposes to hide the column (EMPTY_FILL)

// A pill of "Contenu détecté" that a click turns into an exception (POST .../columns); icon: its eye, open or not
const pillButton = (c, action, style, label, title, icon = "") =>
  `<button class="pill ${style}" data-column="${esc(c.column)}" data-action="${action}" title="${esc(title)}">${icon}${esc(label)}</button>`;

// Seen by the agent, or hidden from it
const EYE = '<svg class="eye" viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="2"/></svg>';
const EYE_OFF = '<svg class="eye" viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z"/><path d="M2.5 13.5l11-11"/></svg>';

// A format the profile detects, its mark before it: applied, or not yet (the data); kept as text or read in another
// format (you)
function formatPill(c, p) {
  const label = formatLabel(p.value), share = p.share != null ? `${pct(p.share)} des valeurs` : "la plupart des valeurs";
  if (p.state === "accepted") return withOrigin("data", pillButton(c, "text", "type", label, `Lue comme ${label} : ${share}. Cliquer pour la garder en texte.`), "Détecté sur les données");
  if (p.current) return withOrigin("you", `<span class="pill type" title="Format revu par toi (le profil détecte ${esc(label)})">${esc(formatLabel(p.current))}</span>`, "Revu par toi");
  if (p.state === "rejected") return withOrigin("you", pillButton(c, "format", "refused", label, `Détectée comme ${label} (${share}), gardée en texte par toi. Cliquer pour la lire comme ${label}.`), "Gardée en texte par toi");
  return withOrigin("data", pillButton(c, "format", "type pending", label, `Détectée comme ${label} (${share}), pas encore appliquée. Cliquer pour l'appliquer.`), "Détecté sur les données");
}

// An empty column, its mark before it: hidden, or not yet (the data); kept visible (you)
function emptyPill(c, p) {
  const label = c.filled ? "quasi vide" : "vide", filled = c.filled ? `remplie à ${pct(c.filled)}` : "jamais remplie";
  if (p.state === "accepted") return withOrigin("data", pillButton(c, "show", "off", `${label} · cachée`, `Cachée à l'agent : ${filled}. Cliquer pour la laisser visible.`, EYE_OFF), "Détecté sur les données");
  if (p.state === "rejected") return withOrigin("you", pillButton(c, "hide", "refused", label, `${filled}, gardée visible par toi. Cliquer pour la cacher.`), "Gardée visible par toi");
  return withOrigin("data", pillButton(c, "hide", "decide pending", label, `${filled}, pas encore cachée. Cliquer pour la cacher.`), "Détecté sur les données");
}

// Personal data, hidden in the whole database at once (personal bar): the pill says its state and makes the exception.
// Grey once hidden (the result wanted), an outline when kept visible on purpose, amber while the agent sees it (to decide);
// its origin: the data (detected by its name or its values), you for the exception
const PERSONAL_PILLS = {
  hidden:  ["data", "show", "off", "personnelle · cachée", "Cachée à l'agent. Cliquer pour la laisser visible.", EYE_OFF],
  kept:    ["you", "hide", "kept", "personnelle · visible exprès", "Gardée visible par toi. Cliquer pour la cacher.", EYE],
  visible: ["data", "hide", "decide", "personnelle · visible par l'agent", "Visible par l'agent, à décider. Cliquer pour la cacher.", EYE],
};

// "Contenu détecté": what the data holds when it differs from the declared type, each pill an exception, its origin before it
function detected(c) {
  const parts = [];
  const format = c.proposals.find((p) => p.kind === "format");
  if (format) parts.push(formatPill(c, format));
  else if (c.format) parts.push(withOrigin("you", `<span class="pill type" title="Format écrit par toi dans tables.yaml">${esc(formatLabel(c.format))}</span>`, "Écrit par toi"));
  if (c.ambiguous) parts.push('<span class="muted-text" title="Aucune valeur ne dit si le jour vient avant le mois : lue en jj/mm, la convention des données vues jusqu\'ici.">jour/mois ?</span>');
  const empty = c.proposals.find((p) => p.kind === "exclude");
  if (empty) parts.push(emptyPill(c, empty));
  if (c.personal) {
    const [origin, ...pill] = PERSONAL_PILLS[c.personal];
    parts.push(withOrigin(origin, pillButton(c, ...pill), origin === "you" ? "Gardée visible par toi" : "Détectée sur les données"));
  }
  return parts.join(" ");
}

// missing: the title of a fill the profile did not measure
function fillBar(filled, missing = "") {
  if (filled == null) return `<span class="muted-text"${missing ? ` title="${esc(missing)}"` : ""}>—</span>`;
  return `<span class="bar"><i class="${filled < LOW_FILL ? "low" : ""}" style="width:${Math.round(filled * 100)}%"></i></span>${pct(filled)}`;
}

// What an older profiler did not measure (detail.outdated, the outdated of a row), each with its own agreement,
// and its short name (the bar on top of the list)
const OUTDATED = { filled: ["le remplissage des dates et des nombres", "n'est pas mesuré", "remplissage"],
  distinct: ["les valeurs distinctes du texte", "ne sont pas mesurées", "valeurs distinctes"],
  partitions: ["les partitions", "ne sont pas lues", "partitions"] };
const outdatedSaid = (missing) => missing.length > 1
  ? `${missing.map((m) => OUTDATED[m][0]).join(" et ")} ne sont pas ${missing.includes("filled") ? "mesurés" : "mesurées"}`
  : missing.length ? `${OUTDATED[missing[0]][0]} ${OUTDATED[missing[0]][1]}` : "";
// "Compléter": make profile --complete, only what is missing
const COMPLETE_TITLE = "Ne mesure que ce qui manque : rien n'est mesuré à nouveau, tables.yaml reste tel quel";

// What is detected in a column, or a quiet dash (hidden on a phone: nothing to say)
function found(c) {
  const pills = detected(c);
  return pills ? `<span class="cell">${pills}</span>` : '<span class="cell nothing"><span class="muted-text">—</span></span>';
}

// Hidden by hand in tables.yaml: no pill of what is detected says so, the name does
const hiddenByHand = (c) => c.excluded && !c.personal && !c.proposals.some((p) => p.kind === "exclude");

function renderProfile(d, el, message) {
  if (!d.profiled) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="profile-one" data-job>Profiler</button></div>
      <p class="placeholder">Pas encore profilée${d.columns != null ? ` · ${plural(d.columns, "colonne")}` : ""}.</p>`;
    $("profile-one").onclick = () => runJob("profile", { tables: [d.name] });
    return;
  }

  // Figures of the profile, one line under the name (the key is said by its pill in the table); the partitions
  // said in the three cases, just above the read mode they explain (nothing on an engine without partitions)
  const r = d.reading, parts = r?.partitioned_by || [];
  const partitions = parts.length ? `partitionnée par ${esc(parts.join(", "))}${r.partitions ? ` (${plural(r.partitions, "partition")})` : ""}`
    : r?.partitions === 0 ? "non partitionnée" : r?.unread ? "partitions non lues" : null;
  const figures = [plural(d.rows, "ligne"), plural(d.columns ?? d.column_facts.length, "colonne"), partitions,
    d.formats ? plural(d.formats, "format") : "aucun format"].filter(Boolean);
  const error = message ? `<p class="note error">${esc(message)}</p>` : "";
  // A profile of an older profiler: what it did not measure, completed without measuring the rest again
  const missing = [...(d.outdated || []), ...(d.reading?.unread ? ["partitions"] : [])].filter((m) => OUTDATED[m]);
  const old = missing.length ? `<div class="note with-action"><span>Profil fait par une version précédente du studio : ${outdatedSaid(missing)}.</span>
      <button class="btn ghost small" id="profile-complete" data-job title="${COMPLETE_TITLE}">Compléter</button></div>` : "";

  // Columns: declared type against what the profile found (distinct values of a text column: on the sample of the profile)
  const lines = d.column_facts.map((c) => `<div class="line${c.excluded ? " hidden-col" : ""}">
      <span class="cell">${esc(c.column)}${c.key ? ` <span class="pill key">${c.key === "unique" ? "clé unique" : "clé"}</span>` : ""}${hiddenByHand(c) ? ` ${withOrigin("you", '<span class="pill off">cachée</span>', "Cachée par toi")}` : ""}</span>
      <span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>
      ${found(c)}
      <span class="fill">${fillBar(c.filled, d.outdated?.includes("filled") ? "Pas mesuré par ce profil" : "")}</span>
      <span class="num"${c.sampled ? ' title="Valeurs distinctes sur l\'échantillon du profil"' : ""}>${c.distinct == null ? "" : fmt.format(c.distinct)}</span>
    </div>`).join("");

  el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="figures">${figures.join(" · ")}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      <button class="btn ghost small" id="profile-again" data-job>Re-profiler</button></div>`
    + `<div class="table-text"><div class="props" aria-label="Propriétés de la table">${typeProp(d.table_type)}${readProp(d.reading)}${asOfProp(d)}</div>`
    + `${readBand(d.reading)}</div>`
    + old + error
    + `<div class="columns">
        <div class="line head"><span>Colonne</span><span>Type en base</span><span>Contenu détecté</span><span class="num">Remplissage</span><span class="num">Distinctes</span></div>
        ${lines}
      </div>`;

  $("profile-again").onclick = () => runJob("profile", { tables: [d.name] });
  if (missing.length) $("profile-complete").onclick = () => runJob("profile", { tables: [d.name], complete: true });
  setupProps(d);
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
  const hidden = c.columns - c.visible;
  return [`${fmt.format(c.described)}/${plural(c.visible, "colonne")} décrite${c.visible > 1 ? "s" : ""}`,
    hidden ? `${fmt.format(hidden)} cachée${hidden > 1 ? "s" : ""}` : null,
    c.rewritten ? `${fmt.format(c.rewritten)} à relire` : null,
    c.categorical ? `${fmt.format(c.categorical)} catégorielle${c.categorical > 1 ? "s" : ""}` : "aucune catégorielle",
    c.edited ? `${fmt.format(c.edited)} corrigée${c.edited > 1 ? "s" : ""} par toi` : null].filter(Boolean).join(" · ");
}

// ── Where a value comes from, the same marks and words on every tab: measured on the data (the profile, the rules,
// the measure of a relation), proposed by the IA (both grey), decided by you (ink); an amber dot, rewritten by the
// IA over a text: to read again; "— avant le suivi", written before the studio kept the origin ──

const PENCIL = '<path d="M10.5 2.5l3 3L6 13l-3.5.5L3 10z"/>';
const ORIGIN = {
  data: { word: "données", says: "mesuré sur les données", path: '<path d="M3.5 13V9M8 13V3.5M12.5 13V7"/>' },
  ai:   { word: "IA", says: "proposé par l'IA", path: '<path d="M8 1.8l1.45 4.2 4.25 1.5-4.25 1.5L8 13.2 6.55 9 2.3 7.5 6.55 6z"/>' },
  you:  { word: "toi", says: "décidé par toi", path: PENCIL },
};
const BEFORE_TRACKING = "Écrite avant que le studio note l'origine";

// The mark of an origin (attrs: data attributes, hidden), its title said on hover
const originMark = (kind, title = ORIGIN[kind].says, attrs = "") =>
  `<svg class="origin ${kind}" viewBox="0 0 16 16" role="img" aria-label="${esc(title)}" ${attrs}><title>${esc(title)}</title>${ORIGIN[kind].path}</svg>`;
// A mark before what it qualifies (a pill)
const withOrigin = (kind, html, title) => `<span class="origin-item">${originMark(kind, title)}${html}</span>`;

// The legend of the tabs that show origins, in the footer (the marks say it on hover too)
const ORIGIN_LEGEND = `<span>Origine :</span>${["data", "ai", "you"].map((kind) =>
  `<span class="${kind}">${originMark(kind)}${ORIGIN[kind].says}</span>`).join("")}`
  + '<span class="review"><span class="todo-dot"></span>à relire</span><span>— avant le suivi</span>';
const ORIGIN_STEPS = ["profile", "semantics", "relations"];

// The marks of a description (column: its name, "" for the table): rewritten by the IA (an amber dot, to read
// again), written by the IA or corrected by you (kept when the table is described again), toggled as it is saved
const rewrittenDot = (column, on) => `<span class="todo-dot" data-dot="${esc(column)}" title="Réécrite par l'IA par-dessus ton texte : à relire" ${on ? "" : "hidden"}></span>`;
const descriptionMarks = (column, rewritten, edited) => rewrittenDot(column, rewritten)
  + originMark("ai", "Rédigée par l'IA", `data-ai="${esc(column)}" ${edited ? "hidden" : ""}`)
  + originMark("you", "Corrigée par toi : gardée quand l'IA redécrit", `data-hand="${esc(column)}" ${edited ? "" : "hidden"}`);

function renderSemantics(d, el, message) {
  if (!d.described) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="describe-one" data-job>Décrire</button></div>
      <p class="placeholder">Pas encore décrite : l'IA rédigera la description de la table et de ses colonnes.</p>`;
    $("describe-one").onclick = () => runJob("semantics", { tables: [d.name] });
    return;
  }
  const lines = d.columns.map((c) => {
    const key = c.key ? ` <span class="pill key">${c.key === "unique" ? "clé unique" : "clé"}</span>` : "";
    if (c.excluded) {
      return `<div class="sline hidden-col"><span class="cell">${esc(c.column)}${key} <span class="pill off">cachée</span></span>
        <span class="cell mono">${esc((c.type || "—").toLowerCase())}</span>
        <span class="muted-text">cachée à l'agent : pas de description</span><span></span></div>`;
    }
    const values = c.distinct == null ? "" : `<span class="values" title="${c.sampled ? "Valeurs distinctes sur l'échantillon du profil" : "Valeurs distinctes de la table"}">${plural(c.distinct, "valeur")}</span>`;
    return `<div class="sline">
      <span class="cell">${esc(c.column)}${key}</span>
      <span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>
      <span class="desc-cell">${descriptionMarks(c.column, c.rewritten, c.edited)}<textarea class="edit" rows="1" data-field="column" data-column="${esc(c.column)}" placeholder="Ajouter une description…" aria-label="Description de ${esc(c.column)}">${esc(c.description)}</textarea></span>
      <label class="switch-cell">${values}<input type="checkbox" class="switch" data-field="categorical" data-column="${esc(c.column)}" ${c.categorical ? "checked" : ""} aria-label="${esc(c.column)} catégorielle"></label>
    </div>`;
  }).join("");

  el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="figures" id="sem-figures">${semanticsFigures(d)}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      <button class="btn ghost small" id="mark-read" title="Les descriptions réécrites par l'IA sortent de « à relire »" ${d.counts.rewritten ? "" : "hidden"}>Tout marquer relu</button>
      <button class="btn ghost small" id="describe-again" data-job title="L'IA réécrit les descriptions de cette table ; celles corrigées par toi sont gardées">Redécrire</button></div>
    <div class="confirm" id="sem-confirm" role="group" aria-label="Redécrire ${esc(d.name)}" hidden></div>
    ${message ? `<p class="note error">${esc(message)}</p>` : ""}
    <div class="table-text">
      <label class="field-label" for="sem-description">Description de la table</label>
      <span class="desc-cell block">${descriptionMarks("", d.description_rewritten, d.description_edited)}<textarea class="edit block" id="sem-description" rows="2" data-field="description">${esc(d.description)}</textarea></span>
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
  el.querySelectorAll("input.switch").forEach((box) => box.addEventListener("change", () => saveSemantics(box, box.checked)));
}

// ── The properties of a table, in a bar under its name in the Contenu card (what it is and how to read it): a pill each
// (its name in grey, its value), the explanation at the top of its menu; a band under the bar only when one asks for
// something ──

const CARET = '<svg width="11" height="11" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>';
const PROPS = ["type", "read", "asof"];

// The text of some HTML, for a tooltip
function plainText(html) {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent;
}

// origin: the mark before the value (data, you; none when nothing was detected nor chosen)
function propPill(name, key, value, { tone = "", title = "", origin = null } = {}) {
  return `<button type="button" class="prop${tone ? ` ${tone}` : ""}" id="${name}-pill" aria-haspopup="true" aria-expanded="false" title="${esc(title)}">
    <span class="prop-key">${key}</span>${tone === "warn" ? '<span class="prop-dot" aria-hidden="true"></span>' : ""}`
    + `${origin ? originMark(origin) : ""}<span class="prop-value">${value}</span>${CARET}</button>`;
}

// One menu open at a time (name: null closes them all); open alone toggles it. While one is open, the card lets it
// go past its bottom (a short card would cut it)
function toggleProp(name, open) {
  let any = false;
  PROPS.forEach((p) => {
    const menu = $(`${p}-menu`);
    if (!menu) return;
    const on = p === name && (open ?? menu.hidden);
    menu.hidden = !on;
    $(`${p}-pill`).setAttribute("aria-expanded", String(on));
    any ||= on;
  });
  $("detail").classList.toggle("menu-open", any);
}
document.addEventListener("click", (e) => { if (!e.target.closest(".prop-wrap")) toggleProp(null); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") toggleProp(null); });

// The type of the table: the one shown, why the rules gave theirs at the top of its menu; another one, or the detection again
function typeProp(t) {
  if (!t) return "";
  const why = t.edited ? `Corrigé par toi · la détection dit : ${typeName(t.detected).toLowerCase()} (${t.reason})` : `Détecté sur les données : ${t.reason}`;
  const option = (type, [name, holds]) => `<button type="button" role="option" class="type-option${t.type === type ? " on" : ""}" data-type="${type}"
      aria-selected="${t.type === type}"><span class="type-check">${t.type === type ? CHECK : ""}</span>
      <span><b>${name}</b><small>${holds}</small></span><span class="type-tag">${t.detected === type ? "détecté" : ""}</span></button>`;
  return `<span class="prop-wrap" id="type-wrap">
      ${propPill("type", "Type", typeName(t.type), { tone: t.type ? "" : "none", title: why, origin: t.edited ? "you" : t.type ? "data" : null })}
      <div class="type-menu" id="type-menu" role="listbox" aria-label="Type de la table" hidden>
        <div class="menu-head">${esc(why)}${ui.data?.summary.relations ? ""
          : '<span class="menu-fact">Les relations ne sont pas encore trouvées (onglet Relations) : le type peut encore changer.</span>'}</div>
        ${Object.entries(TABLE_TYPES).map(([type, words]) => option(type, words)).join("")}
        ${t.edited ? `<button type="button" role="option" class="type-option" data-type=""><span class="type-check"></span>`
          + `<span><b>Revenir à la détection</b><small>${typeName(t.detected).toLowerCase()}</small></span><span></span></button>` : ""}
        <p>Ton choix est écrit dans tables.yaml et passe avant la détection.</p>
      </div>
    </span>`;
}

// A property written in tables.yaml (the type of the table, the date its data stops): the card gets the properties
// back (the type may follow), the list its type; null when refused, the reason next to the save state
async function saveProperty(field, value) {
  const d = ui.detail, state = $("save-state");
  state.textContent = "Enregistrement…";
  state.className = "save-state";
  try {
    const saved = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(d.name)}/properties`,
      { method: "POST", body: { field, value, version: d.version } });
    if (ui.detail?.name !== d.name) return null;
    Object.assign(ui.detail, saved);
    state.textContent = "Enregistré";
    state.className = "save-state ok";
    loadTables();
    return saved;
  } catch (e) {
    state.textContent = e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message;
    state.className = "save-state error";
    return null;
  }
}

// A type chosen (null: the detection again): the card drawn again, its menu closed
async function saveTableType(type) {
  if (!(await saveProperty("table_type", type))) return;
  renderDetail();
  $("save-state").textContent = "Enregistré";
  $("save-state").className = "save-state ok";
}

// The read modes (offline/studio/reading.py): their name as the data teams say it (and tables.yaml, the document of
// the agent), what it means in a few words, what the agent does with each
const READ_MODES = {
  "": ["Aucun", "table lue telle quelle", "ni partitions ni versions à trier : l'agent lit la table telle quelle"],
  last: ["LAST", "dernière partition", "chaque partition est une photo complète : l'agent ne lit que la plus récente"],
  all: ["ALL", "toutes les partitions", "chaque partition ajoute des lignes stockées une fois : l'agent les lit toutes, sans dédoublonner"],
  live: ["LIVE", "dernière version", "plusieurs versions de chaque ligne : l'agent garde la plus récente de chaque clé"],
};
const modeName = (mode) => READ_MODES[mode || ""][0];
const col = (name) => `<span class="mono">${esc(name)}</span>`;

// What the agent does with the table, in a sentence: the top of the menu of the read mode, and the tooltip of its pill
function readingSentence(r) {
  if (r.problems.length) return `${esc(r.problems.join(", "))} : l'agent ne peut pas construire la fiche de la table`;
  if (r.mode === "last") return `une photo complète par ${col(r.partition_column)} : l'agent ne lit que la plus récente`;
  if (r.mode === "all") return `des ajouts par ${col(r.partition_column)}, chaque ligne stockée une fois : l'agent les lit toutes`;
  if (r.mode === "live") return `plusieurs versions par ${names(r.key_columns)} : l'agent garde la plus récente selon ${col(r.version_column)}`;
  if (r.partitioned_by.length) return "l'agent lit toutes les partitions ensemble, comme une seule table";
  return "l'agent lit la table telle quelle";
}

// What the profile says of the partitions, and the mode the data proposes when it is not the one written
function readingFacts(r) {
  const facts = [];
  if (r.partitioned_by.length) facts.push(`Partitionnée par ${names(r.partitioned_by)}${r.partitions ? ` · ${plural(r.partitions, "partition")}` : ""}, d'après le profil.`);
  else if (r.partitions === 0) facts.push("Pas de partition dans la table, d'après le profil.");
  else if (r.unread) facts.push("Partitions pas encore lues : « Compléter » le profil les lit (onglet Contenu).");
  if (r.suggested && r.suggested !== r.mode) facts.push(`Les données proposent ${modeName(r.suggested)} : ${esc(r.why)}.`);
  else if (!r.mode && r.partitioned_by.length && !r.suggested && r.why) facts.push(`Rien à proposer : ${esc(r.why)}.`);
  return facts.map((f) => `<span class="menu-fact">${f}</span>`).join("");
}

// The read mode of the table: its pill (amber when partitioned without a mode, red when the agent can't read it), a menu
// with what it does on top, the four modes, the columns of the one chosen
function readProp(r) {
  if (!r) return "";
  const sentence = readingSentence(r);
  const columns = r.mode === "live" ? `clé ${names(r.key_columns)} · version ${r.version_column ? col(r.version_column) : "—"}`
    : r.mode ? `colonne de partition ${r.partition_column ? col(r.partition_column) : "—"}` : "";
  const option = (mode, [name, means, does]) => {
    const on = (r.mode || "") === mode;
    return `<button type="button" role="option" class="type-option${on ? " on" : ""}" data-mode="${mode}" aria-selected="${on}">
      <span class="type-check">${on ? CHECK : ""}</span>
      <span><b>${name} · ${means}</b><small>${does}</small>${on && columns ? `<span class="option-sub">${columns}<span class="link">Changer</span></span>` : ""}</span>
      <span class="type-tag">${r.suggested === mode ? "proposé" : ""}</span></button>`;
  };
  return `<span class="prop-wrap" id="read-wrap">
      ${propPill("read", "Mode de lecture", modeName(r.mode), { tone: r.problems.length ? "bad" : r.warn ? "warn" : "", title: plainText(sentence),
        origin: r.mode ? "you" : null })}
      <div class="type-menu" id="read-menu" role="listbox" aria-label="Mode de lecture de la table" hidden>
        <div class="menu-head"><b>${modeName(r.mode)}</b> : ${sentence}.${readingFacts(r)}</div>
        ${Object.entries(READ_MODES).map(([mode, words]) => option(mode, words)).join("")}
        <div class="read-fields" id="read-fields" hidden></div>
        <p>Écrit dans tables.yaml : l'agent le lit au prochain index.</p>
      </div>
    </span>`;
}

// The one band under the bar, when the read mode asks for something: a mode the agent can't read (red), a partitioned
// table read without one whose rows may repeat, or a mode the data contradicts (amber); nothing otherwise
function readBand(r) {
  if (!r) return "";
  const band = (tone, title, text, action) => `<div class="band ${tone}" role="status">
      <div class="band-text">${BANG}<div><b>${title}</b><span>${text}</span></div></div>${action}</div>`;
  const apply = `<button type="button" class="btn small" id="read-apply">Appliquer ${modeName(r.suggested)}</button>`;
  const choose = (label) => `<button type="button" class="btn small" data-open-read>${label}</button>`;
  if (r.problems.length) {
    return band("bad", `Mode ${modeName(r.mode)} incomplet`, `${esc(r.problems.join(", "))} : l'agent ne peut pas construire la fiche de la table.`,
      choose("Choisir les colonnes"));
  }
  if (r.warn) {
    const read = r.partitions > 1 ? `les ${fmt.format(r.partitions)} partitions` : "toutes les partitions";
    return band("warn", `Partitionnée par ${names(r.partitioned_by)}${r.partitions ? ` (${plural(r.partitions, "partition")})` : ""}, sans mode de lecture`,
      `L'agent lit ${read} ensemble : une ligne présente dans plusieurs partitions est comptée plusieurs fois. `
      + (r.suggested ? `Proposé : <b>${modeName(r.suggested)}</b>, car ${esc(r.why)}.` : `Rien à proposer : ${esc(r.why)}.`),
      r.suggested ? apply : choose("Choisir un mode"));
  }
  if (r.mode && r.suggested && r.suggested !== r.mode) {
    return band("warn", `Les données ressemblent plutôt à ${modeName(r.suggested)}`,
      `${esc(r.why)}. Le mode écrit est ${modeName(r.mode)}.`, apply);
  }
  return "";
}

// The columns a mode needs, chosen in its menu: the partition (the measured one first), or the key and the version
function readFields(mode, d) {
  const r = d.reading, columns = d.column_facts.map((c) => c.column);
  const choices = (chosen, first = []) => [...new Set([...first, ...columns])]
    .map((c) => `<option value="${esc(c)}"${c === chosen ? " selected" : ""}>${esc(c)}</option>`).join("");
  if (mode === "last" || mode === "all") {
    const chosen = r.partition_column || r.partitioned_by[0] || "";
    return `<label class="read-field">Colonne de partition<select id="read-partition">${chosen ? "" : '<option value="">choisir…</option>'}
        ${choices(chosen, r.partitioned_by)}</select></label>
      <button type="button" class="btn primary small" id="read-save">Enregistrer</button>`;
  }
  const keys = new Set(r.key_columns);
  const version = r.version_column || d.column_facts.find((c) => c.kind === "date" && !r.partitioned_by.includes(c.column))?.column || "";
  return `<span class="read-field">Clé : la ou les colonnes d'une même ligne d'une version à l'autre</span>
    <div class="read-keys">${columns.map((c) => `<label><input type="checkbox" value="${esc(c)}"${keys.has(c) ? " checked" : ""}>${esc(c)}</label>`).join("")}</div>
    <label class="read-field">Version : la plus grande est la plus récente<select id="read-version">${version ? "" : '<option value="">choisir…</option>'}
      ${choices(version)}</select></label>
    <button type="button" class="btn primary small" id="read-save">Enregistrer</button>`;
}

// The date the data stops: a pill (dashed while there is none), a field under it
const AS_OF = "La date de référence de « ce mois-ci », « l'an dernier »… Sans elle, les périodes partent d'aujourd'hui.";
const dayMonthYear = (iso) => iso.split("-").reverse().join("/");
const asOfPill = (d) => (d.data_as_of ? propPill("asof", "Arrêtée au", dayMonthYear(d.data_as_of), { title: AS_OF })
  : `<button type="button" class="prop empty" id="asof-pill" aria-haspopup="true" aria-expanded="false" title="${esc(AS_OF)}">+ Date d'arrêt</button>`);
function asOfProp(d) {
  return `<span class="prop-wrap" id="asof-wrap">
      ${asOfPill(d)}
      <div class="type-menu asof-menu" id="asof-menu" role="dialog" aria-label="Date d'arrêt des données" hidden>
        <label class="read-field" for="sem-as-of">Données arrêtées au
          <input type="date" class="edit date" id="sem-as-of" data-field="data_as_of" value="${esc(d.data_as_of)}"></label>
        <p>${AS_OF}</p>
        <button type="button" class="btn ghost small" id="asof-clear" ${d.data_as_of ? "" : "hidden"}>Retirer la date</button>
      </div>
    </span>`;
}

function setupProps(d) {
  PROPS.forEach((p) => {
    if ($(`${p}-pill`)) $(`${p}-pill`).onclick = (e) => { e.stopPropagation(); toggleProp(p); };
  });
  $("detail").querySelectorAll("[data-type]").forEach((b) => { b.onclick = () => saveTableType(b.dataset.type || null); });
  // A mode with columns: they are chosen below the options, then saved; "Aucun" is saved at once
  const pick = (mode) => {
    $("read-menu").querySelectorAll("[data-mode]").forEach((o) => o.classList.toggle("picked", o.dataset.mode === mode));
    const fields = $("read-fields");
    fields.innerHTML = readFields(mode, d);
    fields.hidden = false;
    $("read-save").onclick = () => saveReadMode(mode === "live"
      ? { mode, key_columns: [...fields.querySelectorAll(".read-keys input:checked")].map((box) => box.value), version_column: $("read-version").value || null }
      : { mode, partition_column: $("read-partition").value || null });
  };
  if (d.reading) {
    $("read-menu").querySelectorAll("[data-mode]").forEach((b) => {
      b.onclick = () => (b.dataset.mode ? pick(b.dataset.mode) : saveReadMode({ mode: null }));
    });
    $("detail").querySelectorAll("[data-open-read]").forEach((b) => {
      b.onclick = (e) => { e.stopPropagation(); toggleProp("read", true); if (d.reading.mode) pick(d.reading.mode); };
    });
    if ($("read-apply")) $("read-apply").onclick = () => saveReadMode({ mode: d.reading.suggested, partition_column: d.reading.partitioned_by[0] });
  }
  // The date: written when it is chosen, its pill redrawn alone (the field keeps the focus while it is typed)
  const asOf = $("sem-as-of");
  if (!asOf) return;
  const saveAsOf = async (value) => {
    if (!(await saveProperty("data_as_of", value)) || !$("asof-pill")) return;
    flashSaved(asOf);
    const open = !$("asof-menu").hidden;
    $("asof-pill").outerHTML = asOfPill(ui.detail);
    $("asof-pill").onclick = (e) => { e.stopPropagation(); toggleProp("asof"); };
    $("asof-pill").setAttribute("aria-expanded", String(open));
    $("asof-clear").hidden = !ui.detail.data_as_of;
  };
  asOf.addEventListener("change", () => saveAsOf(asOf.value));
  $("asof-clear").onclick = () => { asOf.value = ""; saveAsOf(""); toggleProp(null); };
}

// A read mode chosen (mode null: the table read as it is), written in tables.yaml; refused when the agent could not
// read it (a column missing), the reason shown next to the save state
async function saveReadMode(body) {
  const d = ui.detail, state = $("save-state");
  state.textContent = "Enregistrement…";
  state.className = "save-state";
  try {
    const saved = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(d.name)}/read-mode`,
      { method: "POST", body: { ...body, version: d.version } });
    if (ui.detail?.name !== d.name) return;
    ui.detail = { ...ui.detail, ...saved };
    renderDetail();
    $("save-state").textContent = "Enregistré";
    $("save-state").className = "save-state ok";
    loadTables();
  } catch (e) {
    state.textContent = e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message;
    state.className = "save-state error";
  }
}

// The marks of the page after a save, without drawing it again (the focus stays where it is)
function updateMarks(d) {
  const edited = new Set(d.columns.filter((c) => c.edited).map((c) => c.column));
  const rewritten = new Set(d.columns.filter((c) => c.rewritten).map((c) => c.column));
  $("detail").querySelectorAll("[data-dot]").forEach((dot) => {
    dot.hidden = dot.dataset.dot ? !rewritten.has(dot.dataset.dot) : !d.description_rewritten;
  });
  // written by the IA until corrected by you: one mark or the other
  $("detail").querySelectorAll("[data-hand], [data-ai]").forEach((mark) => {
    const column = mark.dataset.hand ?? mark.dataset.ai;
    const yours = column ? edited.has(column) : d.description_edited;
    mark.toggleAttribute("hidden", mark.dataset.hand !== undefined ? !yours : yours);
  });
  $("sem-figures").textContent = semanticsFigures(d);
  $("mark-read").hidden = !d.counts.rewritten;
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

// The strip before an action that costs or exposes: what it does, and before describing again what the LLM keeps,
// the corrections replaced on a click (the box only when there are corrections: edited). job: it starts a job (data-job)
function openConfirm(strip, { text, edited, button, run, job = true }) {
  strip.innerHTML = `<p>${text}</p>
    <span class="confirm-actions">
      ${edited ? `<label class="check"><input type="checkbox" data-replace>Remplacer aussi ${edited > 1 ? `mes ${fmt.format(edited)} corrections` : "ma correction"}</label>` : ""}
      <button class="btn small ghost" type="button" data-cancel>Annuler</button>
      <button class="btn small primary" type="button" data-run${job ? " data-job" : ""}>${button(false)}</button>
    </span>`;
  strip.hidden = false;
  const replace = strip.querySelector("[data-replace]"), go = strip.querySelector("[data-run]");
  if (replace) replace.onchange = () => { go.textContent = button(replace.checked); };
  strip.querySelector("[data-cancel]").onclick = () => closeConfirm(strip);
  go.onclick = () => { closeConfirm(strip); run(Boolean(replace?.checked)); };
  gateButtons();
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

// A field just written: its frame turns green a moment, where the eye is (the header may be out of sight)
function flashSaved(input) {
  if (!input.classList.contains("edit")) return;
  input.classList.add("saved");
  setTimeout(() => input.classList.remove("saved"), 1200);
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
    flashSaved(input);
    loadTables();
  } catch (e) {
    state.textContent = e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message;
    state.className = "save-state error";
  }
}

// ── Relations: the map of the tables, the path of the agent, the evidence ──

const KEY_OVERLAP = 0.9;   // values found under it: the relation is in amber (offline/studio/relations.py)

// The origin of a relation (relationships.yaml) as the marks of the studio say it: its kind, its words, its title
const ORIGINS = { profile: ["data", "mesurée sur les données", "Clé de même nom dont les valeurs se retrouvent (profil)"],
  llm: ["ai", "proposée par l'IA", "Proposée par l'IA à partir des noms et des descriptions"],
  manual: ["you", "ajoutée par toi", "Ajoutée par toi dans le studio"] };

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
  if (ui.step !== "relations") return;
  renderDetail();
  if (ui.path.a && ui.path.b && !ui.path.result) findPath();   // a path asked from another screen (a metric left out)
}

// A click on a table, in the list or on the map, opens its relations and makes it the start of a path:
// the end of the path is chosen in the list "à" only
function selectOnMap(name) {
  ui.current = name;
  ui.relFocus = null;
  ui.path = { a: name, b: null, result: null };
  renderDetail();
}

// The tables of the path shown, and the step of each: a relation is on the path between two steps that follow
const pathLayers = () => new Map((ui.path.result?.layers || []).flatMap((tables, i) => tables.map((t) => [t, i])));
const onPath = (r, layer) => layer.has(r.source) && layer.has(r.target) && Math.abs(layer.get(r.source) - layer.get(r.target)) === 1;

// The SVG of the map: a box per table (its name), a curve per relation from the right side of the table pointing to
// the left side of the one it points to, a dot on the key. The line says what the data showed: solid measured,
// pale not measured yet, amber dashes weak; the table chosen, its relations and their cardinality in orange
function mapSvg() {
  const { nodes, width, height } = ui.rel.layout, layer = pathLayers(), path = layer.size > 0;
  const cur = path ? null : ui.current;   // a path shown alone: no table chosen in orange over it
  const near = new Set(ui.rel.relations.filter((r) => r.source === cur || r.target === cur).flatMap((r) => [r.source, r.target]));
  const pairs = new Map();   // two relations between the same tables: side by side, not one over the other
  const labels = [];   // the cardinalities of the relations of the table chosen, drawn together (cardinalities)
  const edges = ui.rel.relations.map((r, i) => [r, i]).filter(([r]) => nodes[r.source] && nodes[r.target]).map(([r, i]) => {
    const pair = [r.source, r.target].sort().join("|"), n = pairs.get(pair) || 0;
    pairs.set(pair, n + 1);
    const curve = edgePath(nodes[r.source], nodes[r.target], r.source === r.target, n ? (n % 2 ? 6 : -6) * Math.ceil(n / 2) : 0);
    const mine = Boolean(cur) && (r.source === cur || r.target === cur), inPath = onPath(r, layer);
    const cls = ["e", r.weak ? "weak" : r.found == null ? "unchecked" : "", mine ? "on" : "",
      (cur && !mine) || (path && !inPath) ? "dim" : "", inPath ? "path" : ""].join(" ");
    if (mine) labels.push(...cardinality(r, curve, i));
    return `<g class="${cls}" data-rel="${i}" data-s="${esc(r.source)}" data-t="${esc(r.target)}">
      <path class="line" d="${curve.d}"/><circle class="key" cx="${curve.end[0]}" cy="${curve.end[1]}" r="2.5"/>
      <path class="hit" d="${curve.d}"/></g>`;
  }).join("");
  const isolated = new Set(ui.rel.isolated);
  const boxes = Object.entries(nodes).map(([table, n]) => {
    const cls = ["node", table === cur ? "on" : "", near.has(table) && table !== cur ? "near" : "", isolated.has(table) ? "iso" : "",
      layer.has(table) ? "path" : "", (cur && table !== cur && !near.has(table)) || (path && !layer.has(table)) ? "dim" : ""].join(" ");
    const type = ui.data?.db === ui.db ? ui.data.tables.find((t) => t.name === table) : null;
    return `<g class="${cls}" data-table="${esc(table)}" tabindex="0" role="button" aria-label="${esc(table)}">
      ${type ? `<title>${esc(table)} · ${typeName(type.type).toLowerCase()}</title>` : ""}
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="7"/>
      <text x="${n.x + n.w / 2}" y="${n.y + n.h / 2 + 4}" text-anchor="middle">${esc(table)}</text></g>`;
  }).join("");
  return `<svg class="map" id="map" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="Carte des tables et de leurs relations">${edges}${cardinalities(labels)}${boxes}</svg>`;
}

// The curve of a relation, where it starts and where it ends (on the key): left to right between two columns,
// right to left when the table pointed to is on the left, out on the right and back within one column, a loop
// for a table pointing to itself. dy: the place of a second relation between the same tables, beside the first
function edgePath(s, t, self, dy = 0) {
  if (self) {
    const x = s.x + s.w, y = s.y + s.h / 2;
    return { d: `M${x} ${y + 5} C${x + 34} ${y + 22} ${x + 34} ${y - 22} ${x + 2} ${y - 5}`, start: [x, y + 5], end: [x + 2, y - 5], self: true };
  }
  const sy = s.y + s.h / 2 + dy, ty = t.y + t.h / 2 + dy;
  const curve = (x1, x2, bend) => ({ d: `M${x1} ${sy} C${x1 + bend} ${sy} ${x2 - bend} ${ty} ${x2} ${ty}`, start: [x1, sy], end: [x2, ty] });
  if (t.x > s.x + s.w + 8) return curve(s.x + s.w, t.x, Math.max(30, (t.x - s.x - s.w) * 0.45));
  if (t.x + t.w < s.x - 8) return curve(s.x, t.x + t.w, -Math.max(30, (s.x - t.x - t.w) * 0.45));
  const x1 = s.x + s.w, x2 = t.x + t.w, out = Math.max(x1, x2) + 40;
  return { d: `M${x1} ${sy} C${out} ${sy} ${out} ${ty} ${x2} ${ty}`, start: [x1, sy], end: [x2, ty] };
}

// "n" and "1" at the two ends of a relation of the table chosen (how many rows on each side for one on the other):
// where each is written, by the end of the line it belongs to (rel: the index of the relation, toward: the height of
// the other end of the line)
function cardinality(r, curve, rel) {
  const [from, to] = r.links.split("→").map((side) => side.trim());
  if (curve.self) return [{ x: curve.start[0] + 34, y: curve.end[1] - 13, anchor: "end", text: r.links, rel, toward: 0 }];   // over the loop
  const [x1, y1] = curve.start, [x2, y2] = curve.end, right = x2 >= x1;
  return [{ x: x1 + (right ? 6 : -6), y: y1, anchor: right ? "start" : "end", text: from, rel, toward: y2 },
    { x: x2 + (right ? -6 : 6), y: y2, anchor: right ? "end" : "start", text: to, rel, toward: y1 }];
}

// The relations of a table leave from one point and reach another table at one point: a value shared by the lines
// of an end is written once (a fork: "n" once for the two lines leaving a table); two values at the same end, one
// over the lines and one under them, the one of the line going up over. Each label knows its relations, for the
// hover to fade it with them
function cardinalities(labels) {
  const ends = new Map();
  labels.forEach((l) => {
    const key = `${l.x}|${l.y}|${l.anchor}`;
    if (!ends.has(key)) ends.set(key, new Map());
    const values = ends.get(key);
    if (!values.has(l.text)) values.set(l.text, { ...l, rels: [], towards: [] });
    values.get(l.text).rels.push(l.rel);
    values.get(l.text).towards.push(l.toward);
  });
  const mean = (list) => list.reduce((a, b) => a + b, 0) / list.length;
  return [...ends.values()].flatMap((values) => [...values.values()].sort((a, b) => mean(a.towards) - mean(b.towards))
    .map((l, i) => `<text class="card" data-rels="${l.rels.join(" ")}"
    x="${l.x}" y="${l.y + (i ? 13 : -5)}" text-anchor="${l.anchor}">${esc(l.text)}</text>`)).join("");
}

// What a relation says in the bubble of the map: its columns, its cardinality, what the data showed, its origin
function relationTip(r) {
  const measured = r.fanout ? `chaque ligne de ${esc(r.source)} comptée ${decimal(r.fanout)} fois`
    : r.empty ? `${esc(r.source)} est vide : rien à mesurer`
    : r.found == null ? "pas encore mesurée" : `${pct(r.found)} des valeurs trouvées${r.weak ? ", peu" : ""}`;
  const origin = ` · ${ORIGINS[r.origin]?.[1] || "origine d'avant le suivi"}`;
  return `<span class="mono">${esc(r.from)}</span> → <span class="mono">${esc(r.to)}</span><br>${esc(r.links)} · ${measured}${origin}`;
}

// ── The view of the map: the part the svg shows (zoom, drag), kept while the screen draws again ──

// The width the buttons of the zoom take at the right of the map: no table framed under them
const ZOOM_SPACE = 48;

// A box of the map whole and centred, left of the zoom, never larger than 125 %
function frameBox(box, cw, ch) {
  const w = cw - ZOOM_SPACE, scale = Math.min(w / box.w, ch / box.h, 1.25);
  return { scale, x: box.x + box.w / 2 - w / scale / 2, y: box.y + box.h / 2 - ch / scale / 2 };
}

// The view a map opens on: whole when it fits; a big map at the width of the panel (its names readable), from its
// top, the rest a drag away
function firstView(cw, ch) {
  const { width, height } = ui.rel.layout, w = cw - ZOOM_SPACE, scale = Math.max(0.5, Math.min(w / width, 1.25));
  return height * scale <= ch ? frameBox(mapBox(), cw, ch) : { scale, x: (width - w / scale) / 2, y: 0 };
}

// The whole map, or the table chosen and the tables it joins
function mapBox(table = null) {
  const { nodes, width, height } = ui.rel.layout;
  if (!table || !nodes[table]) return { x: 0, y: 0, w: width, h: height };
  const near = new Set([table, ...ui.rel.relations.filter((r) => r.source === table || r.target === table).flatMap((r) => [r.source, r.target])]);
  const boxes = [...near].map((t) => nodes[t]).filter(Boolean);
  const x = Math.min(...boxes.map((n) => n.x)) - 40, y = Math.min(...boxes.map((n) => n.y)) - 30;
  return { x, y, w: Math.max(...boxes.map((n) => n.x + n.w)) + 60 - x, h: Math.max(...boxes.map((n) => n.y + n.h)) + 30 - y };
}

function applyView() {
  const svg = $("map"), v = ui.mapView;
  if (svg && v) svg.setAttribute("viewBox", `${v.x} ${v.y} ${svg.clientWidth / v.scale} ${svg.clientHeight / v.scale}`);
}

// Zoom by a factor around a point of the svg (its centre by default)
function zoomMap(factor, px = null, py = null) {
  const svg = $("map"), v = ui.mapView, w = svg.clientWidth, h = svg.clientHeight;
  const ox = px ?? w / 2, oy = py ?? h / 2, scale = Math.min(2.5, Math.max(0.2, v.scale * factor));
  ui.mapView = { ...v, scale, x: v.x + ox / v.scale - ox / scale, y: v.y + oy / v.scale - oy / scale };
  applyView();
}

// The map once drawn: its height (the whole map at the width of the panel, at most 600 px), its view, the table
// chosen brought into it, then a drag that moves it, the zoom, the hover of a table and the bubble of a relation
function setupMap() {
  const wrap = $("map-wrap"), svg = $("map"), { width, height } = ui.rel.layout, cur = ui.current;
  const cw = wrap.clientWidth || 800;
  svg.style.height = `${Math.max(260, Math.min(600, Math.round(height * Math.min(cw / width, 1.25))))}px`;
  const ch = svg.clientHeight;
  if (ui.mapView?.db !== ui.db) ui.mapView = { db: ui.db, focus: null, ...firstView(cw, ch) };
  const node = cur && ui.rel.layout.nodes[cur], v = ui.mapView;
  if (node && v.focus !== cur) {   // a table chosen out of view: the view goes to it, the zoom stays
    const vw = cw / v.scale, vh = ch / v.scale;
    if (node.x < v.x || node.y < v.y || node.x + node.w > v.x + vw || node.y + node.h > v.y + vh) {
      ui.mapView = { ...v, x: node.x + node.w / 2 - vw / 2, y: node.y + node.h / 2 - vh / 2 };
    }
  }
  ui.mapView.focus = cur;
  applyView();

  $("zoom-in").onclick = () => zoomMap(1.25);
  $("zoom-out").onclick = () => zoomMap(0.8);
  $("zoom-fit").onclick = () => { ui.mapView = { ...ui.mapView, ...frameBox(mapBox(cur), svg.clientWidth, svg.clientHeight) }; applyView(); };
  svg.addEventListener("wheel", (e) => {   // a pinch of the trackpad, or Ctrl and the wheel; the wheel alone scrolls the page
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const box = svg.getBoundingClientRect();
    zoomMap(e.deltaY < 0 ? 1.1 : 0.9, e.clientX - box.left, e.clientY - box.top);
  }, { passive: false });

  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".node, .hit")) return;
    drag = { x: e.clientX, y: e.clientY, view: { ...ui.mapView } };
    svg.setPointerCapture(e.pointerId);
    svg.classList.add("panning");
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    ui.mapView = { ...drag.view, x: drag.view.x - (e.clientX - drag.x) / drag.view.scale, y: drag.view.y - (e.clientY - drag.y) / drag.view.scale };
    applyView();
  });
  const stop = () => { drag = null; svg.classList.remove("panning"); };
  svg.addEventListener("pointerup", stop);
  svg.addEventListener("pointercancel", stop);

  // A table hovered: its relations and the tables they join stand out, the rest fades
  svg.querySelectorAll(".node").forEach((g) => {
    const table = g.dataset.table;
    g.addEventListener("click", () => openTable(table));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter") openTable(table); });
    g.addEventListener("mouseenter", () => {
      svg.classList.add("hovering");
      svg.querySelectorAll(".e").forEach((e) => {
        const mine = e.dataset.s === table || e.dataset.t === table;
        e.classList.toggle("hl", mine);
        if (mine) [e.dataset.s, e.dataset.t].forEach((t) => svg.querySelector(`.node[data-table="${CSS.escape(t)}"]`)?.classList.add("hl"));
      });
      // a cardinality stays while one of its relations does
      svg.querySelectorAll(".card").forEach((c) => c.classList.toggle("hl",
        c.dataset.rels.split(" ").some((i) => svg.querySelector(`.e[data-rel="${i}"]`)?.classList.contains("hl"))));
      g.classList.add("hl");
    });
    g.addEventListener("mouseleave", () => {
      svg.classList.remove("hovering");
      svg.querySelectorAll(".hl").forEach((el) => el.classList.remove("hl"));
    });
  });

  // A relation hovered: its bubble by the pointer; clicked: its table chosen, its line found in the list
  const tip = $("map-tip");
  svg.querySelectorAll(".e").forEach((g) => {
    const r = ui.rel.relations[g.dataset.rel];
    g.addEventListener("mousemove", (e) => {
      const box = wrap.getBoundingClientRect();
      tip.innerHTML = relationTip(r);
      tip.hidden = false;
      tip.style.left = `${Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 8)}px`;
      tip.style.top = `${Math.min(e.clientY - box.top + 14, box.height - tip.offsetHeight - 8)}px`;
    });
    g.addEventListener("mouseleave", () => { tip.hidden = true; });
    g.addEventListener("click", () => {
      ui.relFocus = `${r.from}|${r.to}`;
      ui.current = r.source;
      ui.path = { a: r.source, b: null, result: null };
      renderRows(); renderDetail();
      document.querySelector(".rline.focus")?.scrollIntoView({ block: "nearest" });
    });
  });
}

function pathText() {
  const p = ui.path;
  if (!p.a || !p.b) return "";   // the lists say what is missing ("départ…", "arrivée…")
  if (!p.result) return '<span class="muted-text">…</span>';
  if (p.result.joins == null) return '<span class="slow">aucun chemin : ces deux tables ne sont reliées par aucune suite de relations</span>';
  // The tables of the path are on the map: the path itself in a bubble, on hover, focus or tap
  const steps = p.result.layers.map((tables) => tables.map((t) => `<span class="path-table">${esc(t)}</span>`).join(" | ")).join(" → ");
  return `<span class="muted-text">· <span class="path-joins" tabindex="0">${plural(p.result.joins, "jointure")}`
    + `<span class="path-tip" role="tooltip">${steps}</span></span></span>`
    + (p.result.bridged ? "" : ` <span class="slow">· au-delà de 3 : l'agent ne fera pas ce chemin seul</span>`);
}

function relationLine(r) {
  const [origin, , title] = ORIGINS[r.origin] || [null, null, BEFORE_TRACKING];
  // measured with nothing to compare (its table empty), or never measured
  const found = r.empty ? `<span class="muted-text" title="${esc(r.source)} n'a aucune ligne : rien à comparer, la vérifier à nouveau n'y change rien">table vide</span>`
    : r.found == null ? '<span class="muted-text" title="Pas encore mesurée sur les données">—</span>'
    : `<span class="bar"><i class="${r.found < KEY_OVERLAP ? "low" : ""}" style="width:${Math.round(r.found * 100)}%"></i></span>${pct(r.found)}`;
  const focus = ui.relFocus === `${r.from}|${r.to}`;
  return `<div class="rline${r.weak ? " weak" : ""}${focus ? " focus" : ""}">
    <span class="cell mono rel" title="${esc(r.from)} → ${esc(r.to)}"><span>${esc(r.from)}</span> <span><span class="arrow">→</span> ${esc(r.to)}</span></span>
    <span class="fill">${found}</span>
    ${r.fanout ? `<span><span class="pill decide mono" title="${esc(r.to)} n'est pas une clé : chaque ligne de ${esc(r.source)} comptée ${decimal(r.fanout)} fois en moyenne">${esc(r.links)}</span></span>`
      : `<span><span class="pill links mono" title="Combien de lignes d'un côté pour une ligne de l'autre">${esc(r.links)}</span></span>`}
    <span class="cell" title="${title}">${origin ? `<span class="origin-word ${origin}">${originMark(origin, title)}${ORIGIN[origin].word}</span>`
      : '<span class="muted-text">— avant le suivi</span>'}</span>
    <button class="icon-btn small" data-ignore="${esc(r.from)}|${esc(r.to)}" title="Retirer : la relation sort de la liste et ne reviendra pas" aria-label="Retirer ${esc(r.from)} vers ${esc(r.to)}">✕</button>
  </div>`;
}

function renderRelations(el, message) {
  const rel = ui.rel;
  if (!rel) { el.innerHTML = '<p class="placeholder">Lecture des relations…</p>'; return; }
  const cur = ui.current, tables = ui.data?.tables.map((t) => t.name) || [], layer = pathLayers();
  const mine = (r) => r.source === cur || r.target === cur;
  const shown = (layer.size ? rel.relations.filter((r) => onPath(r, layer)) : cur ? rel.relations.filter(mine) : rel.relations)
    .slice().sort((a, b) => (b.weak - a.weak) || ((a.found == null) - (b.found == null)));
  const ignored = layer.size ? [] : cur ? rel.ignored.filter(mine) : rel.ignored;
  const heading = layer.size ? "Relations du chemin" : cur ? `Relations de ${esc(cur)}` : "Toutes les relations";
  const figures = [plural(rel.relations.length, "relation"),
    rel.isolated.length ? `${plural(rel.isolated.length, "table")} ${rel.isolated.length > 1 ? "isolées" : "isolée"}` : "aucune table isolée",
    rel.weak ? `${fmt.format(rel.weak)} à vérifier` : null]   // the ones never measured: in the note over the map
    .filter(Boolean).join(" · ");
  const options = (value, empty = "—") => `<option value="">${empty}</option>`
    + tables.map((t) => `<option ${t === value ? "selected" : ""}>${esc(t)}</option>`).join("");
  // The form keeps the relation just checked; a new one starts from the table chosen on the map
  const chosen = ui.relCheck ? { source: ui.relCheck.source.split("."), target: ui.relCheck.target.split(".") }
    : { source: [cur, null], target: [null, null] };

  // One bar over the map: the table chosen and the path between two tables. The relations never measured in one note,
  // not on each line of the list. In the map: the zoom at the top right, what the lines mean at the bottom left
  const measure = '<button class="btn small" id="check-relations" data-job title="Mesure sur les données la part des valeurs trouvées de chaque relation, sans l\'IA">';
  const line = (cls) => `<svg width="22" height="8" viewBox="0 0 22 8" aria-hidden="true"><path class="${cls}" d="M1 4 L21 4"/></svg>`;
  const fit = '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
  el.innerHTML = `<div class="detail-head"><h2>Carte des tables</h2><span class="figures">${figures}</span>
      <span class="grow"></span><span class="save-state" id="rel-state">${message ? esc(message) : ""}</span>
      ${rel.unchecked ? "" : `${measure.replace('btn small"', 'btn ghost small"')}Revérifier</button>`}</div>
    <div class="map-tools">
      ${cur ? `<span class="sel-chip">${esc(cur)}<button type="button" id="relations-all" aria-label="Voir toutes les tables">×</button></span><span class="tools-sep"></span>` : ""}
      <span class="muted-text">Chemin</span><select id="path-a" aria-label="Table de départ">${options(ui.path.a, "départ…")}</select>
      <span class="arrow">→</span><select id="path-b" aria-label="Table d'arrivée">${options(ui.path.b, "arrivée…")}</select>
      <span class="path-result">${pathText()}</span>
    </div>
    ${rel.unchecked ? `<div class="rule-note"><span>${rel.unchecked > 1 ? `${fmt.format(rel.unchecked)} relations jamais mesurées` : "Une relation jamais mesurée"} sur les données : `
      + `rien ne prouve encore que leurs valeurs se retrouvent d'une table à l'autre.</span>${measure}Vérifier sur les données</button></div>` : ""}
    <div class="map-wrap" id="map-wrap">${mapSvg()}
      <div class="map-zoom"><button type="button" id="zoom-in" aria-label="Zoomer" title="Zoomer">+</button>`
      + `<button type="button" id="zoom-out" aria-label="Dézoomer" title="Dézoomer">−</button>`
      + `<button type="button" id="zoom-fit" aria-label="Ajuster" title="${cur ? `Ajuster sur ${esc(cur)} et les tables qu'elle joint` : "Ajuster sur toute la carte"}">${fit}</button></div>
      <div class="map-legend"><span>${line("on")}vérifiée</span><span>${line("unchecked")}non mesurée</span><span>${line("weak")}faible</span></div>
      <div class="map-tip" id="map-tip" role="tooltip" hidden></div></div>
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
      <div class="rline head"><span>${heading}</span><span class="num">Valeurs trouvées</span><span>Liens</span><span>Origine</span><span></span></div>
      ${shown.map(relationLine).join("") || `<p class="placeholder">${cur ? "Aucune relation : l'agent ne peut joindre cette table à aucune autre." : "Aucune relation."}</p>`}
      ${ignored.length ? `<details class="ignored"><summary>${plural(ignored.length, "relation retirée")}</summary>
        ${ignored.map((r) => `<div class="rline"><span class="cell mono muted">${esc(r.from)} → ${esc(r.to)}</span><span></span><span></span><span></span>
          <button class="btn ghost small" data-restore="${esc(r.from)}|${esc(r.to)}">Rétablir</button></div>`).join("")}</details>` : ""}
    </div>`;

  setupMap();
  $("path-a").onchange = (e) => { ui.path.a = e.target.value || null; ui.path.result = null; findPath(); };
  $("path-b").onchange = (e) => { ui.path.b = e.target.value || null; ui.path.result = null; findPath(); };
  $("check-relations").onclick = () => runJob("relations/run", { check: true });
  if ($("relations-all")) $("relations-all").onclick = () => { ui.current = null; ui.relFocus = null; ui.path = { a: null, b: null, result: null }; renderRows(); renderDetail(); };
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
  const [state, warn] = added?.fanout ? [`Ajoutée par toi : chaque ligne de ${added.source} comptée ${decimal(added.fanout)} fois en moyenne`, true]
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
  error: '<span class="pill bad" title="L\'agent refusera de démarrer avec cette métrique">erreur</span>', untested: '<span class="pill todo">non testée</span>',
};

function renderMetricRows() {
  const met = ui.met, q = ui.query.toLowerCase();
  const rows = (met?.metrics || []).filter((m) => (!q || m.name.toLowerCase().includes(q))
    && (ui.filter === "all" || m.state === "check" || m.state === "error"));
  const current = (name, kind) => (name === ui.metric && kind === ui.metricKind ? " current" : "");
  const opened = new Set(storedList(openedKey())), fresh = (name) => met.added.includes(name) && !opened.has(name);
  $("rows").innerHTML = rows.map((m) => `<div class="row${current(m.name, "metric")}" data-metric="${esc(m.name)}">
      <span></span><span class="cell" title="${esc(m.name)}">${fresh(m.name) ? '<span class="todo-dot" title="Ajoutée par l\'IA au dernier lancement, pas encore ouverte"></span>' : ""}${esc(m.name)}</span>
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
  if (indexScreen()) {
    const gone = e.target.closest(".row[data-table]");
    if (gone) openTable(gone.dataset.table);
    return;
  }
  const row = e.target.closest(".row[data-kind]");
  if (row) openMetric(row.dataset.name, row.dataset.kind);
});

function openMetric(name, kind) {
  const fromList = !document.body.classList.contains("card-open");
  ui.metric = name; ui.metricKind = kind;
  if (kind === "metric" && ui.met?.added.includes(name)) rememberOpened(name);
  renderRows(); showCurrentRow(); renderDetail();
  if (fromList) showCard();
}

// The metrics the last run added that were opened, per database: their dot goes (only names of the last run kept)
const openedKey = () => `studio.opened.${ui.db}`;
function rememberOpened(name) {
  const opened = storedList(openedKey()).filter((n) => ui.met.added.includes(n));
  if (!opened.includes(name)) store(openedKey(), JSON.stringify([...opened, name]));
}

// Why the generator would not propose a metric today, or why it left a proposal out
const RULES = {
  count: (m) => `Un simple comptage d'une colonne : l'agent sait compter seul, et cette métrique force ${esc(m.tables.join(", ") || "sa table")} dans chaque question qui en parle. L'IA ne la proposerait plus.`,
  same: (m) => `Même formule que ${esc(m.rule_with)} : l'agent ne saura pas laquelle prendre.`,
  distinct: () => "SUM(DISTINCT …) ou AVG(DISTINCT …) : deux lignes de même montant ne compteraient qu'une fois.",
  known: () => "Ce nom est aussi dans les métriques retirées : l'IA ne la reproposera pas.",
};
const SKIPPED = {
  count: "Un simple comptage d'une colonne : l'agent compte seul.",
  same: "Même formule qu'une métrique déjà écrite, sous un autre nom.",
  distinct: "SUM(DISTINCT …) ou AVG(DISTINCT …) : deux lignes de même montant ne compteraient qu'une fois.",
  known: "Même nom qu'une métrique écrite ou retirée.",
  other: "Écartée à la génération.",
};

// The messages of the agent library about a metric (lib/metrics.py, semantic_gen/utils.py) in the words of the
// page, the name of the metric dropped; any other message as it is
const NOT_JOINED = /tables (.+?) not joined to (.+?) by a relationship/;
const IN_FRENCH = [
  [/metric '[^']*' — /, ""],
  [/unknown column '([^']+)'/, "colonne inconnue : $1"],
  [/unknown table '([^']+)'/, "table inconnue : $1"],
  [/expr does not parse[\s\S]*/, "la formule ne se lit pas"],
  [/expr must aggregate \(SUM, COUNT, AVG\.\.\.\)/, "la formule doit agréger les lignes (SUM, COUNT, AVG…)"],
  [/write every column of expr as table\.column/, "écris chaque colonne sous la forme table.colonne"],
  [/write time_dimension as table\.column/, "écris la date sous la forme table.colonne"],
  [/'description' and 'expr' are required/, "la description et la formule sont obligatoires"],
  [NOT_JOINED, "aucune relation ne relie $1 à $2"],
];
const inFrench = (message) => IN_FRENCH.reduce((text, [english, french]) => text.replace(english, french), String(message ?? ""));

// One reason for a proposal left out: its rule, or the check it failed (its tables not joined: said with the tables)
function skippedReason(s) {
  if (s.reason !== "rejected") return SKIPPED[s.reason] || SKIPPED.other;
  const joins = s.detail.match(NOT_JOINED);
  if (joins) {
    return "Ses tables ne se rejoignent pas par une relation que la formule peut suivre : "
      + `<span class="mono">${esc(joins[1])}</span> et <span class="mono">${esc(joins[2])}</span>.`;
  }
  return `Refusée à la vérification : ${esc(inFrench(s.detail))}`;
}

// The path between the tables of a metric left out, on the map of Relations (from its first table)
function showPath(joins) {
  ui.path = { a: joins[2].split(", ")[0], b: joins[1].split(", ")[0], result: null };
  goTo("relations");
}
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
  if (t.error) return checkLine("Formule", '<span class="bad-text">refusée</span>', `<span class="mono">${esc(t.error)}</span>`, true);
  const lines = [checkLine("Formule", `acceptée · ${decimal(t.seconds)} s`, m.joins ? "La base accepte la formule et ses jointures." : "Une seule table : pas de jointure.")];
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
      const joins = s.reason === "rejected" && s.detail.match(NOT_JOINED);
      el.innerHTML = `<div class="detail-head"><h2>${esc(s.name)}</h2><span class="figures">proposée par l'IA · pas ajoutée</span></div>
        <div class="metric-fields"><p class="said">${skippedReason(s)}</p>
        ${s.reason === "other" ? `<pre class="sql">${esc(inFrench(s.detail))}</pre>` : ""}
        ${joins ? '<div><button class="btn ghost small" id="skipped-path">Voir le chemin dans Relations</button></div>' : ""}
        <p class="muted-text">Rien n'a été écrit dans metrics.yaml.</p></div>`;
      if (joins) $("skipped-path").onclick = () => showPath(joins);
      return;
    }
  }
  if (ui.metricKind === "ignored" && met.ignored.includes(ui.metric)) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(ui.metric)}</h2><span class="figures">retirée</span><span class="grow"></span>
        <button class="btn ghost small" id="metric-allow">Autoriser à nouveau</button></div>${note}
      <div class="metric-fields"><p class="said">Retirée : l'IA ne la proposera plus. Sa définition n'est plus dans metrics.yaml.</p>
      <p class="muted-text">« Autoriser à nouveau » la sort des métriques retirées : l'IA pourra la reproposer.</p></div>`;
    $("metric-allow").onclick = () => moveMetric(ui.metric, true);
    return;
  }
  const m = met.metrics.find((x) => x.name === ui.metric);
  if (!m) { el.innerHTML = '<p class="placeholder">Choisis une métrique pour voir sa formule et ses tests.</p>'; return; }

  const figures = [m.tables.join(", "), m.sql ? (m.joins ? plural(m.joins, "jointure") : "sans jointure") : null,
    m.test && !m.test.error ? `testée à ${at(m.test.at)}` : null].filter(Boolean).join(" · ");
  const testing = ui.testing?.current === m.name;
  const error = m.error ? `<div class="error-note"><span>${ERRORS[m.error.kind]}</span><code>${esc(inFrench(m.error.detail))}</code></div>` : "";
  const rule = m.rule ? `<div class="rule-note"><span>${RULES[m.rule](m)}</span><button class="btn small" data-retire="1" title="Elle sort de metrics.yaml et l'IA ne la proposera plus">Retirer</button></div>` : "";
  const dates = dateOptions(m);
  el.innerHTML = `<div class="detail-head"><h2>${esc(m.name)}</h2><span class="figures">${esc(figures)}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      ${m.sql ? `<button class="btn ghost small" id="metric-test" ${testing ? "disabled" : ""}>${testing ? "Test en cours…" : "Tester"}</button>` : ""}
      ${m.rule ? "" : '<button class="btn ghost small" data-retire="1" title="Elle sort de metrics.yaml et l\'IA ne la proposera plus">Retirer</button>'}</div>
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
          <select class="edit-select" id="met-date" data-field="time_dimension">${dates}</select></div>
      </div>
      <p class="note error refused" id="metric-refused" hidden></p>
      ${m.sql ? `<div><span class="field-label">Requête testée</span><pre class="sql">${sqlHtml(m.sql)}</pre></div>` : ""}
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

// The dates a metric can take: the ones of its own tables, then the ones a join would bring (a table more in each
// question that asks for it), "aucune" last; the current one selected
function dateOptions(m) {
  const option = (d) => `<option value="${esc(d)}" ${d === m.time_dimension ? "selected" : ""}>${d ? esc(d) : "aucune"}</option>`;
  const dates = [...new Set([m.time_dimension, ...m.dates].filter(Boolean))];
  const own = dates.filter((d) => m.tables.includes(d.split(".")[0])), joined = dates.filter((d) => !own.includes(d));
  if (!own.length || !joined.length) return [...dates, ""].map(option).join("");
  const via = [...new Set(joined.map((d) => d.split(".")[0]))];
  return `<optgroup label="Dans ${esc(m.tables.join(", "))}">${own.map(option).join("")}</optgroup>`
    + `<optgroup label="Par une jointure (ajoute ${esc(via.join(", "))})">${joined.map(option).join("")}</optgroup>${option("")}`;
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
    if (e.status !== 409) { refused.innerHTML = `Rien n'est écrit : <span class="mono">${esc(inFrench(e.message))}</span>`; refused.hidden = false; }
    return;
  }
  if (field === "expr" || field === "time_dimension") {
    renderRows(); renderDetail();
    testMetrics([name]);
    return;
  }
  // metrics.yaml is read by the agent at startup, not from the index
  state.textContent = "Enregistrée · l'agent la lira à son prochain démarrage";
  state.className = "save-state ok";
  flashSaved(input);
}

// The metrics tested one after the other (a query of counts each); "Arrêter les tests" stops after the current one.
// panel: the run shows in the job panel, then what it found (Tout tester; after: the title of the proposal run
// whose metrics are tested, kept in front). A single test leaves the panel alone
async function testMetrics(names, { panel = false, after = null } = {}) {
  if (ui.testing) return;
  const db = ui.db, tested = [];
  ui.testing = { current: null, stop: false, panel, done: 0, total: names.length, after, job: ui.job?.id ?? null };
  renderActions();
  let message;
  for (const name of names) {
    if (ui.testing.stop || db !== ui.db) break;
    ui.testing.current = name;
    if (panel) renderJob();
    if (ui.step === "metrics") { renderRows(); if (ui.metric === name) renderDetail(); }
    try {
      const met = await api(`/api/db/${encodeURIComponent(db)}/metrics/${encodeURIComponent(name)}/test`, { method: "POST" });
      if (db === ui.db) ui.met = met;
      tested.push(name);
      ui.testing.done = tested.length;
    } catch (e) {
      message = e.status === 503 ? "La base ne répond pas : les métriques ne peuvent pas être testées." : inFrench(e.message);
      break;
    }
  }
  const { stop: stopped, job } = ui.testing;
  ui.testing = null;
  if (panel) {
    // Not over a job started meanwhile, nor on another database
    ui.tested = db === ui.db && (ui.job?.id ?? null) === job
      ? { title: testsSummary(tested, stopped, message, after), job, at: Date.now() / 1000 - ui.clockOffset } : null;
    renderJob();
  }
  if (db !== ui.db || ui.step !== "metrics") return;
  if (ui.data) renderHeader();
  renderRows(); renderDetail(message);
}

// What a run of tests found, from the states of the metrics it tested: [title, halo] for the job panel
function testsSummary(names, stopped, message, after) {
  const states = names.map((name) => ui.met?.metrics.find((m) => m.name === name)?.state);
  const [ok, check, error] = ["ok", "check", "error"].map((state) => states.filter((s) => s === state).length);
  const parts = [after, stopped ? "Tests arrêtés" : null, `${plural(names.length, "métrique")} ${names.length > 1 ? "testées" : "testée"}`,
    ok ? `${fmt.format(ok)} vérifiée${ok > 1 ? "s" : ""}` : null, check ? `${fmt.format(check)} à revoir` : null,
    error ? `${fmt.format(error)} en erreur` : null, message ? esc(message) : null];
  return [parts.filter(Boolean).join(" · "), message ? "failed" : stopped ? "stopped" : ""];
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
  renderBase(); renderTabs(); renderWizard();   // the head of the database, the count of the Fiches, the setup
  if (ui.home || ui.wizard || !indexScreen()) return;
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

// One tile of the Index screen; tone "warn" (amber: to look at) or "bad" (red: the agent won't start)
const indexTile = (label, value, text, tone = "") => `<div class="tile${tone ? ` ${tone}` : ""}">
    <span class="tile-label">${label}</span><span class="tile-value">${value}</span><span class="tile-foot">${text}</span></div>`;

// The verdict of the Index screen (and of a card of the home), in a sentence or two: is the agent ready, what to do
// next, then what the tiles flag in amber. Tone "bad" (red: the agent won't start), "warn" (amber: to do or to
// look at), none when it is ready
function indexVerdict(idx, db = ui.db) {
  const box = (tone, text) => `<p class="verdict${tone ? ` ${tone}` : ""}">${text}</p>`;
  if (!idx.trace) {
    if (!idx.values_at) return box("bad", `Jamais indexée : l'agent ne voit aucune table de ${esc(db)}. « Réindexer » écrit leurs fiches.`);
    const newer = Object.entries(idx.files).filter(([, time]) => time > idx.values_at).map(([file, time]) => `${file} (${when(time)})`);
    return box("warn", `Index construit avant que le studio en garde la trace${newer.length
      ? ` : ${newer.join(" et ")} ${newer.length > 1 ? "ont" : "a"} changé depuis` : ""}. Réindexe une fois pour voir ce que l'agent lit, table par table.`);
  }
  const errors = idx.metrics.errors.map((e) => e.name);
  const changed = idx.tables.filter((t) => t.status === "changed").map((t) => t.name);
  const missing = idx.tables.filter((t) => t.status === "missing").map((t) => t.name);
  const e = idx.eval, documents = idx.documents > 1 ? `Les ${fmt.format(idx.documents)} fiches sont à jour` : "La fiche est à jour";
  let tone = "", text;
  if (errors.length) {
    tone = "bad";
    text = `L'agent ne démarrera pas : il refuse ${errors.length > 1 ? "les métriques" : "la métrique"} ${names(errors)} (détail dans la tuile Métriques).`;
  } else if (changed.length || missing.length || idx.gone.length) {
    tone = "warn";
    const parts = [];
    if (changed.length) {
      parts.push(`${changed.length > 1 ? `${fmt.format(changed.length)} tables ont changé` : "Une table a changé"} depuis l'index (${names(changed)}) : `
        + `l'agent lit encore ${changed.length > 1 ? "leurs anciennes fiches" : "son ancienne fiche"}.`);
    }
    if (missing.length) parts.push(`${names(missing)} ${missing.length > 1 ? "ne sont pas" : "n'est pas"} dans l'index : l'agent ne ${missing.length > 1 ? "les" : "la"} voit pas.`);
    if (idx.gone.length) {
      parts.push(`${names(idx.gone)} ${idx.gone.length > 1 ? "ne sont plus" : "n'est plus"} dans la base : `
        + `${idx.gone.length > 1 ? "leurs fiches partiront" : "sa fiche partira"} à la prochaine indexation.`);
    }
    text = `${parts.join(" ")} ${idx.dataset ? "Réindexe, puis relance l'évaluation." : "Réindexe."}`;
  } else if (!idx.dataset) {
    text = `L'agent est prêt : ${plural(idx.documents, "fiche")} à jour. Pas de questions d'évaluation pour cette base.`;
  } else if (!e) {
    text = `${documents}, mais l'agent n'a jamais été évalué : lance l'évaluation.`;
  } else if (e.at < idx.built_at) {
    tone = "warn";
    text = `${documents}, mais ${e.scored ? `le score de ${pct(e.exact / e.scored)}` : "la dernière évaluation"} date d'avant cet index : `
      + "relance l'évaluation pour le confirmer.";
  } else {
    text = `L'agent est prêt : ${plural(idx.documents, "fiche")} à jour, ${fmt.format(e.exact)} réponse${e.exact > 1 ? "s" : ""} `
      + `exacte${e.exact > 1 ? "s" : ""} sur ${fmt.format(e.scored)}.`;
  }
  // What the tiles flag in amber, in a few words: the agent starts, but it reads less than it could
  const raw = idx.raw.filter((t) => !idx.gone.includes(t));
  const look = [
    raw.length && `${plural(raw.length, "table")} sans description`,
    idx.dropped.length && `${plural(idx.dropped.length, "colonne")} sans valeurs`,
    idx.sample_rows !== idx.sample_rows_now && "les exemples de lignes à réappliquer",
  ].filter(Boolean);
  if (look.length && !errors.length) {
    tone = tone || "warn";
    text += ` À regarder : ${look.length > 1 ? `${look.slice(0, -1).join(", ")} et ${look.at(-1)}` : look[0]}.`;
  }
  return box(tone, text);
}

// The categorical columns, by where the agent finds their values: in the document, searched on demand, or nowhere
// (empty or too varied); one bar of columns, what each part means in a bubble on hover, focus or tap
function valuesTile(idx) {
  const d = idx.dropped, many = d.length > 1;
  const columns = idx.inline.columns + idx.searched.columns + d.length;
  const parts = [
    ["listed", idx.inline.columns, "dans les fiches",
      `Peu de valeurs : les ${fmt.format(idx.inline.values)} sont écrites dans les fiches, l'agent les a toutes sous les yeux.`],
    ["searched", idx.searched.columns, `cherchée${idx.searched.columns > 1 ? "s" : ""} à la demande`,
      `Trop de valeurs pour la fiche : l'agent cherche parmi les ${fmt.format(idx.searched.values)} celles que la question cite.`],
    ["none", d.length, "sans valeurs",
      `${names(d)} ${many ? "sont vides ou trop variées" : "est vide ou trop variée"} : l'agent n'en connaît aucune valeur.`],
  ].filter(([, n]) => n);
  const bar = parts.map(([key, n, label]) => `<span class="${key}" style="width:${n / columns * 100}%" title="${fmt.format(n)} ${label}"></span>`);
  const keys = parts.map(([key, n, label, tip]) => `<span class="split-key ${key}" tabindex="0"><span class="swatch ${key}"></span>
      <span class="key-text"><b>${fmt.format(n)}</b> ${label}</span><span class="key-tip" role="tooltip">${tip}</span></span>`);
  return `<div class="tile wide"><span class="tile-label">Colonnes catégorielles</span>
    <span class="tile-value">${fmt.format(columns)}<small>· ${plural(idx.inline.values + idx.searched.values, "valeur")}</small></span>
    <span class="tile-why">${columns ? "L'agent connaît leurs valeurs réelles : il filtre avec l'orthographe exacte de la base."
      : "Aucune colonne catégorielle : l'agent ne connaît aucune valeur de la base."}</span>
    ${columns ? `<div class="split" role="img" aria-label="${parts.map(([, n, label]) => `${fmt.format(n)} ${label}`).join(", ")}">${bar.join("")}</div>
    <div class="split-keys">${keys.join("")}</div>` : ""}</div>`;
}

// What the agent receives: the documents and their joins, the columns hidden from it, the sample rows, the metrics
// it checks at startup, then the categorical columns
function indexTiles(idx) {
  const raw = idx.raw.filter((t) => !idx.gone.includes(t)), m = idx.metrics, hidden = idx.hidden.length;
  const rows = idx.sample_rows, now = idx.sample_rows_now;
  const perDocument = rows ? `${fmt.format(rows)}<small>par fiche</small>` : "aucune";
  return [
    indexTile("Fiches", fmt.format(idx.documents), raw.length
      ? `${names(raw)} sans description : la structure brute seulement. L'onglet Descriptions les décrit.`
      : `Une par table, toutes décrites, ${idx.joins ? `reliées par ${plural(idx.joins, "jointure")}` : "sans jointure entre elles"}.`, raw.length ? "warn" : ""),
    hidden
      ? indexTile("Cachées à l'agent", `${fmt.format(hidden)}<small>sur ${plural(idx.shown + hidden, "colonne")}</small>`,
        "Vides, techniques ou personnelles : hors des fiches et des exemples de lignes.")
      : indexTile("Cachées à l'agent", "aucune", "L'agent voit toutes les colonnes : rien n'est caché."),
    rows !== now
      ? indexTile("Exemples de lignes", perDocument, `La configuration en demande ${now ? fmt.format(now) : "aucune"} : réindexer pour l'appliquer.`, "warn")
      : indexTile("Exemples de lignes", perDocument,
        rows ? "Envoyés au modèle sous la structure, colonnes cachées retirées." : "Aucune ligne de la base dans les fiches."),
    m.errors.length
      ? indexTile("Métriques", `${fmt.format(m.total - m.errors.length)}<small>valides sur ${fmt.format(m.total)}</small>`, "L'agent refusera de démarrer : "
        + m.errors.map((e) => `<span class="mono">${esc(e.name)}</span> (${esc(inFrench(e.detail))})`).join(", ") + ".", "bad")
      : indexTile("Métriques", m.total ? `${fmt.format(m.total)}<small>valides sur ${fmt.format(m.total)}</small>` : "aucune",
        m.total ? "Leurs colonnes sont toutes dans les fiches : l'agent démarrera." : "Pas de métrique : rien à vérifier."),
    valuesTile(idx),
  ].join("");
}

// The last eval: its score in large, a meter of the exact answers (grey when it measured an older index), the dashboard of the database
function evalTile(idx) {
  if (!idx.dataset) {
    return indexTile("Questions", "aucune", `Pas de questions pour cette base : <span class="mono">offline/eval/datasets/${esc(ui.db.toLowerCase())}.yaml</span>.`);
  }
  const e = idx.eval;
  if (!e) return indexTile("Dernière mesure", "—", "Jamais évaluée : « Évaluer l'agent » lui pose les questions de la base.");
  const built = idx.built_at ?? idx.values_at, before = built != null && e.at < built;
  const rate = e.scored ? e.exact / e.scored : null;
  const text = `${fmt.format(e.exact)} réponses exactes sur ${fmt.format(e.scored)} comparées aux lignes attendues`
    + (e.questions > e.scored ? ` (${fmt.format(e.questions)} questions)` : "");
  return `<div class="tile eval${before ? " warn" : ""}">
    <div class="eval-score"><span class="tile-label">Dernière mesure</span>
      <span class="hero">${rate == null ? "—" : `${fmt.format(Math.round(rate * 100))}<small>%</small>`}</span><span class="hero-unit">résultats exacts</span></div>
    <div class="eval-body"><div class="meter${before ? " stale" : ""}" aria-hidden="true"><span style="width:${(rate ?? 0) * 100}%"></span></div>
      <span>${text}</span><span class="tile-foot">Mesurée le ${when(e.at)}${before ? " · avant cet index : à refaire" : ""}</span></div>
    <a class="btn ghost small" href="/eval/${encodeURIComponent(ui.db)}" target="_blank" rel="noopener">Ouvrir le tableau ↗</a></div>`;
}

// The Vue d'ensemble: what the agent reads of the database and its last eval, its setup on the side. The Fiches:
// the document of the table chosen
function renderIndex(el, message) {
  const idx = ui.idx;
  if (!idx) { el.innerHTML = '<p class="placeholder">Lecture de l\'index…</p>'; return; }
  if (ui.step === "index") {
    if (ui.current) renderDocument(el, message);
    else el.innerHTML = '<p class="placeholder">Choisis une table pour voir sa fiche, telle que l\'agent la lit.</p>';
    return;
  }
  const figures = idx.trace ? `${plural(idx.documents, "fiche")} · ${plural(idx.inline.values + idx.searched.values, "valeur")}`
    : idx.values_at ? `index du ${when(idx.values_at)}` : "jamais indexée";
  el.innerHTML = `<div class="overview">
    <section class="panel overview-main">
      <div class="detail-head"><h2>Ce que l'agent lit</h2><span class="figures">${figures}</span></div>
      ${message ? `<p class="note error">${esc(message)}</p>` : ""}${baseVerdict(idx, ui.data?.db === ui.db ? ui.data.summary : null, ui.db)}
      <div class="dash">${idx.trace ? `<section><h3 class="dash-title">Ce que l'agent reçoit</h3><div class="tiles">${indexTiles(idx)}</div></section>` : ""}
        <section><h3 class="dash-title">Évaluation</h3>${evalTile(idx)}</section></div>
    </section>
    <aside class="panel setup">${setupCard()}</aside>
  </div>`;
  el.querySelectorAll("[data-setup]").forEach((button) => { button.onclick = () => setupGo(button.dataset.setup); });
  $("setup-wizard")?.addEventListener("click", () => openWizard());
}

// The setup of the database on its Vue d'ensemble: each step done or not, opening its tab; the guided setup
function setupCard() {
  if (ui.data?.db !== ui.db || !ui.idx) return '<p class="placeholder">Lecture des étapes…</p>';
  const steps = pipelineSteps({ summary: ui.data.summary, index: ui.idx }), done = steps.filter((p) => p.state === "done").length;
  const need = indexNeed(ui.idx);
  const row = (p) => `<li><button type="button" class="setup-step" data-setup="${p.key}">
      <span class="dot ${p.state}">${p.state === "done" ? CHECK : p.state === "bad" ? BANG : ""}</span>
      <span class="setup-name">${STAGES[p.key]}<small class="${p.state}">${esc(p.tip)}</small></span>
      <span class="setup-go">${p.key !== "index" ? "→" : need === "reindex" ? "Réindexer →" : need === "eval" ? "Évaluer →" : ""}</span></button></li>`;
  return `<div class="setup-head"><h2>Mise en place</h2><span class="muted-text">${fmt.format(done)} ${done > 1 ? "étapes" : "étape"} sur 5</span></div>
    <p class="setup-text">Ce qui prépare la base pour l'agent. Chaque étape s'ouvre dans son onglet.</p>
    <ol class="setup-steps">${steps.map(row).join("")}</ol>
    <div class="setup-foot"><button class="btn" type="button" id="setup-wizard">Lancer la mise en place guidée</button>
      <span>Repasse les étapes une à une, à partir de la première qui n'est pas finie.</span></div>`;
}

// A step of the setup: its tab; the index and the eval are done from here
function setupGo(key) {
  if (key !== "index") { goTo(key); return; }
  const need = indexNeed(ui.idx);
  if (need === "reindex") runJob("index/run", {});
  else if (need === "eval") confirmEval();
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

// A column of the document: its type, its format, its values (listed, searched, none), or hidden, or new in the database;
// hidden or shown since the index: it leaves or enters the document at the next index
function documentLine(c) {
  const type = `<span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>`;
  if (c.entering) return `<div class="xline warn"><span class="cell">${esc(c.column)}</span>${type}<span class="xvalues" title="À la prochaine indexation">entrera dans la fiche</span></div>`;
  if (c.hidden) return `<div class="xline hidden-col"><span class="cell">${esc(c.column)} <span class="pill off">cachée</span></span>${type}<span class="xvalues">hors de la fiche</span></div>`;
  if (c.added) return `<div class="xline warn"><span class="cell">${esc(c.column)}</span>${type}<span class="xvalues">ajoutée depuis l'index</span></div>`;
  const format = c.format ? ` <span class="pill type">${esc(formatLabel(c.format))}</span>` : "";
  if (c.leaving) return `<div class="xline warn"><span class="cell">${esc(c.column)}${format}</span>${type}<span class="xvalues" title="À la prochaine indexation">sortira de la fiche</span></div>`;
  const values = c.gone ? "plus dans la base" : !c.values ? "—" : c.values.dropped ? "aucune valeur indexée"
    : `${fmt.format(c.values.count)} · ${c.values.inline ? "dans la fiche" : "cherchées"}`;
  return `<div class="xline${c.gone || c.values?.dropped ? " warn" : ""}"><span class="cell">${esc(c.column)}${format}</span>${type}<span class="xvalues">${values}</span></div>`;
}

const DOC_VIEW = "studio-doc-view";   // the view of a document chosen: its text (the default) or its columns

function renderDocument(el, message) {
  const d = ui.detail;
  if (!d || d.name !== ui.current) { el.innerHTML = '<p class="placeholder">Lecture de la fiche…</p>'; return; }
  const text = Boolean(d.text) && (ui.docView ?? stored(DOC_VIEW) ?? "text") === "text";   // the view chosen, kept for the next visits
  const views = d.text ? `<div class="seg doc-seg" role="group" aria-label="Vue de la fiche">
      <button type="button" data-doc-view="text" aria-pressed="${text}">Texte</button>
      <button type="button" data-doc-view="columns" aria-pressed="${!text}">Colonnes</button></div>` : "";
  const head = (figures) => `<div class="detail-head"><h2>${esc(d.name)}</h2>${figures ? `<span class="figures">${figures}</span>` : ""}
    <span class="grow"></span>${views}<button class="btn ghost small" id="index-back">← Vue d'ensemble</button></div>`
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
      !d.described && !d.gone && !d.changes.length && "Sans description : l'agent ne lit que la structure brute. L'onglet Descriptions la décrit.",
    ].filter(Boolean);
    el.innerHTML = head(figures) + notes.map((n) => `<div class="rule-note"><span>${n}</span></div>`).join("")
      + (text ? documentText(d)
        : `<div class="columns"><div class="xline head"><span>Colonne de la fiche</span><span>Type</span><span class="num">Valeurs</span></div>
        ${d.columns.map(documentLine).join("")}</div>`);
  }
  $("index-back").onclick = () => goTo("overview");
  el.querySelectorAll("[data-doc-view]").forEach((b) => {
    b.onclick = () => { ui.docView = b.dataset.docView; store(DOC_VIEW, ui.docView); renderDetail(); };
  });
  if ($("doc-copy")) $("doc-copy").onclick = () => copyText($("doc-copy"), d.text);
}

// The icon of a copy, in the corner of a block of code; a tick a moment once copied, red when the browser refuses.
// The clipboard of the browser only exists on https or localhost (the studio may be opened by the name of its host):
// else, or refused, the text is copied the older way, through a field selected out of sight
const COPY_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>';
const COPIED_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
async function copyText(button, text) {
  const say = (state, label) => {
    button.classList.remove("done", "refused");
    if (state) button.classList.add(state);
    button.innerHTML = state === "done" ? COPIED_ICON : COPY_ICON;
    button.title = label;
    button.setAttribute("aria-label", label);
  };
  const byField = () => {
    const field = Object.assign(document.createElement("textarea"), { value: text, readOnly: true });
    field.style.cssText = "position: fixed; top: 0; left: 0; opacity: 0;";
    document.body.append(field);
    field.select();
    const copied = document.execCommand("copy");
    field.remove();
    button.focus();
    return copied;
  };
  let copied;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    copied = byField();
  }
  if (copied) say("done", "Copié");
  else say("refused", "Copie refusée par le navigateur");
  clearTimeout(button.reset);
  button.reset = setTimeout(() => say(null, "Copier le texte"), 1600);
}

// The text of a document as the agent receives it (offline/studio/index.py document_text), in the block of SQL: its code
// coloured (keywords, the type of each column), the comments in grey and never coloured (a description holds "and",
// "on", "as"), the data masked by the studio (‹24 valeurs›, ‹3 lignes d'exemple›) apart; the text copied as it is
// shown. Under it, what
// the text does not show: the columns hidden from the agent, the values searched at each question, the categorical
// columns without a value
function documentText(d) {
  const text = d.text;
  const masked = (html) => html.replace(/‹([^›]*)›/g, '<span class="mask">$1</span>');
  const code = (part) => sqlHtml(part).replace(/^(\s*`[^`]*`\s+)(\w+)/, '$1<span class="sql-kw">$2</span>');   // `col` TYPE
  const comment = (part) => `<span class="c">${masked(esc(part))}</span>`;
  let note = false;   // inside the /* */ block of the sample rows
  const lines = text.split("\n").map((line) => {
    if (line.startsWith("/*")) note = true;
    const [part, ...rest] = line.split("  -- ");
    const out = note || line.startsWith("--") ? comment(line)
      : rest.length ? `${code(part)}  ${comment(`-- ${rest.join("  -- ")}`)}` : code(line);
    if (line.startsWith("*/")) note = false;
    return out;
  });
  return `<p class="doc-text-note">Le texte que l'agent reçoit pour cette table, tel que l'index l'a écrit. `
    + `Les données sont masquées : le studio ne montre jamais une valeur.</p>`
    + `<div class="code-block"><pre class="sql doc-text">${lines.join("\n")}</pre>`
    + `<button type="button" class="copy-btn" id="doc-copy" title="Copier le texte" aria-label="Copier le texte">${COPY_ICON}</button></div>`
    + documentFoot(d.columns);
}

function documentFoot(columns) {
  const col = (c, count) => `<span class="mono">${esc(c.column)}</span>${count ? ` (${fmt.format(c.values.count)})` : ""}`;
  const list = (cs, count) => cs.map((c) => col(c, count)).join(", ");
  const listed = columns.filter((c) => !c.hidden && !c.added && !c.gone);
  const hidden = columns.filter((c) => c.hidden), searched = listed.filter((c) => c.values && !c.values.dropped && !c.values.inline);
  const empty = listed.filter((c) => c.values?.dropped);
  const lines = [
    hidden.length && `Hors de la fiche, ${hidden.length > 1 ? "cachées" : "cachée"} à l'agent : ${list(hidden)}.`,
    searched.length && `Valeurs cherchées à chaque question, pas dans la fiche : ${list(searched, true)}.`,
    empty.length && `${empty.length > 1 ? "Catégorielles" : "Catégorielle"} sans valeur indexée : ${list(empty)}.`,
  ].filter(Boolean);
  return lines.length ? `<div class="doc-foot">${lines.map((l) => `<p>${l}</p>`).join("")}</div>` : "";
}

// ── Home: what the agent knows of every database, then the step that would improve it ──

const HOME = "accueil";
const DIALECTS = { hive: "Hive", impala: "Impala", sqlite: "SQLite", postgresql: "PostgreSQL" };
const CHECK = '<svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 6.2l2.3 2.3 4.7-4.9"/></svg>';
const BANG = '<svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2.8v4"/><circle cx="6" cy="9.2" r="1"/></svg>';
// The five steps of the setup of a database, by the name of what they make
const STAGES = { profile: "Contenu", semantics: "Descriptions", relations: "Relations", metrics: "Métriques", index: "Index et évaluation" };

// What the index of a database needs: "reindex" (never built, built before its trace, files changed since), "eval"
// (no score of this index while there are questions), or nothing
function indexNeed(idx) {
  if (!idx.trace || idx.counts.changed + idx.counts.missing + idx.gone.length) return "reindex";
  const built = idx.built_at ?? idx.values_at;
  return idx.dataset && (!idx.eval || (built != null && idx.eval.at < built)) ? "eval" : null;
}

// The verdict of a database: being prepared while it was never indexed (nothing is broken, the agent just doesn't
// read it yet), else the one of its index
function baseVerdict(idx, s, db) {
  if (idx.trace || idx.values_at) return indexVerdict(idx, db);
  const described = !s ? "" : !s.described ? " Aucune table n'est encore décrite." : s.described < s.tables
    ? ` ${plural(s.described, "table")} sur ${fmt.format(s.tables)} ${s.described > 1 ? "sont décrites" : "est décrite"}.` : "";
  return `<p class="verdict">En préparation : l'agent ne voit pas encore cette base.${described}</p>`;
}

function openHome() {
  if (ui.wizard) leaveWizard();
  ui.home = true;
  document.body.dataset.view = "home";
  $("home").hidden = false;
  if (location.hash !== `#${HOME}`) history.pushState(null, "", `#${HOME}`);
  document.title = "Accueil · Studio sémantique";
  closeConfirm($("confirm"));
  renderJob();
  loadHome();
}

function leaveHome() {
  ui.home = false;
  delete document.body.dataset.view;
  $("home").hidden = true;
}

// Every database, read in the files only: the home opens even when a database can't be reached
async function loadHome() {
  try {
    ui.overview = await api("/api/overview");
  } catch (e) {
    $("bases").innerHTML = `<p class="note error">Impossible de lire les bases : ${esc(e.message)}</p>`;
    return;
  }
  if (ui.home) renderHome();
}

function renderHome() {
  const bases = ui.overview || [];
  $("bases-count").textContent = `${fmt.format(bases.length)} · lues dans les fichiers du studio, sans interroger les bases`;
  $("bases").innerHTML = bases.map(baseCard).join("") || '<p class="placeholder">Aucune base dans la configuration.</p>';
}

// The five steps of a database, each done, partial (to finish or to do again), todo or bad (the agent won't start)
function pipelineSteps(base) {
  const s = base.summary, idx = base.index, e = idx.eval, errors = idx.metrics?.errors || [];
  const built = idx.built_at ?? idx.values_at;
  const toIndex = idx.trace ? idx.counts.changed + idx.counts.missing + idx.gone.length : 0;
  const staleEval = idx.dataset && (!e || (built != null && e.at < built));
  const undescribed = s.tables - s.described;
  return [
    { key: "profile", state: s.profiled === s.tables ? "done" : s.profiled ? "partial" : "todo",
      count: `${fmt.format(s.profiled)}/${fmt.format(s.tables)}`, tip: `${fmt.format(s.profiled)} tables profilées sur ${fmt.format(s.tables)}` },
    { key: "semantics", state: !undescribed && !s.partial && !s.rewritten ? "done" : s.described ? "partial" : "todo",
      count: `${fmt.format(s.described)}/${fmt.format(s.tables)}`,
      tip: `${fmt.format(s.described)} tables décrites sur ${fmt.format(s.tables)}`
        + (s.rewritten ? `, ${fmt.format(s.rewritten)} à relire` : "") + (s.partial ? `, ${fmt.format(s.partial)} à compléter` : "") },
    { key: "relations", state: s.relations || s.tables < 2 ? "done" : "todo", count: s.relations ? fmt.format(s.relations) : "—",
      tip: s.relations ? `${plural(s.relations, "relation")}${s.isolated ? `, ${plural(s.isolated, "table")} sans relation` : ""}` : "Pas encore cherchées" },
    { key: "metrics", state: errors.length ? "bad" : s.metrics ? "done" : "todo",
      count: errors.length ? `${fmt.format(errors.length)} refusée${errors.length > 1 ? "s" : ""}` : s.metrics ? fmt.format(s.metrics) : "—",
      tip: errors.length ? `L'agent refuse ${names(errors.map((m) => m.name)).replace(/<[^>]+>/g, "")}`
        : s.metrics ? `${plural(s.metrics, "métrique")}, toutes valides` : "Pas encore proposées" },
    { key: "index", state: !idx.trace && !idx.values_at ? "todo" : !idx.trace || toIndex || staleEval ? "partial" : "done",
      count: !idx.trace && !idx.values_at ? "jamais" : !idx.trace ? "état inconnu" : toIndex ? `${fmt.format(toIndex)} à réindexer`
        : staleEval ? (e ? "éval à refaire" : "à évaluer") : "à jour",
      tip: !idx.trace && !idx.values_at ? "Jamais indexée" : !idx.trace ? "Index construit avant que le studio en garde la trace"
        : toIndex ? `${plural(toIndex, "fiche")} à mettre à jour` : staleEval ? (e ? "L'évaluation date d'avant l'index" : "Jamais évaluée")
        : `${plural(idx.documents, "fiche")} à jour, évaluées depuis` },
  ];
}

// The first step of the setup that is not done, in words, and the button that does it on this database: its tab, or
// the guided setup while the database is being prepared (never indexed)
function nextStep(base, steps) {
  const s = base.summary, idx = base.index, step = steps.find((p) => p.state !== "done");
  const go = (key, label, action = "") => `<button class="btn primary" type="button" data-step="${key}"${action ? ` data-action="${action}"` : ""}>${label} →</button>`;
  const say = (title, why) => `<span class="lead">Prochaine étape · </span><b>${title}</b> : ${why}`;
  const open = '<button class="btn" type="button" data-step="overview">Ouvrir la base</button>';
  if (!step) return { text: "<b>Tout est à jour</b> : l'agent lit la dernière version de chaque fiche, et son score la mesure.", buttons: open };
  const preparing = !idx.trace && !idx.values_at;
  const tab = (key, label) => open + (preparing
    ? `<button class="btn primary" type="button" data-wizard="1">${steps[0].state === "todo" ? "Lancer" : "Reprendre"} la mise en place →</button>`
    : go(key, label));
  if (step.key === "profile") {
    return { text: say(`Profiler ${plural(s.tables - s.profiled, "table")}`, "le studio ne sait pas encore ce qu'elles contiennent."),
      buttons: tab("profile", "Ouvrir le contenu") };
  }
  if (step.key === "semantics") {
    const undescribed = s.tables - s.described;
    const text = undescribed ? say(`Décrire ${plural(undescribed, "table")}`, "l'agent n'en aurait que la structure brute.")
      : s.rewritten ? say(`Relire ${plural(s.rewritten, "table")}`, "l'IA a réécrit des descriptions.")
      : say(`Compléter ${plural(s.partial, "table")}`, "des colonnes sont arrivées depuis leur description.");
    return { text, buttons: tab("semantics", "Ouvrir les descriptions") };
  }
  if (step.key === "relations") {
    return { text: say("Chercher les relations", "l'agent ne sait pas encore passer d'une table à l'autre."), buttons: tab("relations", "Ouvrir les relations") };
  }
  if (step.key === "metrics") {
    const errors = idx.metrics?.errors || [];
    return { text: errors.length ? say(`Corriger ${names(errors.map((m) => m.name))}`, "l'agent refuse de démarrer tant qu'elle lit une colonne absente des fiches.")
      : say("Proposer des métriques", "sans elles, l'agent refait ses calculs à chaque question."), buttons: tab("metrics", "Ouvrir les métriques") };
  }
  if (preparing) return { text: say("Indexer la base", "l'agent ne la voit pas encore."), buttons: tab("overview", "") };
  const toIndex = idx.trace ? idx.counts.changed + idx.counts.missing + idx.gone.length : 0;
  if (indexNeed(idx) === "reindex") {
    return { text: say("Réindexer", toIndex ? `${plural(toIndex, "fiche")} à mettre à jour : l'agent lit encore l'ancienne version.`
      : "pour voir ce que l'agent lit, table par table."), buttons: open + go("overview", "Réindexer", "reindex") };
  }
  return { text: say("Évaluer l'agent", idx.eval ? "son score date d'avant l'index." : "il n'a jamais été évalué."),
    buttons: open + go("overview", "Évaluer l'agent", "eval") };
}

// The score of the last eval in large, or why there is none
function scoreBlock(idx) {
  const e = idx.eval, rate = e?.scored ? e.exact / e.scored : null;
  const day = new Date((e?.at ?? 0) * 1000).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" });
  const foot = !idx.dataset ? "pas de questions d'évaluation" : !e ? "jamais évaluée" : `${fmt.format(e.exact)} sur ${fmt.format(e.scored)} · mesurés le ${day}`;
  return `<div class="base-score"><span class="tile-label">Résultats exacts</span>
    <span class="hero${rate == null ? " none" : ""}">${rate == null ? "—" : `${fmt.format(Math.round(rate * 100))}<small>%</small>`}</span>
    <span class="hero-unit">${foot}</span></div>`;
}

// Four figures: what the agent reads once indexed, how far the steps went before
function minis(base) {
  const idx = base.index, s = base.summary;
  const mini = (value, label, tone = "") => `<div class="mini${tone ? ` ${tone}` : ""}"><b>${value}</b><span>${label}</span></div>`;
  const many = (n, one, more) => (n > 1 ? more : one);
  if (!idx.trace) {
    return [
      mini(fmt.format(s.profiled), many(s.profiled, "table profilée", "tables profilées")),
      mini(`${fmt.format(s.described)}<small>sur ${fmt.format(s.tables)}</small>`, many(s.described, "table décrite", "tables décrites")),
      mini(s.relations ? fmt.format(s.relations) : "aucune", many(s.relations, "relation", "relations")),
      mini(s.metrics ? fmt.format(s.metrics) : "aucune", many(s.metrics, "métrique", "métriques")),
    ];
  }
  const raw = idx.raw.filter((t) => !idx.gone.includes(t)).length, hidden = idx.hidden.length, m = idx.metrics;
  const valid = m.total - m.errors.length;
  return [
    mini(fmt.format(idx.documents), raw ? `${many(idx.documents, "fiche", "fiches")}, dont ${fmt.format(raw)} sans description`
      : `${many(idx.documents, "fiche", "fiches")}, ${many(idx.documents, "décrite", "toutes décrites")}`, raw ? "warn" : ""),
    mini(idx.joins ? fmt.format(idx.joins) : "aucune", many(idx.joins, "jointure", "jointures")),
    mini(hidden ? fmt.format(hidden) : "aucune",
      hidden ? `${many(hidden, "colonne cachée", "colonnes cachées")} sur ${fmt.format(idx.shown + hidden)}` : "colonne cachée"),
    mini(m.errors.length ? `${fmt.format(valid)}<small>sur ${fmt.format(m.total)}</small>` : m.total ? fmt.format(m.total) : "aucune",
      m.total ? many(valid, "métrique valide", "métriques valides") : "métrique", m.errors.length ? "bad" : ""),
  ];
}

// A dot of the pipeline of a card: its tab (the index and the eval: the Vue d'ensemble)
const pipeStep = (p) => `<li><button type="button" data-step="${p.key === "index" ? "overview" : p.key}" title="${esc(p.tip)}">
    <span class="dot ${p.state}">${p.state === "done" ? CHECK : p.state === "bad" ? BANG : ""}</span>
    <span class="pipe-name">${p.key === "index" ? "Index" : STAGES[p.key]}</span><span class="pipe-count ${p.state}">${p.count}</span></button></li>`;

function baseCard(base) {
  const idx = base.index, s = base.summary, steps = pipelineSteps(base), next = nextStep(base, steps);
  const indexed = idx.trace ? `indexée le ${when(idx.built_at)}` : idx.values_at ? `index du ${when(idx.values_at)}` : "jamais indexée";
  const verdict = baseVerdict(idx, s, base.name);
  return `<article class="base" data-db="${esc(base.name)}">
    <div class="base-head"><h3>${esc(base.name)}</h3><span class="pill kept">${esc(DIALECTS[base.dialect] || base.dialect)}</span>
      <span class="figures">${indexed}</span></div>
    ${verdict}
    <div class="base-figures">${scoreBlock(idx)}<div class="minis">${minis(base).join("")}</div></div>
    <div class="pipeline"><span class="pipeline-title">Pipeline</span><ol class="pipe">${steps.map(pipeStep).join("")}</ol></div>
    <div class="base-next"><p>${next.text}</p>${next.buttons}</div>
  </article>`;
}

// From the home: a tab of a database, then what its button said (a table to open, a job to start), or its guided setup
async function openDb(db, step, { table = null, action = null, wizard = false } = {}) {
  if (db !== ui.db) switchDb(db);
  if (wizard) {
    openWizard();
    await Promise.all([ui.data?.db !== ui.db ? loadTables() : null, ui.idx ? null : loadIndex()]);
    return;
  }
  await goTo(step, true);
  if (table) await openTable(table);
  if (action === "reindex") runJob("index/run", {});
  else if (action === "eval") confirmEval();
}

$("bases").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-step], button[data-wizard]");
  if (!button) return;
  const db = button.closest("[data-db]").dataset.db;
  if (button.dataset.wizard) openDb(db, null, { wizard: true });
  else openDb(db, button.dataset.step, { action: button.dataset.action || null });
});

// The search through the names and descriptions of every database, a moment after the last key
let searchTimer = null, searchSeq = 0;
$("home-search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => runSearch(e.target.value), 150);
});

async function runSearch(query) {
  const seq = ++searchSeq, box = $("search-results");
  if (!query.trim()) { box.hidden = true; box.innerHTML = ""; return; }
  let found;
  try {
    found = await api(`/api/search?q=${encodeURIComponent(query)}`);
  } catch (e) {
    if (seq === searchSeq) { box.hidden = false; box.innerHTML = `<p class="note error">${esc(e.message)}</p>`; }
    return;
  }
  if (seq !== searchSeq) return;   // a later search answered first
  const more = found.total - found.results.length;
  box.hidden = false;
  box.innerHTML = !found.total ? `<div class="results-head">Aucune table ni colonne ne parle de « ${esc(query.trim())} ».</div>`
    : `<div class="results-head">${plural(found.total, "résultat")} dans ${plural(found.bases, "base")} · noms et descriptions, jamais les valeurs</div>`
      + found.results.map(resultRow).join("")
      + (more ? `<div class="results-more">et ${plural(more, "autre")} : précise la recherche pour les voir</div>` : "");
}

const resultRow = (r) => `<button class="result" type="button" data-db="${esc(r.db)}" data-table="${esc(r.table)}">
    <span class="pill db">${esc(r.db)}</span>
    <span class="result-main"><span class="result-path">${esc(r.column ? `${r.table}.${r.column}` : r.table)}</span>
      <span class="result-desc">${esc(r.description || "—")}</span></span>
    <span class="result-tags">${r.column ? "" : '<span class="pill kept">table</span>'}${r.categorical ? '<span class="pill kept">catégorielle</span>' : ""}`
  + `${r.hidden ? '<span class="pill off">cachée à l\'agent</span>' : ""}<span class="result-go">Voir la fiche →</span></span></button>`;

$("search-results").addEventListener("click", (e) => {
  const row = e.target.closest(".result");
  if (row) openDb(row.dataset.db, "index", { table: row.dataset.table });
});

// ── The guided setup: the five steps of a database one after the other, opened on purpose ──

const GUIDE = {
  profile: { title: "Profiler les tables", text: STEPS.profile.text },
  semantics: { title: "Décrire les tables", text: "L'agent choisit ses tables et écrit ses requêtes à partir de ces descriptions. L'IA les rédige à partir de la structure des tables et de leur profil ; tu pourras les relire et les corriger ensuite dans l'onglet Descriptions." },
  relations: { title: "Relier les tables", text: STEPS.relations.text },
  metrics: { title: "Proposer des métriques", text: "L'agent copie ces formules au lieu de refaire ses calculs. L'IA les propose et chacune est testée sur les données, sans jamais en montrer le résultat. Une base peut s'en passer : l'agent calcule alors lui-même." },
  index: { title: "Indexer et évaluer", text: "L'index écrit la fiche de chaque table, ce que l'agent lit vraiment, et les valeurs des colonnes catégorielles lues dans la base. L'évaluation pose ensuite les questions de test à l'agent et compare ses réponses aux lignes attendues." },
};
const GUIDE_STEPS = ["profile", "semantics", "relations", "metrics", "index"];

// The setup opens on the step asked, else on the first that is not done (known once the database is read)
function openWizard(step = null) {
  if (ui.home) leaveHome();
  ui.wizard = { step };
  document.body.dataset.view = "wizard";
  $("wizard").hidden = false;
  $("base-menu").hidden = true;
  remember();
  document.title = `Mise en place · ${ui.db} · Studio sémantique`;
  closeConfirm($("confirm"));
  renderWizard();
  renderJob();
}

function leaveWizard() {
  ui.wizard = null;
  delete document.body.dataset.view;
  $("wizard").hidden = true;
}

$("wizard-quit").addEventListener("click", () => goTo("overview", true));

// The five steps on the left (done or not), the current one on the right: what it does, how far it went, its
// action, what is left; then the way to the step before, the next one, or the tab where it is reviewed
function renderWizard() {
  if (!ui.wizard) return;
  $("wizard-title").innerHTML = `Mise en place de ${esc(ui.db)} <span class="muted-text">· assistant guidé</span>`;
  if (ui.data?.db !== ui.db || !ui.idx) {
    $("wizard-steps").innerHTML = "";
    $("wizard-panel").innerHTML = '<p class="placeholder">Lecture de la base…</p>';
    return;
  }
  const base = { summary: ui.data.summary, index: ui.idx }, steps = pipelineSteps(base);
  if (!ui.wizard.step) ui.wizard.step = (steps.find((p) => p.state !== "done") || steps.at(-1)).key;
  const current = ui.wizard.step, at = GUIDE_STEPS.indexOf(current);
  $("wizard-steps").innerHTML = steps.map((p, i) => {
    const on = p.key === current;
    return `<li><button type="button" class="wizard-step${on ? " on" : ""}" data-guide="${p.key}"${on ? ' aria-current="step"' : ""}>
      <span class="dot ${on ? "current" : p.state}">${!on && p.state === "done" ? CHECK : !on && p.state === "bad" ? BANG : ""}</span>
      <span class="setup-name">${i + 1} · ${STAGES[p.key]}<small class="${p.state}">${esc(p.tip)}</small></span></button></li>`;
  }).join("");
  $("wizard-panel").innerHTML = guideStep(current, base, steps[at]) + guideFoot(at, steps[at]);
  gateButtons();
}

// One step of the setup, from what the studio knows of the database
function guideStep(key, base, p) {
  const s = base.summary, idx = base.index, rows = ui.data.tables;
  const size = (t) => [t.rows != null ? plural(t.rows, "ligne") : null, t.columns != null ? plural(t.columns, "colonne") : null].filter(Boolean).join(" · ");
  const progress = (done, total, label) => `<div class="guide-progress"><div class="guide-count"><span><b>${fmt.format(done)}</b> ${label} sur ${fmt.format(total)}</span>`
    + `<span class="figures">${pct(total ? done / total : null)}</span></div><div class="meter" aria-hidden="true"><span style="width:${total ? done / total * 100 : 0}%"></span></div></div>`;
  const action = (label, job, note = "", primary = true) => `<button class="btn${primary ? " primary" : ""}" type="button" data-guide-run="${job}" data-job>${label}</button>`
    + (note ? `<span class="muted-text">${note}</span>` : "");
  const list = (title, items) => !items.length ? "" : `<div class="guide-list"><div class="guide-list-head">${title} · ${fmt.format(items.length)}</div>`
    + items.map((it) => `<div class="guide-row"><span class="mono">${esc(it.name)}</span><span class="figures">${esc(it.size)}</span>${it.pill}</div>`).join("") + "</div>";
  const done = p.state === "done" ? '<p class="guide-done">Étape terminée : tu peux passer à la suivante.</p>' : "";
  let body = "";
  if (key === "profile") {
    const missing = rows.filter((t) => t.status.profile === "none"), n = missing.length;
    body = progress(s.profiled, s.tables, "tables profilées") + done
      + (n ? `<div class="guide-action">${action(n > 1 ? `Profiler les ${fmt.format(n)} manquantes` : "Profiler la manquante", "profile", "Des comptes seulement, jamais une valeur.")}</div>` : "")
      + list("À profiler", missing.map((t) => ({ name: t.name, size: size(t), pill: '<span class="pill todo">à profiler</span>' })));
  } else if (key === "semantics") {
    const todo = rows.filter((t) => t.status.semantics !== "done"), n = undescribed(s);
    body = progress(s.described - s.partial, s.tables, "tables décrites") + done
      + (n ? `<div class="guide-action">${action(n > 1 ? `Décrire les ${fmt.format(n)} manquantes` : "Décrire la manquante", "semantics",
        "Quelques secondes d'IA par table. Les descriptions corrigées par toi ne sont pas touchées.")}</div>` : "")
      + (s.rewritten ? `<p class="guide-note">${plural(s.rewritten, "table")} ${s.rewritten > 1 ? "ont" : "a"} des descriptions réécrites par l'IA, à relire dans l'onglet Descriptions.</p>` : "")
      + list("À décrire", todo.map((t) => ({ name: t.name, size: size(t),
        pill: t.status.semantics === "none" ? '<span class="pill todo">à décrire</span>' : '<span class="pill decide">à compléter</span>' })));
  } else if (key === "relations") {
    const isolated = rows.filter((t) => t.status.relations !== "done");
    body = progress(s.tables - s.isolated, s.tables, "tables reliées") + done
      + `<div class="guide-action">${action("Chercher les relations", "relations", "L'IA propose les liens, chacun est mesuré sur les données.", p.state !== "done")}</div>`
      + list("Sans relation", isolated.map((t) => ({ name: t.name, size: size(t), pill: '<span class="pill decide">isolée</span>' })));
  } else if (key === "metrics") {
    const errors = idx.metrics?.errors || [];
    body = `<div class="guide-progress"><div class="guide-count"><span>${s.metrics ? `<b>${plural(s.metrics, "métrique")}</b> dans metrics.yaml` : "Aucune métrique pour l'instant"}</span></div></div>` + done
      + `<div class="guide-action">${action(s.metrics ? "Proposer d'autres métriques" : "Proposer des métriques", "metrics",
        "Chaque métrique proposée est testée sur les données.", !s.metrics)}</div>`
      + list("Refusées par l'agent", errors.map((m) => ({ name: m.name, size: inFrench(m.detail), pill: '<span class="pill bad">refusée</span>' })));
  } else {
    const need = indexNeed(idx);
    body = `<div class="guide-body">${baseVerdict(idx, s, ui.db)}</div>` + done
      + `<div class="guide-action">${action(idx.trace || idx.values_at ? "Réindexer" : "Indexer", "index",
        "", need === "reindex")}${idx.dataset ? action("Évaluer l'agent", "eval", "", need === "eval") : ""}</div>`;
  }
  return `<div class="guide-head"><span class="eyebrow">Étape ${GUIDE_STEPS.indexOf(key) + 1} sur 5</span><h1>${GUIDE[key].title}</h1>`
    + `<p>${GUIDE[key].text}</p></div>${body}`;
}

// The way to the step before, the tab where this one is reviewed, the next step (skipped while not done) or the end
function guideFoot(at, p) {
  const prev = GUIDE_STEPS[at - 1], next = GUIDE_STEPS[at + 1], tab = at === 4 ? "overview" : GUIDE_STEPS[at];
  return `<div class="guide-foot">
    ${prev ? `<button class="btn" type="button" data-guide="${prev}">← ${STAGES[prev]}</button>` : ""}
    <button class="btn ghost" type="button" data-tab="${tab}">Voir dans l'onglet ${STEPS[tab].title}</button>
    <span class="grow"></span>
    ${!next ? `<button class="btn${p.state === "done" ? " primary" : ""}" type="button" data-finish="1">Terminer</button>`
      : p.state === "done" ? `<button class="btn primary" type="button" data-guide="${next}">Suivant : ${STAGES[next]} →</button>`
      : `<button class="btn ghost" type="button" data-guide="${next}">Passer cette étape →</button>`}
  </div>`;
}

$("wizard").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button || !ui.wizard) return;
  if (button.dataset.guide) { ui.wizard.step = button.dataset.guide; renderWizard(); $("wizard").scrollIntoView({ block: "start" }); }
  else if (button.dataset.guideRun) runGuide(button.dataset.guideRun);
  else if (button.dataset.tab) goTo(button.dataset.tab, true);
  else if (button.dataset.finish) goTo("overview", true);
});

// The action of a step: the same job as the button of its tab
function runGuide(job) {
  if (job === "profile") runJob("profile", { missing: true });
  else if (job === "semantics") runJob("semantics", {});
  else if (job === "relations") runJob("relations/run", {});
  else if (job === "metrics") runJob("metrics/run", {});
  else if (job === "index") runJob("index/run", {});
  else if (job === "eval") confirmEval();
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
  ui.logOpen = false;   // its detail on demand, or when it fails
  ui.tested = null;   // a CLI job always wins the panel over the last tests
  appendLines(ui.job.lines); ui.since = ui.job.next_line;
  renderJob(); renderRows();
  schedulePoll();
}

// The two actions of the page: on the selection, or on what is missing (which one is primary depends on the step)
function runSelection() {
  if (!ui.selected.size) return;   // greyed with its reason (gateButtons)
  if (ui.step === "profile") runJob("profile", { tables: [...ui.selected] });
  else redescribeSelection();
}
function runMissing() {
  runJob(ui.step === "profile" ? "profile" : "semantics", ui.step === "profile" ? { missing: true } : {});
}
$("run-primary").addEventListener("click", () => {
  if (ui.step === "metrics") runJob("metrics/run", {});
  else if (ui.step === "relations") runJob("relations/run", {});
  else if (ui.step === "profile") runSelection();
  else runMissing();
});
// The eval asks every question of the dataset to the agent and replaces the last results: said before it runs
function confirmEval() {
  const questions = ui.idx?.questions;
  openConfirm($("confirm"), {
    text: `L'agent répond ${questions === 1 ? "à la question " : questions != null ? `aux ${fmt.format(questions)} questions ` : "aux questions "}`
      + `de <span class="mono">offline/eval/datasets/${esc(ui.db.toLowerCase())}.yaml</span>, quelques minutes d'appels à l'IA. `
      + "Les résultats précédents seront remplacés.",
    button: () => "Évaluer l'agent",
    run: () => runJob("eval/run", {}),
  });
}

$("run-secondary").addEventListener("click", () => {
  if (ui.step === "metrics") {
    if (ui.testing) { ui.testing.stop = true; renderJob(); return; }
    testMetrics((ui.met?.metrics || []).filter((m) => m.sql).map((m) => m.name), { panel: true });
  }
  else if (ui.step === "relations") { ui.adding = true; ui.relCheck = null; renderDetail(); $("add-form")?.scrollIntoView({ block: "nearest" }); }
  else if (ui.step === "profile") runMissing();
  else runSelection();
});
$("job-stop").addEventListener("click", async () => {
  // The tests in the panel: they stop after the current one
  if (!ui.job?.running && ui.testing?.panel) { ui.testing.stop = true; renderJob(); return; }
  if (!ui.job) return;
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
  if (!before || before.id !== job.id) { $("job-log").innerHTML = ""; ui.tested = null; }
  appendLines(job.lines);
  ui.since = job.next_line;
  const progressed = !before || before.id !== job.id || before.done.length !== job.done.length
    || before.current !== job.current || before.running !== job.running;
  setJob(job);
  // A job that fails opens its detail: what went wrong is in the log
  if (before?.id === job.id && before.running && !job.running && job.returncode && !job.stopped) ui.logOpen = true;
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
    // Its title counts the relations now known
    if (job.step === "relations" && ended) { await loadRelations(); renderJob(); }
    // The metrics a run adds are tested at once, in the panel after the title of the run: the data says what to look at
    if (job.step === "metrics" && ended) {
      await loadMetrics();
      renderJob();   // its title counts what the run added and skipped
      const added = (ui.met?.added || []).filter((name) => ui.met.metrics.some((m) => m.name === name && m.sql));
      if (ui.step === "metrics" && added.length) testMetrics(added, { panel: true, after: metricsJobTitle(job)[0] });
    }
    // A new index or a new eval: what the agent reads, its score
    if ((job.step === "index" || job.step === "eval") && ended) { await loadIndex(); renderJob(); }
  }
  // The home counts every database: read again when a job ends
  if (ui.home && before?.id === job.id && before.running && !job.running) loadHome();
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

// The screen a job belongs to: over, it shows in full there, as a quiet "Dernier job" line elsewhere
const JOB_SCREEN = { profile: "profile", semantics: "semantics", relations: "relations", metrics: "metrics", index: "overview", eval: "overview" };

// The job panel, one line: the CLI job, or the tests of the metrics while no job runs (their run, then what they found);
// its detail (the log) on demand
function renderJob() {
  // The tests show only while no job started after them: a job that ends during the tests keeps its title and logs
  const job = ui.job, latest = job?.id ?? null;
  const tests = !job?.running && (ui.testing?.panel && ui.testing.job === latest ? "running"
    : ui.tested && ui.tested.job === latest ? "done" : null);
  if (!job && !tests) { $("job").hidden = true; gateButtons(); return; }
  $("job").hidden = false;
  const now = Date.now() / 1000 - ui.clockOffset;   // the clock of the server
  const titles = { semantics: semanticsJobTitle, relations: relationsJobTitle, metrics: metricsJobTitle, profile: profileJobTitle,
    index: indexJobTitle, eval: evalJobTitle };
  let [title, halo] = tests === "running" ? testsRunTitle() : tests === "done" ? ui.tested.title
    : (titles[job.step] || profileJobTitle)(job, now);
  const past = tests === "done" ? ui.step !== "metrics"
    : !tests && !job.running && (JOB_SCREEN[job.step] !== (indexScreen() ? "overview" : ui.step) || job.db !== ui.db);
  if (past) {
    title = tests ? `Derniers tests des métriques, à ${at(ui.tested.at)} · ${title}`
      : `Dernier job sur ${esc(job.db)}, à ${at(job.ended)} · ${title}`;
  }
  $("job").classList.toggle("past", past);
  $("job-title").innerHTML = title;
  $("job-title").title = $("job-title").textContent;   // the whole line on hover: it is cut to one line
  $("job-halo").className = `halo ${halo}`;
  $("job-stop").hidden = tests ? tests === "done" || ui.testing.stop : !job.running || job.stopped;
  $("job-toggle").hidden = Boolean(tests);   // the tests write no log
  $("job-log").hidden = Boolean(tests) || !ui.logOpen;
  $("job-toggle").textContent = ui.logOpen ? "Masquer le détail" : "Détails";
  gateButtons();
}

// The tests running: how far, which metric, after the title of the proposal run whose metrics they test
function testsRunTitle() {
  const t = ui.testing, before = t.after ? `${t.after} · ` : "";
  if (t.stop) return [`${before}Arrêt des tests après la métrique en cours…`, "running"];
  return [`${before}Tests des métriques · ${fmt.format(t.done)}/${fmt.format(t.total)}${t.current ? ` · <b>${esc(t.current)}</b>` : ""}`, "running"];
}

// An index: the values read table after table, then the documents of the agent written, then the value index
function indexJobTitle(job, now) {
  const total = ui.data?.summary.described;
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running && job.current) {
    const waited = now - job.current_since;
    let title = `Lecture des valeurs · <b>${esc(job.current)}</b> depuis ${duration(waited)}${total ? ` (${job.done.length + 1}/${total})` : ""}`;
    if (waited > SLOW_SECONDS) title += ' · <span class="slow">la base met du temps à répondre</span>';
    return [title, "running"];
  }
  if (job.running && job.phase === "documents") return [`Écriture des fiches de l'agent… ${duration(now - job.started)}`, "running"];
  if (job.running && job.phase === "values") return ["Écriture de l'index des valeurs…", "running"];
  if (job.running) return [`Connexion à la base… ${duration(now - job.started)}`, "running"];
  // Before the documents, nothing is written: the agent keeps the previous index
  if (job.stopped) {
    return [job.phase ? '<span class="slow">Index arrêté pendant l\'écriture : il peut être incomplet, réindexe</span>'
      : "Index arrêté · rien n'est écrit : l'agent garde l'index précédent", "stopped"];
  }
  if (job.returncode) return [`Échec de l'index · ${job.phase ? "" : "rien n'est écrit · "}voir les logs`, "failed"];
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
  if (job.stopped) return ["Évaluation arrêtée · les résultats précédents sont gardés", "stopped"];
  if (job.returncode) return ["Échec de l'évaluation · voir les logs", "failed"];
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
  if (job.stopped) return ["Description arrêtée · rien n'est écrit : tables.yaml n'est mis à jour qu'à la fin", "stopped"];
  if (job.returncode) return ["Échec de la description · voir les logs", "failed"];
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

// A relations run: the measured ones, the LLM (phase llm), then the relations without evidence checked on the data
// (phase check, out of to_check); --check: only that. Done: what the search added, what the check found weak
function relationsJobTitle(job, now) {
  const checkOnly = job.command.includes("--check"), n = job.checked, checked = `${plural(n, "relation")} ${n > 1 ? "vérifiées" : "vérifiée"}`;
  const what = checkOnly ? "Vérification des relations" : "Recherche des relations";
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running && job.phase === "llm") return [`L'IA cherche des liens entre les tables… ${duration(now - job.started)}`, "running"];
  if (job.running && job.to_check) {
    return [`Vérification sur les données · ${fmt.format(n)}/${fmt.format(job.to_check)} · ${duration(now - job.started)}`, "running"];
  }
  if (job.running) return [`${what}… ${duration(now - job.started)}`, "running"];
  if (job.stopped) return [`${what} arrêtée · rien n'est écrit : relationships.yaml n'est mis à jour qu'à la fin`, "stopped"];
  if (job.returncode) return [`Échec de la ${checkOnly ? "vérification" : "recherche"} · voir les logs`, "failed"];
  const rel = job.db === ui.db ? ui.rel : null;
  if (checkOnly) return [`Vérification terminée · ${checked}${rel?.weak ? ` · ${fmt.format(rel.weak)} à vérifier` : ""}`, ""];
  const added = job.added ?? 0;
  const found = added > 1 ? `${fmt.format(added)} nouvelles relations` : added ? "1 nouvelle relation" : "aucune nouvelle relation";
  return [`Recherche terminée · ${found}${rel ? ` · ${fmt.format(rel.relations.length)} au total` : ""}`, ""];
}

// A run of the generator: one call of the LLM, then each proposal checked; metrics.yaml written once at the end
function metricsJobTitle(job, now) {
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running) return [`Proposition des métriques de ${esc(job.db)} par l'IA · ${duration(now - job.started)}`, "running"];
  if (job.stopped) return ["Proposition arrêtée · rien n'est écrit : metrics.yaml n'est écrit qu'à la fin", "stopped"];
  if (job.returncode) return ["Échec de la proposition · voir les logs", "failed"];
  const run = job.db === ui.db ? ui.met : null, added = run?.added.length ?? 0, skipped = run?.skipped.length ?? 0;
  return [`Proposition terminée · ${plural(added, "métrique")} ${added > 1 ? "ajoutées" : "ajoutée"}`
    + (skipped ? ` · ${fmt.format(skipped)} écartée${skipped > 1 ? "s" : ""}` : ""), ""];
}

// What the profile detected in the tables a job measured, in the words of the Profiling screen
function detectedText(detected) {
  if (!detected) return "";
  const { personal, formats, empty } = detected;
  const parts = [personal && `${fmt.format(personal)} ${personal > 1 ? "données personnelles" : "donnée personnelle"}`,
    formats && `${fmt.format(formats)} ${formats > 1 ? "formats détectés" : "format détecté"}`,
    empty && `${fmt.format(empty)} ${empty > 1 ? "colonnes vides" : "colonne vide"}`].filter(Boolean);
  return ` · ${parts.join(" · ") || "rien à signaler"}`;
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
    title = `Profilage de ${esc(job.db)} · <b>${esc(job.current)}</b> depuis ${duration(waited)} (${progress})`;
    if (waited > SLOW_SECONDS) title += ' · <span class="slow">la base met du temps à répondre</span>';
  } else if (job.stopped) {
    const done = job.done.length, query = job.cancel === "cancelled" ? " · requête annulée sur la base"
      : job.cancel === "failed" || job.returncode === -9 ? ' · <span class="slow">la requête peut continuer sur la base</span>' : "";
    title = (done ? `Profilage arrêté après ${plural(done, "table")}${total ? ` sur ${fmt.format(total)}` : ""} · les autres n'ont pas changé`
      : "Profilage arrêté pendant la première table · rien n'a changé") + query;
    halo = "stopped";
  } else if (job.returncode) {
    title = "Échec du profiling · voir les logs";
    halo = "failed";
  } else if (!job.done.length && !job.failed.length && job.command.includes("--missing")) {
    title = "Rien à profiler : chaque table a déjà son profil";
  } else {
    title = `Profilage terminé · ${plural(job.done.length, "table")} ${job.done.length > 1 ? "mesurées" : "mesurée"}${detectedText(job.detected)}`;
  }
  if (job.failed.length) title += ` · <span style="color:var(--bad-ink)">échec : ${job.failed.map(esc).join(", ")}</span>`;
  return [title, halo];
}

init();
