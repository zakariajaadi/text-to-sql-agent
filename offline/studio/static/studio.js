// Studio of the semantic layer, three levels as a web app: the home (every database), a database (its tabs),
// an element of it (a table, a metric). The tabs of a database (STEPS) share one layout: its tables on the left
// (a status pill each), the chosen one on the right.
//   Vue d'ensemble: what the agent reads of the database, its last eval, and its setup step by step
//   Périmètre (opened from the Vue d'ensemble, no tab): every table of the database against the ones the agent and
//   the pipeline see, and the index
//   Profilage: what the profile detects in each column (declared type against the data, exceptions)
//   Descriptions: the descriptions the LLM writes, read and corrected in place
//   Jointures: the map of the tables and every join measured on the data
//   Métriques: the metrics on the left instead of the tables, each tested on the data (counts, never a value)
//   Fiches: what the agent reads of each table (its document) against the files today
// The steps in their order, one after the other, only in the guided setup, opened on purpose.
// Everything comes from the API of offline/studio/app.py; the page keeps no state of its own but the
// chosen database, tab, filter and selection.

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("fr-FR");
// A share in percent, rounded; one decimal near the ends, so that 0,4 % or 99,6 % is never said 0 % or 100 %
const pct = (x) => {
  if (x == null) return "—";
  const p = x * 100, tenth = Math.round(p * 10) / 10;
  const shown = p > 0 && p < 1 ? Math.max(0.1, tenth) : p > 99 && p < 100 ? Math.min(99.9, tenth) : Math.round(p);
  return `${new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 }).format(shown)} %`;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
// SQL as the eval dashboard shows it: its keywords in the accent (the list of offline/eval/eval_lib/dashboard_template.html,
// with CREATE TABLE for the documents of the Fiches tab)
const SQL_KW = /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|HAVING|LIMIT|AS|AND|OR|NOT|IN|EXISTS|DISTINCT|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|WITH|UNION|DESC|ASC|IS|NULL|BETWEEN|LIKE|ROUND|CAST|CREATE|TABLE)\b/gi;
const sqlHtml = (sql) => esc(sql).replace(SQL_KW, (m) => `<span class="sql-kw">${m}</span>`);
const plural = (n, word) => `${fmt.format(n)} ${word}${n > 1 ? "s" : ""}`;
// A word of a sentence with what it covers under the pointer; the ones said both in a tab and in the guided setup
const hint = (word, tip) => `<span class="hint" tabindex="0">${word}<span class="hint-tip">${tip}</span></span>`;
const WRITTEN = hint("rédigées", "À partir de la structure des tables et de leur profil");
const TESTED = hint("testées", "Lignes comptées deux fois, mesures vides, dates manquantes");

// The screens of a database: their title, sentence, filter and actions (none: the screen only shows), ai the actions that
// have the IA write (its star before them); TABS, the ones in the bar of tabs
const STEPS = {
  overview: { title: "Vue d'ensemble", key: "index", todo: "À réindexer" },
  scope: {
    title: "Périmètre", key: "scope", todo: "Hors périmètre",
    text: "Les tables de la base que l'agent et le pipeline voient : toutes, celles qui arriveront aussi, ou une liste. Dans une liste, une nouvelle table de la base reste dehors tant que tu ne l'ajoutes pas. Une table retirée garde son profil et ses descriptions, et l'agent la lit jusqu'à la prochaine indexation.",
    primary: () => `Ajouter au périmètre (${scopeSelection().add.length})`,
    secondary: () => `Retirer du périmètre (${scopeSelection().remove.length})`,
  },
  profile: {
    number: 1, title: "Profilage", key: "profile", todo: "À profiler",
    text: `L'agent lit correctement chaque table grâce à son ${hint("profil", "Colonnes vides, dates ou montants enregistrés comme du texte, données personnelles, partitions : lire la dernière photo ou toutes")}, mesuré sur son contenu réel.`,
    primary: (n) => `Profiler la sélection (${n})`, secondary: () => "Profiler les manquantes",
  },
  semantics: {
    number: 2, title: "Descriptions", key: "semantics", todo: "À décrire",
    text: `L'agent choisit ses tables et écrit ses requêtes grâce à ces descriptions, ${WRITTEN} par l'IA et à relire ici.`,
    // tables never described first; then, when only columns are left to describe, the tables that hold them
    primary: () => {
      const s = ui.data?.summary;
      return s && s.tables === s.described && s.partial ? `Compléter les descriptions (${fmt.format(s.partial)})` : "Décrire les manquantes";
    },
    secondary: (n) => `Redécrire la sélection (${n})`, ai: ["primary", "secondary"],
  },
  relations: {
    number: 3, title: "Jointures", key: "relations", todo: "Isolées",
    text: `L'agent passe d'une table à l'autre grâce à ces jointures, ${hint("vérifiées", "Part des valeurs d'une table qu'on retrouve dans l'autre")} sur les données.`,
    primary: () => "Chercher les jointures", secondary: () => "Ajouter une jointure", ai: ["primary"],
  },
  metrics: {
    number: 4, title: "Métriques", key: "metrics", todo: "À revoir",
    text: `L'agent reprend telles quelles ces formules, proposées par l'IA et ${TESTED} sur les données, relues à chaque démarrage, sans réindexer.`,
    primary: () => "Proposer des métriques", secondary: () => (ui.testing ? "Arrêter les tests" : "Tout tester"), ai: ["primary"],
  },
  index: {
    number: 5, title: "Fiches", key: "index", todo: "À réindexer",
    text: `L'agent lit chaque table dans sa ${hint("fiche", "Structure, descriptions, jointures et valeurs des colonnes catégorielles lues dans la base")}. Une correction n'arrive chez l'agent qu'une fois réindexée, à son prochain démarrage.`,
  },
};
const TABS = ["overview", "profile", "semantics", "relations", "metrics", "index"];
// The two tabs that read the index: the whole database, and each table's document
const indexScreen = (step = ui.step) => step === "overview" || step === "index";

const ui = {
  dbs: [], db: null, step: "profile", data: null, filter: "all", query: "", selected: new Set(),
  rel: null, path: { a: null, b: null, result: null }, adding: false,   // the Relations screen
  joins: { db: null, open: new Set(), detail: null, current: null, focus: null, decided: [], shown: 0, next: null },   // its list (joinQueue)
  mapBig: null,                                                        // the map opened large over the page: {height} it left in the panel, opening
  relCheck: null,                                                      // a relation measured, not added (its target is no key)
  met: null, metric: null, metricKind: "metric", testing: null,        // the Metrics screen (testing: {current, stop, panel, done, total})
  showQuery: false,                                                    // the query of a metric shown under its formula
  showChecks: false,                                                   // the checks of a test that passes, unfolded
  draft: null, draftError: null,                                       // the metric open changed, not saved: {metric, fields}; why it was refused
  tested: null,                                                        // what the last tests in the job panel found: {title: [title, halo], job, at}
  idx: null,                                                           // the Index screen
  scope: null,                                                         // the Périmètre screen: every table of the database
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
  $("db-select").onchange = (e) => {
    const db = e.target.value;
    e.target.value = ui.db;   // until a draft of a metric is saved or dropped
    leaveDraft(() => { switchDb(db); goTo(ui.step, true); });
  };
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
  ui.selected.clear(); ui.current = null; ui.detail = null; ui.rel = null; ui.path = { a: null, b: null, result: null }; ui.mapBig = null;
  ui.relCheck = null; closeConfirm($("confirm"));
  ui.met = null; ui.metric = null; if (ui.testing) ui.testing.stop = true; ui.idx = null; ui.scope = null; dropDraft();
  ui.tested = null;   // the tests of the other database leave the panel
}

// The database and its tabs, the sentence, filter and actions of the current tab
function renderStep() {
  const step = STEPS[ui.step];
  $("step-text").innerHTML = step.text || "";
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
  $("list-status").textContent = metrics || ui.step === "scope" ? "État" : ui.step === "index" ? "Index" : "Statut";
  $("filters").querySelector('[data-filter="review"]').hidden = ui.step !== "semantics";
  $("filters").querySelector('[data-filter="unclear"]').hidden = ui.step !== "semantics" || !ui.data?.tables.some((t) => t.unclear);
  $("filters").querySelector('[data-filter="noread"]').hidden = ui.step !== "profile" || !ui.data?.tables.some((t) => t.read_warn);
  $("type-filter-wrap").hidden = ui.step !== "profile";
  $("back-to-list").textContent = metrics ? "← Métriques" : "← Tables";
  closeConfirm($("confirm"));
  renderActions();
  renderJob();   // a job of another screen folds into its quiet line
}

// The bubble of a word (hint), centred under it, slid back inside the panel that holds it (overflow: hidden, the
// setup) or the window when the word ends a narrow line
function placeHint(e) {
  const word = e.target.closest?.(".hint");
  if (!word) return;
  const tip = word.querySelector(".hint-tip");
  tip.style.setProperty("--shift", "0px");
  const box = word.closest(".panel")?.getBoundingClientRect(), r = tip.getBoundingClientRect(), pad = 8;
  const left = Math.max(box?.left ?? 0, 0) + pad, right = Math.min(box?.right ?? innerWidth, innerWidth) - pad;
  tip.style.setProperty("--shift", `${r.right > right ? right - r.right : r.left < left ? left - r.left : 0}px`);
}
document.addEventListener("mouseover", placeHint);
document.addEventListener("focusin", placeHint);

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
  const current = ui.step === "scope" ? "overview" : ui.step;   // the Périmètre opens from the Vue d'ensemble
  $("tabs").innerHTML = TABS.map((key) => `<button type="button" data-step="${key}"${key === current ? ' aria-current="page"' : ""}>`
    + `${STEPS[key].title}${counts[key] != null ? `<span class="tab-count">${rollSlot(`tab-${key}`, fmt.format(counts[key]))}</span>` : ""}</button>`).join("");
  rollFigures($("tabs"));
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
  if (ui.step === "scope") return loadScope();
  if (ui.step === "relations") return loadRelations();
  if (ui.step === "metrics") return loadMetrics();
  if (indexScreen()) return loadIndex();
}

// A tab of the current database (`force`: drawn again though it is the current one, after a change of database).
// The promise ends once the screen is read: its tables when they are not the database's, and its own content
function goTo(step, force = false) {
  if (!STEPS[step]) return;
  if (step === ui.step && !ui.home && !ui.wizard && !force) return;
  if (ui.draft) { leaveDraft(() => goTo(step, force)); return; }   // a metric changed and not saved: asked first
  if (ui.home) leaveHome();
  if (ui.wizard) leaveWizard();
  ui.step = step;
  remember();
  ui.filter = "all"; ui.selected.clear(); ui.detail = null; ui.adding = false; ui.relCheck = null; ui.mapBig = null;
  document.body.classList.remove("map-open");
  [...$("filters").children].forEach((c) => c.classList.toggle("on", c.dataset.filter === "all"));
  renderStep();
  const tables = ui.data?.db !== ui.db ? loadTables() : null;
  const index = !ui.idx && !indexScreen() ? loadIndex() : null;   // the head of the database tells its index
  if (step === "scope" || step === "relations" || step === "metrics" || indexScreen()) {
    if (step === "index" && !force) ui.current = null;   // the Fiches tab opens on its own first row (openFirstRow)
    if (step === "scope" && !force) ui.current = null;   // the Périmètre opens on the database against its scope
    if (step === "relations" && ui.autoOpened) ui.current = null;   // the whole map, unless a table was chosen
    if (ui.data) { renderHeader(); renderRows(); }
    renderDetail();
    return Promise.all([tables, index, loadScreen()]);
  }
  if (ui.data) { renderHeader(); renderRows(); }
  // the card of the other tab fades out while this one is read (renderDetail fades it in): no card of Profilage under
  // the head of Descriptions
  if (ui.current) { $("detail").classList.add("switching"); openTable(ui.current, { again: true }); } else renderDetail();
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
  if (MAP.rowless && ui.step === "relations" && ui.rel) renderDetail();   // the rows of the tables on the map
}

function renderHeader() {
  const { data } = ui, s = data.summary;
  const when = new Date(data.read_at * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  $("refresh").title = data.source === "database"
    ? `Relire la base (${s.tables} tables lues à ${when})` : "Relire la base (tables lues dans les fichiers)";
  if (data.error) showBanner(`La base ne répond pas : les tables viennent de profile.yaml et tables.yaml. ${data.error}`);
  else $("banner").hidden = true;
  const todo = { scope: ui.scope ? ui.scope.tables.filter(outOfScope).length : 0, profile: s.tables - s.profiled, semantics: undescribed(s),
    relations: ui.data.tables.filter((t) => t.status.relations !== "done").length,
    metrics: ui.met ? ui.met.counts.check + ui.met.counts.error : 0,
    index: ui.idx ? ui.idx.counts.changed + ui.idx.counts.missing : 0 }[STEPS[ui.step].key];
  const review = ui.data.tables.filter((t) => t.rewritten).length;
  const unclear = ui.data.tables.filter((t) => t.unclear).length;
  const noread = ui.data.tables.filter((t) => t.read_warn).length;
  const all = ui.step === "metrics" ? ui.met?.metrics.length ?? 0 : ui.step === "scope" ? ui.scope?.tables.length ?? 0 : s.tables;
  const counts = { all, todo, review, unclear, noread };
  const labels = { all: "Toutes", todo: STEPS[ui.step].todo, review: "À relire", unclear: "À préciser", noread: "Sans mode" };
  $("filters").querySelector('[data-filter="noread"]').hidden = ui.step !== "profile" || !noread;
  $("filters").querySelector('[data-filter="unclear"]').hidden = ui.step !== "semantics" || !unclear;
  const noReview = ui.step === "metrics" && !todo;   // no metric to review: no empty filter
  $("filters").querySelector('[data-filter="todo"]').hidden = noReview;
  // the last one given a mode, stated or reviewed: back to every row, not an empty list
  if ((ui.filter === "noread" && !noread) || (ui.filter === "unclear" && !unclear) || (ui.filter === "todo" && noReview)) {
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
    return n ? `<span class="pill done">${plural(n, "jointure")}</span>`
      : '<span class="pill decide" title="Aucune jointure : l\'agent ne peut joindre cette table à aucune autre">isolée</span>';
  }
  if (status === "none") return '<span class="pill todo">à décrire</span>';
  if (status === "partial") return `<span class="pill decide" title="Colonnes ajoutées depuis la dernière description">${table.undescribed} col. à décrire</span>`;
  if (table.unclear) return `<span class="pill decide" title="Colonnes dont l'IA n'a pas su dire le sens ou l'unité : à préciser par toi">${fmt.format(table.unclear)} à préciser</span>`;
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
  if (ui.step === "scope") {
    return (ui.scope?.tables || []).filter((t) => (!q || t.name.toLowerCase().includes(q)) && (ui.filter === "all" || outOfScope(t)));
  }
  const type = ui.step === "profile" ? $("type-filter").value : "";   // the type chosen in the list of the Profilage tab
  const todo = indexScreen() ? (t) => ["changed", "missing"].includes(indexStatus(t.name)) : (t) => t.status[key] !== "done";
  const shown = ui.filter === "review" ? (t) => t.rewritten > 0 : ui.filter === "unclear" ? (t) => t.unclear > 0
    : ui.filter === "noread" ? (t) => t.read_warn : todo;
  return (ui.data?.tables || []).filter((t) => (!q || t.name.toLowerCase().includes(q)) && (ui.filter === "all" || shown(t))
    && (!type || (t.type || "none") === type));
}

// On arrival on a tab with nothing open, its first row opened, as a click would (Profilage, Descriptions, Fiches,
// Métriques): never on Relations, whose screen without a table is the whole map, nor on a phone, where the card would
// take the place of the list. Once a row is open, the list drawn again leaves it be. A table opened so (ui.autoOpened,
// until one is opened by hand) does not follow on Relations, which keeps its whole map
const OPEN_FIRST = ["profile", "semantics", "index", "metrics"];
function openFirstRow() {
  if (!OPEN_FIRST.includes(ui.step) || ui.home || ui.wizard || window.matchMedia("(max-width: 560px)").matches) return;
  if (ui.step === "metrics" ? ui.metric : ui.current) return;
  const row = $("rows").querySelector(":scope > .row");
  if (!row) return;
  if (ui.step === "metrics") openMetric(row.dataset.metric, "metric");
  else { ui.autoOpened = true; openTable(row.dataset.table, { again: true }); }
}

function renderRows() {
  renderPersonal();   // the bars of the personal data and of an older profile: on the Profilage tab only
  renderOutdated();
  if (ui.step === "metrics") { renderMetricRows(); openFirstRow(); return; }
  if (ui.step === "scope") { renderScopeRows(); return; }
  $("list-extra").innerHTML = "";
  $("empty").textContent = "Aucune table ne correspond.";
  const rows = visibleRows();
  $("rows").innerHTML = rows.map((t) => `<div class="row${t.name === ui.current ? " current" : ""}${ui.draft && ui.draft.kind !== "metric" && ui.draft.name === t.name ? " unsaved" : ""}" data-table="${esc(t.name)}">
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
  openFirstRow();
}

// The actions of the current tab, in its bar (the Vue d'ensemble and the Fiches have none: the actions on all the
// database are in its head)
function renderActions() {
  const step = STEPS[ui.step];
  $("run-primary").hidden = !step.primary;
  $("run-secondary").hidden = !step.secondary;
  const label = (which) => `${step.ai?.includes(which) ? AI_STAR : ""}${esc(step[which](ui.selected.size))}`;
  if (step.primary) $("run-primary").innerHTML = label("primary");
  if (step.secondary) $("run-secondary").innerHTML = label("secondary");
  // "Ajouter une jointure", the tests of the metrics and a change of the scope start no job
  $("run-primary").toggleAttribute("data-job", ui.step !== "scope");
  $("run-secondary").toggleAttribute("data-job", !["relations", "metrics", "scope"].includes(ui.step));
  gateButtons();
}

// What a job is, in the title of a button it greys
const JOB_WHAT = { profile: "profilage", semantics: "description", relations: "jointures", metrics: "métriques", index: "index", eval: "évaluation" };

// Why one of the two actions of the page has nothing to do now, or null
function idleReason(button) {
  const s = ui.data?.summary, step = ui.step;
  if (step === "scope" && (button.id === "run-primary" || button.id === "run-secondary")) {
    if (ui.scope && ui.scope.counts.base == null) return "La base ne répond pas : le périmètre se choisit sur la liste de ses tables";
    if (button.id === "run-primary" && !scopeSelection().add.length) return "Coche des tables hors périmètre";
    if (button.id === "run-secondary" && !scopeSelection().remove.length) return "Coche des tables du périmètre";
    return null;
  }
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

// A decision only you can make, waiting on one control (the read mode, the type, the personal data, the scope): a pale
// ring around it until it is made, two pulses the first time it shows in the session, not each time it is drawn again
const decideSeen = new Set();
function toDecide(key) {
  const first = !decideSeen.has(key);
  decideSeen.add(key);
  return first ? " to-decide first" : " to-decide";
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
       <button class="btn small primary${toDecide(`${ui.db}:personal`)}" id="personal-all" data-hide="1">Cacher partout</button>`
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
  if (ui.current) openTable(ui.current, { again: true });
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
$("refresh").addEventListener("click", async () => {
  await loadTables(true);
  if (ui.step === "scope") loadScope();   // the tables the database gained or lost
});

// ── Columns of a table ───────────────────────────────────

// again: the table open drawn again (a tab, a job, a change), or opened by openFirstRow, not a choice of yours
async function openTable(name, { again = false } = {}) {
  if (ui.draft && ui.draft.name !== name) { leaveDraft(() => openTable(name, { again })); return; }
  const fromList = !document.body.classList.contains("card-open");   // on a phone: the card comes in view, once
  if (!again) ui.autoOpened = false;   // chosen by you: it follows on Relations
  if (ui.step === "relations") { selectOnMap(name, again); renderRows(); showCurrentRow(); if (fromList) showCard(); return; }
  ui.current = name;
  if (ui.step === "scope") { renderRows(); showCurrentRow(); renderDetail(); if (fromList) showCard(); return; }   // its state is in the list
  renderRows(); showCurrentRow();
  const step = ui.step, db = encodeURIComponent(ui.db), table = encodeURIComponent(name), shown = ui.detail;
  try {
    ui.detail = await api(step === "index" ? `/api/db/${db}/index/${table}`
      : `/api/db/${db}/tables/${table}${step === "profile" ? "" : "/semantics"}`);
  } catch (e) {
    ui.detail = { name, error: e.message };
  }
  if (ui.current !== name || ui.step !== step) return;
  renderDetail();
  // The descriptions the IA just wrote in the card open as its job ends: their star shines once, drawn again it rests
  if (again && step === "semantics" && shown?.name === name && !ui.detail.error) {
    const fresh = freshDescriptions(shown, ui.detail);
    $("detail").querySelectorAll(".origin.ai[data-ai]").forEach((mark) => mark.classList.toggle("fresh", fresh.has(mark.dataset.ai)));
  }
  if (fromList) showCard();
}

// The descriptions of a table the IA wrote since the card showed it (column: its name, "" for the table)
function freshDescriptions(before, after) {
  const was = new Map((before.columns || []).map((c) => [c.column, c.description]));
  const fresh = new Set((after.columns || []).filter((c) => c.description && !c.edited && c.description !== was.get(c.column)).map((c) => c.column));
  if (after.description && !after.description_edited && after.description !== before.description) fresh.add("");
  return fresh;
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
  el.classList.remove("menu-open", "has-draft");   // drawn again, its menus closed; a draft says itself again (showDraft)
  if (el.classList.contains("switching")) {   // the card of another tab, faded out while this one was read (goTo)
    el.classList.replace("switching", "arriving");
    setTimeout(() => el.classList.remove("arriving"), 250);
  }
  if (ui.step === "relations") renderRelations(el, message);
  else if (ui.step === "scope") renderScope(el, message);
  else if (ui.step === "metrics") renderMetric(el, message);
  else if (indexScreen()) renderIndex(el, message);
  else if (!d) el.innerHTML = '<p class="placeholder">Choisis une table pour voir ses colonnes.</p>';
  else if (d.error) el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2></div><p class="note">${esc(d.error)}</p>`;
  else if (ui.step === "profile") renderProfile(d, el, message);
  else renderSemantics(d, el, message);
  // A card open: on a phone it takes the place of the list. On Relations, a table chosen or a path asked, whose way
  // back is in the card ("← Toutes les jointures")
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
// "Compléter" of the Descriptions: make semantics on one table, additive (offline/semantic_gen/tables.py)
const COMPLETE_DESCRIPTIONS = "L'IA ne décrit que ces colonnes : les autres descriptions, tes corrections et les colonnes catégorielles ne bougent pas";

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

  const shown = draftProps(d);   // its properties as chosen in the draft, over the saved ones
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
    + `<div class="table-text"><div class="props" aria-label="Propriétés de la table">${typeProp(shown.table_type)}${readProp(shown.reading)}${asOfProp(shown)}</div>`
    + `${readBand(shown.reading)}</div>`
    + old + error
    + `<div class="columns">
        <div class="line head"><span>Colonne</span><span>Type en base</span><span>Contenu détecté</span><span class="num">Remplissage</span><span class="num">Distinctes</span></div>
        ${lines}
      </div>${DRAFT_BAR}`;

  $("profile-again").onclick = () => runJob("profile", { tables: [d.name] });
  if (missing.length) $("profile-complete").onclick = () => runJob("profile", { tables: [d.name], complete: true });
  setupProps(d);
  bindDraftBar();
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
    c.unclear ? `${fmt.format(c.unclear)} à préciser` : null,
    c.rewritten ? `${fmt.format(c.rewritten)} à relire` : null,
    c.categorical ? `${fmt.format(c.categorical)} catégorielle${c.categorical > 1 ? "s" : ""}` : "aucune catégorielle",
    c.edited ? `${fmt.format(c.edited)} corrigée${c.edited > 1 ? "s" : ""} par toi` : null].filter(Boolean).join(" · ");
}

// ── Where a value comes from, the same marks and words on every tab: measured on the data (the profile, the rules,
// the measure of a relation) in grey, proposed by the IA in orchid, decided by you (ink); an amber dot, rewritten by the
// IA over a text: to read again; no mark when the origin is unknown (written before the studio kept it) ──

const PENCIL = '<path d="M10.5 2.5l3 3L6 13l-3.5.5L3 10z"/>';
const ORIGIN = {
  data: { word: "données", says: "mesuré sur les données", path: '<path d="M3.5 13V9M8 13V3.5M12.5 13V7"/>' },
  ai:   { word: "IA", says: "proposé par l'IA", path: '<path d="M8 1.8l1.45 4.2 4.25 1.5-4.25 1.5L8 13.2 6.55 9 2.3 7.5 6.55 6z"/>' },
  you:  { word: "toi", says: "décidé par toi", path: PENCIL },
};

// The mark of an origin (attrs: data attributes, hidden), its title said on hover
const originMark = (kind, title = ORIGIN[kind].says, attrs = "") =>
  `<svg class="origin ${kind}" viewBox="0 0 16 16" role="img" aria-label="${esc(title)}" ${attrs}><title>${esc(title)}</title>${ORIGIN[kind].path}</svg>`;
// The star of the IA before a button that has it write: describe, find the joins, propose metrics
const AI_STAR = `<svg class="origin ai" viewBox="0 0 16 16" aria-hidden="true">${ORIGIN.ai.path}</svg>`;
// A mark before what it qualifies (a pill)
const withOrigin = (kind, html, title) => `<span class="origin-item">${originMark(kind, title)}${html}</span>`;

// The legend of the tabs that show origins, in the footer (the marks say it on hover too)
const ORIGIN_LEGEND = `<span>Origine :</span>${["data", "ai", "you"].map((kind) =>
  `<span class="${kind}">${originMark(kind)}${ORIGIN[kind].says}</span>`).join("")}`
  + '<span class="review"><span class="todo-dot"></span>à relire</span>'
  + '<span class="unclear"><span class="unclear-mark" aria-hidden="true">?</span>à préciser</span>';
const ORIGIN_STEPS = ["profile", "semantics", "relations", "metrics"];

// The marks of a description (column: its name, "" for the table): rewritten by the IA (an amber dot, to read
// again), written by the IA or corrected by you (kept when the table is described again), none while it is empty;
// toggled as it is saved
const rewrittenDot = (column, on) => `<span class="todo-dot" data-dot="${esc(column)}" title="Réécrite par l'IA par-dessus ton texte : à relire" ${on ? "" : "hidden"}></span>`;
// A column whose meaning or unit the IA could not tell: a button, pressed it keeps the description as it is (a draft
// field, written by « Enregistrer »); a correction of its text states it too
const UNCLEAR_SAYS = "À préciser : l'IA n'a pas su dire le sens ou l'unité de cette colonne. Écris-le, ou clique si la description te convient telle quelle";
const unclearMark = (column) => `<button type="button" class="unclear-mark" data-unclear="${esc(column)}" title="${UNCLEAR_SAYS}" aria-label="${UNCLEAR_SAYS}">?</button>`;
const descriptionMarks = (column, rewritten, edited, text) => rewrittenDot(column, rewritten)
  + originMark("ai", "Rédigée par l'IA", `data-ai="${esc(column)}" ${edited || !text ? "hidden" : ""}`)
  + originMark("you", "Corrigée par toi : gardée quand l'IA redécrit", `data-hand="${esc(column)}" ${edited && text ? "" : "hidden"}`);

function renderSemantics(d, el, message) {
  if (!d.described) {
    el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="grow"></span>
        <button class="btn primary small" id="describe-one" data-job>Décrire</button></div>
      <p class="placeholder">Pas encore décrite : l'IA rédigera la description de la table et de ses colonnes.</p>`;
    $("describe-one").onclick = () => runJob("semantics", { tables: [d.name] });
    return;
  }
  // The columns the agent sees without a description (arrived since, or emptied): "Compléter" describes those only
  const missing = d.columns.filter((c) => !c.excluded && !c.description).map((c) => c.column);
  // the values of the draft over the saved ones; a change goes to the draft, nothing is written before « Enregistrer »
  const draft = draftFields("semantics", d.name), value = (key, saved) => (key in draft ? draft[key] : saved);
  const change = (key, typed, saved) => setDraft("semantics", d.name, key, typed, saved);
  const lines = d.columns.map((c) => {
    const key = c.key ? ` <span class="pill key">${c.key === "unique" ? "clé unique" : "clé"}</span>` : "";
    if (c.excluded) {
      return `<div class="sline hidden-col"><span class="cell">${esc(c.column)}${key} <span class="pill off">cachée</span></span>
        <span class="cell mono">${esc((c.type || "—").toLowerCase())}</span>
        <span class="muted-text">cachée à l'agent : pas de description</span><span></span></div>`;
    }
    const values = c.distinct == null ? "" : `<span class="values" title="${c.sampled ? "Valeurs distinctes sur l'échantillon du profil" : "Valeurs distinctes de la table"}">${plural(c.distinct, "valeur")}</span>`;
    return `<div class="sline" data-draft-row="${esc(c.column)}">
      <span class="cell">${esc(c.column)}${key}</span>
      <span class="cell mono" title="${esc(c.type || "")}">${esc((c.type || "—").toLowerCase())}</span>
      <span class="desc-cell">${c.unclear ? unclearMark(c.column) : ""}${descriptionMarks(c.column, c.rewritten, c.edited, c.description)}<textarea class="edit" rows="1" data-column="${esc(c.column)}" placeholder="Ajouter une description…" aria-label="Description de ${esc(c.column)}">${esc(value(`column:${c.column}`, c.description))}</textarea></span>
      <label class="switch-cell">${values}<input type="checkbox" class="switch" data-column="${esc(c.column)}" ${value(`categorical:${c.column}`, c.categorical) ? "checked" : ""} aria-label="${esc(c.column)} catégorielle"></label>
    </div>`;
  }).join("");

  el.innerHTML = `<div class="detail-head"><h2>${esc(d.name)}</h2><span class="figures" id="sem-figures">${semanticsFigures(d)}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      <button class="btn ghost small" id="mark-read" title="Les descriptions réécrites par l'IA sortent de « à relire »" ${d.counts.rewritten ? "" : "hidden"}>Tout marquer relu</button>
      <button class="btn ghost small" id="describe-again" data-job title="L'IA réécrit les descriptions de cette table ; celles corrigées par toi sont gardées">${AI_STAR}Redécrire</button></div>
    <div class="confirm" id="sem-confirm" role="group" aria-label="Redécrire ${esc(d.name)}" hidden></div>
    ${missing.length ? `<div class="note with-action"><span>${plural(missing.length, "colonne")} sans description : ${missing.map((c) => `<span class="mono">${esc(c)}</span>`).join(", ")}.</span>
      <button class="btn ghost small" id="describe-missing" data-job title="${COMPLETE_DESCRIPTIONS}">${AI_STAR}Compléter</button></div>` : ""}
    ${message ? `<p class="note error">${esc(message)}</p>` : ""}
    <div class="table-text lead" data-draft="description">
      <label class="field-label" for="sem-description">Description de la table</label>
      <span class="desc-cell block">${descriptionMarks("", d.description_rewritten, d.description_edited, d.description)}<textarea class="edit block" id="sem-description" rows="1" placeholder="Décrire cette table…">${esc(value("description", d.description))}</textarea></span>
    </div>
    <div class="columns">
      <div class="sline head"><span>Colonne</span><span>Type</span><span>Description</span><span class="num">Catégorielle</span></div>
      ${lines}
    </div>
    ${DRAFT_BAR}`;

  // the draft saved or dropped before the question of « Redécrire » (its job would ask after it)
  $("describe-again").onclick = () => leaveDraft(() => redescribeTable(ui.detail));
  if (missing.length) $("describe-missing").onclick = () => runJob("semantics", { tables: [d.name] });
  $("mark-read").onclick = () => markRead(d);
  const columns = new Map(d.columns.map((c) => [c.column, c]));
  el.querySelectorAll("textarea.edit").forEach((area) => {
    fitHeight(area);
    const column = area.dataset.column;
    area.addEventListener("input", () => {
      fitHeight(area);
      if (column) change(`column:${column}`, area.value, columns.get(column).description);
      else change("description", area.value, d.description);
    });
  });
  el.querySelectorAll("input.switch").forEach((box) => box.addEventListener("change", () => {
    change(`categorical:${box.dataset.column}`, box.checked, columns.get(box.dataset.column).categorical);
  }));
  el.querySelectorAll("button.unclear-mark").forEach((mark) => mark.addEventListener("click", () => {
    const key = `unclear:${mark.dataset.unclear}`;
    change(key, key in draftFields("semantics", d.name), true);
  }));
  bindDraftBar();
}

// ── The properties of a table, in a bar under its name in the Profilage card (what it is and how to read it): a pill each
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

// origin: the mark before the value (data, you; none when nothing was detected nor chosen); decide: the ring of a
// decision waiting on it (toDecide)
function propPill(name, key, value, { tone = "", title = "", origin = null, decide = "" } = {}) {
  return `<button type="button" class="prop${tone ? ` ${tone}` : ""}${decide}" id="${name}-pill" aria-haspopup="true" aria-expanded="false" title="${esc(title)}">
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
  return `<span class="prop-wrap" id="type-wrap" data-draft="table_type">
      ${propPill("type", "Type", typeName(t.type), { tone: t.type ? "" : "none", title: why, origin: t.edited ? "you" : t.type ? "data" : null,
        decide: t.type ? "" : toDecide(`${ui.db}:type:${ui.detail?.name}`) })}
      <div class="type-menu" id="type-menu" role="listbox" aria-label="Type de la table" hidden>
        <div class="menu-head">${esc(why)}${ui.data?.summary.relations ? ""
          : '<span class="menu-fact">Les jointures ne sont pas encore trouvées (onglet Jointures) : le type peut encore changer.</span>'}</div>
        ${Object.entries(TABLE_TYPES).map(([type, words]) => option(type, words)).join("")}
        ${t.edited ? `<button type="button" role="option" class="type-option" data-type=""><span class="type-check"></span>`
          + `<span><b>Revenir à la détection</b><small>${typeName(t.detected).toLowerCase()}</small></span><span></span></button>` : ""}
        <p>Ton choix est écrit dans tables.yaml et passe avant la détection.</p>
      </div>
    </span>`;
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
  else if (r.unread) facts.push("Partitions pas encore lues : « Compléter » le profil les lit (onglet Profilage).");
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
  return `<span class="prop-wrap" id="read-wrap" data-draft="read_mode">
      ${propPill("read", "Mode de lecture", modeName(r.mode), { tone: r.problems.length ? "bad" : r.warn ? "warn" : "", title: plainText(sentence),
        origin: r.mode ? "you" : null, decide: r.warn && !r.problems.length ? toDecide(`${ui.db}:read:${ui.detail?.name}`) : "" })}
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
  // LIVE needs the version column, a human choice: its button opens the menu pre-filled instead of saving
  const apply = r.suggested === "live"
    ? `<button type="button" class="btn small" data-open-read data-pick="live">Compléter LIVE</button>`
    : `<button type="button" class="btn small" id="read-apply">Appliquer ${modeName(r.suggested)}</button>`;
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
      <button type="button" class="btn primary small" id="read-save">Choisir</button>`;
  }
  // No key chosen yet: the one the data suggests (live proposed), for the human to confirm or change
  const keys = new Set(r.key_columns.length ? r.key_columns : (r.suggested_key ? [r.suggested_key] : []));
  const version = r.version_column || d.column_facts.find((c) => c.kind === "date" && !r.partitioned_by.includes(c.column))?.column || "";
  return `<span class="read-field">Clé : la ou les colonnes d'une même ligne d'une version à l'autre</span>
    <div class="read-keys">${columns.map((c) => `<label><input type="checkbox" value="${esc(c)}"${keys.has(c) ? " checked" : ""}>${esc(c)}</label>`).join("")}</div>
    <label class="read-field">Version : la plus grande est la plus récente<select id="read-version">${version ? "" : '<option value="">choisir…</option>'}
      ${choices(version)}</select></label>
    <button type="button" class="btn primary small" id="read-save">Choisir</button>`;
}

// The date the data stops: a pill (dashed while there is none), a field under it
const AS_OF = "La date de référence de « ce mois-ci », « l'an dernier »… Sans elle, les périodes partent d'aujourd'hui.";
const dayMonthYear = (iso) => iso.split("-").reverse().join("/");
const asOfPill = (d) => (d.data_as_of ? propPill("asof", "Arrêtée au", dayMonthYear(d.data_as_of), { title: AS_OF })
  : `<button type="button" class="prop empty" id="asof-pill" aria-haspopup="true" aria-expanded="false" title="${esc(AS_OF)}">+ Date d'arrêt</button>`);
function asOfProp(d) {
  return `<span class="prop-wrap" id="asof-wrap" data-draft="data_as_of">
      ${asOfPill(d)}
      <div class="type-menu asof-menu" id="asof-menu" role="dialog" aria-label="Date d'arrêt des données" hidden>
        <label class="read-field" for="sem-as-of">Données arrêtées au
          <input type="date" class="edit date" id="sem-as-of" data-field="data_as_of" value="${esc(d.data_as_of)}"></label>
        <p>${AS_OF}</p>
        <button type="button" class="btn ghost small" id="asof-clear" ${d.data_as_of ? "" : "hidden"}>Retirer la date</button>
      </div>
    </span>`;
}

// The properties of a table as its card shows them: the saved ones, the draft's over them (a mode chosen is no longer
// to decide); the saved value of each, to tell a change from a value put back
const readModeOf = (r) => ({ mode: r?.mode || null, partition_column: r?.partition_column || null,
  key_columns: r?.key_columns || [], version_column: r?.version_column || null });
const savedProperty = (d, key) => ({ table_type: d.table_type?.edited ? d.table_type.type : "", data_as_of: d.data_as_of || "",
  read_mode: readModeOf(d.reading) })[key];
function draftProps(d) {
  const f = draftFields("properties", d.name), shown = { ...d };
  if ("table_type" in f) shown.table_type = { ...d.table_type, type: f.table_type || d.table_type.detected, edited: Boolean(f.table_type) };
  if ("data_as_of" in f) shown.data_as_of = f.data_as_of;
  if ("read_mode" in f && d.reading) shown.reading = { ...d.reading, ...f.read_mode, warn: f.read_mode.mode ? false : d.reading.warn, problems: [] };
  return shown;
}
// A property chosen: to the draft, the card drawn again (its menu closed)
const chooseProperty = (d, key, value) => { setDraft("properties", d.name, key, value, savedProperty(d, key)); renderDetail(); };

function setupProps(d) {
  const shown = draftProps(d);
  PROPS.forEach((p) => {
    if ($(`${p}-pill`)) $(`${p}-pill`).onclick = (e) => { e.stopPropagation(); toggleProp(p); };
  });
  $("detail").querySelectorAll("[data-type]").forEach((b) => { b.onclick = () => chooseProperty(d, "table_type", b.dataset.type); });
  // A mode with columns: they are chosen below the options, then « Choisir »; "Aucun" at once
  const pick = (mode) => {
    $("read-menu").querySelectorAll("[data-mode]").forEach((o) => o.classList.toggle("picked", o.dataset.mode === mode));
    const fields = $("read-fields");
    fields.innerHTML = readFields(mode, shown);
    fields.hidden = false;
    $("read-save").onclick = () => chooseProperty(d, "read_mode", readModeOf(mode === "live"
      ? { mode, key_columns: [...fields.querySelectorAll(".read-keys input:checked")].map((box) => box.value), version_column: $("read-version").value || null }
      : { mode, partition_column: $("read-partition").value || null }));
  };
  if (d.reading) {
    $("read-menu").querySelectorAll("[data-mode]").forEach((b) => {
      b.onclick = () => (b.dataset.mode ? pick(b.dataset.mode) : chooseProperty(d, "read_mode", readModeOf(null)));
    });
    $("detail").querySelectorAll("[data-open-read]").forEach((b) => {
      b.onclick = (e) => { e.stopPropagation(); toggleProp("read", true);
        if (b.dataset.pick) pick(b.dataset.pick); else if (shown.reading.mode) pick(shown.reading.mode); };
    });
    if ($("read-apply")) $("read-apply").onclick = () => savePropertiesNow({ read_mode: { mode: shown.reading.suggested, partition_column: shown.reading.partitioned_by[0] } });
  }
  const asOf = $("sem-as-of");
  if (!asOf) return;
  asOf.addEventListener("change", () => chooseProperty(d, "data_as_of", asOf.value));
  $("asof-clear").onclick = () => chooseProperty(d, "data_as_of", "");
}

// A property written at once by an explicit button (« Appliquer LAST » of the band): the card drawn again; a mode of the
// draft gives way to it, the rest of the draft stays
async function savePropertiesNow(body) {
  const d = ui.detail;
  saving("Enregistrement…");
  try {
    const saved = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(d.name)}/properties`,
      { method: "POST", body: { ...body, version: d.version } });
    if (ui.detail?.name !== d.name) return;
    ui.detail = { ...ui.detail, ...saved };
  } catch (e) {
    const state = $("save-state");
    state.textContent = e.status === 409 ? "tables.yaml a changé : rouvre la table" : e.message;
    state.className = "save-state error";
    return;
  }
  if (ui.draft?.kind === "properties" && "read_mode" in body) {
    delete ui.draft.fields.read_mode;
    if (!Object.keys(ui.draft.fields).length) dropDraft();
  }
  renderDetail();
  saidSaved("Enregistré");
  loadTables();
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
    button: (replace) => `Redécrire ${plural(replace ? written + edited : written, "description")}`, ai: true,
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
    button: () => `Redécrire ${plural(chosen.length, "table")}`, ai: true,
    run,
  });
}

// The strip before an action that costs or exposes: what it does, and before describing again what the LLM keeps,
// the corrections replaced on a click (the box only when there are corrections: edited). job: it starts a job (data-job);
// ai: the IA writes (its star on the button)
function openConfirm(strip, { text, edited, button, run, job = true, ai = false }) {
  const label = (replace) => `${ai ? AI_STAR : ""}${esc(button(replace))}`;
  strip.innerHTML = `<p>${text}</p>
    <span class="confirm-actions">
      ${edited ? `<label class="check"><input type="checkbox" data-replace>Remplacer aussi ${edited > 1 ? `mes ${fmt.format(edited)} corrections` : "ma correction"}</label>` : ""}
      <button class="btn small ghost" type="button" data-cancel>Annuler</button>
      <button class="btn small primary" type="button" data-run${job ? " data-job" : ""}>${label(false)}</button>
    </span>`;
  strip.hidden = false;
  const replace = strip.querySelector("[data-replace]"), go = strip.querySelector("[data-run]");
  if (replace) replace.onchange = () => { go.innerHTML = label(replace.checked); };
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
      { method: "POST", body: { changes: [{ field: "read" }], version: d.version } });
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
    showBanner(`Impossible de lire les jointures : ${e.message}`, true);
    return;
  }
  if (ui.data) { renderHeader(); renderRows(); }
  if (ui.step !== "relations") return;
  renderDetail();
  if (ui.path.a && ui.path.b && !ui.path.result) findPath();   // a path asked from another screen (a metric left out)
}

// A click on a table, in the list or on the map, opens its relations and makes it the start of a path (the end of the
// path is chosen in the list "à" only); a second click on the table chosen goes back to the whole map
function selectOnMap(name, again = false) {
  if (name === ui.current && !ui.path.b && !again) { showAllTables(); return; }
  ui.current = name;
  ui.relFocus = null;
  ui.path = { a: name, b: null, result: null };
  renderDetail();
}
function showAllTables() {
  ui.current = null; ui.relFocus = null; ui.path = { a: null, b: null, result: null };
  renderRows(); renderDetail();
}

// The tables of the path shown, and the step of each: a relation is on the path between two steps that follow
const pathLayers = () => new Map((ui.path.result?.layers || []).flatMap((tables, i) => tables.map((t) => [t, i])));
const onPath = (r, layer) => layer.has(r.source) && layer.has(r.target) && Math.abs(layer.get(r.source) - layer.get(r.target)) === 1;

// What the map last drew: where each table was moved from its place in the layout (a table that moves glides from
// there), the animations of the camera and of the path, a map drawn before the rows of its tables were read; the
// layout it draws (mapLayout) and the one being computed
const MAP = { shift: null, db: null, camera: 0, packet: 0, rowless: false, layout: null, computing: null };
const compactRows = new Intl.NumberFormat("fr-FR", { notation: "compact", maximumFractionDigits: 1 });

// ── The layout of the map: ELK Layered (static/vendor/elk.bundled.js, loaded with the tab) lays out the tables joined
// to others, in layers from left to right, its lines at right angles around the tables. A reference joined by several
// tables is not a box among them, its lines crossing the map: its name is written in a pill under each table joining
// it, and its box waits on a row under the map with the tables joined to none ──
const PILL_USERS = 2;      // a reference joined by this many tables or more: in pills
const PILL_CHAR  = 5.8;    // a character of the pills (monospace, 9.5px)
const PILL_LINE  = 40;     // characters of the pills of a table at most, the others counted ("+2")
const ELK_OPTIONS = {
  "elk.algorithm": "layered", "elk.direction": "RIGHT", "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.spacing.nodeNodeBetweenLayers": "64", "elk.spacing.nodeNode": "14", "elk.layered.spacing.edgeNodeBetweenLayers": "14",
  "elk.spacing.edgeEdge": "6", "elk.layered.spacing.edgeEdgeBetweenLayers": "6", "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.thoroughness": "10", "elk.separateConnectedComponents": "true", "elk.padding": "[top=40,left=40,bottom=40,right=40]",
};

let elkLoading = null;
function loadElk() {
  elkLoading ||= new Promise((resolve, reject) => {
    const script = Object.assign(document.createElement("script"), { src: "/static/vendor/elk.bundled.js" });
    script.onload = () => resolve(new window.ELK());
    script.onerror = () => reject(new Error("elk.bundled.js"));
    document.head.append(script);
  });
  return elkLoading;
}

// The references in pills (joined by PILL_USERS tables or more), and the pills of each table: the ones it joins (a
// reference in pills joining another one carries its pill too, on its box under the map)
function mapPills() {
  const refs = new Set(ui.rel.references || []), users = new Map(), pills = {};
  ui.rel.relations.forEach((r) => {
    if (refs.has(r.target) && r.source !== r.target) users.set(r.target, (users.get(r.target) || new Set()).add(r.source));
  });
  const pilled = new Set([...users].filter(([, u]) => u.size >= PILL_USERS).map(([t]) => t));
  ui.rel.relations.forEach((r) => { if (pilled.has(r.target) && r.source !== r.target) (pills[r.source] ||= new Set()).add(r.target); });
  return { pilled, pills: Object.fromEntries(Object.entries(pills).map(([t, s]) => [t, [...s].sort()])) };
}
const pillLine = (list) => {
  let line = "", k = 0;
  for (const t of list) {
    const next = line ? `${line} · ${t}` : t;
    if (next.length > PILL_LINE && line) break;
    line = next; k++;
  }
  return k < list.length ? `${line} +${list.length - k}` : line;
};

// The layout a map is drawn from, for the database and its relations of now
const layoutKey = () => `${ui.db}|${(ui.rel.references || []).join(",")}|${ui.rel.relations.map((r) => `${r.source}>${r.target}`).join(",")}`;

// {db, key, nodes: {table: {x, y, w, h, line}}, routes: {joinKey: points}, width, height, pilled}: the tables ELK lays
// out, then a row of the tables joined to none (or to references only), a row of the references in pills
async function computeLayout(key) {
  const laid = ui.rel.layout.nodes, { pilled, pills } = mapPills();
  const size = (t) => {
    const line = pills[t] ? pillLine(pills[t]) : "";
    return { w: Math.max(laid[t].w, line ? Math.ceil(PILL_CHAR * line.length + 20) : 0), h: laid[t].h + (line ? 12 : 0), line, refs: pills[t] || [] };
  };
  const core = ui.rel.relations.filter((r) => laid[r.source] && laid[r.target] && r.source !== r.target && !pilled.has(r.source) && !pilled.has(r.target));
  const joined = [...new Set(core.flatMap((r) => [r.source, r.target]))];
  const elk = await loadElk();
  const out = await elk.layout({ id: "map", layoutOptions: ELK_OPTIONS,
    children: joined.map((t) => ({ id: t, width: size(t).w, height: size(t).h })),
    edges: core.map((r) => ({ id: joinKey(r), sources: [r.source], targets: [r.target] })) });
  const nodes = Object.fromEntries(out.children.map((c) => [c.id, { ...size(c.id), x: c.x, y: c.y, w: c.width, h: c.height }]));
  const routes = Object.fromEntries(out.edges.filter((e) => e.sections?.length).map((e) => {
    const s = e.sections[0];
    return [e.id, [s.startPoint, ...(s.bendPoints || []), s.endPoint].map((p) => [p.x, p.y])];
  }));
  const width = Math.max(600, out.width);
  let y = joined.length ? out.height : 40;
  const row = (tables) => {
    let x = 40, h = 0;
    tables.forEach((t) => {
      const s = size(t);
      if (x + s.w > width - 40 && x > 40) { x = 40; y += h + 14; h = 0; }
      nodes[t] = { ...s, x, y };
      x += s.w + 14; h = Math.max(h, s.h);
    });
    if (tables.length) y += h + 40;
  };
  const rest = Object.keys(laid).filter((t) => !nodes[t]);
  row(rest.filter((t) => !pilled.has(t)));
  row(rest.filter((t) => pilled.has(t)));
  return { db: ui.db, key, nodes, routes, width, height: y, pilled };
}

// ELK not loaded: the layout of the server, every table a box, every line drawn from side to side
const serverLayout = (key) => ({ db: ui.db, key, nodes: ui.rel.layout.nodes, routes: {}, width: ui.rel.layout.width,
  height: ui.rel.layout.height, pilled: new Set() });

// The layout to draw: the one of these relations, or while it is computed the last one of this database (its tables
// then glide to their new places); none yet: the map waits for it, drawn again once it is there
function mapLayout() {
  const key = layoutKey();
  if (MAP.layout?.key !== key && MAP.computing !== key) {
    MAP.computing = key;
    computeLayout(key).catch(() => serverLayout(key)).then((layout) => {
      if (MAP.computing !== key) return;
      MAP.computing = null;
      MAP.layout = layout;
      if (ui.step === "relations" && ui.rel && layoutKey() === key) renderDetail();
    });
  }
  return MAP.layout?.db === ui.db ? MAP.layout : null;
}

// The map as it shows: the boxes of the layout, and the table chosen with its neighbours in the middle of the view (its
// tables on its left, the ones it joins on its right, the others gone). The whole map and a path: the lines ELK drew,
// a line to a reference in pills only on the path. A table chosen: each of its relations from the side of one table to
// the side of the other, the relations of a side spread along it by the height of their other end
function mapGeometry(L) {
  const laid = L.nodes, layer = pathLayers(), cur = layer.size ? null : ui.current;
  const nodes = Object.fromEntries(Object.entries(laid).map(([t, n]) => [t, { ...n, dx: n.x, dy: n.y, gone: false }]));
  if (cur && nodes[cur]) {
    const places = centredPlaces(nodes, cur);
    for (const [t, n] of Object.entries(nodes)) {
      const p = places.get(t);
      Object.assign(n, p ? { dx: p.x, dy: p.y } : { gone: true });
    }
  }
  const shown = Object.fromEntries(Object.entries(nodes).filter(([, n]) => !n.gone).map(([t, n]) => [t, { ...n, x: n.dx, y: n.dy }]));
  const list = Object.values(shown), x = Math.min(...list.map((n) => n.x)) - 40, y = Math.min(...list.map((n) => n.y)) - 30;
  const box = { x, y, w: Math.max(...list.map((n) => n.x + n.w)) + 70 - x, h: Math.max(...list.map((n) => n.y + n.h)) + 30 - y };
  const rels = ui.rel.relations, offset = rels.map(() => ({ s: 0, t: 0 })), sides = new Map();
  const side = (s, t) => (t.x > s.x + s.w + 8 ? ["r", "l"] : t.x + t.w < s.x - 8 ? ["l", "r"] : ["r", "r"]);
  const drawn = (r) => shown[r.source] && shown[r.target] && (cur ? r.source === cur || r.target === cur
    : !L.pilled.has(r.source) && !L.pilled.has(r.target) || onPath(r, layer));
  const route = (r) => !cur && L.routes[joinKey(r)];
  rels.forEach((r, i) => {
    if (!drawn(r) || route(r) || r.source === r.target) return;
    const s = shown[r.source], t = shown[r.target], [ss, ts] = side(s, t);
    for (const [key, end, other] of [[`${r.source}|${ss}`, "s", t], [`${r.target}|${ts}`, "t", s]]) {
      if (!sides.has(key)) sides.set(key, []);
      sides.get(key).push({ i, end, y: other.y + other.h / 2 });
    }
  });
  for (const [key, ends] of sides) {
    const h = shown[key.split("|")[0]].h, step = ends.length > 1 ? Math.min(7, (h - 12) / (ends.length - 1)) : 0;
    ends.sort((a, b) => a.y - b.y).forEach((p, j) => { offset[p.i][p.end] = (j - (ends.length - 1) / 2) * step; });
  }
  const edges = rels.map((r, i) => (!drawn(r) ? null : route(r) ? { r, i, ...routePath(route(r)) }
    : { r, i, ...edgePath(shown[r.source], shown[r.target], r.source === r.target, offset[i].s, offset[i].t) }));
  return { nodes, shown, edges, box, cur, layer, width: L.width, height: L.height };
}

// The table chosen in the middle of the view: it stays where it is, the tables that join it in columns on its left, the
// tables it joins on its right; many of them (a hub) in a few balanced columns
function centredPlaces(nodes, cur) {
  const c = nodes[cur], into = [], from = [];
  ui.rel.relations.forEach((r) => {
    if (r.target === cur && r.source !== cur && nodes[r.source] && !into.includes(r.source)) into.push(r.source);
    if (r.source === cur && r.target !== cur && nodes[r.target] && !from.includes(r.target)) from.push(r.target);
  });
  const places = new Map([[cur, { x: c.x, y: c.y }]]), row = c.h + 14;
  const place = (tables, dir) => tables.sort().forEach((t, i) => {
    const per = Math.ceil(tables.length / Math.max(1, Math.round(Math.sqrt(tables.length * row / 230))));
    const col = Math.floor(i / per), inCol = Math.min(per, tables.length - col * per), n = nodes[t];
    places.set(t, { x: dir < 0 ? c.x - 130 - col * 230 - n.w : c.x + c.w + 130 + col * 230,
      y: c.y + c.h / 2 - (inCol * row - 14) / 2 + (i % per) * row });
  });
  place(into, -1);
  place(from.filter((t) => !into.includes(t)), 1);
  return places;
}

// The line of a relation at right angles, as ELK draws the whole map, from the side of a table to the side of the
// other: left to right between two columns, turning half way; right to left when the table pointed to is on the left;
// out on the right and back within one column; a loop for a table pointing to itself. ds, dt: the place of each end
// along its side; dirs: where the line goes out of each end (1 right, -1 left)
function edgePath(s, t, self, ds = 0, dt = 0) {
  const line = (P, dirs) => ({ d: roundedPath(P), start: P[0], end: P.at(-1), dirs });
  if (self) {
    const x = s.x + s.w, y = s.y + s.h / 2;
    return line([[x, y + 5], [x + 18, y + 5], [x + 18, y - 5], [x, y - 5]], [1, 1]);
  }
  const sy = s.y + s.h / 2 + ds, ty = t.y + t.h / 2 + dt;
  const turn = (x1, x2, dirs) => line([[x1, sy], [(x1 + x2) / 2, sy], [(x1 + x2) / 2, ty], [x2, ty]], dirs);
  if (t.x > s.x + s.w + 8) return turn(s.x + s.w, t.x, [1, -1]);
  if (t.x + t.w < s.x - 8) return turn(s.x, t.x + t.w, [-1, 1]);
  const x1 = s.x + s.w, x2 = t.x + t.w, out = Math.max(x1, x2) + 30;
  return line([[x1, sy], [out, sy], [out, ty], [x2, ty]], [1, 1]);
}

// A line ELK routed: its points, the way it goes out of each end (none when it leaves a table up or down)
function routePath(P) {
  const way = (a, b) => (Math.abs(b[0] - a[0]) >= Math.abs(b[1] - a[1]) ? Math.sign(b[0] - a[0]) || 1 : 0);
  return { d: roundedPath(P), start: P[0], end: P.at(-1), dirs: [way(P[0], P[1]), way(P.at(-1), P.at(-2))] };
}

// A line through points at right angles, its corners rounded
function roundedPath(P, r = 6) {
  let d = `M${P[0]}`;
  for (let i = 1; i < P.length - 1; i++) {
    const [p, c, n] = [P[i - 1], P[i], P[i + 1]], d1 = Math.hypot(c[0] - p[0], c[1] - p[1]), d2 = Math.hypot(n[0] - c[0], n[1] - c[1]);
    if (!d1 || !d2) continue;   // a corner of no length: the line goes straight on
    const k = Math.min(r, d1 / 2, d2 / 2);
    d += `L${c[0] + (p[0] - c[0]) * k / d1} ${c[1] + (p[1] - c[1]) * k / d1}Q${c} ${c[0] + (n[0] - c[0]) * k / d2} ${c[1] + (n[1] - c[1]) * k / d2}`;
  }
  return `${d}L${P.at(-1)}`;
}

// What an end of a relation holds, as in the diagrams of databases: many rows a crow's foot (its toes on the table),
// one row a bar; unknown ("?") nothing, nor at an end leaving its table up or down
function endMark(kind, [x, y], dir) {
  if (!dir) return "";
  if (kind === "n") { const q = x + dir * 9; return `M${x} ${y - 5}L${q} ${y}M${x} ${y + 5}L${q} ${y}`; }
  if (kind === "1") { const b = x + dir * 7; return `M${b} ${y - 5}L${b} ${y + 5}`; }
  return "";
}

// The SVG of the map: a box per table (its name, its rows and a gauge of them), a curve per relation whose line says
// what the data showed (solid measured, pale not measured yet, amber dashes weak) and whose ends say how many rows each
// side holds; the table chosen and its relations in orange, a slow flow along them; the path in blue. Each box is drawn
// at its place in the layout and moved to where it shows (the table chosen in the middle): it glides there
function mapSvg(g) {
  const { cur, layer } = g, path = layer.size > 0, rows = ui.data?.db === ui.db ? new Map(ui.data.tables.map((t) => [t.name, t])) : null;
  MAP.rowless = !rows;
  const near = new Set(ui.rel.relations.filter((r) => r.source === cur || r.target === cur).flatMap((r) => [r.source, r.target]));
  const max = Math.max(1, ...[...(rows?.values() || [])].map((t) => t.rows || 0));
  const edges = g.edges.filter(Boolean).map(({ r, i, d, start, end, dirs }) => {
    const mine = Boolean(cur) && (r.source === cur || r.target === cur), inPath = onPath(r, layer);
    const cls = ["e", r.weak ? "weak" : r.found == null && !r.empty ? "unchecked" : "", mine ? "on" : "",
      (cur && !mine) || (path && !inPath) ? "dim" : "", inPath ? "path" : ""].join(" ");
    const [from, to] = r.links.split("→").map((s) => s.trim());
    return `<g class="${cls}" data-rel="${i}" data-s="${esc(r.source)}" data-t="${esc(r.target)}">
      <path class="line" d="${d}"/><path class="mark" d="${endMark(from, start, dirs[0])}${endMark(to, end, dirs[1])}"/>
      <path class="flow" d="${d}"/><path class="hit" d="${d}"/></g>`;
  }).join("");
  const isolated = new Set(ui.rel.isolated);
  const boxes = Object.entries(g.nodes).map(([table, n]) => {
    const cls = ["node", table === cur ? "on" : "", near.has(table) && table !== cur ? "near" : "", isolated.has(table) ? "iso" : "",
      layer.has(table) ? "path" : "", path && !layer.has(table) ? "dim" : "", n.gone ? "gone" : ""].join(" ");
    const t = rows?.get(table), count = t?.rows, share = count ? Math.log10(count + 1) / Math.log10(max + 1) : 0;
    const said = !t ? "" : count == null ? "lignes non comptées" : count ? `${compactRows.format(count)} ligne${count > 1 ? "s" : ""}` : "vide";
    return `<g class="${cls}" data-table="${esc(table)}" data-refs="${esc((n.refs || []).join(" "))}" tabindex="${n.gone ? -1 : 0}" role="button" aria-label="${esc(table)}"
        style="transform:translate(${n.dx - n.x}px,${n.dy - n.y}px)">
      ${t ? `<title>${esc(table)} · ${typeName(t.type).toLowerCase()}${n.refs?.length ? ` · joint ${n.refs.join(", ")}` : ""}</title>` : ""}
      <rect class="box" x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="7"/>
      <text class="nm" x="${n.x + 10}" y="${n.y + 15}">${esc(table)}</text><text class="rows" x="${n.x + 10}" y="${n.y + 27}">${said}</text>
      ${n.line ? `<text class="refs" x="${n.x + 10}" y="${n.y + 40}">${esc(n.line)}</text>` : ""}
      <rect class="gauge-bg" x="${n.x + 8}" y="${n.y + n.h - 5}" width="${n.w - 16}" height="2" rx="1"/>
      <rect class="gauge-fill" x="${n.x + 8}" y="${n.y + n.h - 5}" width="${count ? Math.max(share * (n.w - 16), 2) : 0}" height="2" rx="1"/></g>`;
  }).join("");
  return `<svg class="map" id="map" role="img" aria-label="Carte des tables et de leurs jointures">${edges}${boxes}
    <circle class="packet" id="map-packet" r="4" hidden/></svg>`;
}

// What a relation says by its line: its columns, what the data showed, its origin
function relationTip(r) {
  const measured = r.fanout ? `chaque ligne de ${esc(r.source)} comptée ${decimal(r.fanout)} fois`
    : r.empty ? `${esc(r.source)} est vide : rien à mesurer`
    : r.found == null ? "pas encore mesurée" : `${pct(r.found)} des valeurs trouvées${r.weak ? ", peu" : ""}`;
  const origin = ORIGINS[r.origin] ? ` · ${ORIGINS[r.origin][1]}` : "";
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

// The view of the whole map: whole when it fits; a big map at the width of the panel (its names readable, 50 % at
// least; 75 % large), from its top, the rest a drag away
function wholeView(box, cw, ch, least = 0.5) {
  const fit = frameBox(box, cw, ch);
  if (fit.scale >= least) return fit;
  const scale = Math.max(least, Math.min((cw - ZOOM_SPACE) / box.w, 1.25));
  return { scale, x: box.x + box.w / 2 - (cw - ZOOM_SPACE) / scale / 2, y: box.y };
}

// The view of the table chosen with its neighbours around it: whole while its names stay readable (60 %), else the
// table and its nearest neighbours at 90 %, the mini-map showing the rest
function centredView(g, cw, ch) {
  const fit = frameBox(g.box, cw, ch);
  if (fit.scale >= 0.6 || !g.cur) return fit;
  const n = g.shown[g.cur], scale = 0.9;
  return { scale, x: n.x + n.w / 2 - (cw - ZOOM_SPACE) / scale / 2, y: n.y + n.h / 2 - ch / scale / 2 };
}

// The tables of the path framed
function pathBox(g) {
  const boxes = [...g.layer.keys()].map((t) => g.shown[t]).filter(Boolean);
  const x = Math.min(...boxes.map((n) => n.x)) - 40, y = Math.min(...boxes.map((n) => n.y)) - 30;
  return { x, y, w: Math.max(...boxes.map((n) => n.x + n.w)) + 70 - x, h: Math.max(...boxes.map((n) => n.y + n.h)) + 30 - y };
}

// Where the map goes for what is shown: the table chosen, a path, or the whole map
function viewFor(g, cw, ch) {
  if (g.cur) return centredView(g, cw, ch);
  if (g.layer.size) return frameBox(pathBox(g), cw, ch);
  return ui.mapBig ? wholeView(g.box, cw, ch, 0.75) : wholeView(g.box, cw, ch);
}

function applyView() {
  const svg = $("map"), v = ui.mapView;
  if (!svg || !v) return;
  svg.setAttribute("viewBox", `${v.x} ${v.y} ${svg.clientWidth / v.scale} ${svg.clientHeight / v.scale}`);
  drawMiniMap();
}

// The view goes somewhere: it glides there in a short ease, or jumps without movement
function glideView(target) {
  cancelAnimationFrame(MAP.camera);
  const from = ui.mapView;
  if (stillMotion() || !from) { ui.mapView = { ...from, ...target }; applyView(); return; }
  const t0 = performance.now();
  const step = (now) => {
    const t = Math.min(1, Math.max(0, now - t0) / 420), e = 1 - (1 - t) ** 3;   // a frame's time can precede t0
    ui.mapView = { ...from, x: from.x + (target.x - from.x) * e, y: from.y + (target.y - from.y) * e, scale: from.scale + (target.scale - from.scale) * e };
    applyView();
    if (t < 1) MAP.camera = requestAnimationFrame(step);
  };
  MAP.camera = requestAnimationFrame(step);
}

// Zoom by a factor around a point of the svg (its centre by default)
function zoomMap(factor, px = null, py = null, glide = false) {
  const svg = $("map"), v = ui.mapView, w = svg.clientWidth, h = svg.clientHeight;
  const ox = px ?? w / 2, oy = py ?? h / 2, scale = Math.min(2.5, Math.max(0.2, v.scale * factor));
  const target = { scale, x: v.x + ox / v.scale - ox / scale, y: v.y + oy / v.scale - oy / scale };
  if (glide) glideView(target); else { cancelAnimationFrame(MAP.camera); ui.mapView = { ...v, ...target }; applyView(); }
}

// The map opened large over the page, or back in its panel: the height it leaves there kept (the page does not move
// under it), its view framed again for its new size
function bigMap(open) {
  if (open === Boolean(ui.mapBig)) return;
  ui.mapBig = open ? { height: $("map-frame").offsetHeight, opening: true } : null;
  ui.mapView = null;
  renderDetail();
  $("map-expand")?.focus({ preventScroll: true });
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && ui.mapBig && !e.defaultPrevented) bigMap(false); });

// The whole map small at the bottom right when it passes its frame (a big base, a table chosen with many neighbours):
// the part shown framed in the accent, a click or a drag takes the view there
function drawMiniMap() {
  const mini = $("map-mini"), svg = $("map"), g = MAP.geo, v = ui.mapView;
  if (!mini || !g || !v) return;
  const all = g.box, w = svg.clientWidth / v.scale, h = svg.clientHeight / v.scale;
  const fits = all.x >= v.x - 1 && all.y >= v.y - 1 && all.x + all.w <= v.x + w + 1 && all.y + all.h <= v.y + h + 1;
  mini.hidden = fits;
  if (fits) return;
  const k = Math.min(140 / all.w, 150 / all.h), path = g.layer;
  const lines = g.edges.filter(Boolean).map((e) => `<path class="mini-edge" d="${e.d}"/>`).join("");
  const rects = Object.entries(g.shown).map(([t, n]) => `<rect class="mini-node${path.has(t) ? " path" : t === g.cur ? " on" : ""}"`
    + ` x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="6"/>`).join("");
  const vx = Math.max(all.x, v.x), vy = Math.max(all.y, v.y);
  mini.innerHTML = `<svg width="${all.w * k}" height="${all.h * k}" viewBox="${all.x} ${all.y} ${all.w} ${all.h}" aria-hidden="true">${lines}${rects}
    <rect class="mini-view" x="${vx}" y="${vy}" width="${Math.max(0, Math.min(v.x + w, all.x + all.w) - vx)}" height="${Math.max(0, Math.min(v.y + h, all.y + all.h) - vy)}" rx="8"/></svg>`;
  mini.dataset.k = k;
}

// The path between two tables travelled by a point, join after join, again and again: the way the agent goes
function runPacket(g) {
  cancelAnimationFrame(MAP.packet);
  const dot = $("map-packet"), layers = ui.path.result?.layers || [];
  if (!dot || layers.length < 2 || stillMotion()) return;
  const legs = [];
  let at = layers[0][0];
  for (const next of layers.slice(1)) {
    const k = ui.rel.relations.findIndex((r) => (r.source === at && next.includes(r.target)) || (r.target === at && next.includes(r.source)));
    const line = k >= 0 && $("map").querySelector(`.e[data-rel="${k}"] .line`);
    if (!line) return;
    const r = ui.rel.relations[k];
    legs.push({ line, len: line.getTotalLength(), back: r.source !== at });   // against the line: from its end
    at = r.source === at ? r.target : r.source;
  }
  const leg = 650, total = legs.length * leg + 900, t0 = performance.now();
  const step = (now) => {
    const t = Math.max(0, now - t0) % total, k = Math.floor(t / leg);   // a frame's time can precede t0
    if (k < legs.length) {
      const f = (t - k * leg) / leg, e = f < 0.5 ? 2 * f * f : 1 - (-2 * f + 2) ** 2 / 2, { line, len, back } = legs[k];
      const p = line.getPointAtLength(len * (back ? 1 - e : e));
      dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y); dot.hidden = false;
    } else dot.hidden = true;
    MAP.packet = requestAnimationFrame(step);
  };
  MAP.packet = requestAnimationFrame(step);
}

// A table under the pointer, on the map or in the list: its relations and the tables they join stand out, the rest
// fades; its relations flow. A reference in pills: the tables carrying its pill
function lightTable(table) {
  const svg = $("map");
  if (!svg) return;
  svg.classList.toggle("hovering", Boolean(table));
  svg.querySelectorAll(".hl").forEach((el) => el.classList.remove("hl"));
  if (!table) return;
  svg.querySelectorAll(".e").forEach((e) => {
    if (e.dataset.s !== table && e.dataset.t !== table) return;
    e.classList.add("hl");
    [e.dataset.s, e.dataset.t].forEach((t) => svg.querySelector(`.node[data-table="${CSS.escape(t)}"]`)?.classList.add("hl"));
  });
  svg.querySelectorAll(".node").forEach((n) => { if (n.dataset.table === table || n.dataset.refs.split(" ").includes(table)) n.classList.add("hl"); });
}
$("rows").addEventListener("mouseover", (e) => { if (ui.step === "relations") lightTable(e.target.closest(".row[data-table]")?.dataset.table ?? null); });
$("rows").addEventListener("mouseleave", () => { if (ui.step === "relations") lightTable(null); });

// The map once drawn: its height (the whole map at the width of the panel, at most 600 px), the tables that moved
// gliding from where they were, the view going to what is shown (another table, a path, the whole map), then a drag
// that moves it, the zoom, the hover of a table and the label of a relation, the mini-map, the path travelled
function setupMap(g) {
  const wrap = $("map-wrap"), svg = $("map"), { width, height } = g;
  const cw = wrap.clientWidth || 800;
  svg.style.height = ui.mapBig ? "" : `${Math.max(260, Math.min(600, Math.round(height * Math.min(cw / width, 1.25))))}px`;
  const ch = svg.clientHeight, key = g.cur || (g.layer.size ? `${ui.path.a}>${ui.path.b}` : null);
  MAP.geo = g;

  // the tables that moved since the map was drawn: from where they were to where they go, their relations after them
  const before = MAP.db === ui.db ? MAP.shift : null;
  MAP.db = ui.db;
  MAP.shift = new Map(Object.entries(g.nodes).map(([t, n]) => [t, { x: n.dx - n.x, y: n.dy - n.y, gone: n.gone }]));
  const moved = before && !stillMotion() && [...MAP.shift].some(([t, b]) => { const a = before.get(t); return a && (a.x !== b.x || a.y !== b.y || a.gone !== b.gone); });
  if (moved) {
    const nodes = [...svg.querySelectorAll(".node")];
    nodes.forEach((el) => {
      const a = before.get(el.dataset.table);
      if (!a) return;
      el.style.transition = "none";
      el.style.transform = `translate(${a.x}px,${a.y}px)`;
      el.classList.toggle("gone", a.gone);
    });
    svg.getBoundingClientRect();
    nodes.forEach((el) => {
      const b = MAP.shift.get(el.dataset.table);
      el.style.transition = "";
      el.style.transform = `translate(${b.x}px,${b.y}px)`;
      el.classList.toggle("gone", b.gone);
    });
    svg.querySelectorAll(".e").forEach((e) => e.classList.add("enter"));
  }

  // the view: another database or the map opened large, framed at once; another table, a path or none, it glides there
  if (ui.mapView?.db !== ui.db) ui.mapView = { db: ui.db, focus: key, ...viewFor(g, cw, ch) };
  else if (ui.mapView.focus !== key) { ui.mapView.focus = key; applyView(); glideView(viewFor(g, cw, ch)); }
  applyView();

  $("map-expand").onclick = () => bigMap(!ui.mapBig);
  if (ui.mapBig) { $("map-backdrop").onclick = () => bigMap(false); $("map-close").onclick = () => bigMap(false); }
  $("zoom-in").onclick = () => zoomMap(1.25, null, null, true);
  $("zoom-out").onclick = () => zoomMap(0.8, null, null, true);
  $("zoom-fit").onclick = () => glideView(viewFor(g, svg.clientWidth, svg.clientHeight));
  svg.addEventListener("wheel", (e) => {   // a pinch of the trackpad, or Ctrl and the wheel; the wheel alone scrolls the page
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const box = svg.getBoundingClientRect();
    zoomMap(e.deltaY < 0 ? 1.1 : 0.9, e.clientX - box.left, e.clientY - box.top);
  }, { passive: false });

  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".node, .hit")) return;
    cancelAnimationFrame(MAP.camera);
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

  // A table hovered: its relations and the tables they join stand out; clicked: chosen, clicked again: the whole map
  svg.querySelectorAll(".node").forEach((n) => {
    const table = n.dataset.table;
    n.addEventListener("click", () => openTable(table));
    n.addEventListener("keydown", (e) => { if (e.key === "Enter") openTable(table); });
    n.addEventListener("mouseenter", () => lightTable(table));
    n.addEventListener("mouseleave", () => lightTable(null));
  });

  // A relation hovered: what it says on its line, at its middle; clicked: its table chosen, its line found in the list
  const tip = $("map-tip");
  svg.querySelectorAll(".e").forEach((e) => {
    const r = ui.rel.relations[e.dataset.rel], line = e.querySelector(".line");
    e.addEventListener("mouseenter", () => {
      const p = line.getPointAtLength(line.getTotalLength() / 2), m = line.getScreenCTM(), box = wrap.getBoundingClientRect();
      tip.innerHTML = relationTip(r);
      tip.hidden = false;
      const x = p.x * m.a + m.e - box.left, y = p.y * m.d + m.f - box.top;
      tip.style.left = `${Math.max(tip.offsetWidth / 2 + 8, Math.min(x, box.width - tip.offsetWidth / 2 - 8))}px`;
      tip.style.top = `${Math.max(tip.offsetHeight + 16, y)}px`;
    });
    e.addEventListener("mouseleave", () => { tip.hidden = true; });
    e.addEventListener("click", () => {
      ui.relFocus = `${r.from}|${r.to}`;
      ui.current = r.source;
      ui.path = { a: r.source, b: null, result: null };
      renderRows(); renderDetail();
    });
  });

  // the mini-map: a click or a drag puts the middle of the view there
  const mini = $("map-mini");
  let moving = false;
  const goTo = (e, glide) => {
    const r = mini.querySelector("svg").getBoundingClientRect(), k = Number(mini.dataset.k), v = ui.mapView;
    const x = g.box.x + (e.clientX - r.left) / k, y = g.box.y + (e.clientY - r.top) / k;
    const target = { scale: v.scale, x: x - svg.clientWidth / v.scale / 2, y: y - svg.clientHeight / v.scale / 2 };
    if (glide) glideView(target); else { ui.mapView = { ...v, ...target }; applyView(); }
  };
  mini.addEventListener("pointerdown", (e) => { moving = true; mini.setPointerCapture(e.pointerId); goTo(e, true); });
  mini.addEventListener("pointermove", (e) => { if (moving) goTo(e, false); });
  mini.addEventListener("pointerup", () => { moving = false; });
  runPacket(g);
}

function pathText() {
  const p = ui.path;
  if (!p.a || !p.b) return "";   // the lists say what is missing ("départ…", "arrivée…")
  if (!p.result) return '<span class="muted-text">…</span>';
  if (p.result.joins == null) return '<span class="slow">aucun chemin : ces deux tables ne sont reliées par aucune suite de jointures</span>';
  // The tables of the path are on the map: the path itself in a bubble, on hover, focus or tap
  const steps = p.result.layers.map((tables) => tables.map((t) => `<span class="path-table">${esc(t)}</span>`).join(" | ")).join(" → ");
  return `<span class="muted-text">· <span class="path-joins" tabindex="0">${plural(p.result.joins, "jointure")}`
    + `<span class="path-tip" role="tooltip">${steps}</span></span></span>`
    + (p.result.bridged ? "" : ` <span class="slow">· au-delà de 3 : l'agent ne fera pas ce chemin seul</span>`);
}

// The list of joins under the map, in two parts. "À trancher", in amber: the weak joins (guessed, few of their values
// found: maybe no join at all), worst first, answered by a click or a key (G Garder, R Retirer; the line slides out, the
// next one takes the focus, a toast offers to undo, a progress once started, a closing line when none is left). Then
// every join under the table it leads to, one line per joined table, folded; opened, its tables in columns, top to
// bottom, a partial coverage with its share (the join holds, the agent loses the other rows). A join opens its detail
// under its line (or its group): the share of its values found and what it costs the agent. A join hovered or focused
// lights its edge on the map
const JOIN_CHEVRON = '<svg class="chev" viewBox="0 0 12 12" aria-hidden="true"><path d="M4.5 2.5L8 6L4.5 9.5"/></svg>';
const JOIN_DONE = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7"/><path d="M5 8.2l2 2 4-4.4"/></svg>';
const joinKey = (r) => `${r.from}|${r.to}`;
const joinCol = (ref) => ref.split(".")[1];
const neverMeasured = (r) => r.found == null && !r.empty;
const stillMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const nbPct = (x) => pct(x).replace(" %", " %");   // "18 %" never cut at the end of a line

// The detail of a join: what the data showed and what it costs the agent, where the join comes from
function joinDetail(r, where) {
  const [st, sc] = r.from.split(".");
  const s = esc(r.source), t = esc(r.target), col = `<b>${esc(st)}.${esc(sc)}</b>`, f = r.found;
  let lead, reading = "";
  if (r.empty) lead = `${s} n'a aucune ligne : rien à comparer.`;
  else if (f == null) lead = `Jamais mesurée : rien ne prouve encore que les valeurs de ${col} se retrouvent dans ${t}.`;
  else if (f >= 1) lead = `Toutes les valeurs de ${col} se retrouvent dans ${t}.`;
  else {
    lead = `<span class="${r.weak ? "warn" : ""}">${nbPct(f)}</span> des valeurs de ${col} se retrouvent dans ${t}. `
      + `Les ${nbPct(1 - f)} autres n'y sont pas : l'agent perd ces lignes de ${s} quand il joint les deux tables.`;
    if (r.weak) reading = `Soit la jointure est fausse, soit ${t} ne couvre qu'une partie de ${s}.`;
  }
  const [a, b] = (r.links || "").split(" → ");
  const side = (x) => ({ 1: "une ligne", n: "plusieurs" })[x];
  const rows = r.fanout ? `${t} n'est pas une clé : chaque ligne de ${s} comptée ${decimal(r.fanout)} fois en moyenne.`
    : side(a) && side(b) ? `Pour une même valeur : ${side(a)} dans ${s}, ${side(b)} dans ${t}.` : "";
  const [kind, says] = ORIGINS[r.origin] || [];
  const from = [kind ? `<span>${originMark(kind, says)}${says[0].toUpperCase()}${says.slice(1)}</span>` : "",
    r.kept ? `<span>${originMark("you", "gardée par toi")}Gardée par toi malgré ses valeurs</span>` : ""].join("");
  const k = esc(joinKey(r));
  const acts = where === "queue" ? `<button class="btn ghost small" data-join-act="map" data-key="${k}">Voir sur la carte</button>`
    : (r.weak ? `<button class="btn ghost small" data-join-act="keep" data-key="${k}">Garder</button>` : "")
      + (r.kept ? `<button class="btn ghost small" data-join-act="unkeep" data-key="${k}">Ne plus garder</button>` : "")
      + `<button class="btn ghost small" data-join-act="map" data-key="${k}">Voir sur la carte</button>`
      + `<button class="btn ghost small" data-join-act="remove" data-key="${k}">Retirer</button>`;
  return `<div class="join-detail${r.weak ? " weak" : ""}">
      ${where === "group" && f != null && !r.empty ? `<div class="join-bar" role="img" aria-label="${nbPct(f)} des valeurs retrouvées"><i style="width:${f * 100}%"></i></div>` : ""}
      <p class="lead">${lead}</p>${rows ? `<p>${rows}</p>` : ""}${reading ? `<p class="reading">${reading}</p>` : ""}
      <div class="join-detail-foot">${from}<span class="grow"></span>${acts}</div>
    </div>`;
}

// À trancher: the weak joins, worst first; the progress once an answer is given; a closing line when none is left
function joinQueue(lit, sel) {
  const rel = ui.rel, state = ui.joins;
  const queue = rel.relations.filter((r) => r.weak).sort((a, b) => (a.found ?? 0) - (b.found ?? 0));
  const done = state.decided.length, total = queue.length + done;
  if (!total) return "";
  if (!queue.length) {
    const kept = state.decided.filter((d) => d.verdict === "keep").length;
    return `<div class="join-done">${JOIN_DONE}<span>${total > 1 ? `Les ${total} jointures douteuses sont tranchées.` : "La jointure douteuse est tranchée."}</span>
      <span class="muted-text">${plural(kept, "gardée")}, ${plural(done - kept, "retirée")}</span></div>`;
  }
  const items = queue.map((r) => {
    const k = joinKey(r), open = state.detail?.key === k && state.detail.where === "queue", same = joinCol(r.from) === joinCol(r.to);
    const [kind, , title] = ORIGINS[r.origin] || [];
    return `<div class="join-item${open ? " open" : ""}${lit(r) ? " lit" : ""}" data-key="${esc(k)}"><div class="join-item-in">
        <div class="join-line" tabindex="0" role="button" aria-expanded="${open}" data-rel="${rel.relations.indexOf(r)}">
          ${kind ? originMark(kind, title) : "<span></span>"}
          <span class="names">${esc(r.source)}<span class="arrow">→</span>${esc(r.target)}</span>
          <span class="share">${r.empty ? "vide" : `<span class="join-meter"><i style="width:${(r.found ?? 0) * 100}%"></i></span>${nbPct(r.found)}`}</span>
          <span class="key">${same ? esc(joinCol(r.from)) : `${esc(joinCol(r.from))} → ${esc(joinCol(r.to))}`}</span>
          <span class="answers">
            <button class="btn ghost small" data-join-act="keep" data-key="${esc(k)}" title="Elle reste, sans l'orange : tu la sais juste">Garder<kbd>G</kbd></button>
            <button class="btn ghost small" data-join-act="remove" data-key="${esc(k)}" title="Elle sort de la liste ; tu pourras la rétablir">Retirer<kbd>R</kbd></button>
          </span>
        </div>
        <div class="join-fold"${open ? "" : " inert"}><div class="join-fold-in">${joinDetail(r, "queue")}</div></div>
      </div></div>`;
  }).join("");
  return `<section class="join-queue-wrap${sel ? " sel" : ""}">
    <div class="joins-head"><div class="text"><h3>${queue.length > 1 ? `${queue.length} jointures à trancher` : "Une jointure à trancher"}</h3>
      <p>Devinées par l'IA ou entre des colonnes de noms différents, et peu de leurs valeurs se retrouvent dans la table jointe. Garde celles que tu sais justes, retire les autres.</p></div>
      ${done ? `<div class="join-progress"><span>${done} sur ${total}</span><span class="track"><i style="width:${state.shown * 100}%"></i></span></div>` : ""}</div>
    <div class="join-queue">${items}</div>
    <div class="join-hint" aria-hidden="true"><span><kbd>↑</kbd><kbd>↓</kbd>parcourir</span><span><kbd>Entrée</kbd>voir le détail</span>
      <span><kbd>G</kbd>garder</span><span><kbd>R</kbd>retirer</span></div>
  </section>`;
}

// Every join under the table it leads to, the most joined first; the tables a single join leads to share the last group
function joinGroups(lit, sel, cur, path) {
  const rel = ui.rel, state = ui.joins, by = new Map();
  for (const r of rel.relations) (by.get(r.target) || by.set(r.target, []).get(r.target)).push(r);
  const groups = [...by].filter(([, rs]) => rs.length > 1).sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
    .map(([id, rs]) => ({ id, rs }));
  const single = [...by].filter(([, rs]) => rs.length === 1).map(([, rs]) => rs[0]);
  if (single.length) groups.push({ id: "", rs: single });

  const group = ({ id, rs }) => {
    const cols = rs.map((r) => joinCol(r.to)), count = (c) => cols.filter((x) => x === c).length;
    const key = id ? [...new Set(cols)].sort((a, b) => count(b) - count(a))[0] : null;
    const on = rs.some(lit), open = state.open.has(id) || (on && Boolean(cur) && !state.open.has(`!${id}`));
    const weak = rs.filter((r) => r.weak).length, partial = rs.filter((r) => r.partial).length, never = rs.filter(neverMeasured).length;
    const said = [weak && `<span class="warn">${weak} à trancher</span>`, partial && `${partial} en couverture partielle`,
      never && (never > 1 ? `${never} jamais mesurées` : "1 jamais mesurée")].filter(Boolean).join(" · ");
    const status = `<span class="st">${said || (rs.length > 1 ? "toutes vérifiées" : "vérifiée")}</span>`;
    const title = id ? `<span class="name">${esc(id)}</span><span class="meta">${plural(rs.length, "table")} y mènent, par <span class="mono">${esc(key)}</span></span>`
      : `<span class="name">Les autres</span><span class="meta">${plural(rs.length, "jointure")}, une seule vers chacune de ces tables</span>`;
    const sorted = rs.slice().sort((a, b) => a.source.localeCompare(b.source) || a.target.localeCompare(b.target));
    const here = (r) => state.detail?.key === joinKey(r) && state.detail.where === "group";
    const names = sorted.map((r) => {
      const sc = joinCol(r.from), tc = joinCol(r.to);
      // the key is in the title of the group; a join on other columns says them
      const via = id ? (tc !== key ? `${sc} → ${tc}` : sc !== tc ? sc : "") : "";
      const label = id ? esc(r.source) : `${esc(r.source)}<span class="arrow">→</span>${esc(r.target)}`;
      const cls = ["join-name", r.weak ? "weak" : "", neverMeasured(r) ? "unchecked" : "", lit(r) ? "lit" : "",
        lit(r) && !path && (r.source === cur || (!id && r.target === cur)) ? "me" : "", here(r) ? "open" : ""].filter(Boolean).join(" ");
      const tip = r.weak ? `${pct(r.found)} des valeurs retrouvées` : r.partial ? `${r.target} couvre ${pct(r.found)} de ${r.source}`
        : neverMeasured(r) ? "jamais mesurée" : via ? `par ${via}` : "";   // a key cut at the end of its column
      // what the join costs, after its name: the share a partial coverage keeps, the rows a join on no key counts again
      const cost = r.partial ? nbPct(r.found) : r.fanout ? `×${decimal(r.fanout)}` : "";
      // its origin before it, a kept one yours; an unknown one keeps its place, empty, so that the names stay aligned
      const [kind, , title] = r.kept ? ["you", "", "Gardée par toi malgré ses valeurs"] : ORIGINS[r.origin] || [];
      return `<button class="${cls}" type="button" data-rel="${rel.relations.indexOf(r)}" data-key="${esc(joinKey(r))}" aria-expanded="${here(r)}"${tip ? ` title="${tip}"` : ""}>`
        + `${kind ? originMark(kind, title) : '<span class="slot"></span>'}<span class="label">${label}</span>${via ? `<span class="via">${esc(via)}</span>` : ""}`
        + `${cost ? `<span class="cost">${cost}</span>` : ""}</button>`;
    }).join("");
    const shown = sorted.find(here);
    return `<div class="join-group${open ? " open" : ""}${on ? " lit" : ""}" data-group="${esc(id)}">
        <button class="join-group-head${id && id === cur && !path ? " me" : ""}" type="button" aria-expanded="${open}">${JOIN_CHEVRON}${title}${status}</button>
        <div class="join-fold"${open ? "" : " inert"}><div class="join-fold-in">
          <div class="join-names">${names}</div>
          <div class="join-fold join-group-detail${shown ? " open" : ""}"${shown ? "" : " inert"}><div class="join-fold-in">${shown ? joinDetail(shown, "group") : ""}</div></div>
        </div></div>
      </div>`;
  };
  return `<div class="join-groups${sel ? " sel" : ""}${path ? " path" : ""}">
    <div class="joins-head"><div class="text"><h3>Toutes les jointures</h3><p>Rangées sous la table qu'elles rejoignent.</p></div></div>
    ${groups.map(group).join("") || '<p class="placeholder">Aucune jointure.</p>'}
  </div>`;
}

// A join hovered or focused: its edge and its two tables stand out on the map, the rest fades (as a table hovered there)
function lightJoin(i) {
  const svg = document.querySelector("#map-wrap .map");
  if (!svg) return;
  svg.classList.toggle("hovering", i != null);
  svg.querySelectorAll(".hl").forEach((el) => el.classList.remove("hl"));
  if (i == null) return;
  const r = ui.rel.relations[i];
  svg.querySelector(`.e[data-rel="${i}"]`)?.classList.add("hl");
  [r.source, r.target].forEach((t) => svg.querySelector(`.node[data-table="${CSS.escape(t)}"]`)?.classList.add("hl"));
}

// Opened and closed in place, without drawing the list again: the heights follow their content
function setJoinItem(item, open) {
  item.classList.toggle("open", open);
  item.querySelector(".join-line").setAttribute("aria-expanded", open);
  item.querySelector(".join-fold").inert = !open;
}

function closeJoinDetails(except) {
  document.querySelectorAll(".join-item.open").forEach((o) => { if (o !== except) setJoinItem(o, false); });
  document.querySelectorAll(".join-group-detail.open").forEach((d) => { if (d !== except) { d.classList.remove("open"); d.inert = true; } });
  document.querySelectorAll(".join-name.open").forEach((n) => { n.classList.remove("open"); n.setAttribute("aria-expanded", "false"); });
}

function toggleJoinItem(item) {
  const open = !item.classList.contains("open");
  closeJoinDetails(item);
  setJoinItem(item, open);
  ui.joins.detail = open ? { key: item.dataset.key, where: "queue" } : null;
}

function toggleJoinGroup(group) {
  const id = group.dataset.group, open = !group.classList.contains("open"), state = ui.joins.open;
  group.classList.toggle("open", open);
  group.querySelector(".join-group-head").setAttribute("aria-expanded", open);
  group.querySelector(":scope > .join-fold").inert = !open;
  // "!x": x closed by a click while the table chosen would open it
  if (open) { state.delete(`!${id}`); state.add(id); } else { state.delete(id); state.add(`!${id}`); }
}

function showJoinInGroup(name) {
  const group = name.closest(".join-group"), box = group.querySelector(".join-group-detail"), k = name.dataset.key;
  const again = ui.joins.detail?.key === k && ui.joins.detail.where === "group";
  const swap = box.classList.contains("open") && !again;
  closeJoinDetails(swap ? box : null);
  if (again) { ui.joins.detail = null; return; }
  const inner = box.querySelector(".join-fold-in");
  inner.innerHTML = joinDetail(ui.rel.relations.find((r) => joinKey(r) === k), "group");
  wireJoinActions(inner);
  if (swap && !stillMotion()) inner.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 180, easing: "ease-out" });
  box.classList.add("open"); box.inert = false;
  name.classList.add("open"); name.setAttribute("aria-expanded", "true");
  ui.joins.detail = { key: k, where: "group" };
  // the detail is under the group: brought into view once it has opened
  setTimeout(() => box.scrollIntoView({ block: "nearest", behavior: stillMotion() ? "auto" : "smooth" }), 240);
}

// An answer: the line slides out of the queue, the answer is written, a toast offers to undo it
async function decideJoin(pair, verdict) {
  const r = ui.rel.relations.find((x) => joinKey(x) === pair), state = ui.joins;
  if (!r) return;
  const item = document.querySelector(`.join-item[data-key="${CSS.escape(pair)}"]`);
  const items = [...document.querySelectorAll(".join-item")], at = items.indexOf(item);
  state.next = (items[at + 1] || items[at - 1])?.dataset.key || null;
  if (item && !stillMotion()) { item.classList.add("leaving"); await new Promise((done) => setTimeout(done, 260)); }
  // answered in the queue, its detail leaves with it; under its group, it stays open on what changed
  if (state.detail?.key === pair && (state.detail.where === "queue" || verdict === "remove")) state.detail = null;
  const before = state.decided;
  if (r.weak) state.decided = state.decided.filter((d) => d.pair !== pair).concat({ pair, verdict });
  if (!await moveRelation(pair, false, verdict === "keep" ? "keep" : "ignore")) { state.decided = before; return; }
  showUndo(verdict === "keep" ? "Gardée" : "Retirée", `${r.source} → ${r.target}`, async () => {
    state.decided = state.decided.filter((d) => d.pair !== pair);
    state.next = pair;
    await moveRelation(pair, true, verdict === "keep" ? "keep" : "ignore");
  });
}

function showUndo(word, what, undo) {
  const toast = $("undo-toast");
  toast.innerHTML = `<span>${word}</span><span class="what">${esc(what)}</span><button type="button">Annuler</button>`;
  toast.querySelector("button").onclick = () => { toast.classList.remove("on"); undo(); };
  toast.classList.add("on");
  clearTimeout(showUndo.timer);
  showUndo.timer = setTimeout(() => toast.classList.remove("on"), 6000);
}

function wireJoinActions(root) {
  root.querySelectorAll("[data-join-act]").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const pair = b.dataset.key, act = b.dataset.joinAct;
    if (act === "keep" || act === "remove") { decideJoin(pair, act); return; }
    if (act === "unkeep") { moveRelation(pair, true, "keep"); return; }
    const r = ui.rel.relations.find((x) => joinKey(x) === pair);
    ui.relFocus = pair; ui.current = r.source; ui.path = { a: r.source, b: null, result: null };
    renderRows(); renderDetail();
    $("map-wrap")?.scrollIntoView({ block: "center", behavior: stillMotion() ? "auto" : "smooth" });
  }));
}

function wireJoins(el) {
  el.querySelectorAll("[data-rel]").forEach((j) => {
    const i = Number(j.dataset.rel);
    j.addEventListener("mouseenter", () => lightJoin(i));
    j.addEventListener("mouseleave", () => { if (document.activeElement !== j) lightJoin(null); });
    j.addEventListener("focus", () => lightJoin(i));
    j.addEventListener("blur", () => lightJoin(null));
  });
  el.querySelectorAll(".join-line").forEach((line) => {
    const item = line.closest(".join-item");
    line.addEventListener("click", (e) => { if (!e.target.closest(".answers")) toggleJoinItem(item); });
    // the queue by keys: ↑ ↓ (or j k), Entrée, G, R, Échap
    line.addEventListener("keydown", (e) => {
      if (e.target !== line || e.ctrlKey || e.metaKey || e.altKey) return;
      const lines = [...el.querySelectorAll(".join-line")], at = lines.indexOf(line);
      const go = { ArrowDown: 1, j: 1, ArrowUp: -1, k: -1 }[e.key];
      if (go) { e.preventDefault(); lines[at + go]?.focus(); return; }
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleJoinItem(item); return; }
      if (e.key === "Escape" && item.classList.contains("open")) { toggleJoinItem(item); return; }
      const verdict = { g: "keep", r: "remove" }[e.key.toLowerCase()];
      if (verdict) { e.preventDefault(); decideJoin(item.dataset.key, verdict); }
    });
  });
  el.querySelectorAll(".join-group-head").forEach((h) => h.addEventListener("click", () => toggleJoinGroup(h.closest(".join-group"))));
  el.querySelectorAll(".join-name").forEach((n) => n.addEventListener("click", () => showJoinInGroup(n)));
  wireJoinActions(el);
}

function renderRelations(el, message) {
  const rel = ui.rel;
  if (!rel) { el.innerHTML = '<p class="placeholder">Lecture des jointures…</p>'; return; }
  const layout = mapLayout();
  if (!layout) { el.innerHTML = '<p class="placeholder">Disposition de la carte…</p>'; return; }
  const geo = mapGeometry(layout);
  const cur = ui.current, tables = ui.data?.tables.map((t) => t.name) || [], layer = pathLayers(), path = layer.size > 0;
  const mine = (r) => r.source === cur || r.target === cur;
  // a table chosen (or a path) does not filter the list: its joins stand out there, its groups open
  const lit = (r) => (path ? onPath(r, layer) : cur ? mine(r) : false), sel = path || Boolean(cur);
  const ignored = path ? [] : cur ? rel.ignored.filter(mine) : rel.ignored;
  const state = ui.joins;
  if (state.db !== ui.db) Object.assign(state, { db: ui.db, open: new Set(), detail: null, decided: [], shown: 0 });
  if (cur !== state.current) { state.open = new Set([...state.open].filter((id) => !id.startsWith("!"))); state.current = cur; }
  // a join clicked on the map: its detail open, in the queue or under its group
  const clicked = ui.relFocus && ui.relFocus !== state.focus;
  if (clicked) {
    const r = rel.relations.find((x) => joinKey(x) === ui.relFocus);
    if (r) state.detail = { key: ui.relFocus, where: r.weak ? "queue" : "group" };
    if (r && !r.weak) state.open.add(rel.relations.filter((x) => x.target === r.target).length > 1 ? r.target : "");
  }
  state.focus = ui.relFocus;
  const figures = [plural(rel.relations.length, "jointure"),
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
  const measure = '<button class="btn small" id="check-relations" data-job title="Mesure sur les données la part des valeurs trouvées de chaque jointure, sans l\'IA">';
  const line = (cls) => `<svg width="22" height="8" viewBox="0 0 22 8" aria-hidden="true"><path class="${cls}" d="M1 4 L21 4"/></svg>`;
  const end = (d) => `<svg width="22" height="12" viewBox="0 0 22 12" aria-hidden="true"><path class="end" d="${d}"/></svg>`;
  const fit = '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
  // large: the map, its bar and its note over the page, a title of their own; a block of their height left in the panel
  const big = ui.mapBig;
  const expand = big ? '<path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7"/>' : '<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>';
  document.body.classList.toggle("map-open", Boolean(big));
  // its appearance played once, when it opens: a table chosen in it draws the panel again, not the opening
  const opening = big?.opening ? " opening" : "";
  if (big) big.opening = false;
  el.innerHTML = `<div class="detail-head"><h2>Carte des tables</h2><span class="figures">${figures}</span>
      <span class="grow"></span><span class="save-state" id="rel-state">${message ? esc(message) : ""}</span>
      ${rel.unchecked ? "" : `${measure.replace('btn small"', 'btn ghost small"')}Revérifier</button>`}</div>
    ${big ? `<div class="map-spacer" style="height:${big.height}px"></div><div class="map-backdrop${opening}" id="map-backdrop"></div>` : ""}
    <div class="map-frame${big ? ` big${opening}` : ""}" id="map-frame"${big ? ' role="dialog" aria-modal="true" aria-label="Carte des tables"' : ""}>
    ${big ? `<div class="detail-head"><h2>Carte des tables</h2><span class="figures">${figures}</span><span class="grow"></span>
      <button class="icon-btn" type="button" id="map-close" aria-label="Fermer la carte" title="Fermer (Échap)">✕</button></div>` : ""}
    <div class="map-tools">
      ${cur ? `<span class="sel-chip">${esc(cur)}<button type="button" id="relations-all" aria-label="Voir toutes les tables">×</button></span><span class="tools-sep"></span>` : ""}
      <span class="muted-text">Chemin</span><select id="path-a" aria-label="Table de départ">${options(ui.path.a, "départ…")}</select>
      <span class="arrow">→</span><select id="path-b" aria-label="Table d'arrivée">${options(ui.path.b, "arrivée…")}</select>
      <span class="path-result">${pathText()}</span>
    </div>
    ${rel.unchecked ? `<div class="rule-note"><span>${rel.unchecked > 1 ? `${fmt.format(rel.unchecked)} jointures jamais mesurées` : "Une jointure jamais mesurée"} sur les données : `
      + `rien ne prouve encore que leurs valeurs se retrouvent d'une table à l'autre.</span>${measure}Vérifier sur les données</button></div>` : ""}
    <div class="map-wrap" id="map-wrap">${mapSvg(geo)}
      <div class="map-zoom"><button type="button" id="zoom-in" aria-label="Zoomer" title="Zoomer">+</button>`
      + `<button type="button" id="zoom-out" aria-label="Dézoomer" title="Dézoomer">−</button>`
      + `<button type="button" id="zoom-fit" aria-label="Ajuster" title="${cur ? `Ajuster sur ${esc(cur)} et les tables qu'elle joint` : "Ajuster sur toute la carte"}">${fit}</button></div>
      <div class="map-legend"><span>${line("on")}vérifiée</span><span>${line("unchecked")}non mesurée</span><span>${line("weak")}faible</span>
        <span class="legend-sep"></span><span title="Une table dont plusieurs lignes vont vers une même ligne de l'autre">${end("M2 6H21M2 1L11 6M2 11L11 6")}plusieurs</span>
        <span title="Une seule ligne de la table pour chaque ligne de l'autre">${end("M1 6H20M14 1V11")}une</span>
        ${layout.pilled.size ? `<span class="legend-sep"></span><span title="Un référentiel joint par ${PILL_USERS} tables ou plus : son nom sous chacune d'elles, `
          + `sa boîte sous la carte"><b class="refs-key">nom</b>référentiel</span>` : ""}</div>
      <div class="map-mini" id="map-mini" hidden></div>
      <button class="map-expand" type="button" id="map-expand" aria-label="${big ? "Réduire la carte" : "Agrandir la carte"}" title="${big ? "Réduire (Échap)" : "Agrandir la carte"}">`
      + `<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">${expand}</svg></button>
      <div class="map-tip" id="map-tip" role="tooltip" hidden></div></div>
    </div>
    <form class="add-form" id="add-form" ${ui.adding ? "" : "hidden"}>
      <span class="muted-text">Nouvelle jointure</span>
      <select id="add-source-table" aria-label="Table qui pointe">${options(chosen.source[0])}</select><select id="add-source-column" aria-label="Colonne qui pointe"></select>
      <span class="arrow">→</span>
      <select id="add-target-table" aria-label="Table pointée">${options(chosen.target[0])}</select><select id="add-target-column" aria-label="Colonne pointée"></select>
      <button class="btn small primary" type="submit">Vérifier et ajouter</button>
      <button class="btn small ghost" type="button" id="add-cancel">Annuler</button>
    </form>
    ${ui.adding && ui.relCheck ? relationCheck(ui.relCheck) : ""}
    <div class="joins">
      ${joinQueue(lit, sel)}
      ${joinGroups(lit, sel, cur, path)}
      ${ignored.length ? `<details class="ignored"><summary>${plural(ignored.length, "jointure retirée")}</summary>
        ${ignored.map((r) => `<div class="rline"><span class="cell mono muted">${esc(r.from)} → ${esc(r.to)}</span><span></span><span></span>
          <button class="btn ghost small" data-restore="${esc(r.from)}|${esc(r.to)}">Rétablir</button></div>`).join("")}</details>` : ""}
    </div>`;

  setupMap(geo);
  $("path-a").onchange = (e) => { ui.path.a = e.target.value || null; ui.path.result = null; findPath(); };
  $("path-b").onchange = (e) => { ui.path.b = e.target.value || null; ui.path.result = null; findPath(); };
  $("check-relations").onclick = () => runJob("relations/run", { check: true });
  if ($("relations-all")) $("relations-all").onclick = showAllTables;
  wireJoins(el);
  el.querySelectorAll("[data-restore]").forEach((b) => b.addEventListener("click", () => moveRelation(b.dataset.restore, true)));
  // the progress moves from where it was drawn last; after an answer, the next line of the queue takes the focus
  const total = rel.relations.filter((r) => r.weak).length + state.decided.length, progress = total ? state.decided.length / total : 0;
  const bar = el.querySelector(".join-progress .track i");
  if (bar) requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = `${progress * 100}%`; }));
  state.shown = progress;
  if (state.next) {
    el.querySelector(`.join-item[data-key="${CSS.escape(state.next)}"] .join-line`)?.focus({ preventScroll: true });
    state.next = null;
  }
  if (clicked) el.querySelector(".join-item.open, .join-name.open")?.scrollIntoView({ block: "center" });
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

// A relation ignored or kept (action "ignore" or "keep"), or back from it; true once relationships.yaml is written
async function moveRelation(pair, back, action = "ignore") {
  const [source, target] = pair.split("|");
  try {
    ui.rel = await api(`/api/db/${encodeURIComponent(ui.db)}/relations/${action}${back ? "?back=true" : ""}`,
      { method: "POST", body: { source, target, version: ui.rel.version } });
  } catch (e) {
    renderDetail(e.status === 409 ? "relationships.yaml a changé : recharge la page" : e.message);
    return false;
  }
  if (ui.path.a && ui.path.b) await findPath();
  loadTables(); renderDetail();
  return true;
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
    : [`Ajoutée : ${pct(added.found)} des valeurs trouvées${added.partial ? ` — couverture partielle, l'agent perd les autres lignes de ${added.source}` : ""}`, added.partial];
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

// The state of a metric in the list; none before its test (the tests don't run on arrival: every metric would say it)
const METRIC_PILLS = {
  ok: '<span class="pill done">vérifiée</span>', check: '<span class="pill decide">à revoir</span>',
  error: '<span class="pill bad" title="L\'agent refusera de démarrer avec cette métrique">erreur</span>', untested: "",
};

function renderMetricRows() {
  const met = ui.met, q = ui.query.toLowerCase();
  const rows = (met?.metrics || []).filter((m) => (!q || m.name.toLowerCase().includes(q))
    && (ui.filter === "all" || m.state === "check" || m.state === "error"));
  const current = (name, kind) => (name === ui.metric && kind === ui.metricKind ? " current" : "");
  const opened = new Set(storedList(openedKey())), fresh = (name) => met.added.includes(name) && !opened.has(name);
  $("rows").innerHTML = rows.map((m) => `<div class="row${current(m.name, "metric")}${ui.draft?.kind === "metric" && ui.draft.name === m.name ? " unsaved" : ""}" data-metric="${esc(m.name)}">
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
  if (ui.draft && ui.draft.name !== name) { leaveDraft(() => openMetric(name, kind)); return; }
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
  [NOT_JOINED, "aucune jointure ne relie $1 à $2"],
];
const inFrench = (message) => IN_FRENCH.reduce((text, [english, french]) => text.replace(english, french), String(message ?? ""));

// One reason for a proposal left out: its rule, or the check it failed (its tables not joined: said with the tables)
function skippedReason(s) {
  if (s.reason !== "rejected") return SKIPPED[s.reason] || SKIPPED.other;
  const joins = s.detail.match(NOT_JOINED);
  if (joins) {
    return "Ses tables ne se rejoignent pas par une jointure que la formule peut suivre : "
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
  joins: "Ses tables ne sont reliées par aucune jointure : elle ne peut pas être calculée. La jointure s'ajoute dans l'onglet Jointures.",
};

const at = (seconds) => new Date(seconds * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
const decimal = (x) => new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 }).format(x);
const share = (x, warn) => (x == null ? '<span class="muted-text">—</span>'
  : `<span class="bar"><i class="${warn ? "low" : ""}" style="width:${Math.round(Math.min(x, 1) * 100)}%"></i></span>${pct(x)}`);

// One line of "Sur les données": the check, its measure, what it means; amber when worth a look
const checkLine = (label, measure, text, warn = false) => `<div class="mline${warn ? " warn" : ""}">
    <span>${label}</span><span class="fill">${measure}</span><span class="muted-text">${text}</span></div>`;

// What the test found on the data, a check a line: its verdict, then a sentence with its figures and what they change.
// tone "ok" (green), "warn" (amber: worth a look), "bad" (red: the database refuses the formula)
const dataCheck = (tone, title, text) => `<div class="data-check ${tone}"><span class="check-icon">${tone === "ok" ? CHECK : BANG}</span>
    <div><b>${title}</b><span>${text}</span></div></div>`;

function dataChecks(m) {
  const t = m.test, rows = t.rows, others = (table) => names(m.tables.filter((name) => name !== table));
  const missing = (filled) => fmt.format(Math.max(rows - filled, 0));
  const checks = t.tables.map((c) => (c.counted > c.rows
    ? ["warn", "Lignes comptées plusieurs fois", `Chaque ligne de ${col(c.table)} est comptée ${decimal(c.counted / c.rows)} fois en moyenne : `
      + "la jointure la répète, et ses valeurs entrent plusieurs fois dans le calcul."]
    : c.counted < c.rows
      ? [c.warn ? "warn" : "ok", "Lignes perdues par la jointure", `${plural(c.rows - c.counted, "ligne")} de ${col(c.table)} (${pct(1 - c.counted / c.rows)}) `
        + `sans correspondance dans ${others(c.table)} : elles manquent au calcul.`]
      : ["ok", "Aucune ligne comptée deux fois", `Les ${fmt.format(c.rows)} lignes de ${col(c.table)} sont comptées une fois chacune : `
        + `la jointure avec ${others(c.table)} n'en ajoute ni n'en perd.`]));
  // a measure named when there are several (a ratio), "la valeur calculée" otherwise
  for (const s of t.measures) {
    const what = t.measures.length > 1 ? col(s.sql) : "La valeur calculée";
    checks.push(s.conditional ? ["ok", "Vide hors de sa condition, exprès",
      "La valeur n'est calculée que sur les lignes qui remplissent sa condition (CASE WHEN) : les autres restent vides, c'est voulu."]
      : s.share == null ? ["ok", "Aucune ligne à calculer", "La requête ne trouve aucune ligne sur la base."]
      : s.share === 1 ? ["ok", "Calculée sur toutes les lignes", `${what} est remplie sur les ${fmt.format(rows)} lignes : aucune n'est sautée.`]
      : [s.warn ? "warn" : "ok", `${pct(1 - s.share)} des lignes sautées`, `${what} est vide sur ${missing(s.filled)} lignes : le calcul les saute.`]);
  }
  if (t.date) {
    const d = t.date;
    checks.push(d.share === 1 || d.share == null ? ["ok", "Une date sur chaque ligne",
      `${col(d.column)} est remplie partout : un mois ou une année demandés voient toutes les lignes.`]
      : [d.warn ? "warn" : "ok", d.warn ? "Beaucoup de lignes sans date" : "Une date sur presque toutes les lignes",
        `${col(d.column)} manque sur ${missing(d.filled)} lignes (${pct(1 - d.share)}) : elles comptent dans le total, pas quand on demande un mois ou une année.`]);
  }
  checks.push({
    positive: ["ok", "Résultat positif", "Sur toute la base, le résultat n'est ni vide ni nul."],
    negative: ["ok", "Résultat négatif", "Sur toute la base, le résultat est négatif."],
    zero: ["warn", "Résultat nul", "Zéro sur toute la base : sa condition ne trouve peut-être aucune ligne."],
    empty: ["warn", "Résultat vide", "Aucune ligne à calculer sur toute la base."],
  }[t.sign] || ["ok", "Résultat", ""]);
  // the checks to review first, then the ones that pass
  const review = checks.filter(([tone]) => tone === "warn");
  return [review.length, [...review, ...checks.filter(([tone]) => tone !== "warn")].map((c) => dataCheck(...c)).join("")];
}

// "Sur les données": before the test, one line and its button. After it, one line that says whether it passes, in the
// words of the pills of the list (Vérifiée, À revoir, Erreur), its checks under it: folded when it passes (open on
// demand, kept open from one metric to the next), open when it breaks
function dataSection(m) {
  const testing = ui.testing?.current === m.name, t = m.test;
  const button = (label, ghost) => `<button class="btn small${ghost ? " ghost" : ""}" id="metric-test" ${testing ? "disabled" : ""}>${testing ? "Test en cours…" : label}</button>`;
  if (!t) {
    return `<div class="data-checks"><div class="data-head"><span class="field-label">Sur les données</span>
      <span class="data-said">Pas encore testée</span>${button("Tester", false)}</div></div>`;
  }
  const tested = t.at ? `testée le ${when(t.at)}${t.seconds != null ? `, en ${decimal(t.seconds)} s` : ""}` : "testée";
  const [review, checks] = t.error ? [0, dataCheck("bad", "Formule refusée", `<span class="mono">${esc(t.error)}</span>`)] : dataChecks(m);
  const tone = t.error ? "bad" : review ? "warn" : "ok";
  const title = { ok: "Vérifiée sur les données", warn: `À revoir · ${plural(review, "point")}`, bad: "Erreur · la base refuse la formule" }[tone];
  const open = tone !== "ok" || ui.showChecks;
  const toggle = tone === "ok" ? `<button type="button" class="link-btn" id="checks-toggle" aria-expanded="${open}">`
    + `${open ? "Masquer les contrôles" : "Voir les contrôles"}</button>` : "";
  return `<div class="data-checks"><div class="data-head data-verdict ${tone}"><span class="check-icon">${tone === "ok" ? CHECK : BANG}</span>
      <span class="data-said"><b>${title}</b> · ${tested}<span data-stale hidden> · sur la version enregistrée</span></span>${toggle}${button("Retester", true)}</div>
    ${open ? `<div class="data-details">${checks}</div>` : ""}</div>`;
}

const CROSS = '<svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6"/></svg>';

// The origin of a metric (metrics.yaml): proposed by the IA, its mark before its name; written before the studio noted
// it, no mark (unknown); a field corrected by you, the pencil before its label
const metricOrigin = (m) => (m.origin === "ai" ? originMark("ai", "Proposée par l'IA") : "");
const editedMark = (m, field) => ((m.edited || []).includes(field) ? originMark("you", "Corrigé par toi") : "");

// ── Drafts: what the card open changed and did not save yet (a metric, the descriptions of a table, its properties),
// written at once by « Enregistrer », dropped by « Annuler »; a field put back to its saved value leaves it. Leaving
// the card with a draft asks first ──

const norm = (value) => JSON.stringify(typeof value === "string" ? value.trim() : value ?? "");
const draftFields = (kind, name) => (ui.draft?.kind === kind && ui.draft.name === name ? ui.draft.fields : {});
const dropDraft = () => { ui.draft = null; ui.draftError = null; };

function setDraft(kind, name, key, value, saved) {
  if (ui.draft?.kind !== kind || ui.draft.name !== name) ui.draft = { kind, name, fields: {} };
  if (norm(value) === norm(saved)) delete ui.draft.fields[key];
  else ui.draft.fields[key] = value;
  if (!Object.keys(ui.draft.fields).length) ui.draft = null;
  ui.draftError = null;
  const state = $("save-state");
  if (state) state.textContent = "";   // « Enregistré » of an earlier save no longer true
  showDraft();
}

// What a draft changed, in words for its bar: fields of a metric or properties of a table; the description of a table
// and how many of its columns (a column counted once, its text and its categorical switch changed or not)
const draftColumns = (draft) => new Set(Object.keys(draft.fields).filter((k) => k !== "description").map((k) => k.slice(k.indexOf(":") + 1)));
const draftCount = (draft) => (draft.kind === "semantics" ? ("description" in draft.fields) + draftColumns(draft).size : Object.keys(draft.fields).length);
const DRAFT_WORDS = { description: "description", synonyms: "synonymes", expr: "formule", time_dimension: "date par défaut",
  table_type: "type", read_mode: "mode de lecture", data_as_of: "date d'arrêt" };
function draftWords(draft) {
  const keys = Object.keys(draft.fields);
  if (draft.kind !== "semantics") return Object.keys(DRAFT_WORDS).filter((k) => keys.includes(k)).map((k) => DRAFT_WORDS[k]);
  const columns = draftColumns(draft).size;
  return [keys.includes("description") && "description de la table", columns && plural(columns, "colonne")].filter(Boolean);
}

// The draft told on the card: « modifié » on each field changed (its frame in the accent), the tests of a metric said
// to be on its saved version, the dot of its row in the list, the bar; refused, « Enregistrement refusé » first in the
// bar, the reason under the field of a metric, in the bar otherwise
function showDraft() {
  const draft = ui.draft, fields = draft?.fields || {}, n = draft ? draftCount(draft) : 0, error = ui.draftError, el = $("detail");
  el.classList.toggle("has-draft", n > 0);   // the bar sticks to the bottom of the screen, past the card
  el.querySelectorAll("[data-draft]").forEach((field) => field.classList.toggle("modified", field.dataset.draft in fields));
  el.querySelectorAll("[data-draft-row]").forEach((row) => {
    row.classList.toggle("modified", ["column", "categorical", "unclear"].some((f) => `${f}:${row.dataset.draftRow}` in fields));
  });
  el.querySelectorAll("button.unclear-mark").forEach((mark) => {
    const written = `column:${mark.dataset.unclear}` in fields, kept = `unclear:${mark.dataset.unclear}` in fields;
    mark.classList.toggle("stated", written || kept);
    mark.disabled = written;
    mark.setAttribute("aria-pressed", String(kept));
    mark.title = written ? "Ta correction la sort de « à préciser » à l'enregistrement"
      : kept ? "Gardée telle quelle : sort de « à préciser » à l'enregistrement. Clique pour annuler" : UNCLEAR_SAYS;
  });
  el.querySelectorAll("[data-refused]").forEach((note) => {
    note.hidden = error?.field !== note.dataset.refused;
    if (!note.hidden) note.textContent = `Rien n'est écrit : ${error.message}`;
  });
  el.querySelectorAll("[data-stale]").forEach((said) => { said.hidden = !("expr" in fields || "time_dimension" in fields); });
  document.querySelectorAll("#rows .row.unsaved").forEach((row) => row.classList.remove("unsaved"));
  if (draft) document.querySelector(`#rows .row[data-${draft.kind === "metric" ? "metric" : "table"}="${CSS.escape(draft.name)}"]`)?.classList.add("unsaved");
  if (!$("draft-bar")) return;
  $("draft-bar").hidden = !n;
  if (!n) return;
  const words = draftWords(draft);
  $("draft-said").innerHTML = (error ? `<span class="bad-text">Enregistrement refusé${error.field ? "" : ` : ${esc(error.message)}`}</span> · ` : "")
    + `${plural(n, "modification")} non ${n > 1 ? "enregistrées" : "enregistrée"} : `
    + (words.length > 1 ? `${words.slice(0, -1).join(", ")} et ${words.at(-1)}` : words[0]);
}

// The bar of a draft at the bottom of its card, hidden while nothing is changed
const DRAFT_BAR = `<div class="draft-bar" id="draft-bar" role="region" aria-label="Modifications non enregistrées" hidden>
    <span class="draft-dot"></span><span class="draft-said" id="draft-said"></span>
    <button class="btn ghost small" type="button" id="draft-cancel">Annuler</button>
    <button class="btn primary small" type="button" id="draft-save">Enregistrer</button></div>`;
function bindDraftBar() {
  $("draft-cancel").onclick = () => { dropDraft(); renderRows(); renderDetail(); };
  $("draft-save").onclick = () => saveDraft();
  showDraft();
}

// The draft written at once, or nothing of it; true once written
function saveDraft() {
  const draft = ui.draft;
  if (!draft) return Promise.resolve(true);
  return { metric: saveMetricDraft, semantics: saveSemanticsDraft, properties: savePropertiesDraft }[draft.kind](draft);
}

// A save refused: why, kept on the card with the draft (409: the file changed meanwhile, the page to reload)
function draftRefused(e, field = null, file = "tables.yaml") {
  ui.draftError = e.status === 409 ? { field: null, message: `${file} a changé : recharge la page` } : { field, message: inFrench(e.message) };
  const state = $("save-state");
  if (state) state.textContent = "";
  showDraft();
  return false;
}
const saidSaved = (text) => { const state = $("save-state"); if (state) { state.textContent = text; state.className = "save-state ok"; } };
const saving = (text) => { const state = $("save-state"); if (state) { state.textContent = text; state.className = "save-state"; } };

// A metric: a new formula or date checked by the database first, then tested
async function saveMetricDraft(draft) {
  const changes = Object.fromEntries(Object.entries(draft.fields).map(([f, v]) => [f, f === "synonyms" ? v.join(", ") : v]));
  const checked = "expr" in changes || "time_dimension" in changes;
  saving(checked ? "Vérification sur la base…" : "Enregistrement…");
  try {
    ui.met = await api(`/api/db/${encodeURIComponent(ui.db)}/metrics/${encodeURIComponent(draft.name)}`,
      { method: "POST", body: { changes, version: ui.met.version } });
  } catch (e) {
    return draftRefused(e, "expr" in changes ? "expr" : "time_dimension" in changes ? "time_dimension" : "description", "metrics.yaml");
  }
  dropDraft();
  renderRows(); renderDetail();
  saidSaved("Enregistrée · l'agent la lira à son prochain démarrage");   // metrics.yaml is read at startup, not from the index
  if (checked) testMetrics([draft.name]);
  return true;
}

// The descriptions of a table: "description", "column:<name>", "categorical:<name>"
async function saveSemanticsDraft(draft) {
  const changes = Object.entries(draft.fields).map(([key, value]) => {
    const at = key.indexOf(":");
    return at < 0 ? { field: key, value } : { field: key.slice(0, at), column: key.slice(at + 1), value };
  });
  saving("Enregistrement…");
  try {
    ui.detail = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(draft.name)}/semantics`,
      { method: "POST", body: { changes, version: ui.detail.version } });
  } catch (e) {
    return draftRefused(e);
  }
  dropDraft();
  renderRows(); renderDetail();
  saidSaved("Enregistré");
  loadTables();
  return true;
}

// The properties of a table: its type ("": the detection again), its read mode, the date its data stops
async function savePropertiesDraft(draft) {
  const f = draft.fields;
  saving("Enregistrement…");
  try {
    const saved = await api(`/api/db/${encodeURIComponent(ui.db)}/tables/${encodeURIComponent(draft.name)}/properties`, {
      method: "POST", body: { ...f, ...("table_type" in f ? { table_type: f.table_type || null } : {}), version: ui.detail.version } });
    ui.detail = { ...ui.detail, ...saved };
  } catch (e) {
    return draftRefused(e);
  }
  dropDraft();
  renderRows(); renderDetail();
  saidSaved("Enregistré");
  loadTables();
  return true;
}

// Leaving the card open (another metric or table, a tab, a database, the home, a job on the table) with a draft:
// Rester, Abandonner or Enregistrer first; then() once the draft is gone
function leaveDraft(then) {
  if (!ui.draft) { then(); return; }
  const strip = $("confirm"), n = draftCount(ui.draft);
  strip.innerHTML = `<p>${ui.draft.kind === "metric" ? "" : "La table "}<b>${esc(ui.draft.name)}</b> a ${plural(n, "modification")} non ${n > 1 ? "enregistrées" : "enregistrée"}.</p>
    <span class="confirm-actions"><button class="btn small ghost" type="button" data-stay>Rester</button>
      <button class="btn small" type="button" data-drop>Abandonner</button>
      <button class="btn small primary" type="button" data-save>Enregistrer</button></span>`;
  strip.hidden = false;
  strip.scrollIntoView({ block: "nearest" });
  strip.querySelector("[data-stay]").onclick = () => closeConfirm(strip);
  strip.querySelector("[data-drop]").onclick = () => { closeConfirm(strip); dropDraft(); then(); };
  strip.querySelector("[data-save]").onclick = async () => { closeConfirm(strip); if (await saveDraft()) then(); };
  strip.querySelector("[data-save]").focus();
}
window.addEventListener("beforeunload", (e) => { if (ui.draft) e.preventDefault(); });

// A metric in its card: the values of its draft over the saved ones
const draftValue = (m, field) => (field in draftFields("metric", m.name) ? draftFields("metric", m.name)[field] : m[field]);
const setMetricDraft = (m, field, value) => setDraft("metric", m.name, field, value, m[field]);

// The words of the questions that mean this metric, a pill each. A word another metric has too, as a synonym or as
// its name, in amber with that metric: on that word the agent can't choose between them
function synonymPills(m, words) {
  const others = new Map();
  for (const o of ui.met.metrics) {
    if (o.name === m.name) continue;
    for (const word of [o.name, ...o.synonyms]) if (!others.has(word.trim().toLowerCase())) others.set(word.trim().toLowerCase(), o.name);
  }
  return words.map((word, i) => {
    const other = others.get(word.trim().toLowerCase());
    return `<span class="syn${other ? " shared" : ""}"${other ? ' title="L\'agent ne peut pas choisir entre les deux métriques sur ce mot"' : ""}>${esc(word)}`
      + (other ? `<small>· aussi dans <span class="mono">${esc(other)}</span></small>` : "")
      + `<button type="button" class="syn-x" data-syn="${i}" aria-label="Retirer ${esc(word)}">${CROSS}</button></span>`;
  }).join("");
}

// The synonyms of the draft drawn again, the input kept focused when a word was just added
function drawSynonyms(m, { focus = false } = {}) {
  const box = $("met-syn-box"), words = [...draftValue(m, "synonyms")];
  box.querySelectorAll(".syn").forEach((pill) => pill.remove());
  $("met-syn-add").insertAdjacentHTML("beforebegin", synonymPills(m, words));
  box.querySelectorAll("[data-syn]").forEach((x) => x.addEventListener("click", () => {
    words.splice(Number(x.dataset.syn), 1);
    setMetricDraft(m, "synonyms", words);
    drawSynonyms(m);
  }));
  if (focus) $("met-syn-add").focus();
}

// What was typed in the input of the synonyms, added to the draft: one word or several separated by commas, a word
// already there (whatever its case) not twice
function addSynonyms(m) {
  const input = $("met-syn-add"), words = [...draftValue(m, "synonyms")], known = new Set(words.map((w) => w.toLowerCase()));
  let added = false;
  for (const word of input.value.split(",").map((w) => w.trim())) {
    if (word && !known.has(word.toLowerCase())) { known.add(word.toLowerCase()); words.push(word); added = true; }
  }
  input.value = "";
  if (added) { setMetricDraft(m, "synonyms", words); drawSynonyms(m, { focus: true }); }
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
        ${joins ? '<div><button class="btn ghost small" id="skipped-path">Voir le chemin dans Jointures</button></div>' : ""}
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

  const error = m.error ? `<div class="error-note"><span>${ERRORS[m.error.kind]}</span><code>${esc(inFrench(m.error.detail))}</code></div>` : "";
  const rule = m.rule ? `<div class="rule-note"><span>${RULES[m.rule](m)}</span><button class="btn small" data-retire="1" title="Elle sort de metrics.yaml et l'IA ne la proposera plus">Retirer</button></div>` : "";
  const expr = draftValue(m, "expr"), refused = (field) => `<p class="note error refused" data-refused="${field}" hidden></p>`;
  // the query on demand, in the style of the formula: its joins are those of the tables of the header, by their relations
  const query = m.sql ? `<div class="query-line">
      <button type="button" class="link-btn" id="met-query">${ui.showQuery ? "Masquer la requête" : "Voir la requête"}</button></div>
    ${ui.showQuery ? `<pre class="sql">${sqlHtml(m.sql)}</pre>` : ""}` : "";
  el.innerHTML = `<div class="detail-head"><h2 class="origin-item">${metricOrigin(m)}${esc(m.name)}</h2><span class="figures">${esc(m.tables.join(", "))}</span>
      <span class="grow"></span><span class="save-state" id="save-state"></span>
      ${m.rule ? "" : '<button class="btn ghost small quiet" data-retire="1" title="Elle sort de metrics.yaml et l\'IA ne la proposera plus">Retirer</button>'}</div>
    ${note}${error}${rule}
    <div class="metric-fields">
      <div class="lead" data-draft="description"><label class="field-label" for="met-description">Description</label>
        <span class="desc-cell block">${editedMark(m, "description")}<textarea class="edit block" id="met-description" rows="1" placeholder="Dire ce que mesure cette métrique…">${esc(draftValue(m, "description"))}</textarea></span>${refused("description")}</div>
      <div data-draft="synonyms"><label class="field-label" for="met-syn-add">${editedMark(m, "synonyms")}Synonymes</label>
        <div class="syn-box" id="met-syn-box"><input type="text" id="met-syn-add" placeholder="+ ajouter" autocomplete="off"></div></div>
      <div class="formula-line">
        <div data-draft="expr"><label class="field-label" for="met-expr">${editedMark(m, "expr")}Formule</label>
          <div class="code-edit"><pre class="sql" id="met-expr-hl" aria-hidden="true">${sqlHtml(expr)}\n</pre>
            <textarea class="sql" id="met-expr" rows="1" spellcheck="false">${esc(expr)}</textarea></div>
          ${refused("expr")}${query}</div>
        <div data-draft="time_dimension"><label class="field-label" for="met-date">${editedMark(m, "time_dimension")}Date par défaut</label>
          <select class="edit-select" id="met-date">${dateOptions(m, draftValue(m, "time_dimension"))}</select>${refused("time_dimension")}</div>
      </div>
    </div>
    ${m.sql ? dataSection(m) : ""}
    ${DRAFT_BAR}`;

  // every change goes to the draft, nothing is written before « Enregistrer »
  const description = $("met-description");
  fitHeight(description);
  description.addEventListener("input", () => { fitHeight(description); setMetricDraft(m, "description", description.value); });
  const formula = $("met-expr");
  formula.addEventListener("input", () => { $("met-expr-hl").innerHTML = `${sqlHtml(formula.value)}\n`; setMetricDraft(m, "expr", formula.value); });
  $("met-date").addEventListener("change", (e) => setMetricDraft(m, "time_dimension", e.target.value));
  drawSynonyms(m);
  $("met-syn-add").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addSynonyms(m); } });
  $("met-syn-add").addEventListener("blur", () => addSynonyms(m));
  $("met-syn-box").addEventListener("click", (e) => { if (e.target === e.currentTarget) $("met-syn-add").focus(); });
  bindDraftBar();
  if ($("met-query")) $("met-query").onclick = () => { ui.showQuery = !ui.showQuery; renderDetail(); };
  if ($("checks-toggle")) $("checks-toggle").onclick = () => { ui.showChecks = !ui.showChecks; renderDetail(); };
  if ($("metric-test")) $("metric-test").onclick = () => testMetrics([m.name]);
  el.querySelectorAll("[data-retire]").forEach((b) => b.addEventListener("click", () => moveMetric(m.name, false)));
}

// The dates a metric can take: the ones of its own tables, then the ones a join would bring (a table more in each
// question that asks for it), "aucune" last; the one of the draft, else the saved one, selected
function dateOptions(m, chosen = m.time_dimension) {
  const option = (d) => `<option value="${esc(d)}" ${d === chosen ? "selected" : ""}>${d ? esc(d) : "aucune"}</option>`;
  const dates = [...new Set([m.time_dimension, ...m.dates].filter(Boolean))];
  const own = dates.filter((d) => m.tables.includes(d.split(".")[0])), joined = dates.filter((d) => !own.includes(d));
  if (!own.length || !joined.length) return [...dates, ""].map(option).join("");
  const via = [...new Set(joined.map((d) => d.split(".")[0]))];
  return `<optgroup label="Dans ${esc(m.tables.join(", "))}">${own.map(option).join("")}</optgroup>`
    + `<optgroup label="Par une jointure (ajoute ${esc(via.join(", "))})">${joined.map(option).join("")}</optgroup>${option("")}`;
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
  if (ui.draft?.kind === "metric" && ui.draft.name === name) dropDraft();   // retired: its draft goes with it
  loadTables(); renderRows(); renderDetail();
}

// ── Périmètre: every table of the database against the ones the agent and the pipeline see, and the index ──

async function loadScope() {
  const db = ui.db;
  try {
    const scope = await api(`/api/db/${encodeURIComponent(db)}/scope`);
    if (db !== ui.db) return;
    ui.scope = scope;
  } catch (e) {
    showBanner(`Impossible de lire le périmètre : ${e.message}`, true);
    return;
  }
  if (ui.step !== "scope" || ui.home || ui.wizard) return;
  if (ui.data) renderHeader();
  renderRows(); renderDetail();
}

// A table of the database the scope leaves out (the filter "Hors périmètre")
const outOfScope = (t) => !t.in_scope && t.in_base !== false;

// The tables checked, by what a button does with them: the ones out of the scope to add, the ones in it to take out
function scopeSelection() {
  const checked = (ui.scope?.tables || []).filter((t) => ui.selected.has(t.name));
  return { add: checked.filter((t) => !t.in_scope && t.in_base).map((t) => t.name), remove: checked.filter((t) => t.in_scope).map((t) => t.name) };
}

// A pill only where the scope and what the agent reads differ: a table added not indexed yet, one taken out it still
// reads, one the database no longer has. A table of the scope as indexed has none: the state of its document is the
// Fiches tab's
const STILL_INDEXED = '<span class="pill decide" title="L\'agent lit encore sa fiche : elle partira à la prochaine indexation">encore indexée</span>';
const SCOPE_PILLS = {
  missing: '<span class="pill todo" title="Pas dans l\'index : l\'agent ne voit pas encore cette table">à indexer</span>',
  leaving: STILL_INDEXED,
};
function scopePill(t) {
  if (t.state !== "gone") return SCOPE_PILLS[t.state] || "";
  if (t.indexed) return STILL_INDEXED;
  return t.in_scope ? '<span class="pill todo" title="Dans le périmètre, plus dans la base : l\'agent et le pipeline l\'ignorent">à retirer</span>' : "";
}

// The tables of the scope, then the other tables of the database, then the ones it no longer has, quieter
function renderScopeRows() {
  const rows = visibleRows();
  const row = (t) => `<div class="row${t.in_scope && t.in_base !== false ? "" : " aside"}${t.name === ui.current ? " current" : ""}" data-table="${esc(t.name)}">
      <input type="checkbox" ${ui.selected.has(t.name) ? "checked" : ""} aria-label="Sélectionner ${esc(t.name)}">
      <span class="cell" title="${esc(t.name)}">${esc(t.name)}</span>${scopePill(t)}</div>`;
  const group = (title, list) => (list.length ? `<div class="list-group">${title} · ${fmt.format(list.length)}</div>${list.map(row).join("")}` : "");
  $("rows").innerHTML = group("Dans le périmètre", rows.filter((t) => t.in_scope && t.in_base !== false))
    + group("Hors périmètre", rows.filter(outOfScope)) + group("Plus dans la base", rows.filter((t) => t.in_base === false));
  $("list-extra").innerHTML = "";
  $("empty").textContent = ui.scope ? "Aucune table ne correspond." : "Lecture de la base…";
  $("empty").hidden = rows.length > 0;
  $("select-all").checked = rows.length > 0 && rows.every((t) => ui.selected.has(t.name));
  renderActions();
}

// The card of the Périmètre: the table chosen, or the database against its scope and its index
function renderScope(el, message) {
  const sc = ui.scope;
  if (!sc) { el.innerHTML = '<p class="placeholder">Lecture de la base…</p>'; return; }
  const t = ui.current && sc.tables.find((r) => r.name === ui.current);
  el.innerHTML = (t ? scopeCard(t, sc) : scopeSummary(sc)).replace("<!--message-->", message ? `<p class="note error">${esc(message)}</p>` : "");
  $("scope-back")?.addEventListener("click", () => { ui.current = null; renderRows(); renderDetail(); });
  $("scope-overview")?.addEventListener("click", () => goTo("overview"));
  $("scope-toggle")?.addEventListener("click", (e) => (e.currentTarget.dataset.add ? changeScope({ add: [t.name] }) : confirmRemove([t.name])));
  el.querySelectorAll("[data-scope-all]").forEach((b) => { b.onclick = () => chooseMode(b.dataset.scopeAll === "1"); });
}

// How the scope is chosen: every table, the ones to come too, or a list (lib/semantics.scope_origin). Before a choice:
// nothing (every table, the step to do)
function scopeMode(sc) {
  const off = sc.counts.base == null ? ' disabled title="La base ne répond pas"' : "";
  const mode = (all, label) => `<button type="button" data-scope-all="${all ? 1 : 0}" aria-pressed="${sc.origin === (all ? "all" : "list")}"${off}>${label}</button>`;
  const says = {
    none: "Aucun périmètre choisi : l'agent voit toutes les tables de la base, même celles qui y arriveront. À choisir :",
    all: "Toutes les tables de la base, celles qui y arriveront aussi.",
    list: "Les tables de la liste seulement : une table qui arrive dans la base reste dehors.",
  }[sc.origin];
  return `<div class="note with-action scope-mode"><span>${says}</span>
    <div class="seg scope-seg${sc.origin === "none" ? toDecide(`${ui.db}:scope`) : ""}" role="group" aria-label="Tables lues">${mode(true, "Toutes les tables, même les futures")}${mode(false, "Une liste de tables")}</div></div>`;
}

// Every table or a list: a list keeps the tables of today; every table brings in the ones out of the scope, said before
function chooseMode(all) {
  const sc = ui.scope, outside = sc.tables.filter(outOfScope).length;
  if (sc.origin === (all ? "all" : "list")) return;
  if (!all || !outside) { changeScope({ all }); return; }
  openConfirm($("confirm"), {
    text: `<b>Toutes les tables</b> : ${outside > 1 ? `les ${fmt.format(outside)} tables hors périmètre y entrent` : "la table hors périmètre y entre"}, `
      + "et celles qui arriveront dans la base aussi. Le pipeline pourra les profiler et les décrire, l'agent les lire une fois indexées.",
    button: () => "Toutes les tables", run: () => changeScope({ all: true }), job: false,
  });
}

// What differs between the scope and what the agent reads, in a sentence or two (a document changed since the index is
// the Fiches tab's)
function scopeVerdict(sc) {
  const named = (test) => sc.tables.filter(test).map((t) => t.name);
  const missing = named((t) => t.state === "missing"), leaving = named((t) => t.state === "leaving");
  const gone = named((t) => t.state === "gone");
  const goneInScope = named((t) => t.state === "gone" && t.in_scope);
  const many = (list, one, more) => (list.length > 1 ? more : one);
  const parts = [];
  if (!sc.trace) parts.push("L'index n'a pas de trace : réindexe pour comparer le périmètre à ce que l'agent lit.");
  if (missing.length) {
    parts.push(`${names(missing)} ${many(missing, "est", "sont")} dans le périmètre mais pas dans l'index : l'agent ne ${many(missing, "la", "les")} voit pas encore. `
      + `${many(missing, "Profile-la et décris-la", "Profile-les et décris-les")}, puis réindexe.`);
  }
  if (leaving.length) {
    parts.push(`${names(leaving)} ${many(leaving, "est sortie", "sont sorties")} du périmètre : l'agent ${many(leaving, "la", "les")} lit jusqu'à la prochaine indexation.`);
  }
  if (gone.length) {
    const remove = goneInScope.length === gone.length ? many(gone, "retire-la", "retire-les") : `retire ${names(goneInScope)}`;
    parts.push(`${names(gone)} ${many(gone, "n'est plus", "ne sont plus")} dans la base${goneInScope.length ? ` : ${remove} du périmètre` : ""}.`);
  }
  if (!parts.length) {
    const outside = (sc.counts.outside || 0);
    return `<p class="verdict">Le périmètre et l'index concordent : l'agent lit les ${plural(sc.counts.scope, "table")} du périmètre`
      + `${outside ? `, aucune des ${fmt.format(outside)} autres de la base` : ""}.</p>`;
  }
  const reindex = sc.trace && (leaving.length || sc.tables.some((t) => t.state === "gone" && t.indexed));
  return `<p class="verdict warn">${parts.join(" ")}${reindex ? " Réindexe pour que l'agent lise le périmètre tel qu'il est." : ""}</p>`;
}

function scopeSummary(sc) {
  const c = sc.counts, outside = (c.outside || 0) + (c.leaving || 0);
  const read = sc.source === "database" ? `lue à ${at(sc.read_at)}` : "lue dans les fichiers";
  return `<div class="detail-head"><h2>La base et l'index</h2><span class="figures">${read}</span>
      <span class="grow"></span><button class="btn ghost small" type="button" id="scope-overview">← Vue d'ensemble</button></div><!--message-->
    ${scopeMode(sc)}${scopeVerdict(sc)}
    <div class="dash"><div class="tiles">
      ${indexTile("Dans le périmètre", fmt.format(c.scope) + (c.base == null ? "" : `<small class="of">/${fmt.format(c.base)}</small>`),
        c.base == null ? "Ce que l'agent et le pipeline voient. La base ne répond pas : les tables viennent des fichiers."
          : `Ce que l'agent et le pipeline voient, sur les ${plural(c.base, "table")} de la base.`)}
      ${indexTile("Dans l'index", sc.trace ? fmt.format(c.indexed) : "—",
        sc.trace ? `${c.indexed > 1 ? "Fiches" : "Fiche"} que l'agent lit aujourd'hui.` : "Pas de trace : réindexer pour la voir.")}
      ${indexTile("Hors périmètre", fmt.format(outside), outside
        ? "Le pipeline ne les lit pas, et une requête de l'agent qui les lit est bloquée." : "Toutes les tables de la base sont dans le périmètre.")}
    </div></div>`;
}

// What a table is in the database, the scope, the files and the index, and the button that moves it in or out
const SCOPE_STATES = {
  fresh: () => "Dans le périmètre et dans l'index : l'agent lit sa fiche à jour.",
  changed: () => "Dans le périmètre. Ses fichiers ont changé depuis l'index : l'agent lit encore l'ancienne fiche, réindexe pour la nouvelle.",
  missing: (t) => `Dans le périmètre, pas encore dans l'index : l'agent ne la voit pas. ${t.described ? "Réindexe pour qu'il la lise."
    : t.profiled ? "Décris-la, puis réindexe." : "Profile-la et décris-la, puis réindexe."}`,
  unknown: () => "Dans le périmètre. L'index n'en a pas de trace : réindexe pour savoir ce que l'agent lit.",
  outside: () => "Hors périmètre : le pipeline ne la lit pas, et une requête de l'agent qui la lit est bloquée.",
  leaving: () => "Retirée du périmètre, encore dans l'index : l'agent la lit jusqu'à la prochaine indexation.",
  gone: (t) => `Plus dans la base${t.indexed ? " : sa fiche partira à la prochaine indexation" : ""}.`
    + `${t.in_scope ? " L'agent et le pipeline l'ignorent : retire-la du périmètre." : ""}`,
};
function scopeCard(t, sc) {
  const facts = [t.columns != null && plural(t.columns, "colonne"), t.profiled ? "profilée" : "non profilée", t.described ? "décrite" : "non décrite"];
  const tone = ["changed", "missing", "leaving", "gone"].includes(t.state) ? " warn" : "";
  const kept = !t.in_scope && t.in_base && (t.profiled || t.described)
    ? `<p class="note">${t.profiled && t.described ? "Son profil et ses descriptions restent" : t.profiled ? "Son profil reste" : "Ses descriptions restent"}
      dans les fichiers : ajoutée de nouveau, elle les retrouve.</p>` : "";
  const off = sc.counts.base == null ? " disabled title=\"La base ne répond pas\"" : "";
  const action = t.in_scope ? `<button class="btn" type="button" id="scope-toggle"${off}>Retirer du périmètre</button>`
    : t.in_base ? `<button class="btn primary" type="button" id="scope-toggle" data-add="1"${off}>Ajouter au périmètre</button>` : "";
  return `<div class="detail-head"><h2>${esc(t.name)}</h2><span class="figures">${facts.filter(Boolean).join(" · ")}</span>
      <span class="grow"></span><button class="btn ghost small" type="button" id="scope-back">← La base et l'index</button></div><!--message-->
    <p class="verdict${tone}">${SCOPE_STATES[t.state](t)}</p>${kept}
    ${action ? `<div class="scope-actions">${action}</div>` : ""}`;
}

// A change of the scope, written to scope.yaml: every table ({all: true}), tables added or taken out ({add, remove}: a
// list), or the scope of today as a list ({all: false}); every screen reads it again
async function changeScope(body) {
  const version = ui.scope?.version ?? ui.idx?.scope.version;   // the guided setup changes it before the screen is read
  try {
    ui.scope = await api(`/api/db/${encodeURIComponent(ui.db)}/scope`, { method: "POST", body: { ...body, version } });
  } catch (e) {
    showBanner(e.status === 409 ? "scope.yaml a changé : relis la base avant de recommencer." : e.message, true);
    return;
  }
  ui.selected.clear();
  if (ui.step === "scope") { renderRows(); renderDetail(); }
  await loadTables();   // the other tabs, the head and the tabs of the database
  loadIndex();
}

// Tables taken out of the scope, said before: their files stay, the agent loses them at the next index
function confirmRemove(tables) {
  if (!tables.length) return;
  const one = tables.length === 1;
  openConfirm($("confirm"), {
    text: `<b>Retirer ${one ? `<span class="mono">${esc(tables[0])}</span>` : plural(tables.length, "table")} du périmètre</b> : `
      + `${one ? "elle ne sera plus profilée, décrite ni indexée" : "elles ne seront plus profilées, décrites ni indexées"}, et l'agent `
      + `${one ? "la" : "les"} perdra à la prochaine indexation. ${one ? "Son profil et ses descriptions restent" : "Leurs profils et descriptions restent"} dans les fichiers.`
      + (["all", "none"].includes(ui.scope.origin) ? " Le périmètre devient une liste : une table qui arrivera dans la base restera dehors." : ""),
    button: () => "Retirer", run: () => changeScope({ remove: tables }), job: false,
  });
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
  if (ui.current) openTable(ui.current, { again: true }); else renderDetail();
}

const indexStatus = (name) => ui.idx?.tables.find((t) => t.name === name)?.status;
const INDEX_PILLS = {
  fresh: '<span class="pill done">à jour</span>',
  changed: '<span class="pill decide" title="Les fichiers ont changé depuis l\'index : l\'agent lit encore l\'ancienne fiche">modifiée</span>',
  missing: '<span class="pill todo" title="Pas dans l\'index : l\'agent ne voit pas cette table">à indexer</span>',
  unknown: '<span class="pill todo" title="Index construit avant que le studio en garde la trace">état inconnu</span>',
};

// The tables the index holds that leave it at the next index: the database no longer has them, or they are out of the scope
const leavingIndex = (idx) => [...idx.gone, ...idx.unscoped];

// The tables leaving the index: under the tables, quieter
function renderGone() {
  const group = (title, list) => (list.length ? `<div class="list-group">${title} · ${fmt.format(list.length)}</div>`
    + list.map((name) => `<div class="row aside${name === ui.current ? " current" : ""}" data-table="${esc(name)}">
        <span></span><span class="cell" title="${esc(name)}">${esc(name)}</span><span class="pill todo">à retirer</span></div>`).join("") : "");
  $("list-extra").innerHTML = group("Plus dans la base", ui.idx?.gone || []) + group("Hors du périmètre", ui.idx?.unscoped || []);
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

// The tables whose document the next index changes, in sentences: changed since, not indexed yet, leaving it
function indexChanges(idx) {
  const changed = idx.tables.filter((t) => t.status === "changed").map((t) => t.name);
  const missing = idx.tables.filter((t) => t.status === "missing").map((t) => t.name);
  const parts = [];
  if (changed.length) {
    parts.push(`${names(changed)} ${changed.length > 1 ? "ont" : "a"} changé depuis l'index : `
      + `l'agent lit encore ${changed.length > 1 ? "leurs anciennes fiches" : "son ancienne fiche"}.`);
  }
  if (missing.length) parts.push(`${names(missing)} ${missing.length > 1 ? "ne sont pas" : "n'est pas"} dans l'index : l'agent ne ${missing.length > 1 ? "les" : "la"} voit pas.`);
  if (idx.gone.length) {
    parts.push(`${names(idx.gone)} ${idx.gone.length > 1 ? "ne sont plus" : "n'est plus"} dans la base : `
      + `${idx.gone.length > 1 ? "leurs fiches partiront" : "sa fiche partira"} à la prochaine indexation.`);
  }
  if (idx.unscoped.length) {
    parts.push(`${names(idx.unscoped)} ${idx.unscoped.length > 1 ? "sont sorties" : "est sortie"} du périmètre : `
      + `${idx.unscoped.length > 1 ? "leurs fiches partiront" : "sa fiche partira"} à la prochaine indexation.`);
  }
  return parts;
}

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
  const errors = idx.metrics.errors.map((e) => e.name), changes = indexChanges(idx);
  const e = idx.eval, documents = idx.documents > 1 ? `Les ${fmt.format(idx.documents)} fiches sont à jour` : "La fiche est à jour";
  let tone = "", text;
  if (errors.length) {
    tone = "bad";
    text = `L'agent ne démarrera pas : il refuse ${errors.length > 1 ? "les métriques" : "la métrique"} ${names(errors)} (détail dans l'onglet Métriques).`;
  } else if (changes.length) {
    tone = "warn";
    text = `${changes.join(" ")} ${idx.dataset ? "Réindexe, puis relance l'évaluation." : "Réindexe."}`;
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
  const raw = idx.raw.filter((t) => !leavingIndex(idx).includes(t));
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

// ── The Vue d'ensemble: the eval of the agent, what it reads in six cards, the setup on the side ──

// What moved since the overview was last drawn: the database, its index, its eval (what plays when the page is drawn
// again), the default word of each card, the table under the pointer ({tables, col, q, metric}), the observer that
// draws the chart and the joins again at another width
const OV = { db: null, built: null, eval: null, notes: {}, focus: null, resize: null, lens: { g: 0, gTo: 0, at: {}, to: {}, raf: 0 }, orders: null };
const ROLLED = new Map();   // the text each rolling figure last showed, by its key: {text, db}
const DIGITS = [..."0123456789"].map((d) => `<span>${d}</span>`).join("");
const stillPage = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const rollSlot = (key, text) => `<span class="roll" data-roll="${key}">${esc(text)}</span>`;

// The figures of a part of the page roll from what they showed to their new value, digit by digit, the units first
// (another database, a new index, a new eval); on the same database, a figure that changed says by how much for a few
// seconds (its [data-delta]). Drawn the first time, a figure just shows
function rollFigures(root) {
  root.querySelectorAll("[data-roll]").forEach((el) => {
    const key = el.dataset.roll, text = el.textContent, before = ROLLED.get(key);
    ROLLED.set(key, { text, db: ui.db });
    if (!before || before.text === text || stillPage()) return;
    const change = Number(text.replace(/\D/g, "")) - Number(before.text.replace(/\D/g, "")), delta = root.querySelector(`[data-delta="${key}"]`);
    if (delta && before.db === ui.db && change) { delta.textContent = `${change > 0 ? "+" : "−"}${fmt.format(Math.abs(change))}`; delta.hidden = false; }
    rollDigits(el, text, before.text);
  });
}

// The digits of a figure rolled from the old text to the new one, the units first
function rollDigits(el, text, before) {
  const chars = [...text], old = [...before], shift = chars.length - old.length;
  el.setAttribute("aria-label", text);
  el.innerHTML = chars.map((c) => (/\d/.test(c) ? `<span class="roll-digit" aria-hidden="true"><span>${DIGITS}</span></span>`
    : `<span aria-hidden="true">${esc(c)}</span>`)).join("");
  [...el.children].forEach((child, i) => {
    if (!child.classList.contains("roll-digit")) return;
    const strip = child.firstChild, was = old[i - shift];
    strip.style.transition = "none";
    strip.style.transform = `translateY(${-(/\d/.test(was ?? "") ? Number(was) : 0) * 1.2}em)`;
    strip.getBoundingClientRect();
    strip.style.transition = "";
    strip.style.transitionDelay = `${(chars.length - 1 - i) * 55}ms`;
    strip.style.transform = `translateY(${-Number(chars[i]) * 1.2}em)`;
  });
}

// A figure that follows the pointer (the legend of the values) rolls to its new value, its digits going on from where
// they are while it keeps its shape (drawn again only when it gains or loses a digit); still without movement
function rollTo(el, text) {
  const was = el.getAttribute("aria-label") ?? el.textContent, parts = [...el.children], chars = [...text];
  if (was === text) return;
  if (stillPage()) { el.removeAttribute("aria-label"); el.textContent = text; return; }
  const same = parts.length === chars.length && parts.every((p, i) => (p.classList.contains("roll-digit") ? /\d/.test(chars[i]) : p.textContent === chars[i]));
  if (!same) { rollDigits(el, text, was); return; }
  el.setAttribute("aria-label", text);
  parts.forEach((p, i) => {
    if (!p.classList.contains("roll-digit")) return;
    p.firstChild.style.transitionDelay = `${(chars.length - 1 - i) * 55}ms`;
    p.firstChild.style.transform = `translateY(${-Number(chars[i]) * 1.2}em)`;
  });
}

// What the agent reads, six figures: the structure (the tables of its documents, their columns, joins), then the data
// and the calculations (sample rows, values of the categorical columns, metrics). Each with a word under it (amber or
// red only when that figure needs looking at) and what it is in a bubble, what is wrong first. What to do about them
// is said by the Mise en place
function agentFigures(idx) {
  const sc = idx.scope, m = idx.metrics, hidden = idx.hidden.length, d = idx.dropped, leaving = leavingIndex(idx);
  const changed = idx.tables.filter((t) => t.status === "changed").length, missing = idx.tables.filter((t) => t.status === "missing").length;
  const raw = idx.raw.filter((t) => !leaving.includes(t)), rows = idx.sample_rows, now = idx.sample_rows_now;
  const tables = [
    changed && `${plural(changed, "fiche")} ${changed > 1 ? "anciennes" : "ancienne"}`,
    missing && `${fmt.format(missing)} à indexer`,
    leaving.length && `${fmt.format(leaving.length)} à retirer`,
    raw.length && `${fmt.format(raw.length)} sans description`,
  ].filter(Boolean);
  const outside = sc.base == null ? null : sc.base - sc.tables;
  const values = idx.inline.values + idx.searched.values, columns = idx.inline.columns + idx.searched.columns;
  const example = sampleTable(idx);
  return [
    { key: "tables", value: idx.documents, label: idx.documents > 1 ? "tables" : "table",
      note: tables.length ? tables.join(" · ")
        : sc.origin === "all" || sc.origin === "none" ? "toutes, même les futures"
        : outside ? `sur ${fmt.format(sc.base)} dans la base` : outside === 0 ? "toutes celles d'aujourd'hui" : "celles de la liste",
      tone: tables.length ? "warn" : "",
      issue: [...indexChanges(idx), raw.length && `${names(raw)} sans description : l'agent n'en a que la structure brute.`].filter(Boolean).join(" "),
      tip: "Une fiche par table du périmètre : sa description, ses colonnes, ses jointures. Une barre par table, haute de ses lignes." },
    { key: "columns", value: idx.shown, label: idx.shown > 1 ? "colonnes" : "colonne",
      note: hidden ? `${fmt.format(hidden)} ${hidden > 1 ? "cachées" : "cachée"}` : "aucune cachée",
      tip: "Les colonnes écrites dans les fiches, hachurées celles cachées. Une colonne vide, technique ou personnelle peut être cachée : "
        + "l'agent ne la voit plus, ni dans les exemples de lignes." + (hidden ? ` Cachées : ${names(idx.hidden)}.` : "") },
    { key: "joins", value: idx.joins, label: idx.joins > 1 ? "jointures" : "jointure",
      note: idx.joins ? "entre les tables" : "tables isolées",
      tip: "Les jointures écrites dans les fiches, un arc chacune entre ses deux tables : l'agent relie les tables par ces colonnes, sans les deviner." },
    { key: "samples", value: rows, label: rows > 1 ? "lignes d'exemple" : "ligne d'exemple",
      note: rows !== now ? "à réappliquer" : !rows ? "aucune ligne de la base" : example ? `par fiche, ici <b>${esc(example.name)}</b>` : "par fiche",
      tone: rows !== now ? "warn" : "",
      issue: rows !== now ? `La configuration en demande ${now ? fmt.format(now) : "aucune"} par fiche : réindexer pour l'appliquer.` : "",
      tip: "Des lignes réelles de chaque table sous sa structure, pour que le modèle voie à quoi ressemblent les données. "
        + "Les colonnes cachées en sont retirées." },
    { key: "values", value: values, label: values > 1 ? "valeurs connues" : "valeur connue",
      note: d.length ? `${plural(d.length, "colonne")} sans valeurs` : columns ? `dans ${plural(columns, "colonne")}` : "aucune colonne catégorielle",
      tone: d.length ? "warn" : "",
      issue: d.length ? `${names(d)} ${d.length > 1 ? "sont vides ou trop variées" : "est vide ou trop variée"} : l'agent n'en connaît aucune valeur.` : "",
      tip: "Les valeurs réelles des colonnes catégorielles, une barre par colonne : l'agent filtre avec l'orthographe exacte de la base.", extra: valuesSplit(idx) },
    { key: "metrics", value: m.total, label: m.total > 1 ? "métriques" : "métrique",
      note: m.errors.length ? `${fmt.format(m.errors.length)} refusée${m.errors.length > 1 ? "s" : ""}`
        : m.total ? (m.total > 1 ? "toutes valides" : "valide") : "aucune écrite",
      tone: m.errors.length ? "bad" : "",
      issue: m.errors.length ? `L'agent ne démarrera pas : il refuse ${names(m.errors.map((e) => e.name))}.` : "",
      tip: "Les calculs métier écrits une fois pour toutes. L'agent vérifie leurs colonnes à son démarrage." },
  ];
}
const CARD_GROUPS = [["Structure", ["tables", "columns", "joins"]], ["Données et calculs", ["samples", "values", "metrics"]]];

// The categorical columns in the bubble of the values: one bar, the columns without values left as the empty track,
// then where the agent finds the values (in the document, searched on demand)
function valuesSplit(idx) {
  const columns = idx.inline.columns + idx.searched.columns + idx.dropped.length;
  if (!columns) return "";
  const parts = [
    ["listed", idx.inline.columns, idx.inline.values, "dans les fiches", "de peu de valeurs"],
    ["searched", idx.searched.columns, idx.searched.values, "cherchées à la demande", "trop riches pour la fiche"],
  ].filter(([, n]) => n);
  const bar = parts.map(([key, n]) => `<span class="${key}" style="width:${n / columns * 100}%"></span>`).join("");
  const keys = parts.map(([key, n, values, where, why]) => `<span class="swatch ${key}"></span><b>${fmt.format(values)}</b>
    <span>${where}, ${plural(n, "colonne")} ${why}</span>`).join("");
  return `<div class="split" aria-hidden="true">${bar}</div><div class="split-keys">${keys}</div>`;
}

// A card of what the agent reads: its figure (rolling), its name (what it is in a bubble), the word under it, its picture
function readCard(f, viz) {
  return `<article class="panel read-card" data-card="${f.key}">
    <div class="read-top"><span class="read-value">${rollSlot(f.key, fmt.format(f.value))}</span><span class="read-label" tabindex="0">${f.label}</span>
      <div class="figure-tip" role="tooltip">${f.issue ? `<p class="issue">${f.issue}</p>` : ""}<p>${f.tip}</p>${f.extra || ""}</div>
      <span class="delta" data-delta="${f.key}" hidden></span></div>
    <div class="read-note${f.tone ? ` ${f.tone}` : ""}"><span>${f.note}</span></div>
    <div class="read-viz">${viz}</div></article>`;
}

// A bar of a card, its height a share of the tallest (at least a sliver), the table (and column) it draws in data-*
const vizBar = (h, cls, attrs, i) => `<span class="b${cls ? ` ${cls}` : ""}" style="height:${Math.max(h * 100, 6).toFixed(1)}%;--i:${i}" ${attrs}></span>`;
// rows and values on a log scale: a table of 8 rows stays visible next to one of 600 000
const logShare = (n, max) => (n ? Math.log10(n + 1) / Math.log10(max + 1) : 0);

// The picture of each card, its bars in its order (cardOrders); the bars of the tables, the columns and the values
// placed by the lens (placeLens)
function cardViz(key, idx) {
  const c = idx.cards, leaving = leavingIndex(idx), tables = OV.orders[key === "columns" ? "columns" : "tables"];
  const status = Object.fromEntries(idx.tables.map((t) => [t.name, t.status]));
  if (key === "tables") {
    const max = Math.max(1, ...tables.map((t) => t.rows || 0));
    return `<div class="bars lens" data-lens="tables">${tables.map((t, i) => vizBar(Math.max(logShare(t.rows, max), 0.06),
      leaving.includes(t.name) ? "leaving" : status[t.name] === "changed" ? "changed" : t.rows ? "" : "nil", `data-table="${esc(t.name)}"`, i)).join("")}</div>`;
  }
  if (key === "columns") {
    const max = Math.max(1, ...tables.map((t) => t.kinds.length + t.hidden));
    return `<div class="bars lens" data-lens="columns">${tables.map((t, i) => `<span class="b stack" style="height:${Math.max((t.kinds.length + t.hidden) / max * 100, 6).toFixed(1)}%;--i:${i}" data-table="${esc(t.name)}">`
      + `${t.hidden ? `<i class="hid" style="flex-grow:${t.hidden}"></i>` : ""}${t.kinds.length ? `<i style="flex-grow:${t.kinds.length}"></i>` : ""}</span>`).join("")}</div>`;
  }
  if (key === "joins") return '<div class="arcs-box" data-arcs></div>';
  if (key === "samples") { const t = sampleTable(idx); return `<div class="sgrid" data-samples="${esc(t?.name || "")}">${sampleGrid(t, idx.sample_rows)}</div>`; }
  if (key === "values") {
    const cols = OV.orders.values;
    if (!cols.length) return "";
    const max = Math.max(1, ...cols.map((v) => v.count)), total = (where) => cols.filter((v) => v.where === where).reduce((n, v) => n + v.count, 0);
    const bars = cols.map((v, i) => vizBar(v.count ? logShare(v.count, max) : 0.3, v.where, `data-table="${esc(v.table)}" data-col="${esc(v.column)}"`, i)).join("");
    return `<div class="bars short lens" data-lens="values">${bars}</div><div class="viz-keys"><span data-kind="listed"><i class="sw listed"></i>dans les fiches <b class="roll">${fmt.format(total("listed"))}</b></span>`
      + `<span data-kind="searched"><i class="sw searched"></i>cherchées <b class="roll">${fmt.format(total("searched"))}</b></span></div>`;
  }
  // the metrics, the refused ones first, all of them: more than the card holds scroll (wireMetricList)
  const list = [...c.metrics].sort((a, b) => b.refused - a.refused);
  return `<ul class="mlist">${list.map((m) => `<li data-metric="${esc(m.name)}"><span class="m-dot${m.refused ? " bad" : ""}"></span>`
    + `<span class="nm">${esc(m.name)}</span><span class="tb">${esc(m.tables.join(", "))}</span></li>`).join("")}</ul>`;
}

// The table whose sample rows the card draws when no table is under the pointer: the largest one with a few columns
function sampleTable(idx) {
  const tables = idx.cards?.tables || [];
  return [...tables].filter((t) => t.kinds.length >= 6 && t.rows).sort((a, b) => b.rows - a.rows)[0]
    || [...tables].sort((a, b) => b.kinds.length - a.kinds.length)[0];
}

// Its header then its sample rows, a cell per column shown (a text wider than a date, wider than a number), ten at
// most: a wide table says how many more
const SAMPLE_CELLS = 10;
function sampleGrid(table, rows) {
  if (!table) return "";
  const more = table.kinds.length - SAMPLE_CELLS;
  const cells = [...table.kinds.slice(0, SAMPLE_CELLS)].map((k) => `<i style="flex-grow:${{ n: 1, d: 1.6, s: 2.2 }[k]}"></i>`).join("");
  return `<div class="srow hdr">${cells}${more > 0 ? `<span class="srow-more">+${fmt.format(more)}</span>` : ""}</div>`
    + `<div class="srow">${cells}${more > 0 ? '<span class="srow-more"></span>' : ""}</div>`.repeat(Math.max(rows, 0));
}

// ── The lens of the bars, a Dock (Furnas 1986): at rest past 40 bars a silhouette of one piece, the bars touching;
// the pointer near, the bars under it widen and part, the far ones close up, the total width the same. The table
// under the pointer is found in every card, the lens opening there in each (OV.lens.at: its place per card) ──

// The place of n bars in W px, the lens at c (an index, the bars' middles at 0, 1, 2…) with a strength g (0 at rest,
// 1 open). As a fisheye menu (Bederson, 2000): each bar near the pointer gains the same pixels on both sides of it,
// up to T px for the one under it (16, or 1.8 times its width at rest when that is already wide), e^(−(i−c)²/2σ²)
// of it further; what they gain, each side of the pointer takes from its own bars at rest, so the point under the
// pointer stays under it and the ends stay at the ends (near an end, the lens moves in a little instead; it spreads
// over a third of the card at most, so that elsewhere both sides hold their half). Each bar
// gives up half of the gap on each of its sides; extra: a gap more after a bar (the next table in the values), kept
// out of the silhouette
function lensLayout(n, W, c, g, extra = []) {
  const rest = n > 40 ? 0 : 2, base = (W - (n - 1) * rest) / Math.max(n, 1), least = 0.6;
  const top = Math.max(0, Math.max(16, base * 1.8) - base) + (2 - rest);
  // the spread of the lens: wider with more bars, yet what it gains (top·σ·√2π) no more than a third of the card,
  // so that a side has room for its half
  const sigma = Math.max(1.5, Math.min(n / 20, (0.35 * W) / ((top || 1) * 2.5)));
  const s = Array.from({ length: n }, (_, i) => Math.exp(-((i - c) ** 2) / (2 * sigma * sigma)) * g);
  const gain = s.map((v) => top * v);
  const u = Math.max(1e-6, Math.min(n - 1e-6, c + 0.5)), i0 = Math.floor(u), f0 = u - i0;
  // what each side gains; near an end, its side too short, the lens moves in a little (as the Dock's ends move out)
  // rather than squeezing: the bars it widens still cover the pointer
  let left = gain.slice(0, i0).reduce((a, b) => a + b, 0) + f0 * gain[i0];
  let right = gain.slice(i0 + 1).reduce((a, b) => a + b, 0) + (1 - f0) * gain[i0];
  const needL = least * u + left, needR = least * (n - u) + right, fit = Math.min(1, Math.max(0, W - least * n) / ((left + right) || 1));
  left *= fit; right *= fit;
  const P = fit < 1 ? least * u + left : Math.min(Math.max((u / n) * W, needL), W - needR);
  const rL = (P - left) / u, rR = (W - P - right) / (n - u);
  const slot = gain.map((gn, i) => (i < i0 ? rL + gn * fit : i > i0 ? rR + gn * fit : f0 * (rL + gn * fit) + (1 - f0) * (rR + gn * fit)));
  const gap = (i) => (i < 0 || i >= n - 1 ? 0 : rest + (2 - rest) * s[i] + (rest ? extra[i] || 0 : 0));
  const x = [], w = [];
  slot.reduce((at, sl, i) => {
    x.push(at + gap(i - 1) / 2);
    w.push(Math.max(sl - (gap(i - 1) + gap(i)) / 2, 0.5));
    return at + sl;
  }, 0);
  return { x, w, s, rest };
}
// The order of each card: the tables as the joins put them (the most joined in the middle) while they are few; past
// 40, each card by its own size, largest first (a curve, rank by size, instead of a ridge of heights at random): the
// tables by rows, the tables by columns, the categorical columns by values. The joins keep the order of the tables
const SORTED = 40;
function cardOrders(idx) {
  const tables = idx.cards.tables, by = (list, size) => (list.length > SORTED ? [...list].sort((a, b) => size(b) - size(a)) : list);
  const values = [], middle = [];   // in the order of the tables, where the columns of each table sit (their middle)
  tables.forEach((t) => {
    const start = values.length;
    t.values.forEach(([column, count, where]) => values.push({ table: t.name, column, count, where }));
    middle.push(values.length > start ? (start + values.length - 1) / 2 : start - 0.5);
  });
  const sortedValues = values.length > SORTED;
  return { tables: by(tables, (t) => t.rows ?? -1), columns: by(tables, (t) => t.kinds.length + t.hidden), values: by(values, (v) => v.count), joins: tables, middle, sortedValues };
}

const lerpAt = (list, c) => {
  const a = Math.max(0, Math.min(list.length - 1, Math.floor(c))), b = Math.min(list.length - 1, a + 1);
  return list[a] + (list[b] - list[a]) * Math.max(0, c - a);
};

// Where a card opens its lens for a table, c its place in the order of the tables (between two as the pointer moves):
// a card in that order at the same place, the values in it around the columns of the tables there; a sorted card
// at the table, found by its name (in the values, its largest column); null when it has none of it
function placeIn(key, table, c) {
  const o = OV.orders;
  if (o[key] === o.joins) return c;
  if (key === "values" && !o.sortedValues) return o.values.length ? lerpAt(o.middle, c) : null;
  const at = o[key].findIndex((x) => (x.table ?? x.name) === table);
  return at < 0 ? null : at;
}

// The bars of the three cards and the dots of the joins where the lens puts them, each card around its own place for
// the table followed
function placeLens(animateArcs = false) {
  const idx = ui.idx, root = document.querySelector(".overview");
  if (!root || !idx?.cards) return;
  const { at, g } = OV.lens, values = OV.orders.values;
  const place = (key, extra) => {
    const box = root.querySelector(`[data-lens="${key}"]`);
    if (!box) return;
    const L = lensLayout(OV.orders[key].length, box.clientWidth, at[key] ?? 0, g, extra);
    box.classList.toggle("sil", L.rest === 0);
    [...box.children].forEach((el, i) => {
      el.style.left = `${L.x[i].toFixed(2)}px`;
      el.style.width = `${Math.max(L.w[i], 0.5).toFixed(2)}px`;
      el.style.setProperty("--r", `${(L.rest ? 2 : 2 * L.s[i]).toFixed(1)}px`);
    });
  };
  place("tables");
  place("columns");
  // in the order of the tables, a gap more between the columns of two tables
  if (values.length) place("values", OV.orders.sortedValues ? [] : values.map((v, j) => (values[j + 1] && values[j + 1].table !== v.table ? 2 : 0)));
  drawArcs(idx, animateArcs);
}

// The lens follows the pointer in a short ease, each card toward its place, opens and closes; still without movement
const LENS_CARDS = ["tables", "columns", "values", "joins"];
function runLens() {
  cancelAnimationFrame(OV.lens.raf);
  const step = () => {
    const l = OV.lens, k = stillPage() ? 1 : 0.3, kg = stillPage() ? 1 : 0.18;
    let moving = Math.abs(l.gTo - l.g) > 0.005;
    l.g += (l.gTo - l.g) * kg;
    for (const key of LENS_CARDS) {
      if (l.to[key] == null) continue;
      l.at[key] = (l.at[key] ?? l.to[key]) + (l.to[key] - (l.at[key] ?? l.to[key])) * k;
      if (Math.abs(l.to[key] - l.at[key]) > 0.01) moving = true;
    }
    if (!moving) { l.g = l.gTo; Object.assign(l.at, l.to); }
    placeLens();
    if (moving) l.raf = requestAnimationFrame(step);
  };
  OV.lens.raf = requestAnimationFrame(step);
}

// The pointer over a card of bars or over the joins: where it is in the card at rest (as the Dock does, not in the
// bars it moves) gives the bar followed and the card's lens there; the other cards open theirs on the same table
function aimLens(e, box) {
  const key = box.dataset.lens || "joins", o = OV.orders, list = o[key], r = box.getBoundingClientRect();
  const f = Math.max(0, Math.min(0.9999, (e.clientX - r.left) / r.width)), u = f * list.length - 0.5;
  const item = list[Math.max(0, Math.min(list.length - 1, Math.round(u)))], table = item.table ?? item.name;
  const focus = key === "values" ? { tables: [table], col: item.column } : { tables: [table] };
  const c = list === o.joins ? u : o.joins.findIndex((t) => t.name === table), l = OV.lens;
  for (const other of LENS_CARDS) {
    const at = other === key ? u : placeIn(other, table, c);
    if (at != null) l.to[other] = at;
  }
  if (l.g < 0.01) Object.assign(l.at, l.to);   // opening: the lens starts where the pointer is
  l.gTo = 1;
  if (JSON.stringify(focus) !== JSON.stringify(OV.focus)) { OV.focus = focus; applyOvFocus(); }
  runLens();
}

// The joins as arcs over the line of tables, in the order of the other cards; a table without a join a hollow dot.
// Each arc as high as the root of its span at rest (the joins of a hub nest, the near ones low, the far ones high),
// lighter when they are many; the dot of a table larger the more it is joined. The dots sit under the bars of the
// tables as the lens places them (only the feet of the arcs move); those of the tables followed in the accent
function drawArcs(idx, animate) {
  const box = document.querySelector("[data-arcs]");
  if (!box || !idx.cards) return;
  const tables = idx.cards.tables.map((t) => t.name), joins = idx.cards.joins, W = box.clientWidth, H = box.clientHeight || 72, y = H - 5;
  const L = lensLayout(tables.length, W, OV.lens.at.joins ?? 0, OV.lens.g), rest = lensLayout(tables.length, W, 0, 0);
  const at = new Map(tables.map((t, i) => [t, i])), x = (i) => L.x[i] + L.w[i] / 2, x0 = (i) => rest.x[i] + rest.w[i] / 2, degree = new Map();
  joins.forEach(([a, b]) => new Set([a, b]).forEach((t) => degree.set(t, (degree.get(t) || 0) + 1)));
  const span = Math.max(1, ...joins.map(([a, b]) => Math.abs(x0(at.get(a) ?? 0) - x0(at.get(b) ?? 0))));
  const set = new Set(OV.focus?.tables || []), near = new Set();
  const paths = joins.map(([a, b], k) => {
    const ia = at.get(a), ib = at.get(b);
    if (ia == null || ib == null) return "";
    const xa = x(ia), xb = x(ib), on = set.has(a) || set.has(b);
    if (on) near.add(a).add(b);
    const h = (6 + (H - 18) * Math.sqrt(Math.abs(x0(ib) - x0(ia)) / span)) / 0.75;   // a cubic curve peaks at 3/4 of its handles
    const d = a === b ? `M${xa - 3},${y - 1} C${xa - 10},${y - 20} ${xa + 10},${y - 20} ${xa + 3},${y - 1}` : `M${xa},${y} C${xa},${y - h} ${xb},${y - h} ${xb},${y}`;
    return `<path pathLength="1" class="${animate ? "draw" : ""}${on ? " on" : ""}" data-s="${esc(a)}" data-t="${esc(b)}" d="${d}" style="--i:${k}"/>`;
  }).join("");
  const dots = tables.map((t, i) => `<circle class="${degree.has(t) ? "" : "lone"}${set.has(t) ? " on" : near.has(t) ? " near" : ""}" cx="${x(i).toFixed(2)}" cy="${y}"`
    + ` r="${(2.6 + Math.min(3, (degree.get(t) || 0) / 8) + 1.5 * L.s[i]).toFixed(1)}" data-table="${esc(t)}"/>`).join("");
  // many joins (a hub): finer and lighter, a veil that lets the others show
  const many = joins.length > SORTED, alpha = many ? Math.min(1, Math.max(0.12, 8 / joins.length)) : Math.min(1, Math.max(0.3, 16 / Math.max(joins.length, 1)));
  box.innerHTML = `<svg class="arcs" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="--arc-alpha:${alpha.toFixed(2)};--arc-width:${many ? 0.9 : 1.2}" aria-hidden="true">`
    + `${paths}${dots}</svg>`;
}

const LEVELS = [["simple", "Simples"], ["medium", "Moyennes"], ["hard", "Difficiles"], ["other", "Autres"]];
const LEVEL = { simple: "Simple", medium: "Moyenne", hard: "Difficile" };
const QUESTION_HINT = "Survole une case : la question, et ses tables s'allument dans les cartes en dessous.";

// The last eval: the share of exact answers, its questions by difficulty (green exact, red wrong, hollow not scored),
// the score of each eval over time. Whether it still measures this index is said by the Mise en place
function evalPanel(idx, db) {
  const head = (extra) => `<div class="detail-head"><h2>Évaluation</h2>${extra}</div>`;
  if (!idx.dataset) {
    return `<section class="panel eval-panel">${head("")}<p class="panel-text">Pas de questions pour cette base : `
      + `<span class="mono">offline/eval/datasets/${esc(db.toLowerCase())}.yaml</span>.</p></section>`;
  }
  const e = idx.eval;
  if (!e) return `<section class="panel eval-panel">${head(`<span class="muted-text">${plural(idx.questions, "question")}</span>`)}<p class="panel-text">Jamais évaluée.</p></section>`;
  const unscored = e.questions - e.scored;
  const count = unscored ? `<span class="muted-text" title="${plural(unscored, "question")} sans lignes attendues : l'agent y répond, rien ne le note.">`
    + `${plural(e.questions, "question")}, ${fmt.format(e.scored)} notée${e.scored > 1 ? "s" : ""}</span>` : `<span class="muted-text">${plural(e.questions, "question")}</span>`;
  const link = `<span class="grow"></span><a class="btn ghost small" href="/eval/${encodeURIComponent(db)}" target="_blank" rel="noopener">Ouvrir le tableau ↗</a>`;
  const results = (e.results || []).map((r, i) => ({ ...r, i, level: LEVEL[r.d] ? r.d : "other" }));
  const cell = (r) => `<span class="cell-q ${r.x === 1 ? "exact" : r.x === 0 ? "wrong" : "unscored"}" data-q="${r.i}" style="--i:${r.i}"></span>`;
  const levels = LEVELS.map(([key, label]) => {
    const rs = results.filter((r) => r.level === key), scored = rs.filter((r) => r.x != null), exact = scored.filter((r) => r.x === 1).length;
    return rs.length ? `<span class="lvl-head"><span class="lvl-name">${label}</span>`
      + `<span class="lvl-ratio${exact < scored.length ? " miss" : ""}">${scored.length ? `${exact}/${scored.length}` : "—"}</span></span>`
      + `<span class="cells">${rs.map(cell).join("")}</span>` : "";
  }).join("");
  const trend = (e.history || []).length > 1
    ? `<div class="ev-trend-head"><b>Au fil des évaluations</b><span class="grow"></span><span class="lg"><i class="lg-line"></i>score</span>`
      + `<span class="lg"><i class="lg-idx"></i>réindexation</span></div><div class="ev-chart" id="ev-chart"></div>`
    : '<p class="ev-trend-empty">Une seule évaluation pour l\'instant : la courbe du score viendra avec les suivantes.</p>';
  return `<section class="panel eval-panel">${head(count + link)}
    <div class="ev-body"><div class="ev-main">
        <div class="ev-score" id="ev-score">${e.scored ? `${rollSlot("score", fmt.format(Math.round(e.exact / e.scored * 100)))}<small>%</small>` : "—"}</div>
        <div class="ev-sub">réponses exactes</div><div class="ev-when">Mesurée le ${when(e.measured_at)}</div>
        ${levels ? `<div class="ev-levels">${levels}</div>` : ""}</div>
      <div class="ev-trend">${trend}</div></div>
    ${results.length ? `<div class="ev-question" id="ev-question">${QUESTION_HINT}</div>` : ""}</section>`;
}

const shortDay = (seconds) => new Date(seconds * 1000).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" });

// The score of each eval, evenly spaced (runs months apart stay readable), the indexes between them as diamonds placed
// by date, the last point in the accent; as high as the column of the score beside it, its dates on the line of its
// last questions. mode: "draw" the line from the left (another database), "extend" its last segment (a new eval),
// "mark" the last diamond (a new index), "" still
function drawEvalChart(idx, mode) {
  const box = $("ev-chart");
  if (!box) return;
  const points = idx.eval.history, W = Math.max(box.clientWidth, 200), H = Math.max(box.clientHeight, 168), L = 40, R = 14, T = 10, B = 24;
  const builds = [...(idx.builds || []), ...(idx.built_at ? [idx.built_at] : [])].filter((t) => t >= points[0].at);
  const after = builds.filter((t) => t >= points.at(-1).at);
  const step = (W - L - R) / Math.max(points.length - 1 + (after.length ? 0.6 : 0), 1), X = (i) => L + i * step;
  const xAt = (t) => {
    for (let k = 0; k < points.length - 1; k++) if (t < points[k + 1].at) return X(k) + (t - points[k].at) / (points[k + 1].at - points[k].at) * step;
    return X(points.length - 1) + step * 0.6 * (after.indexOf(t) + 1) / (after.length + 0.5);   // after the last eval, in its slot
  };
  const pcts = points.map((p) => p.exact / p.scored * 100), lo = Math.min(70, Math.floor((Math.min(...pcts) - 4) / 10) * 10);
  const Y = (v) => T + (100 - v) / (100 - lo) * (H - T - B), base = H - B, every = 100 - lo > 50 ? 20 : 10;
  const pts = points.map((p, i) => [X(i), Y(pcts[i])]);
  const line = (list) => list.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const grid = [];
  for (let v = lo; v <= 100; v += every) grid.push(`<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 8}" y="${Y(v) + 3.5}" text-anchor="end">${v} %</text>`);
  const markers = builds.map((t, i) => {
    const x = xAt(t), fresh = mode === "mark" && i === builds.length - 1;
    return `<g class="c-idx${fresh ? " fade" : ""}"><line x1="${x}" x2="${x}" y1="${T}" y2="${base}"/>`
      + `<rect class="${fresh ? "pop" : ""}" x="${x - 3.5}" y="${base - 3.5}" width="7" height="7" transform="rotate(45 ${x} ${base})"/>`
      + `<circle class="c-hit" cx="${x}" cy="${base}" r="9" data-ix="${i}"/></g>`;
  }).join("");
  const label = Math.ceil(points.length / 8);
  const axis = points.map((p, i) => ((points.length - 1 - i) % label ? "" : `<text x="${pts[i][0]}" y="${H - 4}" text-anchor="middle">${shortDay(p.at)}</text>`)).join("");
  const head = mode === "extend" ? pts.slice(0, -1) : pts, tail = mode === "extend" ? pts.slice(-2) : null;
  const dots = pts.map(([x, y], i) => {
    const last = i === pts.length - 1, pop = last && (mode === "draw" || mode === "extend");
    const delay = mode === "draw" ? 200 + i * 120 : mode === "extend" && last ? 700 : 0;
    return `${last ? `<circle class="c-halo${pop ? " pop" : ""}" cx="${x}" cy="${y}" r="9" style="animation-delay:${delay}ms"/>` : ""}`
      + `<circle class="c-dot${last ? " last" : ""}${pop ? " pop" : mode === "draw" ? " fade" : ""}" cx="${x}" cy="${y}" r="${last ? 4.5 : 3.5}" style="animation-delay:${delay}ms"/>`
      + `<circle class="c-hit" cx="${x}" cy="${y}" r="12" data-ev="${i}"/>`;
  }).join("");
  box.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Score des évaluations successives">
    <defs><linearGradient id="ev-area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity="0.16"/>
      <stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>
    <g class="c-grid">${grid.join("")}</g>${markers}
    <path class="c-area${mode === "draw" || mode === "extend" ? " fade" : ""}" d="${line(pts)}L${pts.at(-1)[0]},${base}L${pts[0][0]},${base}Z"/>
    <path class="c-line${mode === "draw" ? " draw" : ""}" pathLength="1" d="${line(head)}"/>${tail ? `<path class="c-line draw" pathLength="1" d="${line(tail)}"/>` : ""}
    <g class="c-axis">${axis}</g>${dots}</svg><div class="chart-tip" hidden></div>`;
  box.onpointermove = (ev) => {
    const hit = ev.target.closest(".c-hit"), tip = box.querySelector(".chart-tip");
    if (!hit) { tip.hidden = true; return; }
    if (hit.dataset.ev != null) {
      const p = points[Number(hit.dataset.ev)];
      tip.innerHTML = `<b>${fmt.format(Math.round(p.exact / p.scored * 100))} %</b> · ${fmt.format(p.exact)} sur ${fmt.format(p.scored)} · le ${when(p.at)}`;
    } else tip.innerHTML = `Index reconstruit le <b>${when(builds[Number(hit.dataset.ix)])}</b>`;
    tip.hidden = false;
    const r = hit.getBoundingClientRect(), b = box.getBoundingClientRect();
    tip.style.left = `${Math.min(r.left - b.left + r.width / 2 + 10, b.width - tip.offsetWidth)}px`;
    tip.style.top = `${r.top - b.top - 30}px`;
  };
  box.onpointerleave = () => { box.querySelector(".chart-tip").hidden = true; };
}

// While the agent is evaluated on this database: its questions fill in as they are asked, the one asked now pulses
// and the score waits faded; the answers land once the eval ends (the page drawn again)
function evalProgress() {
  const levels = document.querySelector(".ev-levels"), job = ui.job;
  if (!levels) return;
  const running = Boolean(job?.running && job.step === "eval" && job.db === ui.db);
  const now = running && job.progress ? job.progress[0] - 1 : -1;
  levels.classList.toggle("asking", running);
  $("ev-score")?.classList.toggle("stale", running);
  levels.querySelectorAll(".cell-q").forEach((cell) => {
    const i = Number(cell.dataset.q);
    cell.classList.toggle("asked", running && i < now);
    cell.classList.toggle("now", running && i === now);
  });
}

// The table under the pointer (a bar, a dot of the joins, a column of values), the tables of a question of the eval
// or of a metric: followed in every card
function ovPointer(e) {
  const idx = ui.idx;
  if (!idx?.cards || e.target.closest("[data-lens], [data-arcs]")) return;   // the lens follows those (aimLens)
  const q = e.target.closest(".cell-q"), met = e.target.closest("[data-metric]");
  const col = e.target.closest(".read-viz [data-col]"), table = e.target.closest(".read-viz [data-table]");
  const focus = q ? { tables: idx.eval.results[Number(q.dataset.q)].tables, q: Number(q.dataset.q) }
    : met ? { tables: idx.cards.metrics.find((m) => m.name === met.dataset.metric)?.tables || [], metric: met.dataset.metric }
    : col ? { tables: [col.dataset.table], col: col.dataset.col }
    : table ? { tables: [table.dataset.table] } : null;
  if (JSON.stringify(focus) === JSON.stringify(OV.focus)) return;
  OV.focus = focus;
  applyOvFocus();
}

function applyOvFocus() {
  const idx = ui.idx, root = document.querySelector(".overview");
  if (!root || !idx?.cards) return;
  const f = OV.focus, set = new Set(f?.tables || []);
  root.querySelectorAll(".read-viz").forEach((v) => v.classList.toggle("focusing", Boolean(f) && !v.querySelector("[data-samples]")));
  root.querySelectorAll(".read-viz [data-table]").forEach((el) => {
    el.classList.toggle("on", set.has(el.dataset.table) && (!f.col || !el.dataset.col || el.dataset.col === f.col));
  });
  drawArcs(idx, false);   // the joins of the tables followed, in the accent
  root.querySelectorAll("[data-metric]").forEach((li) => {
    const m = idx.cards.metrics.find((x) => x.name === li.dataset.metric);
    li.classList.toggle("on", Boolean(f) && (f.metric ? f.metric === m.name : m.tables.some((t) => set.has(t))));
  });
  if (f && !f.metric) revealMetric(root);
  root.querySelectorAll(".cell-q").forEach((c) => c.classList.toggle("on", f?.q === Number(c.dataset.q)));
  const one = set.size === 1 ? idx.cards.tables.find((t) => set.has(t.name)) : null, grid = root.querySelector("[data-samples]");
  const shown = one || sampleTable(idx);
  if (grid && shown && grid.dataset.samples !== shown.name) { grid.innerHTML = sampleGrid(shown, idx.sample_rows); grid.dataset.samples = shown.name; }
  const question = $("ev-question"), r = f?.q != null ? idx.eval.results[f.q] : null;
  if (question) {
    question.innerHTML = !r ? QUESTION_HINT : `<span class="pill todo">${LEVEL[r.d] || "Question"}</span><span class="q">${esc(r.q)}</span>`
      + (r.x === 1 ? '<span class="ok">exacte</span>' : r.x === 0 ? '<span class="ko">fausse</span>' : "<span>non notée</span>")
      + `<span>· ${r.latency != null ? `${String(r.latency.toFixed(1)).replace(".", ",")} s · ` : ""}${r.tables.length ? `lit ${names(r.tables)}` : "aucune table lue"}</span>`;
    question.classList.toggle("on", Boolean(r));
  }
  ovNotes(idx, f, one);
}

// ── The list of the metrics, longer than its card: still under the pointer in its middle, it scrolls when the pointer
// nears its top or bottom edge, faster nearer the edge, and with the wheel; no bar, a fade on the side that has more ──
const MLIST_EDGE = 18, MLIST_SPEED = 360, MLIST_FADE = 14;   // px from an edge where it scrolls; px/s on the edge; px
function wireMetricList(root) {
  const box = root.querySelector(".mlist");
  if (!box) return;
  let x = 0, y = 0, inside = false, raf = 0, last = 0, pos = 0;
  const fades = () => {
    box.classList.toggle("more-up", box.scrollTop > 1);
    box.classList.toggle("more-down", box.scrollTop < box.scrollHeight - box.clientHeight - 1);
  };
  // the push of the pointer: −1 on the top edge, 1 on the bottom one, 0 in the middle
  const push = () => {
    const r = box.getBoundingClientRect(), up = y - r.top, down = r.bottom - y;
    return up < MLIST_EDGE ? -(1 - up / MLIST_EDGE) : down < MLIST_EDGE ? 1 - down / MLIST_EDGE : 0;
  };
  const step = (t) => {
    const p = push(), max = box.scrollHeight - box.clientHeight;
    if (!inside || !p || (p < 0 ? box.scrollTop <= 0 : box.scrollTop >= max - 1)) { raf = 0; return; }   // max: rounded
    if (Math.abs(box.scrollTop - pos) > 2) pos = box.scrollTop;   // the wheel moved it meanwhile
    pos = Math.min(max, Math.max(0, pos + Math.sign(p) * p * p * MLIST_SPEED * Math.min(Math.max(t - last, 0), 50) / 1000));
    last = t;
    box.scrollTop = pos;
    raf = requestAnimationFrame(step);
  };
  box.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;   // a finger scrolls it itself
    x = e.clientX; y = e.clientY; inside = true;
    if (!raf && push()) { pos = box.scrollTop; last = performance.now(); raf = requestAnimationFrame(step); }
  });
  box.addEventListener("pointerleave", () => { inside = false; });
  // the list moving under a still pointer: the metric now under it followed
  box.addEventListener("scroll", () => {
    fades();
    const under = inside && document.elementFromPoint(x, y);
    if (under) ovPointer({ target: under });
  });
  fades();
}

// The first metric of the tables followed elsewhere (a bar, a join, a question of the eval) brought into the list,
// just under its top fade; left where it is when it shows already
function revealMetric(root) {
  const box = root.querySelector(".mlist"), li = box?.querySelector("li.on");
  if (!li) return;
  const top = li.offsetTop, bottom = top + li.offsetHeight;
  if (top >= box.scrollTop + MLIST_FADE && bottom <= box.scrollTop + box.clientHeight - MLIST_FADE) return;
  box.scrollTo({ top: top - MLIST_FADE, behavior: stillPage() ? "auto" : "smooth" });
}

// The legend of the values: the counts of the base, or of what the pointer follows, the kind it has none of faded
function valueKeys(values, following) {
  document.querySelectorAll('[data-card="values"] [data-kind]').forEach((el) => {
    const of = values.filter((v) => v.where === el.dataset.kind);
    rollTo(el.querySelector(".roll"), fmt.format(of.reduce((n, v) => n + v.count, 0)));
    el.classList.toggle("off", following && !of.length);
  });
}

// The word under each card: its own, or what it is for the tables under the pointer
function ovNotes(idx, f, one) {
  const note = (key, html, tone = "") => {
    const el = document.querySelector(`[data-card="${key}"] .read-note`);
    if (!el) return;
    el.className = `read-note${tone ? ` ${tone}` : ""}`;
    el.innerHTML = `<span>${html}</span>`;
  };
  if (!f) { Object.entries(OV.notes).forEach(([key, n]) => note(key, n.note, n.tone)); valueKeys(OV.orders?.values || [], false); return; }
  const set = new Set(f.tables), list = idx.cards.tables.filter((t) => set.has(t.name));
  const who = `<b>${f.metric ? esc(f.metric) : f.q != null ? "La question" : esc(f.tables.join(", "))}</b>`, lead = `${who} · `;
  const status = one && idx.tables.find((t) => t.name === one.name)?.status;
  note("tables", lead + (one ? (one.rows ? plural(one.rows, "ligne") : one.rows === 0 ? "vide" : "lignes non comptées") + (status === "changed" ? ", fiche ancienne" : "")
    : plural(list.length, "table")), "focus");
  const cols = list.reduce((n, t) => n + t.kinds.length, 0), hid = list.reduce((n, t) => n + t.hidden, 0);
  note("columns", lead + plural(cols, "colonne") + (hid ? `, ${fmt.format(hid)} cachée${hid > 1 ? "s" : ""}` : ""), "focus");
  const joins = idx.cards.joins.filter(([a, b]) => (one ? set.has(a) || set.has(b) : set.has(a) && set.has(b)));
  note("joins", lead + (joins.length ? `${plural(joins.length, "jointure")}${one ? "" : " entre elles"}` : "aucune jointure"), "focus");
  if (one) note("samples", `${lead}${fmt.format(idx.sample_rows)} × ${plural(one.kinds.length, "colonne")}`, "focus");
  else note("samples", OV.notes.samples.note, OV.notes.samples.tone);
  // over the values only the name: their counts go to the legend, said here only when there are none
  const values = list.flatMap((t) => t.values.map(([column, count, where]) => ({ column, count, where }))).filter((v) => !f.col || v.column === f.col);
  note("values", (f.col ? `<b>${esc(f.tables[0])}.${esc(f.col)}</b>` : who) + (!values.length ? " · aucune colonne catégorielle"
    : values.every((v) => v.where === "dropped") ? " · aucune valeur indexée" : ""), "focus");
  valueKeys(values, true);
  const metrics = f.metric ? idx.cards.metrics.filter((m) => m.name === f.metric) : idx.cards.metrics.filter((m) => m.tables.some((t) => set.has(t)));
  note("metrics", lead + (f.metric ? esc(metrics[0]?.tables.join(", ") || "aucune table lue")
    : metrics.length ? plural(metrics.length, "métrique") : "aucune métrique"), "focus");
}

// The Vue d'ensemble: the last eval and its trend, what the agent reads, the setup on the side (the one place that
// says what to do). Another database draws its bars, joins, line and gauge; a new index marks its diamond and says by
// how much each figure changed; a new eval extends the line and lands its answers. The Fiches: the document of a table
function renderIndex(el, message) {
  const idx = ui.idx;
  if (!idx) { el.innerHTML = '<p class="placeholder">Lecture de l\'index…</p>'; return; }
  if (ui.step === "index") {
    if (ui.current) renderDocument(el, message);
    else el.innerHTML = '<p class="placeholder">Choisis une table pour voir sa fiche, telle que l\'agent la lit.</p>';
    return;
  }
  const mode = OV.db !== ui.db ? "draw" : OV.eval !== idx.eval?.measured_at ? "extend" : OV.built !== idx.built_at ? "mark" : "";
  Object.assign(OV, { db: ui.db, eval: idx.eval?.measured_at, built: idx.built_at, focus: mode === "draw" ? null : OV.focus });
  cancelAnimationFrame(OV.lens.raf);
  OV.lens = { g: 0, gTo: 0, at: {}, to: {}, raf: 0 };
  OV.orders = idx.cards ? cardOrders(idx) : null;
  const cards = idx.trace && idx.cards, figures = cards ? agentFigures(idx) : [];
  OV.notes = Object.fromEntries(figures.map((f) => [f.key, { note: f.note, tone: f.tone }]));
  const read = cards ? CARD_GROUPS.map(([title, keys]) => `<h3 class="read-group">${title}</h3><div class="read-cards">`
      + `${keys.map((key) => readCard(figures.find((f) => f.key === key), cardViz(key, idx))).join("")}</div>`).join("")
    : `<p class="panel reads-empty">${idx.values_at ? `Index du ${when(idx.values_at)}, construit avant que le studio en garde la trace : `
      + "ses chiffres viendront à la prochaine indexation." : "Pas encore indexée : l'agent ne voit aucune table de cette base."}</p>`;
  const reads = `<section class="reads"><div class="reads-head"><h2>Ce que l'agent lit</h2><span class="grow"></span>
      ${idx.trace ? '<button class="btn ghost small" type="button" data-setup="documents">Lire les fiches →</button>' : ""}</div>
      ${message ? `<p class="note error">${esc(message)}</p>` : ""}${read}</section>`;
  const evaluated = idx.dataset && idx.eval;   // an eval to show leads the page; none, it follows what the agent reads
  el.innerHTML = `<div class="overview${mode === "draw" ? " drawing" : mode === "extend" ? " landed" : ""}">
    <div class="overview-main">${evaluated ? evalPanel(idx, ui.db) + reads : reads + evalPanel(idx, ui.db)}</div>
    <aside class="panel setup">${setupCard()}</aside>
  </div>`;
  const overview = el.querySelector(".overview");
  drawEvalChart(idx, mode);
  wireMetricList(overview);
  applyOvFocus();
  placeLens(mode === "draw");   // after the focus: its arcs drawn once, from the left on another database
  rollFigures(el);
  evalProgress();
  overview.addEventListener("pointerover", ovPointer);
  overview.querySelectorAll("[data-lens], [data-arcs]").forEach((box) => {
    box.addEventListener("pointermove", (e) => aimLens(e, box));
    box.addEventListener("pointerleave", () => { OV.lens.gTo = 0; runLens(); });   // closes where it is
  });
  overview.addEventListener("pointerleave", () => { if (OV.focus) { OV.focus = null; applyOvFocus(); } });
  // another width (the window, the side panel folding under): the chart, the bars and the joins placed again, still
  OV.resize?.disconnect();
  let width = overview.querySelector(".overview-main").clientWidth;
  OV.resize = new ResizeObserver(([entry]) => {
    if (Math.round(entry.contentRect.width) === Math.round(width) || !document.body.contains(overview)) return;
    width = entry.contentRect.width;
    drawEvalChart(ui.idx, ""); placeLens(); applyOvFocus();
  });
  OV.resize.observe(overview.querySelector(".overview-main"));
  el.querySelectorAll("[data-setup]").forEach((button) => { button.onclick = () => setupGo(button.dataset.setup); });
  $("setup-wizard")?.addEventListener("click", () => openWizard());
}

// The steps of the setup on a gauge, one segment each, coloured by its state (green done, amber to finish, red the
// agent won't start, grey to do), the number done in its middle
function setupGauge(steps) {
  const cx = 100, cy = 100, r = 86, gap = 3, span = (180 - gap * (steps.length - 1)) / steps.length;
  const p = (a) => `${(cx + r * Math.cos(a * Math.PI / 180)).toFixed(2)},${(cy - r * Math.sin(a * Math.PI / 180)).toFixed(2)}`;
  const arcs = steps.map((s, i) => {
    const a1 = 180 - i * (span + gap), d = `M${p(a1)} A${r},${r} 0 0 1 ${p(a1 - span)}`;
    return `<path class="arc-track" d="${d}"/><path class="arc-seg ${s.state}" pathLength="1" d="${d}" style="--i:${i}"/>`;
  }).join("");
  const done = steps.filter((s) => s.state === "done").length;
  return `<div class="gauge"><svg viewBox="0 0 200 108" aria-hidden="true">${arcs}</svg>
    <div class="gauge-text"><span class="gauge-big">${rollSlot("setup-done", fmt.format(done))}<small> sur ${fmt.format(steps.length)}</small></span>
      <span class="gauge-sub">étapes faites</span></div></div>`;
}

// The setup of the database on its Vue d'ensemble: its gauge, each step on a line, opening its tab (the index: the
// Fiches); the step to do now open in place, with why and the button that does it (the index and the eval too); the
// guided setup
function setupCard() {
  if (ui.data?.db !== ui.db || !ui.idx) return '<p class="placeholder">Lecture des étapes…</p>';
  const base = { summary: ui.data.summary, index: ui.idx }, steps = pipelineSteps(base), next = nextStep(base, steps);
  const dot = (p) => `<span class="dot ${p.state}">${p.state === "done" ? CHECK : p.state === "bad" ? BANG : ""}</span>`;
  const count = (p) => `<span class="setup-count ${p.state}">${esc(p.count)}</span>`;
  const row = (p) => (p.key === next?.key
    ? `<li class="setup-open"><div class="setup-line">${dot(p)}<span class="setup-name">${STAGES[p.key]}</span>${count(p)}</div>
        <p>${next.detail}</p><button class="btn primary" type="button" data-setup="${p.key}">${next.label} →</button></li>`
    : `<li><button type="button" class="setup-step" data-setup="${p.key === "index" ? "documents" : p.key}" title="${esc(p.tip)}">${dot(p)}
        <span class="setup-name">${STAGES[p.key]}</span>${count(p)}<span class="setup-go">→</span></button></li>`);
  return `<div class="setup-head"><h2 title="Ce qui prépare la base pour l'agent. Chaque étape s'ouvre dans son onglet.">Mise en place</h2></div>
    ${setupGauge(steps)}
    <ol class="setup-steps">${steps.map(row).join("")}</ol>
    <div class="setup-foot"><button class="btn" type="button" id="setup-wizard"
      title="Repasse les étapes une à une, à partir de la première qui n'est pas finie.">Lancer la mise en place guidée</button></div>`;
}

// A step of the setup: its tab, the documents in the Fiches; the index and the eval are done from here
function setupGo(key) {
  if (key === "documents") { goTo("index"); return; }
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
      d.unscoped && "Hors du périmètre : sa fiche partira à la prochaine indexation.",
      d.changes.length && `Depuis l'index : ${d.changes.map((c) => CHANGES[c.kind](c)).join(" · ")}. L'agent lit encore l'ancienne fiche.`,
      !d.described && !d.gone && !d.unscoped && !d.changes.length && "Sans description : l'agent ne lit que la structure brute. L'onglet Descriptions la décrit.",
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
  return `<p class="doc-text-note">Le texte que l'agent reçoit pour cette table, tel que l'index l'a écrit.</p>`
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
// The steps of the setup of a database, by the name of what they make
const STAGES = { scope: "Périmètre", profile: "Profilage", semantics: "Descriptions", relations: "Jointures", metrics: "Métriques", index: "Index et évaluation" };

// What the index of a database needs: "reindex" (never built, built before its trace, files changed since), "eval"
// (no score of this index while there are questions), or nothing
function indexNeed(idx) {
  if (!idx.trace || idx.counts.changed + idx.counts.missing + leavingIndex(idx).length) return "reindex";
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
  if (ui.draft) { leaveDraft(openHome); return; }
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
  $("bases-count").textContent = fmt.format(bases.length);
  $("bases").innerHTML = bases.map(baseCard).join("") || '<p class="placeholder">Aucune base dans la configuration.</p>';
}

// The steps of a database, each done, partial (to finish or to do again), todo or bad (the agent won't start). The
// Périmètre is done once chosen, every table or a list (lib/semantics.scope_origin)
function pipelineSteps(base) {
  const s = base.summary, idx = base.index, e = idx.eval, errors = idx.metrics?.errors || [], sc = idx.scope;
  const built = idx.built_at ?? idx.values_at;
  const toIndex = idx.trace ? idx.counts.changed + idx.counts.missing + leavingIndex(idx).length : 0;
  const staleEval = idx.dataset && (!e || (built != null && e.at < built));
  const undescribed = s.tables - s.described;
  const of = sc.base == null ? "" : ` sur ${fmt.format(sc.base)}`;
  return [
    { key: "scope", state: { all: "done", list: "done", none: "todo" }[sc.origin],
      count: { all: "toutes", list: `${fmt.format(sc.tables)}${sc.base == null ? "" : `/${fmt.format(sc.base)}`}`, none: "à choisir" }[sc.origin],
      tip: { all: "Toutes les tables, celles qui arriveront aussi", list: `${plural(sc.tables, "table")}${of} : une nouvelle reste dehors`,
        none: "Aucun périmètre choisi : toutes les tables, celles qui arriveront aussi" }[sc.origin] },
    { key: "profile", state: s.profiled === s.tables ? "done" : s.profiled ? "partial" : "todo",
      count: `${fmt.format(s.profiled)}/${fmt.format(s.tables)}`, tip: `${fmt.format(s.profiled)} tables profilées sur ${fmt.format(s.tables)}` },
    { key: "semantics", state: !undescribed && !s.partial && !s.rewritten && !s.unclear ? "done" : s.described ? "partial" : "todo",
      count: `${fmt.format(s.described)}/${fmt.format(s.tables)}`,
      tip: `${fmt.format(s.described)} tables décrites sur ${fmt.format(s.tables)}`
        + (s.unclear ? `, ${fmt.format(s.unclear)} à préciser` : "") + (s.rewritten ? `, ${fmt.format(s.rewritten)} à relire` : "") + (s.partial ? `, ${fmt.format(s.partial)} à compléter` : "") },
    { key: "relations", state: s.relations || s.tables < 2 ? "done" : "todo", count: s.relations ? fmt.format(s.relations) : "—",
      tip: s.relations ? `${plural(s.relations, "jointure")}${s.isolated ? `, ${plural(s.isolated, "table")} sans jointure` : ""}` : "Pas encore cherchées" },
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

// The step of the setup to do now: the first one that stops the agent, else the first one not done. What to do and why,
// short for a card of the home (title, why) and in a sentence for the Mise en place (detail), and the button that does
// it (label; action for the index and the eval); null when everything is done
function nextStep(base, steps) {
  const s = base.summary, idx = base.index;
  const step = steps.find((p) => p.state === "bad") || steps.find((p) => p.state !== "done");
  if (!step) return null;
  const next = (title, why, label, { action = "", detail = `${title} : ${why}` } = {}) => ({ key: step.key, title, why, detail, label, action });
  if (step.key === "scope") {
    return next("Choisir le périmètre", "toutes les tables, celles qui arriveront aussi, ou une liste.", "Ouvrir le périmètre",
      { detail: "Aucun périmètre choisi : l'agent voit toutes les tables de la base, celles qui y arriveront aussi." });
  }
  if (step.key === "profile") {
    return next(`Profiler ${plural(s.tables - s.profiled, "table")}`, "le studio ne sait pas encore ce qu'elles contiennent.", "Ouvrir le profilage");
  }
  if (step.key === "semantics") {
    const undescribed = s.tables - s.described;
    return undescribed ? next(`Décrire ${plural(undescribed, "table")}`, "l'agent n'en aurait que la structure brute.", "Ouvrir les descriptions")
      : s.unclear ? next(`Préciser ${plural(s.unclear, "table")}`, "l'IA n'a pas su dire le sens ou l'unité de certaines colonnes.", "Ouvrir les descriptions")
      : s.rewritten ? next(`Relire ${plural(s.rewritten, "table")}`, "l'IA a réécrit des descriptions.", "Ouvrir les descriptions")
      : next(`Compléter ${plural(s.partial, "table")}`, "des colonnes sont arrivées depuis leur description.", "Ouvrir les descriptions");
  }
  if (step.key === "relations") return next("Chercher les jointures", "l'agent ne sait pas encore passer d'une table à l'autre.", "Ouvrir les jointures");
  if (step.key === "metrics") {
    const errors = idx.metrics?.errors || [];
    return errors.length
      ? next(`Corriger ${names(errors.map((m) => m.name))}`, "l'agent refuse de démarrer tant qu'elle lit une colonne absente des fiches.", "Ouvrir les métriques",
        { detail: `L'agent ne démarrera pas : il refuse ${errors.map((e) => `<span class="mono">${esc(e.name)}</span> (${esc(inFrench(e.detail))})`).join(", ")}.` })
      : next("Proposer des métriques", "sans elles, l'agent refait ses calculs à chaque question.", "Ouvrir les métriques");
  }
  if (!idx.trace && !idx.values_at) return next("Indexer la base", "l'agent ne la voit pas encore.", "Indexer", { action: "reindex" });
  if (indexNeed(idx) === "reindex") {
    const toIndex = idx.trace ? idx.counts.changed + idx.counts.missing + leavingIndex(idx).length : 0;
    return next("Réindexer", toIndex ? `${plural(toIndex, "fiche")} à mettre à jour : l'agent lit encore l'ancienne version.`
      : "pour voir ce que l'agent lit, table par table.", "Réindexer", {
      action: "reindex",
      detail: toIndex ? indexChanges(idx).join(" ")
        : "L'index a été construit avant que le studio en garde la trace : réindexe une fois pour voir ce que l'agent lit, table par table.",
    });
  }
  const e = idx.eval;
  return next("Évaluer l'agent", e ? "son score date d'avant l'index." : "il n'a jamais été évalué.", "Évaluer l'agent", {
    action: "eval",
    detail: !e ? "L'agent n'a jamais été évalué : l'évaluation lui pose les questions de la base."
      : `${e.scored ? `Le score de ${pct(e.exact / e.scored)}` : "La dernière évaluation"} date d'avant l'index du ${when(idx.built_at ?? idx.values_at)} : `
        + "relance l'évaluation pour le confirmer.",
  });
}

// The next step on a card of the home, in words, then the base to open and the button that does it: its tab, the index
// or the eval on the Vue d'ensemble, or the guided setup while the database is being prepared (never indexed)
function nextBlock(base, steps) {
  const idx = base.index, next = nextStep(base, steps);
  const open = '<button class="btn" type="button" data-step="overview">Ouvrir la base</button>';
  if (!next) return `<p><b>Tout est à jour</b> : l'agent lit la dernière version de chaque fiche, et son score la mesure.</p>${open}`;
  const button = !idx.trace && !idx.values_at
    ? `<button class="btn primary" type="button" data-wizard="1">${steps[0].state === "todo" ? "Lancer" : "Reprendre"} la mise en place →</button>`
    : `<button class="btn primary" type="button" data-step="${next.action ? "overview" : next.key}"${next.action ? ` data-action="${next.action}"` : ""}>${next.label} →</button>`;
  return `<p><span class="lead">Prochaine étape · </span><b>${next.title}</b> : ${next.why}</p>${open}${button}`;
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
      mini(s.relations ? fmt.format(s.relations) : "aucune", many(s.relations, "jointure", "jointures")),
      mini(s.metrics ? fmt.format(s.metrics) : "aucune", many(s.metrics, "métrique", "métriques")),
    ];
  }
  const raw = idx.raw.filter((t) => !leavingIndex(idx).includes(t)).length, hidden = idx.hidden.length, m = idx.metrics;
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
  const idx = base.index, s = base.summary, steps = pipelineSteps(base);
  const indexed = idx.trace ? `indexée le ${when(idx.built_at)}` : idx.values_at ? `index du ${when(idx.values_at)}` : "jamais indexée";
  const verdict = baseVerdict(idx, s, base.name);
  return `<article class="base" data-db="${esc(base.name)}">
    <div class="base-head"><h3>${esc(base.name)}</h3><span class="pill kept">${esc(DIALECTS[base.dialect] || base.dialect)}</span>
      <span class="figures">${indexed}</span></div>
    ${verdict}
    <div class="base-figures">${scoreBlock(idx)}<div class="minis">${minis(base).join("")}</div></div>
    <div class="pipeline"><span class="pipeline-title">Pipeline</span><ol class="pipe">${steps.map(pipeStep).join("")}</ol></div>
    <div class="base-next">${nextBlock(base, steps)}</div>
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
    : `<div class="results-head">${plural(found.total, "résultat")} dans ${plural(found.bases, "base")}</div>`
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

// ── The guided setup: the steps of a database one after the other, opened on purpose ──

const GUIDE = {
  scope: { title: "Choisir les tables", text: "Les tables de la base que l'agent et le pipeline voient : toutes, celles qui arriveront aussi, ou une liste. Une table hors de la liste n'est jamais profilée, décrite ni indexée, et une requête de l'agent qui la lit est bloquée." },
  profile: { title: "Profiler les tables", text: STEPS.profile.text },
  semantics: { title: "Décrire les tables", text: `L'agent choisit ses tables et écrit ses requêtes grâce à ces descriptions, ${WRITTEN} par l'IA et à relire ensuite dans l'onglet Descriptions.` },
  relations: { title: "Relier les tables", text: STEPS.relations.text },
  metrics: { title: "Proposer des métriques", text: `L'agent reprend telles quelles ces formules, proposées par l'IA et ${TESTED} sur les données. Une base peut s'en passer : l'agent calcule alors lui-même.` },
  index: { title: "Indexer et évaluer", text: "L'index écrit la fiche de chaque table, ce que l'agent lit vraiment, et les valeurs des colonnes catégorielles lues dans la base. L'évaluation pose ensuite les questions de test à l'agent et compare ses réponses aux lignes attendues." },
};
const GUIDE_STEPS = ["scope", "profile", "semantics", "relations", "metrics", "index"];

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

// The steps on the left (done or not), the current one on the right: what it does, how far it went, its
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
  const action = (label, job, note = "", primary = true) => `<button class="btn${primary ? " primary" : ""}" type="button" data-guide-run="${job}" data-job>`
    + `${["semantics", "relations", "metrics"].includes(job) ? AI_STAR : ""}${label}</button>`
    + (note ? `<span class="muted-text">${note}</span>` : "");
  const list = (title, items) => !items.length ? "" : `<div class="guide-list"><div class="guide-list-head">${title} · ${fmt.format(items.length)}</div>`
    + items.map((it) => `<div class="guide-row"><span class="mono">${esc(it.name)}</span><span class="figures">${esc(it.size)}</span>${it.pill}</div>`).join("") + "</div>";
  const done = p.state === "done" ? '<p class="guide-done">Étape terminée : tu peux passer à la suivante.</p>' : "";
  let body = "";
  if (key === "scope") {
    // The choice itself, every table or a list; the tables of a list are chosen on the Périmètre
    const sc = idx.scope, off = sc.base == null ? ' disabled title="La base ne répond pas"' : "";
    const choice = (mode, label, primary) => `<button class="btn${primary ? " primary" : ""}" type="button" data-guide-run="scope-${mode}"${off}>${label}</button>`;
    const said = {
      none: "Aucun périmètre choisi : l'agent voit toutes les tables de la base, même celles qui y arriveront.",
      all: `Toutes les tables de la base${sc.base == null ? "" : ` (${fmt.format(sc.base)})`}, celles qui y arriveront aussi.`,
      list: `<b>${fmt.format(sc.tables)}</b> ${sc.tables > 1 ? "tables" : "table"} dans la liste${sc.base == null ? "" : ` sur ${fmt.format(sc.base)}`} : une table qui arrive dans la base reste dehors.`,
    }[sc.origin];
    const choices = {
      none: choice("all", "Toutes les tables, même les futures", true) + choice("list", "Une liste de tables", false)
        + '<span class="muted-text">La liste part des tables d\'aujourd\'hui : tu retires ensuite les autres dans le Périmètre.</span>',
      all: choice("list", "Passer à une liste de tables", false),
      list: "",
    }[sc.origin];
    body = `<div class="guide-progress"><div class="guide-count"><span>${said}</span></div></div>` + done
      + (choices ? `<div class="guide-action">${choices}</div>` : "");
  } else if (key === "profile") {
    const missing = rows.filter((t) => t.status.profile === "none"), n = missing.length;
    body = progress(s.profiled, s.tables, "tables profilées") + done
      + (n ? `<div class="guide-action">${action(n > 1 ? `Profiler les ${fmt.format(n)} manquantes` : "Profiler la manquante", "profile")}</div>` : "")
      + list("À profiler", missing.map((t) => ({ name: t.name, size: size(t), pill: '<span class="pill todo">à profiler</span>' })));
  } else if (key === "semantics") {
    const todo = rows.filter((t) => t.status.semantics !== "done"), n = undescribed(s);
    body = progress(s.described - s.partial, s.tables, "tables décrites") + done
      + (n ? `<div class="guide-action">${action(n > 1 ? `Décrire les ${fmt.format(n)} manquantes` : "Décrire la manquante", "semantics",
        "Quelques secondes d'IA par table. Les descriptions corrigées par toi ne sont pas touchées.")}</div>` : "")
      + (s.unclear ? `<p class="guide-note">${plural(s.unclear, "table")} ${s.unclear > 1 ? "ont" : "a"} des colonnes dont l'IA n'a pas su dire le sens ou l'unité, à préciser dans l'onglet Descriptions.</p>` : "")
      + (s.rewritten ? `<p class="guide-note">${plural(s.rewritten, "table")} ${s.rewritten > 1 ? "ont" : "a"} des descriptions réécrites par l'IA, à relire dans l'onglet Descriptions.</p>` : "")
      + list("À décrire", todo.map((t) => ({ name: t.name, size: size(t),
        pill: t.status.semantics === "none" ? '<span class="pill todo">à décrire</span>' : '<span class="pill decide">à compléter</span>' })));
  } else if (key === "relations") {
    const isolated = rows.filter((t) => t.status.relations !== "done");
    body = progress(s.tables - s.isolated, s.tables, "tables reliées") + done
      + `<div class="guide-action">${action("Chercher les jointures", "relations", "L'IA propose les jointures, chacune est mesurée sur les données.", p.state !== "done")}</div>`
      + list("Sans jointure", isolated.map((t) => ({ name: t.name, size: size(t), pill: '<span class="pill decide">isolée</span>' })));
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
  return `<div class="guide-head"><span class="eyebrow">Étape ${GUIDE_STEPS.indexOf(key) + 1} sur ${GUIDE_STEPS.length}</span><h1>${GUIDE[key].title}</h1>`
    + `<p>${GUIDE[key].text}</p></div>${body}`;
}

// The way to the step before, the tab where this one is reviewed, the next step (skipped while not done) or the end
function guideFoot(at, p) {
  const prev = GUIDE_STEPS[at - 1], next = GUIDE_STEPS[at + 1], tab = GUIDE_STEPS[at] === "index" ? "overview" : GUIDE_STEPS[at];
  return `<div class="guide-foot">
    ${prev ? `<button class="btn" type="button" data-guide="${prev}">← ${STAGES[prev]}</button>` : ""}
    <button class="btn ghost" type="button" data-tab="${tab}">${tab === "scope" ? "Choisir dans le Périmètre" : `Voir dans l'onglet ${STEPS[tab].title}`}</button>
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
  if (job === "scope-all") changeScope({ all: true });
  else if (job === "scope-list") changeScope({ all: false });
  else if (job === "profile") runJob("profile", { missing: true });
  else if (job === "semantics") runJob("semantics", {});
  else if (job === "relations") runJob("relations/run", {});
  else if (job === "metrics") runJob("metrics/run", {});
  else if (job === "index") runJob("index/run", {});
  else if (job === "eval") confirmEval();
}

// ── Jobs ─────────────────────────────────────────────────

async function runJob(step, request) {
  // a job reads or rewrites the files a draft would change: the draft is saved or dropped first
  if (ui.draft) { leaveDraft(() => runJob(step, request)); return; }
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
  if (ui.step === "scope") changeScope({ add: scopeSelection().add });
  else if (ui.step === "metrics") runJob("metrics/run", {});
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
  if (ui.step === "scope") confirmRemove(scopeSelection().remove);
  else if (ui.step === "metrics") {
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
  if (job.step === "eval" && job.db === ui.db && ui.step === "overview") evalProgress();   // the question asked now
  if (progressed && job.db === ui.db) {
    await loadTables();
    // Only the table the job wrote gets its detail reloaded: a description being typed in another one stays
    const wasDone = before?.id === job.id && before.done.includes(ui.current);
    const ended = before?.id === job.id && before.running && !job.running;
    if (ui.current && job.step === ui.step) {
      if (job.step === "profile" && job.done.includes(ui.current) && !wasDone) openTable(ui.current, { again: true });
      if (job.step === "semantics" && ended && job.started_tables.includes(ui.current)) openTable(ui.current, { again: true });
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
    // The Périmètre tells what each table has: profiled, described, indexed
    if (ended && ui.step === "scope") await loadScope();
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
  // what an incremental index spared: the values of the tables whose data did not change, the documents left as they were
  const spared = [job.values_kept ? `valeurs reprises pour ${plural(job.values_kept, "table")} (données inchangées)` : null,
    job.embedded ? (job.embedded[0] ? `${plural(job.embedded[0], "fiche")} sur ${fmt.format(job.embedded[1])} ${job.embedded[0] > 1 ? "recalculées" : "recalculée"}`
      : "aucune fiche recalculée (textes inchangés)") : null]
    .filter(Boolean).map((s) => ` · ${s}`).join("");
  return [`Index terminé${counts}${spared} · l'agent le lira à son prochain démarrage`, ""];
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
  const unclear = job.unclear ? ` · <b>${plural(job.unclear, "colonne")} à préciser</b>` : "";
  if (job.command.includes("--fresh")) {
    // Described again: what the LLM changed, to read again, and the corrections made by hand it kept
    const which = sent > 2 ? plural(sent, "table") : job.started_tables.map(esc).join(", ");
    const changed = job.rewritten ? `${plural(job.rewritten, "description")} ${job.rewritten > 1 ? "réécrites" : "réécrite"}, à relire`
      : "aucune description changée";
    const kept = job.kept ? ` · ${job.kept > 1 ? `tes ${fmt.format(job.kept)} corrections gardées` : "ta correction gardée"}` : "";
    return [`Description terminée · ${which} · <b>${changed}</b>${kept}${unclear}`, ""];
  }
  return [`Description terminée · ${plural(sent, "table")} ${sent > 1 ? "décrites" : "décrite"} · à relire${unclear}`, ""];
}

// A relations run: the measured ones, the LLM (phase llm), then the relations without evidence checked on the data
// (phase check, out of to_check); --check: only that. Done: what the search added, what the check found weak
function relationsJobTitle(job, now) {
  const checkOnly = job.command.includes("--check"), n = job.checked, checked = `${plural(n, "jointure")} ${n > 1 ? "vérifiées" : "vérifiée"}`;
  const what = checkOnly ? "Vérification des jointures" : "Recherche des jointures";
  if (job.running && job.stopped) return [`Arrêt en cours… ${duration(now - job.stopped_at)}`, "running"];
  if (job.running && job.phase === "llm") return [`L'IA cherche des jointures entre les tables… ${duration(now - job.started)}`, "running"];
  if (job.running && job.to_check) {
    return [`Vérification sur les données · ${fmt.format(n)}/${fmt.format(job.to_check)} · ${duration(now - job.started)}`, "running"];
  }
  if (job.running) return [`${what}… ${duration(now - job.started)}`, "running"];
  if (job.stopped) return [`${what} arrêtée · rien n'est écrit : relationships.yaml n'est mis à jour qu'à la fin`, "stopped"];
  if (job.returncode) return [`Échec de la ${checkOnly ? "vérification" : "recherche"} · voir les logs`, "failed"];
  const rel = job.db === ui.db ? ui.rel : null;
  if (checkOnly) return [`Vérification terminée · ${checked}${rel?.weak ? ` · ${fmt.format(rel.weak)} à vérifier` : ""}`, ""];
  const added = job.added ?? 0;
  const found = added > 1 ? `${fmt.format(added)} nouvelles jointures` : added ? "1 nouvelle jointure" : "aucune nouvelle jointure";
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
