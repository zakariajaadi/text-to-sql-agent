/* Welcome screen title: "Interrogez <database>" between the avatar and the description, the database name in a
   moving primary gradient. The name is the one the header shows: Chainlit's profile selector (#chat-profiles)
   with several databases, our badge (db_badge.js, #db-badge) with a single one. Both appear after the first
   render, and the welcome screen comes back on each new chat: re-checked on every DOM change. Styles live in
   custom.css (#welcome-title). */
(() => {
  const TITLE_ID = "welcome-title";

  const currentDatabase = () =>
    (document.getElementById("chat-profiles") || document.getElementById("db-badge"))?.textContent.trim() || "";

  function buildTitle() {
    const title = document.createElement("h1");
    title.id = TITLE_ID;
    title.append("Interrogez ");
    const name = document.createElement("span");
    name.className = "welcome-title-name";
    title.append(name);
    return title;
  }

  function ensureTitle() {
    const welcome = document.getElementById("welcome-screen");
    const database = currentDatabase();
    const description = welcome?.querySelector(".prose");
    if (!description || !database) return;
    let title = document.getElementById(TITLE_ID);
    if (!title || title.parentElement !== description.parentElement) {
      title?.remove();
      title = buildTitle();
      description.before(title);
    }
    const name = title.querySelector(".welcome-title-name");
    if (name.textContent !== database) name.textContent = database;
  }

  ensureTitle();
  new MutationObserver(ensureTitle).observe(document.body, { childList: true, subtree: true, characterData: true });
})();
