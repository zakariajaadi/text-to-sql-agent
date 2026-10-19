/* Header database badge: Chainlit hides the chat profile selector when there is a single profile, so a lone
   database (e.g. a private cluster's) would leave no trace of which database the agent queries. With one
   profile, show it where the selector sits, in the selector's own style (icon + name), minus the chevron:
   there is nothing to pick. With several profiles, Chainlit's selector is there and this script does nothing.
   The profiles come from the same settings endpoint the Chainlit frontend reads. */
(() => {
  const BADGE_ID = "db-badge";

  // The selector's own classes (Chainlit's #chat-profiles button), without its border and hover
  const BADGE_CLASS = "flex h-10 items-center px-3 py-2 w-fit text-muted-foreground font-semibold text-lg";

  let profile = null; // {name, icon} of the single profile, null until loaded or with several profiles

  async function loadProfile() {
    const language = navigator.language || "en-US";
    const response = await fetch(`/project/settings?language=${encodeURIComponent(language)}`);
    if (!response.ok) return;
    const profiles = (await response.json()).chatProfiles || [];
    if (profiles.length === 1) profile = profiles[0];
  }

  function buildBadge() {
    const badge = document.createElement("div");
    badge.id = BADGE_ID;
    badge.className = BADGE_CLASS;
    badge.title = profile.name;

    const content = document.createElement("div");
    content.className = "flex items-center gap-2";
    if (profile.icon) {
      const icon = document.createElement("img");
      icon.src = profile.icon;
      icon.alt = "";
      icon.className = "w-6 h-6 rounded-md object-cover";
      content.append(icon);
    }
    const name = document.createElement("span");
    name.textContent = profile.name;
    content.append(name);

    badge.append(content);
    return badge;
  }

  // The header is re-rendered by React (new chat, thread pages): put the badge back whenever it is missing,
  // in the header's left group, right after the new chat button, where the selector would be
  function ensureBadge() {
    if (!profile || document.getElementById("chat-profiles")) return;
    const left = document.querySelector("#header > div:first-child");
    if (!left || left.querySelector(`#${BADGE_ID}`)) return;
    left.append(buildBadge());
  }

  loadProfile()
    .then(() => {
      if (!profile) return;
      ensureBadge();
      new MutationObserver(ensureBadge).observe(document.body, { childList: true, subtree: true });
    })
    .catch(() => {}); // no badge rather than a broken page
})();
