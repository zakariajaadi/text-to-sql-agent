/* Chainlit loads a single custom JS file (custom_js in .chainlit/config.toml): this one loads the UI scripts.
   mascot.js: the welcome screen robot. db_badge.js: the database name in the header when there is a single one.
   welcome_title.js: the welcome screen title. composer_light.js: places the composer's light on the dot grid. */
for (const src of ["/public/mascot.js", "/public/db_badge.js", "/public/welcome_title.js", "/public/composer_light.js"]) {
  const script = document.createElement("script");
  script.src = src;
  document.head.append(script);
}
