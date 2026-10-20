# Earlier versions of the agent logo

Kept to step back if needed. Three levels, newest first.

## Database-cylinder head standing on short legs (before the legless icons)

`cylinder_legs/` holds the five icons as they were. To go back:

1. Copy `cylinder_legs/agent.svg` over `public/icons/agent.svg`, `cylinder_legs/sql_agent.svg` over
   `public/avatars/sql_agent.svg`, and the other three over the ones in `public/`.
2. In `public/mascot.js`: `ORIGIN_Y = 14.3` and `HOME_LEG_EXT = 1 / 3`, so the mascot at rest stands on its short
   legs again like the icon.

## Robot head with the orange/amber gradient (before the database-cylinder head)

`robot_gradient/` holds `agent.svg`, `sql_agent.svg` and `mascot.js.txt` as they were. To go back:

1. Copy `robot_gradient/agent.svg` over `public/icons/agent.svg`, `robot_gradient/sql_agent.svg` over
   `public/avatars/sql_agent.svg`, and `robot_gradient/mascot.js.txt` over `public/mascot.js`.
2. In `public/custom.css`, drop `.mascot-svg .m-rim` from the white stroke rule (harmless if left).
3. `favicon.svg`, `logo_dark.svg` and `logo_light.svg` in this folder are the flat robot: copy them over the
   ones in `public/` (they never had the gradient robot).

## Flat orange robot (before the gradient)

`agent.svg`, `sql_agent.svg`, `favicon.svg`, `logo_dark.svg` and `logo_light.svg` in this folder. To go back:

1. Copy each over its file: `agent.svg` → `public/icons/`, `sql_agent.svg` → `public/avatars/`, the other three
   → `public/`.
2. `git checkout HEAD -- public/mascot.js` (the committed mascot has the flat robot head).
3. In `public/custom.css`, `#welcome-screen img.rounded-full.mascot-home`: back to `background: #E86441;`, and
   `.mascot-away` back to `background-color: transparent;` with `transition: background-color 400ms, outline-color 400ms;`
   (drop the `--home-from` / `--home-to` properties and their `@property` rules). `.mascot-svg .m-face`: back to
   `fill: #E86441;`.

Then delete this folder.
