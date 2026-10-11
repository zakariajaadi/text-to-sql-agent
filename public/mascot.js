/* Welcome screen mascot: after a few seconds of inactivity, the robot of the agent avatar wakes up, looks around,
   jumps out of its circle unfolding its arms and legs, lands on the message composer, runs to its left corner and
   sits on the edge, legs dangling. At rest it swings its legs, blinks and follows the pointer, or the text being
   typed, with its eyes. Plays once per page load, only on the welcome screen. Inactivity means no click, key,
   scroll or touch: a pointer resting or drifting over the page does not count. Styles live in custom.css.
   Replay from the console: sqlAgentMascot.replay() */
(() => {
  const IDLE_MS = 5000;

  // The robot is the agent icon (public/icons/agent.svg): a 24-unit grid drawn at 4/3 in the 64px avatar, so at
  // scale 1 its head covers the avatar's exactly. Poses are in these units.
  const UNIT = 4 / 3, ORIGIN = 16;
  const toPx = (u) => ORIGIN + u * UNIT;
  const FORWARD = -1; // it only ever goes left: forward is -x, so the icon never needs mirroring
  const HIP_Y = 20, GROUND = 27, LEG = 3.6, ARM = 5, TOE = 1.1, HANG = 7.1;
  const FEET_ANCHOR = GROUND + 0.9, SEAT_ANCHOR = HIP_Y + 0.9; // outer edge of the strokes
  const SIDE_HIPS = [13.5, 10.5], FRONT_HIPS = [14.5, 9.5]; // [back leg, front leg]
  const SHOULDERS = [[19.6, 16.8], [4.4, 16.8]]; // [back arm, front arm], right under the ears
  const TUCK = [[-0.5, 4.6], [1.8, 4.2]]; // feet relative to the hips mid-air, x forward

  const SCALE = 1.3; // on the composer
  const LAND_DX = -140; // landing point, from the avatar center
  const SEAT_DX = 22; // hips at rest, from the composer's left edge
  const JUMP_MS = 700, JUMP_HEIGHT = 55;
  const RUN_SPEED = 95, RUN_ACCEL = 500, RUN_DECEL = 380; // px/s, px/s²
  const STRIDE = 11, STANCE = 0.4, STEP_LIFT = 2.5; // gait, in units: a foot stays planted for 40% of a cycle

  // Antenna: damped spring (deg) pushed by the robot's horizontal acceleration
  const ANTENNA_K = (2 * Math.PI * 2.5) ** 2, ANTENNA_DAMP = 2 * 0.22 * Math.sqrt(ANTENNA_K), ANTENNA_PUSH = 8;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const SVG = `
    <svg class="mascot-svg" width="64" height="64" viewBox="0 0 64 64">
      <g transform="translate(${ORIGIN} ${ORIGIN}) scale(${UNIT})" fill="none" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round">
        <path class="m-leg m-back"/><path class="m-leg m-front"/>
        <g class="m-head">
          <path class="m-arm m-back"/><path class="m-arm m-front"/>
          <g class="m-antenna"><path d="M12 8V4H8"/></g>
          <rect class="m-face" width="16" height="12" x="4" y="8" rx="2"/>
          <path class="m-ears" d="M2 14h2M20 14h2"/>
          <g class="m-eyes"><g class="m-lids"><path d="M15 13v2M9 13v2"/></g></g>
        </g>
      </g>
    </svg>`;

  let welcome = null; // the #welcome-screen the mascot belongs to
  let m = null; // the running mascot: DOM nodes, world state, pose
  let idleTimer = null;
  let played = false;
  let pointer = null, typingUntil = 0;

  const findWelcome = () => document.getElementById("welcome-screen");
  const avatarOf = (screen) => screen.querySelector("img.rounded-full");

  function geometry() {
    const avatar = avatarOf(welcome).getBoundingClientRect();
    const composer = document.getElementById("message-composer").getBoundingClientRect();
    return { cx: avatar.left + avatar.width / 2, avatarTop: avatar.top, left: composer.left, top: composer.top };
  }

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (t) => Math.min(1, Math.max(0, t));
  const easeOut = (t) => 1 - (1 - t) ** 3;
  const easeInOut = (t) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2);
  const easeOutBack = (t) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2;
  const smoothstep = (t) => t * t * (3 - 2 * t);

  function standingPose() {
    return {
      bodyY: 0, lean: 0, squash: 0, // squash < 0 stretches
      hips: [...SIDE_HIPS], feet: SIDE_HIPS.map((h) => [h, GROUND]), toe: TOE, legExt: 1,
      arms: [14, -14], armExt: 1, // degrees from hanging down, positive towards +x
      eyes: [0, 0], eyeTarget: [0, 0],
    };
  }

  // Two-bone IK: the knee, bent forward, and the foot pulled back within reach
  function solveLeg(hip, foot, l1, l2) {
    let dx = foot[0] - hip[0], dy = foot[1] - hip[1];
    const reach = l1 + l2 - 1e-3, dist = Math.hypot(dx, dy) || 1e-6;
    if (dist > reach) { dx *= reach / dist; dy *= reach / dist; }
    const d = Math.min(dist, reach), a = Math.atan2(dy, dx);
    const b = Math.acos(clamp01((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)));
    const k = a - FORWARD * b;
    return [[hip[0] + l1 * Math.cos(k), hip[1] + l1 * Math.sin(k)], [hip[0] + dx, hip[1] + dy]];
  }

  function render(dt) {
    const { state: s, pose: p } = m;
    const ax = toPx(12), ay = toPx(s.anchor);
    m.el.style.transform = `translate(${s.x - ax}px, ${s.y - ay}px)`;
    m.inner.style.transformOrigin = `${ax}px ${ay}px`;
    m.inner.style.transform = `scale(${s.scale})`;

    if (dt > 0) {
      const sp = m.spring, vx = (s.x - sp.lastX) / dt;
      sp.accel += ((vx - sp.vx) / dt - sp.accel) * 0.3;
      sp.lastX = s.x;
      sp.vx = vx;
      sp.speed += (-ANTENNA_K * sp.angle - ANTENNA_DAMP * sp.speed - ANTENNA_PUSH * sp.accel) * dt;
      sp.angle = Math.max(-35, Math.min(35, sp.angle + sp.speed * dt));
      const follow = 1 - Math.exp(-dt * 14);
      p.eyes = p.eyes.map((e, i) => e + (p.eyeTarget[i] - e) * follow);
    }
    m.antenna.setAttribute("transform", `rotate(${m.spring.angle} 12 8)`);
    m.eyes.setAttribute("transform", `translate(${p.eyes[0]} ${p.eyes[1]})`);
    m.head.setAttribute(
      "transform",
      `translate(0 ${p.bodyY}) rotate(${p.lean} 12 ${HIP_Y}) ` +
        `translate(12 ${HIP_Y}) scale(${1 + p.squash * 0.6} ${1 - p.squash}) translate(-12 ${-HIP_Y})`,
    );

    for (const i of [0, 1]) {
      const hip = [p.hips[i], HIP_Y + p.bodyY], ext = p.legExt;
      const target = [lerp(hip[0], p.feet[i][0], ext), lerp(hip[1], p.feet[i][1], ext)];
      const [knee, foot] = solveLeg(hip, target, LEG * ext, LEG * ext);
      m.legs[i].setAttribute("d", ext < 0.05 ? "" : `M${hip}L${knee}L${foot}l${FORWARD * p.toe * ext} 0`);

      const sh = SHOULDERS[i], a = (p.arms[i] * Math.PI) / 180, l = ARM * p.armExt;
      m.arms[i].setAttribute("d", p.armExt < 0.05 ? "" : `M${sh}L${sh[0] + l * Math.sin(a)},${sh[1] + l * Math.cos(a)}`);
    }
  }

  const ABORT = Symbol("abort");
  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

  // Calls step(dt, elapsed seconds) then renders, every frame, until step returns true
  async function frames(step) {
    const start = performance.now();
    let last = start;
    for (;;) {
      await nextFrame();
      if (!m || m.cancelled) throw ABORT; // teardown() cancels then clears m
      const now = performance.now(), dt = Math.min(1 / 30, Math.max(1 / 240, (now - last) / 1000));
      last = now;
      const done = step(dt, (now - start) / 1000);
      render(dt);
      if (done) return;
    }
  }
  const tween = (ms, step) => frames((dt, elapsed) => {
    const t = Math.min(1, (elapsed * 1000) / ms);
    step(t, dt);
    return t === 1;
  });

  // Running: each foot stays planted while the body passes over it, then swings forward,
  // so the cycle follows the distance covered and the feet never slide
  function gait(p, distance, speed) {
    const cycles = distance / STRIDE, w = smoothstep(clamp01(speed * 2.5)), ground = STRIDE * STANCE;
    p.feet = SIDE_HIPS.map((h, i) => {
      const phase = (cycles + i * 0.5) % 1;
      let rel, lift = 0;
      if (phase < STANCE) {
        rel = ground / 2 - (phase / STANCE) * ground;
      } else {
        const u = (phase - STANCE) / (1 - STANCE);
        rel = -ground / 2 + ground * easeInOut(u);
        lift = STEP_LIFT * Math.sin(Math.PI * u);
      }
      return [h + FORWARD * rel * w, GROUND - lift * w];
    });
    p.bodyY = 0.6 * Math.cos(4 * Math.PI * (cycles - STANCE / 2)) * w;
    p.lean = FORWARD * 9 * speed;
    p.arms = [14, -14].map((rest, i) => rest + 50 * w * Math.sin(2 * Math.PI * (cycles + i * 0.5 + 0.25)));
  }

  function blinkLater(ms) {
    const current = m;
    setTimeout(async () => {
      if (m !== current || current.cancelled) return;
      const blinks = Math.random() < 0.2 ? 2 : 1;
      for (let i = 0; i < blinks; i++) {
        current.lids.classList.add("is-closed");
        await new Promise((resolve) => setTimeout(resolve, 110));
        current.lids.classList.remove("is-closed");
        await new Promise((resolve) => setTimeout(resolve, 140));
      }
      blinkLater(2500 + Math.random() * 3500);
    }, ms);
  }

  // At rest: read the composer while typing, otherwise follow the pointer
  function restingLook() {
    if (performance.now() < typingUntil) return [1.8, 1.2];
    if (!pointer) return [1.6, 1.2];
    const s = m.state, headY = s.y - 6 * UNIT * s.scale;
    const dx = pointer[0] - s.x, dy = pointer[1] - headY, dist = Math.hypot(dx, dy) || 1;
    const reach = 2 * Math.min(1, dist / 160);
    return [(dx / dist) * reach, (dy / dist) * reach];
  }

  async function play() {
    played = true;
    const avatar = avatarOf(welcome);
    const el = document.createElement("div");
    el.className = "mascot";
    el.innerHTML = `<div class="mascot-inner">${SVG}</div>`;
    m = {
      el,
      inner: el.firstElementChild,
      head: el.querySelector(".m-head"),
      antenna: el.querySelector(".m-antenna"),
      eyes: el.querySelector(".m-eyes"),
      lids: el.querySelector(".m-lids"),
      legs: [...el.querySelectorAll(".m-leg")],
      arms: [...el.querySelectorAll(".m-arm")],
      state: { x: 0, y: 0, scale: 1, anchor: FEET_ANCHOR },
      pose: { ...standingPose(), legExt: 0, armExt: 0 },
      spring: { angle: 0, speed: 0, lastX: 0, vx: 0, accel: 0 },
      cancelled: false,
    };
    const { state: s, pose: p } = m;
    const stayHome = () => {
      const g = geometry();
      s.x = g.cx;
      s.y = g.avatarTop + toPx(FEET_ANCHOR);
    };
    const stay = (ms) => tween(ms, stayHome);

    try {
      // Take over the avatar: the head drawn exactly over the icon, the avatar becomes a plain disc behind it
      stayHome();
      m.spring.lastX = s.x;
      render(0);
      document.body.appendChild(el);
      avatar.classList.add("mascot-home");

      // Wakes up and looks around, then at where it is going
      await stay(300);
      blinkLater(0);
      await stay(300);
      p.eyeTarget = [-1.8, 0];
      await stay(400);
      p.eyeTarget = [1.8, 0];
      await stay(400);
      p.eyeTarget = [FORWARD * 1.8, 1.4];
      await stay(250);
      await tween(170, (t) => { stayHome(); p.squash = 0.16 * easeOut(t); });

      // Jump: stretch on take-off, unfold arms and legs on the way up (with a little overshoot),
      // reach for the ground before landing
      const landX = (g) => Math.max(g.cx + LAND_DX, g.left + SEAT_DX + 70);
      await tween(JUMP_MS, (t) => {
        const g = geometry(), y0 = g.avatarTop + toPx(FEET_ANCHOR);
        s.x = lerp(g.cx, landX(g), t);
        s.y = lerp(y0, g.top, t) - JUMP_HEIGHT * 4 * t * (1 - t);
        s.scale = lerp(1, SCALE, easeInOut(t));
        p.squash = t < 0.2 ? -0.12 * (1 - t / 0.2) : t > 0.8 ? (-0.06 * (t - 0.8)) / 0.2 : 0;
        p.legExt = p.armExt = t < 0.1 ? 0 : easeOutBack(clamp01((t - 0.1) / 0.3));
        const reach = easeInOut(clamp01((t - 0.6) / 0.4));
        p.feet = SIDE_HIPS.map((h, i) => [h + FORWARD * lerp(TUCK[i][0], 0, reach), lerp(HIP_Y + TUCK[i][1], GROUND, reach)]);
        p.arms = [lerp(150, 20, reach), lerp(-150, -20, reach)];
        if (t > 0.25 && !el.classList.contains("is-out")) {
          el.classList.add("is-out");
          avatar.classList.add("mascot-away");
        }
      });

      // Landing: the hips drop and the knees bend, then spring back
      await tween(320, (t) => {
        const g = geometry();
        s.x = landX(g);
        s.y = g.top;
        const absorb = t < 0.55 ? Math.sin((Math.PI * t) / 0.55) : -0.15 * Math.sin((Math.PI * (t - 0.55)) / 0.45);
        p.bodyY = 2.2 * absorb;
        p.squash = 0.1 * Math.max(0, absorb);
        p.feet = SIDE_HIPS.map((h) => [h, GROUND]);
        p.arms = [20 + 25 * absorb, -20 - 25 * absorb];
      });

      // Run to the left corner: accelerate, then brake to stop right on the seat
      const g0 = geometry();
      let rel = landX(g0) - g0.left, speed = 0, distance = 0;
      p.eyeTarget = [FORWARD * 1.8, 0.5];
      await frames((dt) => {
        const g = geometry(), remaining = rel - SEAT_DX;
        speed = Math.min(RUN_SPEED, speed + RUN_ACCEL * dt, Math.sqrt(2 * RUN_DECEL * Math.max(0, remaining)));
        const step = Math.min(remaining, speed * dt);
        rel -= step;
        distance += step / (UNIT * s.scale);
        s.x = g.left + rel;
        s.y = g.top;
        gait(p, distance, speed / RUN_SPEED);
        return remaining - step < 0.2;
      });
      const stopFeet = p.feet.map((f) => [...f]), stopArms = [...p.arms], stopBodyY = p.bodyY;
      await tween(220, (t) => {
        const g = geometry(), e = easeOut(t);
        s.x = g.left + SEAT_DX;
        s.y = g.top;
        p.feet = stopFeet.map((f, i) => [lerp(f[0], SIDE_HIPS[i], e), lerp(f[1], GROUND, e)]);
        p.arms = stopArms.map((a, i) => lerp(a, i ? -14 : 14, e));
        p.bodyY = lerp(stopBodyY, 0, e);
        p.lean = 0;
      });
      p.eyeTarget = [0, 0.3];
      await tween(200, () => {});

      // Hop up and sit on the edge, facing us, the legs hanging over the composer
      await tween(460, (t) => {
        const g = geometry(), e = easeInOut(t), tuck = 2.5 * Math.sin(Math.PI * t);
        s.x = g.left + SEAT_DX;
        s.y = g.top - 10 * Math.sin(Math.PI * t);
        s.anchor = lerp(FEET_ANCHOR, SEAT_ANCHOR, e);
        p.hips = [0, 1].map((i) => lerp(SIDE_HIPS[i], FRONT_HIPS[i], e));
        p.feet = p.hips.map((h) => [h, lerp(GROUND, HIP_Y + HANG, e) - tuck]);
        p.toe = lerp(TOE, 0, e);
        p.arms = [lerp(14, 30, e), lerp(-14, -30, e)];
        p.squash = t < 0.25 ? -0.08 * (1 - t / 0.25) : 0;
      });
      await tween(200, (t) => {
        const g = geometry();
        s.x = g.left + SEAT_DX;
        s.y = g.top;
        p.squash = 0.12 * Math.sin(Math.PI * t);
      });

      // Rest: swing the legs, follow the composer if it moves, look around
      let time = 0;
      await frames((dt) => {
        const g = geometry();
        s.x = g.left + SEAT_DX;
        s.y = g.top;
        time += dt;
        p.feet = FRONT_HIPS.map((h, i) => {
          const a = 0.2 * Math.sin(2 * Math.PI * 0.7 * time + i * Math.PI);
          return [h + HANG * Math.sin(a), HIP_Y + HANG * Math.cos(a)];
        });
        p.eyeTarget = restingLook();
        return false;
      });
    } catch (e) {
      if (e !== ABORT) throw e;
    }
  }

  function teardown() {
    if (!m) return;
    const { el } = m;
    m.cancelled = true;
    m = null;
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }

  function armIdleTimer() {
    clearTimeout(idleTimer);
    if (m || played || !welcome) return;
    idleTimer = setTimeout(() => {
      const input = document.getElementById("chat-input");
      if (document.hidden || !welcome || (input && input.value.trim())) return armIdleTimer();
      play();
    }, IDLE_MS);
  }

  // Chainlit is a React app: watch the DOM for the welcome screen coming and going
  let scheduled = false;
  new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      const current = findWelcome();
      if (current === welcome) return;
      teardown();
      welcome = current;
      armIdleTimer();
    });
  }).observe(document.documentElement, { childList: true, subtree: true });

  for (const type of ["pointerdown", "keydown", "wheel", "touchstart", "input"]) {
    window.addEventListener(type, armIdleTimer, { capture: true, passive: true });
  }
  window.addEventListener("pointermove", (e) => { pointer = [e.clientX, e.clientY]; }, { passive: true });
  window.addEventListener("input", (e) => {
    if (e.target.id === "chat-input") typingUntil = performance.now() + 1500;
  }, true);
  document.addEventListener("visibilitychange", armIdleTimer);

  window.sqlAgentMascot = {
    replay() {
      teardown();
      played = false;
      if (welcome) avatarOf(welcome)?.classList.remove("mascot-home", "mascot-away");
      armIdleTimer();
    },
  };
})();
