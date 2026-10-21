/* Light of the message composer on the dot grid: custom.css draws a layer of primary dots (body::before) masked
   by an ellipse around the composer, sized and placed by the variables set here. The composer moves (welcome
   screen in the middle, bottom of the page in a conversation, grows with its text) and Chainlit replaces it on
   a new chat: its box is read on every frame, the variables written only when it changed. */
(() => {
  let last = "";

  function place() {
    const composer = document.getElementById("message-composer");
    if (composer) {
      const box = composer.getBoundingClientRect();
      const values = [
        `${box.left + box.width / 2}px`,   // center
        `${box.top + box.height / 2}px`,
        `${box.width * 0.75}px`,           // radii: past the composer's sides, 140px above and below it
        `${box.height / 2 + 140}px`,
      ];
      const key = values.join(" ");
      if (key !== last) {
        last = key;
        const style = document.body.style;
        ["--lit-cx", "--lit-cy", "--lit-rx", "--lit-ry"].forEach((name, i) => style.setProperty(name, values[i]));
      }
    }
    requestAnimationFrame(place);
  }

  requestAnimationFrame(place);
})();
