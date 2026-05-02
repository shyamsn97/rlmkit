// main.js
// Bootstraps the boids simulation per CONTRACT.
// Imports exactly as specified.
import { Flock } from "./flock.js";

function init() {
  const canvas = document.getElementById("boids");
  if (!canvas) {
    console.error("Canvas #boids not found");
    return;
  }
  const ctx = canvas.getContext("2d", { alpha: false });

  // Track devicePixelRatio for HiDPI handling
  let dpr = Math.max(1, window.devicePixelRatio || 1);

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    // CSS size
    const cssW = Math.max(1, window.innerWidth);
    const cssH = Math.max(1, window.innerHeight);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    // Backing store size
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    // Scale drawing so 1 unit = 1 CSS pixel
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Optional: ensure crisp rendering for trails
    ctx.imageSmoothingEnabled = false;
  }

  window.addEventListener("resize", resize, { passive: true });
  resize();

  // Create flock with fixed count; pass logical (CSS) dimensions
  const BOID_COUNT = 300;
  const flock = new Flock(BOID_COUNT, canvas.width / dpr, canvas.height / dpr);

  // Animation loop
  let last = performance.now();
  function frame(now) {
    const elapsed = (now - last) / 1000; // seconds
    last = now;
    const dt = Math.min(0.05, elapsed); // clamp dt to avoid large jumps

    // Draw semi-transparent black rect to create trails (in CSS pixels)
    ctx.fillStyle = "rgba(0,0,0,0.15)";
    ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr);

    // Update flock using logical canvas size
    flock.update(canvas.width / dpr, canvas.height / dpr, dt);

    // Draw flock; boids draw in CSS pixels because we set transform
    flock.draw(ctx);

    requestAnimationFrame(frame);
  }

  // Start loop
  last = performance.now();
  requestAnimationFrame(frame);
}

// Bootstrap when DOM is ready
if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  // already parsed
  init();
}
