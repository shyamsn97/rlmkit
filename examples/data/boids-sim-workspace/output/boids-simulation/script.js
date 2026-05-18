import { Boid } from './boid.js';
import { vec, randRange } from './util.js';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false });

let W = canvas.width = window.innerWidth;
let H = canvas.height = window.innerHeight;

let boids = [];
let running = true;
let targetCount = 400;
const defaults = { count: 400 };

const countSlider = document.getElementById('countSlider');
const countLabel = document.getElementById('countLabel');
const toggleBtn = document.getElementById('toggleBtn');
const fpsLabel = document.getElementById('fps');

function resize() {
  W = canvas.width = window.innerWidth;
  H = canvas.height = window.innerHeight;
  // update boid boundaries
  for (let b of boids) { b.w = W; b.h = H; }
}
window.addEventListener('resize', resize);

function makeBoid(i) {
  return new Boid(Math.random()*W, Math.random()*H, W, H, i);
}

function populate(n) {
  boids = [];
  for (let i=0;i<n;i++) boids.push(makeBoid(i));
}

countSlider.addEventListener('input', (e) => {
  targetCount = parseInt(e.target.value, 10);
  countLabel.textContent = targetCount;
});

countSlider.addEventListener('change', (e) => {
  // re-create to reset ids and distribution
  populate(targetCount);
});

toggleBtn.addEventListener('click', () => {
  running = !running;
  toggleBtn.textContent = running ? 'Pause' : 'Play';
});

populate(defaults.count);
countLabel.textContent = defaults.count;
countSlider.value = defaults.count;

let last = performance.now();
let fps = 0;
let frames = 0;
let acum = 0;

function step(now) {
  const dt = Math.min(50, now - last); // clamp delta
  last = now;
  if (running) {
    // update
    for (let b of boids) {
      b.flock(boids);
      b.update();
      b.edges();
    }
  }

  // draw background
  // clear with a slightly darkened rectangle for trailing effect:
  ctx.fillStyle = '#071019';
  ctx.fillRect(0,0,W,H);

  // draw boids
  for (let b of boids) b.draw(ctx);

  // FPS calc
  frames++;
  acum += dt;
  if (acum >= 250) {
    fps = Math.round(frames / (acum/1000));
    frames = 0;
    acum = 0;
    fpsLabel.textContent = `FPS: ${fps}`;
    // also show count
    countLabel.textContent = boids.length;
  }

  requestAnimationFrame(step);
}

// start the loop
requestAnimationFrame(step);

// expose a global helper for quick debugging in the console
window.boids_sim = {
  setCount(n) { populate(n); targetCount = n; countSlider.value = n; countLabel.textContent = n; },
  pause() { running = false; toggleBtn.textContent = 'Play'; },
  play() { running = true; toggleBtn.textContent = 'Pause'; },
  list() { return boids; }
};
