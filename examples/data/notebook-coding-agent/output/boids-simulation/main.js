import { Simulation } from './sim.js';

const canvas = document.getElementById('c');

function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext('2d');
  // Use CSS pixel coordinates; let the context scale to device pixels
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

resize();
window.addEventListener('resize', resize, { passive: true });

const sim = new Simulation(canvas);
sim.start();
