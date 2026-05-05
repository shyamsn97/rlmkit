import { Simulation } from "./simulation.js";
import { Renderer } from "./renderer.js";

const canvas = document.getElementById("boids-canvas");

function fitCanvas(canvas){
  const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
  const w = window.innerWidth, h = window.innerHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
  return { ctx, width: w, height: h, dpr };
}

if (!canvas) {
  throw new Error("Canvas with id 'boids-canvas' not found.");
}

let { ctx, width, height } = fitCanvas(canvas);

const sim = new Simulation(width, height, 300, 240); // 300 boids, fast
const renderer = new Renderer(ctx);
renderer.setSize(width, height);

function onResize(){
  const res = fitCanvas(canvas);
  ctx = res.ctx;
  width = res.width;
  height = res.height;
  renderer.setContext && renderer.setContext(ctx); // if renderer supports swapping context
  renderer.setSize(width, height);
  if (typeof sim.resize === "function") {
    sim.resize(width, height);
  }
}

window.addEventListener("resize", onResize, { passive: true });

let last = performance.now();
function frame(t){
  const dt = Math.min((t - last) / 1000, 0.05);
  last = t;
  sim.update(dt);
  if (typeof renderer.clear === "function") renderer.clear();
  renderer.draw(sim.boidsArray);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
