import { Boid, SPEED } from './boid.js';

export class Simulation {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.boids = [];
    this.running = false;
    this._last = 0;
    this._initBoids(300); // 100s of fast, colorful boids
  }

  _cssSize() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    return { w: this.canvas.width / dpr, h: this.canvas.height / dpr };
  }

  _initBoids(count) {
    const { w, h } = this._cssSize();
    this.boids.length = 0;
    for (let i = 0; i < count; i++) {
      const x = Math.random() * w;
      const y = Math.random() * h;
      const a = Math.random() * Math.PI * 2;
      const vx = Math.cos(a) * SPEED;
      const vy = Math.sin(a) * SPEED;
      const hue = Math.floor(Math.random() * 360);
      this.boids.push(new Boid(x, y, vx, vy, hue));
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this._last = performance.now();
    const loop = (t) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (t - this._last) / 1000); // cap dt for stability
      this._last = t;
      this.update(dt);
      this.draw();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
  }

  update(dt) {
    const { w, h } = this._cssSize();
    const arr = this.boids;
    for (let i = 0; i < arr.length; i++) {
      arr[i].update(dt, arr, w, h);
    }
  }

  draw() {
    const { w, h } = this._cssSize();
    const ctx = this.ctx;
    // Clear with slight trail for visual appeal; comment next two lines to disable trails
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    ctx.fillRect(0, 0, w, h);

    // If first frame (no trail base), ensure fully cleared
    // This avoids initial alpha stacking if canvas was uninitialized
    // Detect by checking a simple sentinel flag
    if (!this._clearedOnce) {
      ctx.clearRect(0, 0, w, h);
      this._clearedOnce = true;
    }

    for (let i = 0; i < this.boids.length; i++) {
      this.boids[i].draw(ctx);
    }
  }
}
