(() => {
  'use strict';

  function getCanvas() {
    return document.getElementById('boidsCanvas');
  }

  function start() {
    const canvas = getCanvas();
    if (!canvas) return; // If index violates contract, fail gracefully.
    const ctx = canvas.getContext('2d');

    // HiDPI handling
    const state = { w: 0, h: 0, dpr: 1 };
    function resize() {
      const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
      state.dpr = dpr;
      const cssW = Math.max(1, Math.floor(window.innerWidth));
      const cssH = Math.max(1, Math.floor(window.innerHeight));
      state.w = cssW;
      state.h = cssH;
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // Draw in CSS pixels
    }
    window.addEventListener('resize', resize);
    resize();

    // Simulation parameters
    const COUNT = Math.max(320, Math.min(800, Math.floor((state.w * state.h) / 6000)));
    const VIEW_RADIUS = 42;
    const AVOID_RADIUS = 18;
    const MAX_SPEED = 2.4;
    const MIN_SPEED = 0.8;
    const MAX_FORCE = 0.05;

    const ALIGNMENT = 0.9;
    const COHESION = 0.6;
    const SEPARATION = 1.2;

    const cellSize = VIEW_RADIUS | 0;

    function randRange(a, b) { return a + Math.random() * (b - a); }

    // Boid data
    const boids = new Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      const angle = randRange(0, Math.PI * 2);
      const speed = randRange(MIN_SPEED, MAX_SPEED);
      boids[i] = {
        x: Math.random() * state.w,
        y: Math.random() * state.h,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        hue: (Math.random() * 360) | 0,
        size: randRange(2.0, 3.2),
      };
    }

    // Spatial hash grid
    function buildGrid() {
      const grid = new Map();
      for (let i = 0; i < boids.length; i++) {
        const b = boids[i];
        const cx = (b.x / cellSize) | 0;
        const cy = (b.y / cellSize) | 0;
        const key = cx + ',' + cy;
        let cell = grid.get(key);
        if (!cell) { cell = []; grid.set(key, cell); }
        cell.push(i);
      }
      return grid;
    }

    function limit(vx, vy, max) {
      const s2 = vx * vx + vy * vy;
      if (s2 > max * max) {
        const s = Math.sqrt(s2);
        const f = max / s;
        return [vx * f, vy * f];
      }
      return [vx, vy];
    }

    function wrap(b) {
      if (b.x < 0) b.x += state.w;
      else if (b.x >= state.w) b.x -= state.w;
      if (b.y < 0) b.y += state.h;
      else if (b.y >= state.h) b.y -= state.h;
    }

    function step(dt) {
      const grid = buildGrid();

      for (let i = 0; i < boids.length; i++) {
        const b = boids[i];
        const cx = (b.x / cellSize) | 0;
        const cy = (b.y / cellSize) | 0;

        // Neighborhood accumulators
        let count = 0;
        let avgVX = 0, avgVY = 0;
        let centerX = 0, centerY = 0;
        let repelX = 0, repelY = 0;

        // Check neighbor cells
        for (let gy = cy - 1; gy <= cy + 1; gy++) {
          for (let gx = cx - 1; gx <= cx + 1; gx++) {
            const key = gx + ',' + gy;
            const cell = grid.get(key);
            if (!cell) continue;
            for (let k = 0; k < cell.length; k++) {
              const j = cell[k];
              if (j === i) continue;
              const o = boids[j];
              let dx = o.x - b.x;
              let dy = o.y - b.y;

              // Account for wrapping shortest vector
              if (dx > state.w * 0.5) dx -= state.w;
              else if (dx < -state.w * 0.5) dx += state.w;
              if (dy > state.h * 0.5) dy -= state.h;
              else if (dy < -state.h * 0.5) dy += state.h;

              const d2 = dx * dx + dy * dy;
              if (d2 < VIEW_RADIUS * VIEW_RADIUS) {
                count++;
                avgVX += o.vx;
                avgVY += o.vy;
                centerX += (b.x + dx);
                centerY += (b.y + dy);
                if (d2 < AVOID_RADIUS * AVOID_RADIUS && d2 > 0.0001) {
                  const d = Math.sqrt(d2);
                  const inv = 1 / d;
                  const strength = (AVOID_RADIUS - d) * inv;
                  repelX -= dx * strength;
                  repelY -= dy * strength;
                }
              }
            }
          }
        }

        let ax = 0, ay = 0;

        if (count > 0) {
          // Alignment
          avgVX /= count; avgVY /= count;
          [avgVX, avgVY] = limit(avgVX, avgVY, MAX_SPEED);
          const steerAX = avgVX - b.vx;
          const steerAY = avgVY - b.vy;
          [ax, ay] = [ax + steerAX * ALIGNMENT, ay + steerAY * ALIGNMENT];

          // Cohesion
          centerX /= count; centerY /= count;
          let toCX = centerX - b.x;
          let toCY = centerY - b.y;
          // Wrap shortest path
          if (toCX > state.w * 0.5) toCX -= state.w;
          else if (toCX < -state.w * 0.5) toCX += state.w;
          if (toCY > state.h * 0.5) toCY -= state.h;
          else if (toCY < -state.h * 0.5) toCY += state.h;
          const magC = Math.hypot(toCX, toCY) || 1;
          toCX = (toCX / magC) * MAX_SPEED - b.vx;
          toCY = (toCY / magC) * MAX_SPEED - b.vy;
          ax += toCX * COHESION;
          ay += toCY * COHESION;

          // Separation
          ax += repelX * SEPARATION;
          ay += repelY * SEPARATION;
        }

        // Limit steering force
        [ax, ay] = limit(ax, ay, MAX_FORCE);

        // Integrate velocity
        b.vx += ax;
        b.vy += ay;
        [b.vx, b.vy] = limit(b.vx, b.vy, MAX_SPEED);

        // Enforce a minimum speed to keep boids lively
        const sp = Math.hypot(b.vx, b.vy) || 1e-6;
        if (sp < MIN_SPEED) {
          const f = MIN_SPEED / sp;
          b.vx *= f; b.vy *= f;
        }

        // Integrate position
        b.x += b.vx;
        b.y += b.vy;
        wrap(b);

        // Subtle hue drift
        b.hue = (b.hue + 0.2) % 360;
      }
    }

    function draw() {
      ctx.clearRect(0, 0, state.w, state.h); // Let CSS dark background show through
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      for (let i = 0; i < boids.length; i++) {
        const b = boids[i];
        const angle = Math.atan2(b.vy, b.vx);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const size = b.size;

        const noseX = b.x + cos * size * 3.0;
        const noseY = b.y + sin * size * 3.0;
        const leftX = b.x + Math.cos(angle + 2.5) * size * 2.0;
        const leftY = b.y + Math.sin(angle + 2.5) * size * 2.0;
        const rightX = b.x + Math.cos(angle - 2.5) * size * 2.0;
        const rightY = b.y + Math.sin(angle - 2.5) * size * 2.0;

        ctx.fillStyle = `hsl(${b.hue | 0}, 85%, 60%)`;
        ctx.beginPath();
        ctx.moveTo(noseX, noseY);
        ctx.lineTo(leftX, leftY);
        ctx.lineTo(rightX, rightY);
        ctx.closePath();
        ctx.fill();
      }
    }

    let last = performance.now();
    function tick(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      step(dt);
      draw();
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // Start immediately if canvas is present (as per contract), or on DOM ready otherwise.
  const c = getCanvas();
  if (c) start();
  else {
    document.addEventListener('DOMContentLoaded', () => {
      if (getCanvas()) start();
    });
  }
})();