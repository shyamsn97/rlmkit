// flock.js
// Exports: class Flock
// Imports required by CONTRACT:
import { Boid } from "./boid.js";
import { SpatialHash } from "./utils.js";

export class Flock {
  // count: number of boids
  // width,height: initial canvas size (used for initial placement)
  constructor(count = 100, width = 800, height = 600) {
    this.boids = [];
    // cell size tuned to typical perception radius (~60)
    this.grid = new SpatialHash(64);
    this.width = width;
    this.height = height;

    for (let i = 0; i < count; i++) {
      const x = Math.random() * width;
      const y = Math.random() * height;
      // small random velocity
      const angle = Math.random() * Math.PI * 2;
      const speed = 20 + Math.random() * 80;
      const vx = Math.cos(angle) * speed;
      const vy = Math.sin(angle) * speed;
      const hue = Math.floor(Math.random() * 360);
      const color = `hsl(${hue} 90% 60%)`; // vivid HSL color

      // Prefer calling constructor with (x,y,vx,vy,color) — extra args are safe in JS.
      let b;
      try {
        b = new Boid(x, y, vx, vy, color);
      } catch (e) {
        // If constructor throws, try fewer args
        try {
          b = new Boid(x, y, color);
          if (typeof b.vx !== "undefined") { b.vx = vx; b.vy = vy; }
          if (typeof b.color === "undefined") b.color = color;
        } catch (e2) {
          // Last resort: call with position only and then set props
          b = new Boid(x, y);
          if (typeof b.vx !== "undefined") { b.vx = vx; b.vy = vy; }
          b.color = b.color || color;
        }
      }

      // Ensure boid has x,y properties for the spatial hash
      if (typeof b.x === "undefined") b.x = x;
      if (typeof b.y === "undefined") b.y = y;

      this.boids.push(b);
    }
  }

  // Rebuild the spatial hash and update boids.
  // update(width, height, dt)
  update(width, height, dt) {
    // update internal canvas size
    this.width = width;
    this.height = height;

    // Rebuild / clear grid each frame per CONTRACT
    if (this.grid && typeof this.grid.clear === "function") {
      this.grid.clear();
    } else {
      // If grid missing clear, recreate
      this.grid = new SpatialHash(64);
    }

    // Insert all boids into the grid
    for (const b of this.boids) {
      // Ensure the boid has numeric x/y
      if (typeof b.x !== "number" || typeof b.y !== "number") continue;
      // SpatialHash.insert expects an item with x,y
      try {
        this.grid.insert(b);
      } catch (e) {
        // ignore insertion errors
      }
    }

    // For each boid, query neighbors and call its update method
    for (const b of this.boids) {
      if (typeof b.x !== "number" || typeof b.y !== "number") continue;

      // Determine radius to query: try boid.perception or fallback to 60
      let r = 60;
      if (typeof b.perception === "number" && b.perception > 0) {
        r = b.perception;
      } else if (typeof b.perceptionRadius === "number" && b.perceptionRadius > 0) {
        r = b.perceptionRadius;
      }

      // If there's a separation radius, ensure it's included
      if (typeof b.separationRadius === "number" && b.separationRadius > r) {
        r = Math.max(r, b.separationRadius);
      }

      let neighbors = [];
      try {
        const q = this.grid.query(b.x, b.y, r);
        if (Array.isArray(q)) {
          // filter out the boid itself if present
          neighbors = q.filter(item => item !== b);
        }
      } catch (e) {
        neighbors = [];
      }

      // Try calling update with several common signatures to be robust:
      // Preferred: update(neighbors, dt, width, height)
      // Fallback: update(dt, width, height)
      // Fallback: update(neighbors)
      try {
        if (typeof b.update === "function") {
          // Attempt preferred signature
          try {
            b.update(neighbors, dt, width, height);
          } catch (e1) {
            try {
              b.update(dt, width, height);
            } catch (e2) {
              try {
                b.update(neighbors);
              } catch (e3) {
                try {
                  b.update();
                } catch (e4) {
                  // give up for this boid
                }
              }
            }
          }
        }
      } catch (e) {
        // ignore update errors per-boid
      }

      // Ensure wrapping (toroidal) in case boid.update doesn't handle edges
      if (typeof b.x === "number" && typeof b.y === "number") {
        if (b.x < 0) b.x += width;
        if (b.x >= width) b.x -= width;
        if (b.y < 0) b.y += height;
        if (b.y >= height) b.y -= height;
      }
    }
  }

  // Draw all boids using their draw(ctx) method or fallback rendering.
  draw(ctx) {
    for (const b of this.boids) {
      try {
        if (b && typeof b.draw === "function") {
          b.draw(ctx);
        } else if (b && typeof b.x === "number" && typeof b.y === "number") {
          // Fallback: draw a small triangle oriented by velocity if available.
          const x = b.x, y = b.y;
          let vx = (typeof b.vx === "number") ? b.vx : 0;
          let vy = (typeof b.vy === "number") ? b.vy : -1;
          const speed = Math.hypot(vx, vy) || 1;
          vx /= speed; vy /= speed;
          // triangle size
          const size = 6;
          const angle = Math.atan2(vy, vx);
          ctx.save();
          ctx.translate(x, y);
          ctx.rotate(angle);
          ctx.beginPath();
          ctx.moveTo(size, 0);
          ctx.lineTo(-size * 0.6, size * 0.6);
          ctx.lineTo(-size * 0.6, -size * 0.6);
          ctx.closePath();
          ctx.fillStyle = b.color || "white";
          ctx.fill();
          ctx.restore();
        }
      } catch (e) {
        // ignore drawing errors per-boid
      }
    }
  }
}
