
// boid.js
// Simple Boid implementation suitable for use in a flock simulation.
// Exports a Boid class with methods:
//  - constructor(x, y, opts={})
//  - update()
//  - applyForce(force)
//  - edges(width, height)  // wrap-around behavior
//  - flock(boids)           // compute steering from nearby boids and apply
//  - render(ctx)           // optional: draw on a CanvasRenderingContext2D
//
// Options can include:
//  - maxSpeed (default 3)
//  - maxForce (default 0.05)
//  - perceptionRadius (default 50)
//  - separationRadius (default 25)

class Vector {
  constructor(x=0, y=0) {
    this.x = x;
    this.y = y;
  }
  add(v) { this.x += v.x; this.y += v.y; return this; }
  sub(v) { this.x -= v.x; this.y -= v.y; return this; }
  mult(n) { this.x *= n; this.y *= n; return this; }
  div(n) { if (n !== 0) { this.x /= n; this.y /= n; } return this; }
  mag() { return Math.hypot(this.x, this.y); }
  setMag(n) { const m = this.mag(); if (m !== 0) this.mult(n / m); return this; }
  normalize() { const m = this.mag(); if (m !== 0) this.div(m); return this; }
  limit(max) { if (this.mag() > max) this.setMag(max); return this; }
  copy() { return new Vector(this.x, this.y); }
  static sub(a, b) { return new Vector(a.x - b.x, a.y - b.y); }
  static add(a, b) { return new Vector(a.x + b.x, a.y + b.y); }
  static dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
}

// Default options
const DEFAULTS = {
  maxSpeed: 3,
  maxForce: 0.05,
  perceptionRadius: 50,
  separationRadius: 25,
  separationWeight: 1.5,
  alignmentWeight: 1.0,
  cohesionWeight: 1.0
};

class Boid {
  constructor(x=0, y=0, opts = {}) {
    const o = Object.assign({}, DEFAULTS, opts);
    this.pos = new Vector(x, y);
    // random initial velocity
    const angle = Math.random() * Math.PI * 2;
    this.vel = new Vector(Math.cos(angle), Math.sin(angle));
    this.vel.setMag(o.maxSpeed * (0.5 + Math.random()*0.5));
    this.acc = new Vector(0, 0);

    // parameters
    this.maxSpeed = o.maxSpeed;
    this.maxForce = o.maxForce;
    this.perceptionRadius = o.perceptionRadius;
    this.separationRadius = o.separationRadius;
    this.separationWeight = o.separationWeight;
    this.alignmentWeight = o.alignmentWeight;
    this.cohesionWeight = o.cohesionWeight;
  }

  applyForce(force) {
    // force is a Vector
    this.acc.add(force);
  }

  update() {
    this.vel.add(this.acc);
    this.vel.limit(this.maxSpeed);
    this.pos.add(this.vel);
    // reset acceleration
    this.acc.mult(0);
  }

  edges(width, height) {
    // wrap-around
    if (this.pos.x < 0) this.pos.x += width;
    if (this.pos.y < 0) this.pos.y += height;
    if (this.pos.x >= width) this.pos.x -= width;
    if (this.pos.y >= height) this.pos.y -= height;
  }

  // Steering behaviors
  separation(boids) {
    const steering = new Vector(0, 0);
    let total = 0;
    for (const other of boids) {
      const d = Vector.dist(this.pos, other.pos);
      if (other !== this && d < this.separationRadius && d > 0) {
        const diff = Vector.sub(this.pos, other.pos);
        diff.normalize();
        diff.div(d); // weight by distance
        steering.add(diff);
        total++;
      }
    }
    if (total > 0) {
      steering.div(total);
      steering.setMag(this.maxSpeed);
      steering.sub(this.vel);
      steering.limit(this.maxForce);
    }
    return steering;
  }

  alignment(boids) {
    const steering = new Vector(0, 0);
    let total = 0;
    for (const other of boids) {
      const d = Vector.dist(this.pos, other.pos);
      if (other !== this && d < this.perceptionRadius) {
        steering.add(other.vel);
        total++;
      }
    }
    if (total > 0) {
      steering.div(total);
      steering.setMag(this.maxSpeed);
      steering.sub(this.vel);
      steering.limit(this.maxForce);
    }
    return steering;
  }

  cohesion(boids) {
    const steering = new Vector(0, 0);
    let total = 0;
    for (const other of boids) {
      const d = Vector.dist(this.pos, other.pos);
      if (other !== this && d < this.perceptionRadius) {
        steering.add(other.pos);
        total++;
      }
    }
    if (total > 0) {
      steering.div(total);
      steering.sub(this.pos);
      steering.setMag(this.maxSpeed);
      steering.sub(this.vel);
      steering.limit(this.maxForce);
    }
    return steering;
  }

  flock(boids) {
    const sep = this.separation(boids).mult(this.separationWeight);
    const ali = this.alignment(boids).mult(this.alignmentWeight);
    const coh = this.cohesion(boids).mult(this.cohesionWeight);

    this.applyForce(sep);
    this.applyForce(ali);
    this.applyForce(coh);
  }

  // Optional canvas rendering helper
  render(ctx, options = {}) {
    // ctx: CanvasRenderingContext2D
    const size = options.size || 6;
    const angle = Math.atan2(this.vel.y, this.vel.x);

    ctx.save();
    ctx.translate(this.pos.x, this.pos.y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.6, size * 0.6);
    ctx.lineTo(-size * 0.6, -size * 0.6);
    ctx.closePath();
    ctx.fillStyle = options.color || '#ffffff';
    ctx.fill();
    ctx.restore();
  }
}

// Export for CommonJS and ES modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Boid, Vector };
} else {
  // attach to window for browser usage
  if (typeof window !== 'undefined') {
    window.Boid = Boid;
    window.Vector = Vector;
  }
  // also provide named exports if supported
  try {
    if (typeof define === 'function' && define.amd) {
      define(function() { return { Boid: Boid, Vector: Vector }; });
    }
  } catch (e) {}
}
