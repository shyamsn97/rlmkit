import { vec, add, sub, div, mul, mag, normalize, limit, dist2, randRange, hueColor } from './util.js';

export class Boid {
  constructor(x, y, w, h, id=0) {
    this.pos = vec(x !== undefined ? x : Math.random()*w, y !== undefined ? y : Math.random()*h);
    // velocity randomized for fast motion
    const ang = Math.random()*Math.PI*2;
    this.vel = vec(Math.cos(ang)*randRange(1.5,4.0), Math.sin(ang)*randRange(1.5,4.0));
    this.acc = vec(0,0);
    this.id = id;
    this.w = w;
    this.h = h;
    // tuned parameters for a lively, fast swarm
    this.maxSpeed = randRange(3.0, 5.5);
    this.maxForce = 0.05;
    this.perception = 48;        // for alignment/cohesion
    this.sepPerception = 26;     // for separation
    this.hue = (id * 37) % 360;  // varying hue per boid id
  }

  applyForce(f) {
    this.acc = add(this.acc, f);
  }

  edges() {
    // wrap-around edges
    if (this.pos.x < 0) this.pos.x += this.w;
    if (this.pos.y < 0) this.pos.y += this.h;
    if (this.pos.x >= this.w) this.pos.x -= this.w;
    if (this.pos.y >= this.h) this.pos.y -= this.h;
  }

  align(boids) {
    let steering = vec(0,0), total = 0;
    for (let other of boids) {
      if (other === this) continue;
      let d2 = dist2(this.pos, other.pos);
      if (d2 < this.perception*this.perception) {
        steering = add(steering, other.vel);
        total++;
      }
    }
    if (total > 0) {
      steering = div(steering, total);
      steering = normalize(steering);
      steering = mul(steering, this.maxSpeed);
      steering = sub(steering, this.vel);
      steering = limit(steering, this.maxForce);
    }
    return steering;
  }

  cohesion(boids) {
    let steering = vec(0,0), total = 0;
    for (let other of boids) {
      if (other === this) continue;
      let d2 = dist2(this.pos, other.pos);
      if (d2 < this.perception*this.perception) {
        steering = add(steering, other.pos);
        total++;
      }
    }
    if (total > 0) {
      steering = div(steering, total);
      steering = sub(steering, this.pos);
      steering = normalize(steering);
      steering = mul(steering, this.maxSpeed);
      steering = sub(steering, this.vel);
      steering = limit(steering, this.maxForce);
    }
    return steering;
  }

  separation(boids) {
    let steering = vec(0,0), total = 0;
    for (let other of boids) {
      if (other === this) continue;
      let dx = this.pos.x - other.pos.x;
      let dy = this.pos.y - other.pos.y;
      let d2 = dx*dx + dy*dy;
      if (d2 < this.sepPerception*this.sepPerception && d2 > 0) {
        let diff = vec(dx, dy);
        let inv = 1 / Math.sqrt(d2);
        diff = mul(diff, inv); // normalize
        diff = div(diff, Math.sqrt(d2)); // stronger when closer
        steering = add(steering, diff);
        total++;
      }
    }
    if (total > 0) {
      steering = div(steering, total);
      steering = normalize(steering);
      steering = mul(steering, this.maxSpeed);
      steering = sub(steering, this.vel);
      steering = limit(steering, this.maxForce * 1.5);
    }
    return steering;
  }

  flock(boids) {
    // weights chosen for vivid, slightly chaotic flocking
    const align = this.align(boids);
    const coh = this.cohesion(boids);
    const sep = this.separation(boids);
    this.applyForce(mul(align, 1.0));
    this.applyForce(mul(coh, 0.7));
    this.applyForce(mul(sep, 1.6));
  }

  update() {
    this.vel = add(this.vel, this.acc);
    this.vel = limit(this.vel, this.maxSpeed);
    this.pos = add(this.pos, this.vel);
    this.acc = vec(0,0);
  }

  draw(ctx) {
    // Draw as a rotated triangle pointing in velocity direction
    const angle = Math.atan2(this.vel.y, this.vel.x);
    ctx.save();
    ctx.translate(this.pos.x, this.pos.y);
    ctx.rotate(angle);
    // color reacts to speed for dynamic hues
    const speed = mag(this.vel);
    const hue = (this.hue + speed*18) % 360;
    ctx.fillStyle = hueColor(hue, '75%', '55%');
    ctx.beginPath();
    ctx.moveTo(10, 0);
    ctx.lineTo(-6, 4);
    ctx.lineTo(-6, -4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}
