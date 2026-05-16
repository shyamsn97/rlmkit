// Boid class: position, velocity, acceleration, color
function randRange(a, b) { return a + Math.random() * (b - a); }

class Boid {
  constructor(x, y, id) {
    this.pos = createVec(x, y);
    // fast initial speeds for a lively simulation
    var angle = Math.random() * Math.PI * 2;
    this.vel = createVec(Math.cos(angle) * randRange(1.5, 3.5), Math.sin(angle) * randRange(1.5, 3.5));
    this.acc = createVec(0, 0);
    this.id = id || 0;
    this.maxSpeed = randRange(3.5, 6.0); // faster boids
    this.maxForce = 0.12;
    // colorful hue based on id to spread colors
    this.hue = Math.floor((this.id * 47) % 360);
    this.size = randRange(3.5, 7.0);
  }

  update(boids, width, height) {
    this.applyBehaviors(boids);
    // integrate
    this.vel = add(this.vel, this.acc);
    this.vel = limit(this.vel, this.maxSpeed);
    this.pos = add(this.pos, this.vel);
    // wrap edges
    this.pos = wrapPosition(this.pos, width, height);
    // reset acceleration
    this.acc = createVec(0, 0);
  }

  applyForce(f) {
    this.acc = add(this.acc, f);
  }

  applyBehaviors(boids) {
    var alignment = this.align(boids);
    var cohesion = this.cohere(boids);
    var separation = this.separate(boids);
    // weights tuning for fast, colorful swarm
    alignment = mult(alignment, 1.0);
    cohesion = mult(cohesion, 0.85);
    separation = mult(separation, 1.5);

    this.applyForce(alignment);
    this.applyForce(cohesion);
    this.applyForce(separation);
  }

  // Steering behaviors
  align(boids) {
    var perception = 50;
    var steering = createVec(0, 0);
    var total = 0;
    for (var other of boids) {
      var d = dist(this.pos, other.pos);
      if (other !== this && d < perception) {
        steering = add(steering, other.vel);
        total++;
      }
    }
    if (total > 0) {
      steering = div(steering, total);
      steering = setMag(steering, this.maxSpeed);
      steering = sub(steering, this.vel);
      steering = limit(steering, this.maxForce);
    }
    return steering;
  }

  cohere(boids) {
    var perception = 60;
    var steering = createVec(0, 0);
    var total = 0;
    for (var other of boids) {
      var d = dist(this.pos, other.pos);
      if (other !== this && d < perception) {
        steering = add(steering, other.pos);
        total++;
      }
    }
    if (total > 0) {
      steering = div(steering, total);
      steering = sub(steering, this.pos);
      steering = setMag(steering, this.maxSpeed);
      steering = sub(steering, this.vel);
      steering = limit(steering, this.maxForce);
    }
    return steering;
  }

  separate(boids) {
    var perception = 28;
    var steering = createVec(0, 0);
    var total = 0;
    for (var other of boids) {
      var d = dist(this.pos, other.pos);
      if (other !== this && d < perception && d > 0) {
        var diff = sub(this.pos, other.pos);
        diff = div(diff, d); // weight by distance
        steering = add(steering, diff);
        total++;
      }
    }
    if (total > 0) {
      steering = div(steering, total);
      steering = setMag(steering, this.maxSpeed);
      steering = sub(steering, this.vel);
      steering = limit(steering, this.maxForce * 1.5);
    }
    return steering;
  }

  // draw a triangle representing the boid
  draw(ctx) {
    var theta = Math.atan2(this.vel.y, this.vel.x);
    ctx.save();
    ctx.translate(this.pos.x, this.pos.y);
    ctx.rotate(theta);
    // color bright and colorful
    ctx.fillStyle = 'hsl(' + this.hue + ', 85%, 55%)';
    ctx.beginPath();
    ctx.moveTo(this.size * 1.8, 0);
    ctx.lineTo(-this.size, this.size * 0.9);
    ctx.lineTo(-this.size, -this.size * 0.9);
    ctx.closePath();
    ctx.fill();
    // faint stroke for contrast
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 0.6;
    ctx.stroke();
    ctx.restore();
  }
}
