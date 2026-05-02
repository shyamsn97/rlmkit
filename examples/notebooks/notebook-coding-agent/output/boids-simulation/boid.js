export class Boid {
  // Classic Boid with separation, alignment, cohesion, toroidal wrap, and colorful HSL
  constructor(x, y, vx = 0, vy = 0) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.ax = 0;
    this.ay = 0;

    // Per-spec defaults (units: pixels, seconds)
    this.perception = 60;        // cohesion / alignment radius
    this.separationRadius = 24;  // separation radius
    this.maxSpeed = 200;         // px / s
    this.maxForce = 100;         // px / s^2

    // Random vivid color per boid
    const h = Math.floor(Math.random() * 360);
    const s = 80 + Math.floor(Math.random() * 15); // 80-95%
    const l = 45 + Math.floor(Math.random() * 15); // 45-60%
    this.color = `hsl(${h} ${s}% ${l}%)`;

    // For drawing size
    this.size = 6 + Math.random() * 6; // 6-12 px
  }

  // Helper: limit magnitude of vector (x,y) to max
  static limit(x, y, max) {
    const mag = Math.hypot(x, y);
    if (mag > max && mag > 0) {
      const f = max / mag;
      return [x * f, y * f];
    }
    return [x, y];
  }

  // Update boid physics.
  // Signature: update(dtSeconds, width, height, neighborsArray)
  // neighborsArray: array of candidate Boid objects (may include self)
  update(dt, width, height, neighbors = []) {
    // Reset acceleration
    this.ax = 0;
    this.ay = 0;

    // Accumulators for behaviors
    let alignX = 0, alignY = 0, alignCount = 0;
    let cohX = 0, cohY = 0, cohCount = 0;
    let sepX = 0, sepY = 0, sepCount = 0;

    const px = this.x, py = this.y;

    for (let other of neighbors) {
      if (other === this) continue;
      // Handle toroidal distances: compute shortest vector considering wrap
      let dx = other.x - px;
      let dy = other.y - py;

      // Adjust for wrap in X
      if (dx > width / 2) dx -= width;
      else if (dx < -width / 2) dx += width;
      // Adjust for wrap in Y
      if (dy > height / 2) dy -= height;
      else if (dy < -height / 2) dy += height;

      const dist = Math.hypot(dx, dy);
      if (dist <= 0) continue;

      if (dist < this.perception) {
        // Alignment: steer towards average heading (velocity)
        alignX += other.vx;
        alignY += other.vy;
        alignCount += 1;

        // Cohesion: steer towards average position
        cohX += other.x + (dx - (other.x - other.x)); // keep using dx,dy for wrap
        cohY += other.y + (dy - (other.y - other.y));
        // Actually we want local relative positions, so sum vector from self to other (dx,dy)
        cohCount += 1;
      }

      if (dist < this.separationRadius) {
        // Separation: steer away inversely proportional to distance
        sepX -= dx / dist;
        sepY -= dy / dist;
        sepCount += 1;
      }
    }

    // Alignment force
    if (alignCount > 0) {
      alignX /= alignCount;
      alignY /= alignCount;
      // desired = normalize(align) * maxSpeed
      let [axDesired, ayDesired] = Boid.limit(alignX, alignY, this.maxSpeed);
      // steer = desired - velocity
      let steerX = axDesired - this.vx;
      let steerY = ayDesired - this.vy;
      // limit steer to maxForce
      [steerX, steerY] = Boid.limit(steerX, steerY, this.maxForce);
      // weight
      const alignWeight = 1.0;
      this.ax += steerX * alignWeight;
      this.ay += steerY * alignWeight;
    }

    // Cohesion force
    if (cohCount > 0) {
      // cohX/cohY currently are sums of offset vectors (we used dx,dy idea), but to be safe:
      // We'll recompute average relative vector to neighbors using the neighbors list again (cheap).
      let avgDX = 0, avgDY = 0, ccount = 0;
      for (let other of neighbors) {
        if (other === this) continue;
        let dx = other.x - px;
        let dy = other.y - py;
        if (dx > width / 2) dx -= width;
        else if (dx < -width / 2) dx += width;
        if (dy > height / 2) dy -= height;
        else if (dy < -height / 2) dy += height;
        const dist = Math.hypot(dx, dy);
        if (dist <= 0) continue;
        if (dist < this.perception) {
          avgDX += dx;
          avgDY += dy;
          ccount += 1;
        }
      }
      if (ccount > 0) {
        avgDX /= ccount;
        avgDY /= ccount;
        // Desired = towards average position: normalize(avgDX,avgDY) * maxSpeed
        let [desX, desY] = Boid.limit(avgDX, avgDY, this.maxSpeed);
        // steer = desired - velocity
        let steerX = desX - this.vx;
        let steerY = desY - this.vy;
        [steerX, steerY] = Boid.limit(steerX, steerY, this.maxForce);
        const cohesionWeight = 0.6;
        this.ax += steerX * cohesionWeight;
        this.ay += steerY * cohesionWeight;
      }
    }

    // Separation force
    if (sepCount > 0) {
      sepX /= sepCount;
      sepY /= sepCount;
      // steer towards sep vector (already points away)
      // scale to maxSpeed then subtract velocity
      let [desX, desY] = Boid.limit(sepX, sepY, this.maxSpeed);
      let steerX = desX - this.vx;
      let steerY = desY - this.vy;
      [steerX, steerY] = Boid.limit(steerX, steerY, this.maxForce);
      const separationWeight = 1.8;
      this.ax += steerX * separationWeight;
      this.ay += steerY * separationWeight;
    }

    // Small random wander to keep things lively
    const jitter = 10; // px/s^2 small
    this.ax += (Math.random() - 0.5) * jitter;
    this.ay += (Math.random() - 0.5) * jitter;

    // Apply acceleration (units: px/s^2). Integrate to velocity with dt seconds.
    this.vx += this.ax * dt;
    this.vy += this.ay * dt;

    // Limit speed
    [this.vx, this.vy] = Boid.limit(this.vx, this.vy, this.maxSpeed);

    // Integrate position
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    // Toroidal wrap-around
    if (this.x < 0) this.x += width;
    else if (this.x >= width) this.x -= width;
    if (this.y < 0) this.y += height;
    else if (this.y >= height) this.y -= height;
  }

  // Draw a filled triangle pointing along velocity.
  draw(ctx) {
    // If nearly still, use a small default heading to avoid NaN
    let angle = Math.atan2(this.vy, this.vx);
    if (!isFinite(angle)) angle = 0;

    const s = this.size;
    ctx.save();
    // Because of toroidal, boids might be drawn across edges by Flock -- but Flock.draw will call draw for each boid at its current position(s).
    ctx.translate(this.x, this.y);
    ctx.rotate(angle);

    ctx.fillStyle = this.color;
    ctx.beginPath();
    // triangle pointing to +x
    ctx.moveTo(s, 0);
    ctx.lineTo(-s * 0.75, -s * 0.6);
    ctx.lineTo(-s * 0.75, s * 0.6);
    ctx.closePath();
    ctx.fill();

    // subtle stroke for contrast
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.stroke();

    ctx.restore();
  }
}
