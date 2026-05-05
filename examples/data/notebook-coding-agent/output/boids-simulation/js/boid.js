// output/boids-simulation/js/boid.js
// Implements Boid and spawnBoids per contract.

export class Boid {
  constructor(x, y, angle, speed, color) {
    this.x = x;
    this.y = y;
    this.vx = Math.cos(angle) * speed;
    this.vy = Math.sin(angle) * speed;
    this.size = 3.5;
    this.color = color;
  }

  /**
   * Update this boid's velocity and position.
   * boids: array of Boid
   * width, height: world dimensions
   * dt: timestep (s)
   * speed: desired constant speed for boids
   */
  update(boids, width, height, dt, speed) {
    // Internal constants
    const PERCEPTION = 55;
    const SEPARATION_DIST = 20;
    const ALIGN_W = 1.0;
    const COHESION_W = 0.8;
    const SEPARATION_W = 1.6;
    const STEER = 2.0; // blend rate (1/s)

    const PERCEPTION_R2 = PERCEPTION * PERCEPTION;
    const SEPARATION_R2 = SEPARATION_DIST * SEPARATION_DIST;

    // Accumulators (avoid allocations inside loop)
    let alignVX = 0, alignVY = 0, alignCount = 0;
    let cohX = 0, cohY = 0, cohCount = 0;
    let sepX = 0, sepY = 0, sepCount = 0;

    // Gather neighbors
    for (let i = 0, n = boids.length; i < n; i++) {
      const other = boids[i];
      if (other === this) continue;
      let dx = other.x - this.x;
      let dy = other.y - this.y;

      // Account for toroidal wrap when measuring distance (shortest distance)
      // Use simple wrap: consider offsets of -width, 0, +width and -height, 0, +height
      // but to keep it cheap, adjust dx/dy by wrapping if over half-dimension
      if (dx > width * 0.5) dx -= width;
      else if (dx < -width * 0.5) dx += width;
      if (dy > height * 0.5) dy -= height;
      else if (dy < -height * 0.5) dy += height;

      const dist2 = dx * dx + dy * dy;
      if (dist2 > PERCEPTION_R2) continue;

      // Alignment: average neighbor velocity
      alignVX += other.vx;
      alignVY += other.vy;
      alignCount++;

      // Cohesion: average neighbor position (we'll turn into vector toward avg)
      cohX += other.x + ((other.x - this.x) < -width * 0.5 ? width : ((other.x - this.x) > width * 0.5 ? -width : 0));
      cohY += other.y + ((other.y - this.y) < -height * 0.5 ? height : ((other.y - this.y) > height * 0.5 ? -height : 0));
      cohCount++;

      // Separation: only for close neighbors
      if (dist2 < SEPARATION_R2 && dist2 > 0) {
        const dist = Math.sqrt(dist2);
        // vector pointing away, weighted by inverse distance (closer -> stronger)
        sepX += (this.x - other.x);
        sepY += (this.y - other.y);
        // NOTE: sepCount used to normalize later
        sepCount++;
      }
    }

    // Helper to normalize and scale vector to magnitude 'speed'
    function normalizedToSpeed(ix, iy) {
      const m = Math.sqrt(ix * ix + iy * iy);
      if (m === 0) return [0, 0];
      const s = speed / m;
      return [ix * s, iy * s];
    }

    // Compute desired vectors for each rule (as velocities of magnitude 'speed')
    let desiredAlignX = 0, desiredAlignY = 0;
    if (alignCount > 0) {
      const avgVX = alignVX / alignCount;
      const avgVY = alignVY / alignCount;
      [desiredAlignX, desiredAlignY] = normalizedToSpeed(avgVX, avgVY);
    }

    let desiredCohX = 0, desiredCohY = 0;
    if (cohCount > 0) {
      // average neighbor position
      const avgX = cohX / cohCount;
      const avgY = cohY / cohCount;
      // vector from this boid toward average position (consider wrapping shortest path)
      let vx = avgX - this.x;
      let vy = avgY - this.y;
      // adjust for wrap-around shortest
      if (vx > width * 0.5) vx -= width;
      else if (vx < -width * 0.5) vx += width;
      if (vy > height * 0.5) vy -= height;
      else if (vy < -height * 0.5) vy += height;
      [desiredCohX, desiredCohY] = normalizedToSpeed(vx, vy);
    }

    let desiredSepX = 0, desiredSepY = 0;
    if (sepCount > 0) {
      // average separation vector (points away from neighbors)
      const avgSepX = sepX / sepCount;
      const avgSepY = sepY / sepCount;
      [desiredSepX, desiredSepY] = normalizedToSpeed(avgSepX, avgSepY);
    }

    // If no neighbors at all, desired = current velocity normalized to speed
    let desiredCurrX = 0, desiredCurrY = 0;
    [desiredCurrX, desiredCurrY] = normalizedToSpeed(this.vx, this.vy);

    // Combine weighted desires. If a particular desire is zero (no neighbors), it contributes nothing.
    let combinedX =
      (desiredAlignX * ALIGN_W) +
      (desiredCohX * COHESION_W) +
      (desiredSepX * SEPARATION_W);

    let combinedY =
      (desiredAlignY * ALIGN_W) +
      (desiredCohY * COHESION_W) +
      (desiredSepY * SEPARATION_W);

    // If combined is zero (no neighbors), fallback to current direction
    if (combinedX === 0 && combinedY === 0) {
      combinedX = desiredCurrX;
      combinedY = desiredCurrY;
    } else {
      // It's often good to normalize the combined vector to speed as well
      const cmag = Math.sqrt(combinedX * combinedX + combinedY * combinedY);
      if (cmag > 0) {
        const s = speed / cmag;
        combinedX *= s;
        combinedY *= s;
      }
    }

    // Blend current velocity toward desired using STEER rate
    this.vx += (combinedX - this.vx) * STEER * dt;
    this.vy += (combinedY - this.vy) * STEER * dt;

    // Renormalize to exact 'speed' magnitude
    const mag = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    if (mag > 0) {
      const scale = speed / mag;
      this.vx *= scale;
      this.vy *= scale;
    } else {
      // If zero (unlikely), give a small random jitter to avoid lock
      const ang = Math.random() * Math.PI * 2;
      this.vx = Math.cos(ang) * speed;
      this.vy = Math.sin(ang) * speed;
    }

    // Update position
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    // Wrap around edges with margin = 2 * size
    const margin = 2 * this.size;
    if (this.x < -margin) this.x += width + margin * 2;
    else if (this.x > width + margin) this.x -= width + margin * 2;
    if (this.y < -margin) this.y += height + margin * 2;
    else if (this.y > height + margin) this.y -= height + margin * 2;
  }
}

/**
 * Spawn n boids randomly across the world.
 * Returns an array of Boid instances.
 */
export function spawnBoids(n, width, height, speed) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const angle = Math.random() * Math.PI * 2;
    const hue = Math.floor(Math.random() * 360);
    const color = `hsl(${hue},90%,60%)`;
    out[i] = new Boid(x, y, angle, speed, color);
  }
  return out;
}
