export const SPEED = 300; // constant px/sec, not based on size

export class Boid {
  constructor(x, y, vx, vy, hue) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.hue = hue; // 0..360
  }

  update(dt, boids, w, h) {
    const PERCEPTION = 55;
    const SEP_DIST = 20;

    const ALIGN_FORCE = 0.8;
    const COHESION_FORCE = 0.5;
    const SEPARATION_FORCE = 1.6;

    let steerAx = 0, steerAy = 0;
    let alignCount = 0, cohCount = 0, sepCount = 0;

    let avgVx = 0, avgVy = 0;       // alignment
    let centerX = 0, centerY = 0;   // cohesion
    let repelX = 0, repelY = 0;     // separation

    const n = boids.length;
    for (let i = 0; i < n; i++) {
      const other = boids[i];
      if (other === this) continue;
      const dx = other.x - this.x;
      const dy = other.y - this.y;
      const d2 = dx*dx + dy*dy;
      if (d2 < PERCEPTION * PERCEPTION) {
        const d = Math.sqrt(d2) || 1e-6;

        // Alignment: match velocity
        avgVx += other.vx;
        avgVy += other.vy;
        alignCount++;

        // Cohesion: steer toward center
        centerX += other.x;
        centerY += other.y;
        cohCount++;

        // Separation: steer away if too close
        if (d < SEP_DIST) {
          // Inverse square falloff
          const inv = 1.0 / (d2 + 1e-6);
          repelX -= dx * inv;
          repelY -= dy * inv;
          sepCount++;
        }
      }
    }

    if (alignCount > 0) {
      avgVx /= alignCount;
      avgVy /= alignCount;
      // Desired is the normalized average scaled to SPEED
      const mag = Math.hypot(avgVx, avgVy) || 1e-6;
      const desAx = (avgVx / mag) * SPEED - this.vx;
      const desAy = (avgVy / mag) * SPEED - this.vy;
      steerAx += desAx * ALIGN_FORCE;
      steerAy += desAy * ALIGN_FORCE;
    }

    if (cohCount > 0) {
      centerX /= cohCount;
      centerY /= cohCount;
      const toCx = centerX - this.x;
      const toCy = centerY - this.y;
      const mag = Math.hypot(toCx, toCy) || 1e-6;
      // Desired toward center at SPEED
      const desVx = (toCx / mag) * SPEED;
      const desVy = (toCy / mag) * SPEED;
      steerAx += (desVx - this.vx) * COHESION_FORCE;
      steerAy += (desVy - this.vy) * COHESION_FORCE;
    }

    if (sepCount > 0) {
      // Normalize separation vector
      const mag = Math.hypot(repelX, repelY) || 1e-6;
      steerAx += (repelX / mag) * SPEED * SEPARATION_FORCE;
      steerAy += (repelY / mag) * SPEED * SEPARATION_FORCE;
    }

    // Gentle bias toward center of the screen to keep flock together
    const cx = w * 0.5, cy = h * 0.5;
    const toCenterX = cx - this.x, toCenterY = cy - this.y;
    steerAx += toCenterX * 0.2;
    steerAy += toCenterY * 0.2;

    // Integrate velocity with steering
    this.vx += steerAx * dt;
    this.vy += steerAy * dt;

    // Enforce constant speed
    let vm = Math.hypot(this.vx, this.vy);
    if (vm < 1e-6) {
      // Re-seed a random heading if we somehow stop
      const a = Math.random() * Math.PI * 2;
      this.vx = Math.cos(a) * SPEED;
      this.vy = Math.sin(a) * SPEED;
    } else {
      this.vx = (this.vx / vm) * SPEED;
      this.vy = (this.vy / vm) * SPEED;
    }

    // Integrate position
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    // Wrap around edges
    if (this.x < 0) this.x += w;
    else if (this.x >= w) this.x -= w;
    if (this.y < 0) this.y += h;
    else if (this.y >= h) this.y -= h;
  }

  draw(ctx) {
    // Draw a small triangle pointing along velocity
    const angle = Math.atan2(this.vy, this.vx);
    const len = 10;  // tip-to-tail in CSS px
    const wing = 6;  // half-width

    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(angle);

    ctx.beginPath();
    ctx.moveTo(len * 0.6, 0);
    ctx.lineTo(-len * 0.5, wing * 0.5);
    ctx.lineTo(-len * 0.5, -wing * 0.5);
    ctx.closePath();

    ctx.fillStyle = `hsl(${this.hue}, 90%, 60%)`;
    ctx.strokeStyle = `hsl(${this.hue}, 90%, 30%)`;
    ctx.lineWidth = 1.0;
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }
}
