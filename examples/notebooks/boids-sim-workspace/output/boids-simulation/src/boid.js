// Boid class, uses global Vec
(function(global){
  const Vec = global.Vec;

  class Boid {
    constructor(x,y, w,h){
      this.pos = new Vec(x,y);
      const angle = Math.random() * Math.PI * 2;
      this.vel = new Vec(Math.cos(angle), Math.sin(angle));
      this.acc = new Vec();
      this.maxForce = 0.05;
      this.maxSpeed = 2.5;
      this.size = 3 + Math.random()*3;
      this.hue = Math.floor(Math.random()*360);
      // bounds for wrapping
      this._w = w;
      this._h = h;
    }

    edges(){
      if(this.pos.x < -10) this.pos.x = this._w + 10;
      if(this.pos.x > this._w + 10) this.pos.x = -10;
      if(this.pos.y < -10) this.pos.y = this._h + 10;
      if(this.pos.y > this._h + 10) this.pos.y = -10;
    }

    align(boids, perception){
      const steering = new Vec();
      let total = 0;
      for(const other of boids){
        const d = this.pos.dist(other.pos);
        if(other !== this && d < perception){
          steering.add(other.vel);
          total++;
        }
      }
      if(total > 0){
        steering.div(total);
        steering.setMag(this.maxSpeed);
        steering.sub(this.vel);
        steering.limit(this.maxForce);
      }
      return steering;
    }

    cohesion(boids, perception){
      const steering = new Vec();
      let total = 0;
      for(const other of boids){
        const d = this.pos.dist(other.pos);
        if(other !== this && d < perception){
          steering.add(other.pos);
          total++;
        }
      }
      if(total > 0){
        steering.div(total);
        steering.sub(this.pos);
        steering.setMag(this.maxSpeed);
        steering.sub(this.vel);
        steering.limit(this.maxForce);
      }
      return steering;
    }

    separation(boids, perception){
      const steering = new Vec();
      let total = 0;
      for(const other of boids){
        const d = this.pos.dist(other.pos);
        if(other !== this && d < perception/1.5){
          const diff = Vec.sub(this.pos, other.pos);
          if(d>0) diff.div(d);
          steering.add(diff);
          total++;
        }
      }
      if(total > 0){
        steering.div(total);
        steering.setMag(this.maxSpeed);
        steering.sub(this.vel);
        steering.limit(this.maxForce * 1.5);
      }
      return steering;
    }

    applyBehaviors(boids, params){
      const align = this.align(boids, params.perception).mul(params.alignWeight);
      const coh = this.cohesion(boids, params.perception).mul(params.cohesionWeight);
      const sep = this.separation(boids, params.perception).mul(params.separationWeight);
      this.acc.add(align).add(coh).add(sep);
    }

    update(speedMultiplier){
      this.vel.add(this.acc);
      this.vel.limit(this.maxSpeed * speedMultiplier);
      this.pos.add(this.vel);
      this.acc.mul(0);
    }

    draw(ctx, showDebug=false){
      const angle = Math.atan2(this.vel.y, this.vel.x);
      ctx.save();
      ctx.translate(this.pos.x, this.pos.y);
      ctx.rotate(angle);
      // color varies with speed for a "colorful" effect
      const speed = this.vel.mag();
      const light = Math.min(70, 20 + speed * 20);
      ctx.fillStyle = `hsl(${this.hue} 80% ${light}%)`;
      ctx.beginPath();
      ctx.moveTo(this.size * 2, 0);
      ctx.lineTo(-this.size, this.size);
      ctx.lineTo(-this.size, -this.size);
      ctx.closePath();
      ctx.fill();
      if(showDebug){
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.beginPath();
        ctx.arc(0,0,10,0,Math.PI*2);
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  global.Boid = Boid;
})(window);
