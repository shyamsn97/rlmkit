// Simple 2D vector utility usable as a global "Vec"
(function(global){
  class Vec {
    constructor(x=0,y=0){ this.x = x; this.y = y; }
    copy(){ return new Vec(this.x,this.y); }
    add(v){ this.x += v.x; this.y += v.y; return this; }
    sub(v){ this.x -= v.x; this.y -= v.y; return this; }
    mul(s){ this.x *= s; this.y *= s; return this; }
    div(s){ if(s!==0){ this.x /= s; this.y /= s; } return this; }
    mag(){ return Math.hypot(this.x, this.y); }
    setMag(m){ const d = this.mag() || 1; this.mul(m / d); return this; }
    normalize(){ const m = this.mag() || 1; this.div(m); return this; }
    limit(max){ if(this.mag() > max) this.setMag(max); return this; }
    dist(v){ return Math.hypot(this.x - v.x, this.y - v.y); }
    static sub(a,b){ return new Vec(a.x - b.x, a.y - b.y); }
    static add(a,b){ return new Vec(a.x + b.x, a.y + b.y); }
  }
  global.Vec = Vec;
})(window);
