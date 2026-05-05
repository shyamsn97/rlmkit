export class Renderer {
  constructor(ctx) { this.ctx = ctx; this.width = 0; this.height = 0; }
  setSize(width, height){ this.width = width; this.height = height; }
  clear(){
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    // Optional subtle background:
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, this.width, this.height);
  }
  draw(boids){
    const ctx = this.ctx;
    for (let i=0;i<boids.length;i++){
      const b = boids[i];
      const angle = Math.atan2(b.vy, b.vx);
      const s = b.size || 3.5;
      const head = s * 2.2;
      const wing = s * 1.2;
      const x = b.x, y = b.y;

      // Triangle points oriented along 'angle'
      const cos = Math.cos(angle), sin = Math.sin(angle);
      const x1 = x + cos * head,      y1 = y + sin * head;        // nose
      const x2 = x - cos * head + -sin * wing, y2 = y - sin * head +  cos * wing; // left
      const x3 = x - cos * head +  sin * wing, y3 = y - sin * head + -cos * wing; // right

      ctx.fillStyle = b.color || "#fff";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x3, y3);
      ctx.closePath();
      ctx.fill();
    }
  }
}
