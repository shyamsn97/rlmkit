// Flock manages many Boid instances
class Flock {
  constructor(width, height, num) {
    this.width = width;
    this.height = height;
    this.boids = [];
    for (var i = 0; i < num; i++) {
      var x = Math.random() * width;
      var y = Math.random() * height;
      this.boids.push(new Boid(x, y, i));
    }
  }

  addBoid(b) {
    this.boids.push(b);
  }

  // Run one simulation step: update all boids then draw them
  run(ctx) {
    // update
    for (var b of this.boids) {
      b.update(this.boids, this.width, this.height);
    }
    // draw
    for (var b of this.boids) {
      b.draw(ctx);
    }
  }
}
