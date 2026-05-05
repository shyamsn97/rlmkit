import { Boid, spawnBoids } from "./boid.js";

export class Simulation {
  constructor(width, height, count = 300, speed = 240) {
    this.width = width;
    this.height = height;
    this.speed = speed;
    this.boids = spawnBoids(count, width, height, speed);
  }

  update(dt) {
    const b = this.boids, w = this.width, h = this.height, s = this.speed;
    for (let i = 0; i < b.length; i++) b[i].update(b, w, h, dt, s);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    // no re-spawn; boids will wrap into new bounds naturally
  }

  get boidsArray() { return this.boids; }
}

export { Simulation };
