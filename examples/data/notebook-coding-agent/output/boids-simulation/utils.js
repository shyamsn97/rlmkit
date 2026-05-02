// utils.js
// Helpers for the boids simulation per CONTRACT:
// - export function randRange(min, max)
// - export class SpatialHash with constructor(cellSize), clear(), insert(item), query(x,y,r)

export function randRange(min, max) {
  return Math.random() * (max - min) + min;
}

export class SpatialHash {
  constructor(cellSize) {
    if (!(cellSize > 0)) throw new Error("cellSize must be > 0");
    this.cellSize = cellSize;
    this.buckets = new Map(); // key: "cx,cy" -> Array of items
  }

  _cellCoords(x, y) {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    return [cx, cy];
  }

  _key(cx, cy) {
    return cx + "," + cy;
  }

  clear() {
    this.buckets.clear();
  }

  // item must have numeric .x and .y properties
  insert(item) {
    const [cx, cy] = this._cellCoords(item.x, item.y);
    const key = this._key(cx, cy);
    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = [];
      this.buckets.set(key, bucket);
    }
    bucket.push(item);
  }

  // Return items whose positions are within radius r of (x,y).
  // Scans neighboring cells that could contain points within r.
  query(x, y, r) {
    const [cx, cy] = this._cellCoords(x, y);
    const cellRadius = Math.ceil(r / this.cellSize);
    const r2 = r * r;
    const results = [];
    const seen = new Set();

    for (let dx = -cellRadius; dx <= cellRadius; dx++) {
      for (let dy = -cellRadius; dy <= cellRadius; dy++) {
        const key = this._key(cx + dx, cy + dy);
        const bucket = this.buckets.get(key);
        if (!bucket) continue;
        for (const item of bucket) {
          // dedupe same object reference
          if (seen.has(item)) continue;
          const ix = item.x;
          const iy = item.y;
          const ddx = ix - x;
          const ddy = iy - y;
          if (ddx * ddx + ddy * ddy <= r2) {
            results.push(item);
            seen.add(item);
          }
        }
      }
    }

    return results;
  }
}
