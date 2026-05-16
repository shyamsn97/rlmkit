// Utility vector helpers (plain functions using {x,y} objects)
function createVec(x = 0, y = 0) { return {x: x, y: y}; }
function add(a, b) { return {x: a.x + b.x, y: a.y + b.y}; }
function sub(a, b) { return {x: a.x - b.x, y: a.y - b.y}; }
function mult(v, n) { return {x: v.x * n, y: v.y * n}; }
function div(v, n) { return {x: v.x / n, y: v.y / n}; }
function mag(v) { return Math.sqrt(v.x * v.x + v.y * v.y); }
function setMag(v, m) { var g = mag(v) || 1; return mult(div(v, g), m); }
function normalize(v) { var m = mag(v) || 1; return {x: v.x / m, y: v.y / m}; }
function limit(v, max) {
  var m = mag(v);
  if (m > max) return setMag(v, max);
  return {x: v.x, y: v.y};
}
function dist(a, b) { var dx = a.x - b.x, dy = a.y - b.y; return Math.sqrt(dx*dx + dy*dy); }

// Wrap position across edges (toroidal world)
function wrapPosition(pos, width, height) {
  var x = pos.x, y = pos.y;
  if (x < 0) x += width;
  else if (x >= width) x -= width;
  if (y < 0) y += height;
  else if (y >= height) y -= height;
  return {x: x, y: y};
}
