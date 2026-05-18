// Small 2D vector utilities using plain objects {x,y}
export function vec(x=0,y=0){return {x, y};}
export function add(a,b){return {x: a.x + b.x, y: a.y + b.y};}
export function sub(a,b){return {x: a.x - b.x, y: a.y - b.y};}
export function mul(v,s){return {x: v.x * s, y: v.y * s};}
export function div(v,s){return {x: v.x / s, y: v.y / s};}
export function mag(v){return Math.hypot(v.x, v.y);}
export function mag2(v){return v.x*v.x + v.y*v.y;}
export function normalize(v){let m = mag(v)||1; return {x: v.x/m, y: v.y/m};}
export function limit(v, max){ let m = mag(v); if (m > max) return mul(normalize(v), max); return v; }
export function dist2(a,b){ let dx = a.x-b.x, dy = a.y-b.y; return dx*dx + dy*dy; }
export function randRange(a,b){ return a + Math.random()*(b-a); }
export function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
export function hueColor(h, s='80%', l='55%'){ return `hsl(${h} ${s} ${l})`; }
