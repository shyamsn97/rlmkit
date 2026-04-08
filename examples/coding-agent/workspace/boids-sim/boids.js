// boids.js — Craig Reynolds' Boids Flocking Simulation

const canvas = document.getElementById('canvas') || document.querySelector('canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Track mouse position
const mouse = { x: null, y: null };
canvas.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});
canvas.addEventListener('mouseleave', () => {
    mouse.x = null;
    mouse.y = null;
});

// --- PARAMETERS ---
const params = {
    count: 200,
    separation: 1.5,
    alignment: 1.0,
    cohesion: 1.0,
    visualRange: 75,
    maxSpeed: 4,
    minSpeed: 1.5,
    trail: 0,
    predator: false,
    wrapEdges: true,
    margin: 100,
    turnFactor: 0.3
};

// --- BOID CLASS ---
class Boid {
    constructor(x, y, vx, vy) {
        this.x = x !== undefined ? x : Math.random() * canvas.width;
        this.y = y !== undefined ? y : Math.random() * canvas.height;
        this.vx = vx !== undefined ? vx : (Math.random() - 0.5) * 2;
        this.vy = vy !== undefined ? vy : (Math.random() - 0.5) * 2;
        this.hue = Math.random() * 360;
        this.history = [];
    }
}

// --- SPATIAL HASH GRID ---
class SpatialHashGrid {
    constructor(cellSize) {
        this.cellSize = cellSize;
        this.cells = new Map();
    }

    clear() {
        this.cells.clear();
    }

    _key(x, y) {
        const cx = Math.floor(x / this.cellSize);
        const cy = Math.floor(y / this.cellSize);
        return cx + ',' + cy;
    }

    insert(boid) {
        const key = this._key(boid.x, boid.y);
        if (!this.cells.has(key)) {
            this.cells.set(key, []);
        }
        this.cells.get(key).push(boid);
    }

    query(boid, range) {
        const neighbors = [];
        const r = range;
        const cx = Math.floor(boid.x / this.cellSize);
        const cy = Math.floor(boid.y / this.cellSize);
        const cellRange = Math.ceil(r / this.cellSize);

        for (let dx = -cellRange; dx <= cellRange; dx++) {
            for (let dy = -cellRange; dy <= cellRange; dy++) {
                const key = (cx + dx) + ',' + (cy + dy);
                const cell = this.cells.get(key);
                if (!cell) continue;
                for (let i = 0; i < cell.length; i++) {
                    const other = cell[i];
                    if (other === boid) continue;
                    const distX = other.x - boid.x;
                    const distY = other.y - boid.y;
                    const distSq = distX * distX + distY * distY;
                    if (distSq < r * r) {
                        neighbors.push(other);
                    }
                }
            }
        }
        return neighbors;
    }
}

let grid = new SpatialHashGrid(params.visualRange);
let boids = [];

function initBoids(count) {
    boids = [];
    for (let i = 0; i < count; i++) {
        boids.push(new Boid());
    }
}

initBoids(params.count);

// --- RULES ---
function applyRules(boid, neighbors) {
    let sepX = 0, sepY = 0;
    let alignX = 0, alignY = 0;
    let cohX = 0, cohY = 0;
    let alignCount = 0;
    let cohCount = 0;

    const sepRange = params.visualRange * 0.4;
    const sepRangeSq = sepRange * sepRange;

    for (let i = 0; i < neighbors.length; i++) {
        const other = neighbors[i];
        const dx = boid.x - other.x;
        const dy = boid.y - other.y;
        const distSq = dx * dx + dy * dy;

        // Separation: steer away from very close boids
        if (distSq < sepRangeSq && distSq > 0) {
            sepX += dx / distSq;
            sepY += dy / distSq;
        }

        // Alignment: average heading
        alignX += other.vx;
        alignY += other.vy;
        alignCount++;

        // Cohesion: average position
        cohX += other.x;
        cohY += other.y;
        cohCount++;
    }

    // Apply separation
    boid.vx += sepX * params.separation;
    boid.vy += sepY * params.separation;

    // Apply alignment
    if (alignCount > 0) {
        alignX /= alignCount;
        alignY /= alignCount;
        boid.vx += (alignX - boid.vx) * params.alignment * 0.05;
        boid.vy += (alignY - boid.vy) * params.alignment * 0.05;
    }

    // Apply cohesion
    if (cohCount > 0) {
        cohX /= cohCount;
        cohY /= cohCount;
        boid.vx += (cohX - boid.x) * params.cohesion * 0.005;
        boid.vy += (cohY - boid.y) * params.cohesion * 0.005;
    }
}

function applyEdgeBehavior(boid) {
    if (params.wrapEdges) {
        if (boid.x < 0) boid.x += canvas.width;
        if (boid.x > canvas.width) boid.x -= canvas.width;
        if (boid.y < 0) boid.y += canvas.height;
        if (boid.y > canvas.height) boid.y -= canvas.height;
    } else {
        const m = params.margin;
        const t = params.turnFactor;
        if (boid.x < m) boid.vx += t;
        if (boid.x > canvas.width - m) boid.vx -= t;
        if (boid.y < m) boid.vy += t;
        if (boid.y > canvas.height - m) boid.vy -= t;
    }
}

function applyMouseInteraction(boid) {
    if (mouse.x === null || mouse.y === null) return;
    const dx = boid.x - mouse.x;
    const dy = boid.y - mouse.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (params.predator) {
        // Flee from mouse
        if (dist < 150 && dist > 0) {
            const force = (150 - dist) / 150;
            boid.vx += (dx / dist) * force * 0.8;
            boid.vy += (dy / dist) * force * 0.8;
        }
    } else {
        // Gently attracted to mouse
        if (dist < 200 && dist > 0) {
            boid.vx -= (dx / dist) * 0.15;
            boid.vy -= (dy / dist) * 0.15;
        }
    }
}

function limitSpeed(boid) {
    const speed = Math.sqrt(boid.vx * boid.vx + boid.vy * boid.vy);
    if (speed > params.maxSpeed) {
        boid.vx = (boid.vx / speed) * params.maxSpeed;
        boid.vy = (boid.vy / speed) * params.maxSpeed;
    }
    if (speed < params.minSpeed) {
        boid.vx = (boid.vx / speed) * params.minSpeed;
        boid.vy = (boid.vy / speed) * params.minSpeed;
    }
}

// --- RENDERING ---
function drawBoid(boid) {
    const angle = Math.atan2(boid.vy, boid.vx);
    const size = 8;
    const color = `hsl(${boid.hue}, 80%, 55%)`;

    // Draw trail
    if (params.trail > 0 && boid.history.length > 1) {
        for (let i = 1; i < boid.history.length; i++) {
            const alpha = i / boid.history.length * 0.6;
            ctx.beginPath();
            ctx.moveTo(boid.history[i - 1][0], boid.history[i - 1][1]);
            ctx.lineTo(boid.history[i][0], boid.history[i][1]);
            ctx.strokeStyle = `hsla(${boid.hue}, 80%, 55%, ${alpha})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    // Draw triangle / arrow
    ctx.save();
    ctx.translate(boid.x, boid.y);
    ctx.rotate(angle);

    ctx.beginPath();
    ctx.moveTo(size, 0);           // tip
    ctx.lineTo(-size * 0.6, size * 0.4);  // bottom-left
    ctx.lineTo(-size * 0.3, 0);           // indent
    ctx.lineTo(-size * 0.6, -size * 0.4); // top-left
    ctx.closePath();

    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
}

// --- ANIMATION LOOP ---
let paused = false;
let lastTime = performance.now();
let frameCount = 0;
let fps = 0;
let fpsTimer = 0;

function update() {
    if (!paused) {
        // Rebuild spatial grid
        grid = new SpatialHashGrid(params.visualRange);
        for (let i = 0; i < boids.length; i++) {
            grid.insert(boids[i]);
        }

        for (let i = 0; i < boids.length; i++) {
            const boid = boids[i];
            const neighbors = grid.query(boid, params.visualRange);

            applyRules(boid, neighbors);
            applyMouseInteraction(boid);
            applyEdgeBehavior(boid);
            limitSpeed(boid);

            // Update position
            boid.x += boid.vx;
            boid.y += boid.vy;

            // Update trail history
            if (params.trail > 0) {
                boid.history.push([boid.x, boid.y]);
                if (boid.history.length > params.trail) {
                    boid.history.shift();
                }
            } else {
                boid.history.length = 0;
            }
        }
    }

    // Render
    if (params.trail > 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    for (let i = 0; i < boids.length; i++) {
        drawBoid(boids[i]);
    }

    // FPS calculation
    const now = performance.now();
    frameCount++;
    fpsTimer += now - lastTime;
    lastTime = now;
    if (fpsTimer >= 1000) {
        fps = frameCount;
        frameCount = 0;
        fpsTimer = 0;
    }

    const statsEl = document.getElementById('stats');
    if (statsEl) {
        statsEl.textContent = `FPS: ${fps} | Boids: ${boids.length}`;
    }

    requestAnimationFrame(update);
}

// --- UI CONTROLS ---
function bindSlider(id, paramKey, parser) {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(id + 'Val');
    if (!slider) return;

    slider.value = params[paramKey];
    if (valSpan) valSpan.textContent = params[paramKey];

    slider.addEventListener('input', () => {
        const val = parser ? parser(slider.value) : parseFloat(slider.value);
        params[paramKey] = val;
        if (valSpan) valSpan.textContent = val;

        // Special handling for count
        if (paramKey === 'count') {
            while (boids.length < val) {
                boids.push(new Boid());
            }
            while (boids.length > val) {
                boids.pop();
            }
        }
    });
}

bindSlider('count', 'count', v => parseInt(v));
bindSlider('separation', 'separation', v => parseFloat(v));
bindSlider('alignment', 'alignment', v => parseFloat(v));
bindSlider('cohesion', 'cohesion', v => parseFloat(v));
bindSlider('visualRange', 'visualRange', v => parseFloat(v));
bindSlider('maxSpeed', 'maxSpeed', v => parseFloat(v));
bindSlider('trail', 'trail', v => parseInt(v));

// Checkboxes
const predatorCb = document.getElementById('predator');
if (predatorCb) {
    predatorCb.checked = params.predator;
    predatorCb.addEventListener('change', () => {
        params.predator = predatorCb.checked;
    });
}

const wrapEdgesCb = document.getElementById('wrapEdges');
if (wrapEdgesCb) {
    wrapEdgesCb.checked = params.wrapEdges;
    wrapEdgesCb.addEventListener('change', () => {
        params.wrapEdges = wrapEdgesCb.checked;
    });
}

// Reset button
const resetBtn = document.getElementById('resetBtn');
if (resetBtn) {
    resetBtn.addEventListener('click', () => {
        initBoids(params.count);
    });
}

// Pause button
const pauseBtn = document.getElementById('pauseBtn');
if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
        paused = !paused;
        pauseBtn.textContent = paused ? 'Resume' : 'Pause';
    });
}

// Toggle controls panel
const toggleBtn = document.getElementById('toggle-btn');
const controlsPanel = document.getElementById('controls');
if (toggleBtn && controlsPanel) {
    toggleBtn.addEventListener('click', () => {
        if (controlsPanel.style.display === 'none') {
            controlsPanel.style.display = '';
        } else {
            controlsPanel.style.display = 'none';
        }
    });
}

// Start animation
requestAnimationFrame(update);
