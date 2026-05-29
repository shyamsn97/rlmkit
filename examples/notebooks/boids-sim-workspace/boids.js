(function () {
  // Boids simulation on a canvas with id="boids"
  var canvas = document.getElementById('boids');
  if (!canvas) {
    // Fail-safe: create canvas if not present (but per spec, it's in index.html)
    canvas = document.createElement('canvas');
    canvas.id = 'boids';
    document.body.appendChild(canvas);
  }
  var ctx = canvas.getContext('2d', { alpha: false });

  var width = 0, height = 0, dpr = Math.max(1, window.devicePixelRatio || 1);

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    width = window.innerWidth | 0;
    height = window.innerHeight | 0;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // render in CSS pixel coords
  }

  window.addEventListener('resize', resize);

  // Parameters
  var NUM_BOIDS = 350;
  var PERCEPTION = 45;      // neighbor perception radius
  var SEPARATION_R = 20;    // separation radius
  var COHESION_W = 0.005;
  var ALIGN_W = 0.05;
  var SEPARATE_W = 0.15;
  var MAX_SPEED = 3.2;
  var MAX_FORCE = 0.06;     // gentle steering
  var EDGE_MARGIN = 12;     // wrap-around padding
  var TRI_LEN = 8;          // visual triangle length
  var TRI_WING = 4.5;       // visual wing half-span

  // Grid-based neighbor search for performance
  var GRID_SIZE = PERCEPTION | 0;
  function gridKey(cx, cy) { return (cx << 16) ^ cy; } // fast int key

  var boids = [];
  function randRange(a, b) { return a + Math.random() * (b - a); }
  function mag(x, y) { return Math.sqrt(x * x + y * y); }
  function limit(vx, vy, max) {
    var m = vx * vx + vy * vy;
    if (m > max * max) {
      m = Math.sqrt(m);
      var s = max / (m || 1);
      vx *= s; vy *= s;
    }
    return [vx, vy];
  }
  function setMag(vx, vy, m) {
    var cur = Math.sqrt(vx * vx + vy * vy) || 1;
    var s = m / cur;
    return [vx * s, vy * s];
  }

  function makeBoid(i) {
    var angle = randRange(0, Math.PI * 2);
    var speed = randRange(0.5 * MAX_SPEED, MAX_SPEED);
    return {
      x: randRange(0, width),
      y: randRange(0, height),
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      ax: 0,
      ay: 0,
      hueBase: (Math.random() * 360) | 0,
      id: i
    };
  }

  function rebuildGrid() {
    grid = Object.create(null);
    var gs = GRID_SIZE;
    for (var i = 0; i < boids.length; i++) {
      var b = boids[i];
      var cx = (b.x / gs) | 0;
      var cy = (b.y / gs) | 0;
      var key = gridKey(cx, cy);
      var cell = grid[key];
      if (!cell) grid[key] = cell = [];
      cell.push(i);
    }
  }

  function neighbors(index, outIdx) {
    // Collect neighbor indices from 9 cells (self + 8 around)
    var b = boids[index];
    var gs = GRID_SIZE;
    var cx = (b.x / gs) | 0;
    var cy = (b.y / gs) | 0;
    var k = 0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        var key = gridKey(cx + dx, cy + dy);
        var cell = grid[key];
        if (!cell) continue;
        for (var j = 0; j < cell.length; j++) {
          var idx = cell[j];
          if (idx !== index) {
            outIdx[k++] = idx;
          }
        }
      }
    }
    return k; // number of items populated in outIdx
  }

  // Simulation buffers
  var grid = Object.create(null);
  var neighIdx = new Int32Array(1024); // grows if needed per frame

  function step(dt) {
    // Build grid for neighbor queries
    rebuildGrid();

    // If many neighbors, ensure buffer large enough
    var potentialMax = boids.length * 9; // loose upper bound (very conservative)
    if (neighIdx.length < potentialMax) {
      neighIdx = new Int32Array(potentialMax);
    }

    var pr = PERCEPTION;
    var sr = SEPARATION_R;
    var pr2 = pr * pr;
    var sr2 = sr * sr;

    for (var i = 0; i < boids.length; i++) {
      var b = boids[i];

      var alignX = 0, alignY = 0;
      var cohX = 0, cohY = 0;
      var sepX = 0, sepY = 0;
      var countA = 0, countC = 0, countS = 0;

      // Collect neighbor indices
      var nCount = neighbors(i, neighIdx);

      for (var n = 0; n < nCount; n++) {
        var j = neighIdx[n];
        var o = boids[j];
        var dx = o.x - b.x;
        var dy = o.y - b.y;
        var d2 = dx * dx + dy * dy;

        if (d2 < pr2) {
          // Alignment and cohesion consider neighbors within perception
          alignX += o.vx;
          alignY += o.vy;
          cohX += o.x;
          cohY += o.y;
          countA++;
          countC++;
        }
        if (d2 < sr2 && d2 > 0.0001) {
          // Separation stronger at closer distances
          var inv = 1 / Math.sqrt(d2);
          // Weight by 1/d to push harder for very close neighbors
          sepX -= dx * inv;
          sepY -= dy * inv;
          countS++;
        }
      }

      // Reset acceleration
      var ax = 0, ay = 0;

      // Alignment
      if (countA > 0) {
        alignX /= countA; alignY /= countA;
        var aVec = setMag(alignX, alignY, MAX_SPEED);
        aVec[0] -= b.vx; aVec[1] -= b.vy;
        aVec = limit(aVec[0], aVec[1], MAX_FORCE);
        ax += aVec[0] * ALIGN_W;
        ay += aVec[1] * ALIGN_W;
      }

      // Cohesion
      if (countC > 0) {
        cohX = (cohX / countC) - b.x;
        cohY = (cohY / countC) - b.y;
        var cVec = setMag(cohX, cohY, MAX_SPEED);
        cVec[0] -= b.vx; cVec[1] -= b.vy;
        cVec = limit(cVec[0], cVec[1], MAX_FORCE);
        ax += cVec[0] * COHESION_W;
        ay += cVec[1] * COHESION_W;
      }

      // Separation
      if (countS > 0) {
        sepX /= countS; sepY /= countS;
        var sVec = setMag(sepX, sepY, MAX_SPEED);
        sVec[0] -= b.vx; sVec[1] -= b.vy;
        sVec = limit(sVec[0], sVec[1], MAX_FORCE * 1.2);
        ax += sVec[0] * SEPARATE_W;
        ay += sVec[1] * SEPARATE_W;
      }

      // Apply acceleration
      b.vx += ax;
      b.vy += ay;

      // Limit speed
      var vLimited = limit(b.vx, b.vy, MAX_SPEED);
      b.vx = vLimited[0]; b.vy = vLimited[1];

      // Integrate
      b.x += b.vx * dt;
      b.y += b.vy * dt;

      // Wrap-around edges with margin
      if (b.x < -EDGE_MARGIN) b.x = width + EDGE_MARGIN;
      else if (b.x > width + EDGE_MARGIN) b.x = -EDGE_MARGIN;
      if (b.y < -EDGE_MARGIN) b.y = height + EDGE_MARGIN;
      else if (b.y > height + EDGE_MARGIN) b.y = -EDGE_MARGIN;
    }
  }

  function draw() {
    // Clear with dark background (canvas is opaque by default due to alpha:false)
    ctx.fillStyle = '#0b0e11';
    ctx.fillRect(0, 0, width, height);

    for (var i = 0; i < boids.length; i++) {
      var b = boids[i];
      var ang = Math.atan2(b.vy, b.vx);
      var ca = Math.cos(ang), sa = Math.sin(ang);

      // Triangle points oriented by velocity
      var len = TRI_LEN, wing = TRI_WING;
      var x = b.x, y = b.y;

      var tipX = x + ca * len;
      var tipY = y + sa * len;
      var tailX = x - ca * (len * 0.6);
      var tailY = y - sa * (len * 0.6);

      // Wings using a perpendicular vector
      var leftX = tailX + (-sa) * wing;
      var leftY = tailY + (ca) * wing;
      var rightX = tailX - (-sa) * wing;
      var rightY = tailY - (ca) * wing;

      // Color by base hue and speed
      var speed = Math.sqrt(b.vx * b.vx + b.vy * b.vy);
      var hue = (b.hueBase + speed * 40) % 360;
      var light = 45 + Math.min(15, speed * 3.5); // subtle speed-based lightness

      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(leftX, leftY);
      ctx.lineTo(rightX, rightY);
      ctx.closePath();

      ctx.fillStyle = 'hsl(' + hue.toFixed(0) + ', 80%,' + light.toFixed(0) + '%)';
      ctx.strokeStyle = 'hsla(' + hue.toFixed(0) + ', 90%, 70%, 0.25)';
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
    }
  }

  // Main loop
  var last = 0;
  function loop(ts) {
    if (!last) last = ts;
    var dt = (ts - last) / 16.6667; // normalize to ~60 FPS steps
    if (dt > 2.5) dt = 2.5; // clamp for stability if tab was inactive
    last = ts;

    step(dt);
    draw();
    requestAnimationFrame(loop);
  }

  // Init
  resize();
  boids.length = 0;
  for (var i = 0; i < NUM_BOIDS; i++) {
    boids.push(makeBoid(i));
  }

  requestAnimationFrame(loop);
})();