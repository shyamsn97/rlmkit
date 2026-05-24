(function () {
  var App = window.BoidsApp;
  if (!App) App = (window.BoidsApp = {});

  function torusDelta(d, size) {
    // Minimal signed delta on a torus in [-size/2, size/2]
    return d - Math.round(d / size) * size;
  }

  function setMag(vx, vy, mag) {
    var m = Math.hypot(vx, vy);
    if (m > 0) {
      var s = mag / m;
      return [vx * s, vy * s];
    }
    return [0, 0];
  }

  function limit(vx, vy, max) {
    if (App.util && App.util.limitMag) {
      var out = App.util.limitMag(vx, vy, max);
      return [out[0], out[1]];
    }
    var m = Math.hypot(vx, vy);
    if (m > max && m > 0) {
      var s = max / m;
      return [vx * s, vy * s];
    }
    return [vx, vy];
  }

  function steerToward(currentVx, currentVy, desiredVx, desiredVy, maxForce) {
    var sx = desiredVx - currentVx;
    var sy = desiredVy - currentVy;
    var m = Math.hypot(sx, sy);
    if (m > maxForce && m > 0) {
      var s = maxForce / m;
      sx *= s;
      sy *= s;
    }
    return [sx, sy];
  }

  var rand = (App.util && App.util.rand) ? App.util.rand : function (min, max) {
    return Math.random() * (max - min) + min;
  };

  // Boid class
  App.Boid = function Boid(x, y) {
    this.x = x || 0;
    this.y = y || 0;

    var angle = Math.random() * Math.PI * 2;
    var speed = rand(0.5, 1.5);
    this.vx = Math.cos(angle) * speed;
    this.vy = Math.sin(angle) * speed;

    this.ax = 0;
    this.ay = 0;

    this.hue = Math.floor(Math.random() * 360);
  };

  App.Boid.prototype.update = function (boids, params, width, height) {
    var perception = params.perception || 50;
    var separationDist = params.separationDist || 18;
    var maxSpeed = params.maxSpeed || 3;
    var maxForce = params.maxForce || 0.05;

    var sumAlignX = 0, sumAlignY = 0, countAlign = 0;
    var sumCohX = 0, sumCohY = 0, countCoh = 0;
    var sumSepX = 0, sumSepY = 0, countSep = 0;

    for (var i = 0; i < boids.length; i++) {
      var b = boids[i];
      if (b === this) continue;

      var dx = b.x - this.x;
      var dy = b.y - this.y;
      dx = torusDelta(dx, width);
      dy = torusDelta(dy, height);

      var d = Math.hypot(dx, dy);
      if (d < perception) {
        // Alignment: average neighbor velocity
        sumAlignX += b.vx;
        sumAlignY += b.vy;
        countAlign++;

        // Cohesion: average offset toward neighbors (torus-aware)
        sumCohX += dx;
        sumCohY += dy;
        countCoh++;
      }
      if (d > 0 && d < separationDist) {
        // Separation: weight inversely by distance
        var inv = 1 / d;
        sumSepX -= dx * inv;
        sumSepY -= dy * inv;
        countSep++;
      }
    }

    var steerX = 0, steerY = 0;

    // Alignment steering
    if (countAlign > 0) {
      var avgVx = sumAlignX / countAlign;
      var avgVy = sumAlignY / countAlign;
      var desiredA = setMag(avgVx, avgVy, maxSpeed);
      var sA = steerToward(this.vx, this.vy, desiredA[0], desiredA[1], maxForce);
      steerX += sA[0];
      steerY += sA[1];
    }

    // Cohesion steering
    if (countCoh > 0) {
      var toCenterX = sumCohX / countCoh;
      var toCenterY = sumCohY / countCoh;
      var desiredC = setMag(toCenterX, toCenterY, maxSpeed);
      var sC = steerToward(this.vx, this.vy, desiredC[0], desiredC[1], maxForce);
      steerX += sC[0];
      steerY += sC[1];
    }

    // Separation steering (stronger)
    if (countSep > 0) {
      var sepX = sumSepX / countSep;
      var sepY = sumSepY / countSep;
      var desiredS = setMag(sepX, sepY, maxSpeed);
      var sS = steerToward(this.vx, this.vy, desiredS[0], desiredS[1], maxForce);
      steerX += sS[0] * 1.5;
      steerY += sS[1] * 1.5;
    }

    // Apply acceleration (clamped by maxForce in steering)
    this.ax = steerX;
    this.ay = steerY;

    // Integrate velocity
    this.vx += this.ax;
    this.vy += this.ay;

    // Limit speed
    var limited = limit(this.vx, this.vy, maxSpeed);
    this.vx = limited[0];
    this.vy = limited[1];

    // Update position with toroidal wrap-around
    this.x += this.vx;
    this.y += this.vy;

    if (this.x < 0) this.x += width;
    else if (this.x >= width) this.x -= width;
    if (this.y < 0) this.y += height;
    else if (this.y >= height) this.y -= height;
  };

  App.Boid.prototype.draw = function (ctx) {
    var angle = Math.atan2(this.vy, this.vx);
    var r = 3.2; // boid size

    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(angle);

    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(-r * 0.7, r * 0.5);
    ctx.lineTo(-r * 0.5, 0);
    ctx.lineTo(-r * 0.7, -r * 0.5);
    ctx.closePath();

    ctx.fillStyle = 'hsl(' + this.hue + ', 85%, 55%)';
    ctx.fill();

    // Optional stroke for visibility
    ctx.lineWidth = 0.75;
    ctx.strokeStyle = 'hsla(' + this.hue + ', 85%, 25%, 0.9)';
    ctx.stroke();

    ctx.restore();
  };
})();