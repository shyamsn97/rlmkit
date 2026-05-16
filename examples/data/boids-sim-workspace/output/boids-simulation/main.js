// Entry: setup canvas, create flock, animate with FPS display
(function () {
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  var overlayFps = document.getElementById('fps');
  var overlayCount = document.getElementById('count');

  var dpr = Math.max(1, window.devicePixelRatio || 1);
  function resize() {
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (flock) {
      flock.width = window.innerWidth;
      flock.height = window.innerHeight;
    }
  }
  window.addEventListener('resize', resize);

  var numBoids = 400; // configurable count (hundreds)
  var flock = new Flock(window.innerWidth || 800, window.innerHeight || 600, numBoids);

  // nice additive trail effect: clear fully each frame for crisp, fast visuals
  function clear() {
    ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
  }

  var last = performance.now();
  var frames = 0;
  var fps = 0;
  var fpsTimer = performance.now();

  function loop(now) {
    var dt = now - last;
    last = now;
    frames++;
    if (now - fpsTimer >= 250) { // update fps several times a second
      fps = Math.round((frames * 1000) / (now - fpsTimer));
      fpsTimer = now;
      frames = 0;
      overlayFps.textContent = 'FPS: ' + fps;
    }
    overlayCount.textContent = 'Boids: ' + flock.boids.length;

    // update & draw
    clear();
    flock.run(ctx);

    requestAnimationFrame(loop);
  }

  // boot
  resize();
  requestAnimationFrame(loop);

  // Expose for console tuning
  window._flock = flock;
  window._canvas = canvas;
})();
