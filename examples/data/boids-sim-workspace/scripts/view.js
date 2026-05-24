(function () {
  var App = (window.BoidsApp = window.BoidsApp || {});

  // Resize canvas to window size with device pixel ratio for crisp rendering.
  App.resizeCanvas = function (canvas, state) {
    if (!canvas) return;

    var dpr = window.devicePixelRatio || 1;
    var width = Math.max(1, Math.floor(window.innerWidth || canvas.clientWidth || 1));
    var height = Math.max(1, Math.floor(window.innerHeight || canvas.clientHeight || 1));

    // Set the internal pixel buffer size
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    // Leave CSS sizing to stylesheet (fullscreen), avoid overriding style.width/height here.

    // Update state
    if (state) {
      state.width = width;
      state.height = height;
      state.dpr = dpr;
    }

    // Ensure context transform matches DPR so we can use CSS pixel coordinates when drawing
    var ctx = (state && state.ctx) ? state.ctx : canvas.getContext('2d');
    if (ctx && ctx.setTransform) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if ('imageSmoothingEnabled' in ctx) ctx.imageSmoothingEnabled = false;
    }
  };

  // Clear and render all boids
  App.render = function (state) {
    if (!state || !state.canvas || !state.ctx) return;
    var ctx = state.ctx;
    var dpr = state.dpr || 1;

    // Re-assert DPR transform in case anything changed
    if (ctx.setTransform) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear the visible area in CSS pixel units
    ctx.clearRect(0, 0, state.width, state.height);

    var boids = state.boids || [];
    for (var i = 0; i < boids.length; i++) {
      var boid = boids[i];
      if (boid && typeof boid.draw === 'function') {
        boid.draw(ctx);
      }
    }
  };
})();