(function () {
  window.addEventListener('load', function () {
    const App = window.BoidsApp || (window.BoidsApp = {});
    const state = App.state || (App.state = {});
    const params = App.params || (App.params = { numBoids: 300 });

    const canvas = document.getElementById('boids');
    if (!canvas) return;

    state.canvas = canvas;
    state.ctx = canvas.getContext('2d');

    // Initial DPR-aware sizing
    if (typeof App.resizeCanvas === 'function') {
      App.resizeCanvas(canvas, state);
    } else {
      // Fallback sizing if resize helper not yet available
      state.dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * state.dpr);
      canvas.height = Math.floor(window.innerHeight * state.dpr);
      canvas.style.width = window.innerWidth + 'px';
      canvas.style.height = window.innerHeight + 'px';
      state.width = canvas.width;
      state.height = canvas.height;
    }

    // Create boids within current bounds
    const count = Math.max(1, Math.floor(params.numBoids || 250));
    state.boids = Array.from({ length: count }, function () {
      const x = Math.random() * state.width;
      const y = Math.random() * state.height;
      return new App.Boid(x, y);
    });

    // Animation loop
    function frame(ts) {
      const last = state.lastTime || ts;
      state.lastTime = ts;

      const boids = state.boids;
      for (let i = 0; i < boids.length; i++) {
        boids[i].update(boids, params, state.width, state.height);
      }

      App.render(state);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    // Handle window resize
    window.onresize = function () {
      App.resizeCanvas(canvas, state);
    };
  });
})();