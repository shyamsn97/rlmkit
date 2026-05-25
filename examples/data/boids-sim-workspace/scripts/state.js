(function (global) {
  // Single global namespace
  var App = (global.BoidsApp = global.BoidsApp || {});

  // Simulation parameters
  App.params = {
    numBoids: 320,        // 250-400
    maxSpeed: 3.2,        // 2-4 px/frame
    maxForce: 0.06,       // 0.03-0.08
    perception: 50,       // 35-60
    separationDist: 18    // 12-24
  };

  // Global state
  App.state = {
    boids: [],
    canvas: null,
    ctx: null,
    dpr: 1,
    width: 0,
    height: 0,
    lastTime: 0
  };

  // Utilities
  function rand(min, max) {
    return Math.random() * (max - min) + min;
  }

  function clamp(v, min, max) {
    return v < min ? min : (v > max ? max : v);
  }

  function limitMag(vx, vy, max) {
    var mag = Math.hypot(vx, vy);
    if (mag > max && mag > 0) {
      var s = max / mag;
      return [vx * s, vy * s];
    }
    return [vx, vy];
  }

  App.util = {
    rand: rand,
    clamp: clamp,
    limitMag: limitMag
  };
})(window);