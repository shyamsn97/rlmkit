// Main simulation script. Assumes Vec and Boid are on window.
(function(global){
  const canvas = document.getElementById('flock');
  const ctx = canvas.getContext('2d', { alpha: true });
  const boids = [];
  let w = canvas.width = innerWidth;
  let h = canvas.height = innerHeight;

  const params = {
    count: 300,
    speed: 2.5,
    perception: 50,
    alignWeight: 1.0,
    cohesionWeight: 0.8,
    separationWeight: 1.6
  };

  // UI elements
  const countEl = document.getElementById('count');
  const speedEl = document.getElementById('speed');
  const perceptionEl = document.getElementById('perception');
  const pauseBtn = document.getElementById('pause');
  const info = document.getElementById('info');

  let running = true;
  let lastTime = performance.now();

  function resize(){
    w = canvas.width = innerWidth;
    h = canvas.height = innerHeight;
    for(const b of boids){ b._w = w; b._h = h; }
  }
  addEventListener('resize', resize);

  function createBoids(n){
    boids.length = 0;
    for(let i=0;i<n;i++){
      const x = Math.random()*w;
      const y = Math.random()*h;
      boids.push(new Boid(x,y,w,h));
    }
  }

  // controls
  countEl.addEventListener('input', (e)=>{
    params.count = +e.target.value;
    updateCount();
  });
  speedEl.addEventListener('input', (e)=>{
    params.speed = +e.target.value;
  });
  perceptionEl.addEventListener('input', (e)=>{
    params.perception = +e.target.value;
  });
  pauseBtn.addEventListener('click', ()=>{
    running = !running;
    pauseBtn.textContent = running ? 'Pause' : 'Resume';
    if(running) lastTime = performance.now();
  });

  canvas.addEventListener('click', (ev)=>{
    // add a few boids where clicked
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    for(let i=0;i<6;i++){
      boids.push(new Boid(x + (Math.random()-0.5)*20, y + (Math.random()-0.5)*20, w,h));
    }
    // trim to max
    if(boids.length > 1200) boids.splice(0, boids.length - 1200);
  });

  function updateCount(){
    const diff = params.count - boids.length;
    if(diff > 0){
      for(let i=0;i<diff;i++){
        boids.push(new Boid(Math.random()*w, Math.random()*h, w,h));
      }
    }else if(diff < 0){
      boids.splice(diff);
    }
  }

  function step(ts){
    const dt = (ts - lastTime) / 16.666; // relative to 60fps
    lastTime = ts;
    if(!running){
      requestAnimationFrame(step);
      return;
    }

    // clear with slight alpha for trails
    ctx.fillStyle = 'rgba(4,6,12,0.25)';
    ctx.fillRect(0,0,w,h);

    // update & draw boids
    const localBoids = boids; // alias
    for(const b of localBoids){
      b.applyBehaviors(localBoids, params);
      b.update(params.speed * dt);
      b.edges();
      b.draw(ctx);
    }

    // UI update
    info.textContent = `Boids: ${boids.length}`;
    requestAnimationFrame(step);
  }

  // initial setup
  createBoids(params.count);
  requestAnimationFrame((t)=>{ lastTime = t; requestAnimationFrame(step); });

  // expose for debugging
  global.SIM = { boids, params, canvas, ctx, createBoids, updateCount };
})(window);
