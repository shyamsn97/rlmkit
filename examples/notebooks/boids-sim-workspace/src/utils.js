// utils.js
// Small utility helpers used across the project.

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function isObject(val) {
  return val !== null && typeof val === 'object' && !Array.isArray(val);
}

function deepClone(obj) {
  // Simple deep clone handling plain objects and arrays.
  if (obj === null || typeof obj !== 'object') return obj;
  if (Array.isArray(obj)) return obj.map(deepClone);
  const out = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      out[key] = deepClone(obj[key]);
    }
  }
  return out;
}

function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = Math.max(0, decimals);
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function uid(prefix = '') {
  // Simple unique id using timestamp + random
  return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

function throttle(fn, wait) {
  let last = 0;
  return function(...args) {
    const now = Date.now();
    if (now - last >= wait) {
      last = now;
      return fn.apply(this, args);
    }
  };
}

function debounce(fn, wait) {
  let timeout = null;
  return function(...args) {
    const ctx = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => fn.apply(ctx, args), wait);
  };
}

module.exports = {
  clamp,
  isObject,
  deepClone,
  formatBytes,
  uid,
  throttle,
  debounce,
};
