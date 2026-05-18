// util.js - shared utility functions
// Auto-generated to satisfy the project's util.js contract.

'use strict';

/**
 * noop - a no-operation function.
 */
function noop() {}

/**
 * isObject - returns true if value is a plain object (and not null/array).
 */
function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * deepClone - simple deep clone for JSON-safe structures.
 * Note: does not clone functions, Dates, RegExps, Maps, Sets, etc.
 */
function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * isEmpty - returns true if value is null/undefined/empty string/empty array/empty object.
 */
function isEmpty(value) {
  if (value == null) return true;
  if (typeof value === 'string') return value.trim().length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (isObject(value)) return Object.keys(value).length === 0;
  return false;
}

/**
 * once - ensures a function is only called once.
 */
function once(fn) {
  let called = false;
  let result;
  return function(...args) {
    if (!called) {
      called = true;
      result = fn.apply(this, args);
    }
    return result;
  };
}

/**
 * promisifySafe - wraps a function that may throw or return a promise and returns a promise
 * that always resolves to [err, result] (Node-style tuple).
 */
function promisifySafe(fn) {
  return function(...args) {
    try {
      const res = fn.apply(this, args);
      if (res && typeof res.then === 'function') {
        return res.then(r => [null, r]).catch(e => [e]);
      }
      return Promise.resolve([null, res]);
    } catch (err) {
      return Promise.resolve([err]);
    }
  };
}

/**
 * wrapAsync - converts an async function to a Node-style callback-taking function.
 * wrapAsync(asyncFn) -> (args..., cb) where cb(err, result)
 */
function wrapAsync(asyncFn) {
  return function(...args) {
    const cb = typeof args[args.length - 1] === 'function' ? args.pop() : null;
    const promise = Promise.resolve().then(() => asyncFn.apply(this, args));
    if (cb) {
      promise.then(res => cb(null, res)).catch(err => cb(err));
    }
    return promise;
  };
}

/**
 * formatBytes - human-readable byte size formatter.
 */
function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = Math.max(0, decimals);
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(k));
  const value = parseFloat((bytes / Math.pow(k, i)).toFixed(dm));
  return `${value} ${sizes[i]}`;
}

/**
 * debounce - returns a debounced version of fn
 */
function debounce(fn, wait = 100) {
  let timeout = null;
  return function(...args) {
    const ctx = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => fn.apply(ctx, args), wait);
  };
}

/**
 * throttle - returns a throttled version of fn
 */
function throttle(fn, limit = 100) {
  let inThrottle = false;
  let lastArgs = null;
  return function(...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
        if (lastArgs) {
          fn.apply(this, lastArgs);
          lastArgs = null;
        }
      }, limit);
    } else {
      lastArgs = args;
    }
  };
}

/**
 * simple assert - throws if condition false
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

module.exports = {
  noop,
  isObject,
  deepClone,
  isEmpty,
  once,
  promisifySafe,
  wrapAsync,
  formatBytes,
  debounce,
  throttle,
  assert
};
