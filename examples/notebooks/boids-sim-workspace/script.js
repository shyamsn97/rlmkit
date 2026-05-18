/*
  script.js
  - Lightweight app bootstrap and utilities.
  - Exposes a global `App` with init(), on(), off(), emit().
  - Initializes on DOMContentLoaded.
  - Theme toggle (light/dark) persisted to localStorage key "theme".
  - Simple form validator: data-validate attributes (required, email) and shows inline errors.
  - fetchJson(url, opts) helper that returns parsed JSON or throws an Error with message.
  - Graceful no-conflict: if window.App already exists, preserve it under App._previous.
*/

(function (global) {
  'use strict';

  // Simple event emitter
  function Emitter() {
    this._handlers = Object.create(null);
  }
  Emitter.prototype.on = function (evt, fn) {
    if (!this._handlers[evt]) this._handlers[evt] = [];
    this._handlers[evt].push(fn);
    return fn;
  };
  Emitter.prototype.off = function (evt, fn) {
    if (!this._handlers[evt]) return;
    if (!fn) { delete this._handlers[evt]; return; }
    this._handlers[evt] = this._handlers[evt].filter(function (h) { return h !== fn; });
  };
  Emitter.prototype.emit = function (evt) {
    var args = Array.prototype.slice.call(arguments, 1);
    var handlers = this._handlers[evt] || [];
    handlers.forEach(function (h) {
      try { h.apply(null, args); } catch (e) { console.error('Event handler error for', evt, e); }
    });
  };

  // Utilities
  function qs(sel, ctx) { return (ctx || document).querySelector(sel); }
  function qsa(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  function addClass(el, c) { if (!el) return; el.classList.add(c); }
  function removeClass(el, c) { if (!el) return; el.classList.remove(c); }

  function fetchJson(url, opts) {
    opts = opts || {};
    return fetch(url, opts).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (txt) {
          var err = new Error('Network error: ' + res.status + ' ' + res.statusText);
          err.status = res.status;
          err.body = txt;
          throw err;
        });
      }
      return res.text().then(function (text) {
        if (!text) return null;
        try { return JSON.parse(text); }
        catch (e) {
          var err = new Error('Invalid JSON response');
          err.cause = e;
          err.text = text;
          throw err;
        }
      });
    });
  }

  // Theme toggle
  var THEME_KEY = 'theme';
  function applyTheme(theme) {
    var root = document.documentElement;
    if (theme === 'dark') {
      addClass(root, 'theme-dark');
      removeClass(root, 'theme-light');
    } else {
      addClass(root, 'theme-light');
      removeClass(root, 'theme-dark');
    }
  }
  function getStoredTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function storeTheme(theme) {
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* ignore */ }
  }
  function toggleTheme() {
    var cur = getStoredTheme() || (document.documentElement.classList.contains('theme-dark') ? 'dark' : 'light');
    var next = cur === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    storeTheme(next);
    App.emit('theme:change', next);
    return next;
  }

  // Form validation: supports data-validate="required|email" on input elements.
  function validateField(field) {
    var rules = (field.getAttribute('data-validate') || '').split('|').map(function (r) { return r.trim(); }).filter(Boolean);
    var val = field.value != null ? String(field.value).trim() : '';
    var errors = [];
    rules.forEach(function (rule) {
      if (rule === 'required') {
        if (!val) errors.push('This field is required.');
      } else if (rule === 'email') {
        // simple email pattern
        var ok = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val);
        if (val && !ok) errors.push('Please enter a valid email address.');
      } else if (rule.indexOf('min:') === 0) {
        var n = parseInt(rule.split(':')[1], 10) || 0;
        if (val.length < n) errors.push('Please enter at least ' + n + ' characters.');
      } else if (rule.indexOf('max:') === 0) {
        var n2 = parseInt(rule.split(':')[1], 10) || 0;
        if (val.length > n2) errors.push('Please enter at most ' + n2 + ' characters.');
      }
    });
    return errors;
  }

  function showFieldErrors(field, errors) {
    // Try to find or create an .error-message element after the field
    var wrap = field.parentNode;
    var existing = wrap && wrap.querySelector && wrap.querySelector('.error-message');
    if (existing) {
      existing.textContent = errors.join(' ');
      if (errors.length) addClass(existing, 'visible'); else removeClass(existing, 'visible');
    } else if (wrap) {
      var el = document.createElement('div');
      el.className = 'error-message' + (errors.length ? ' visible' : '');
      el.textContent = errors.join(' ');
      wrap.appendChild(el);
    }
    if (errors.length) addClass(field, 'has-error'); else removeClass(field, 'has-error');
  }

  function validateForm(form) {
    var fields = qsa('[data-validate]', form);
    var allErrors = {};
    var hasErrors = false;
    fields.forEach(function (f) {
      var errs = validateField(f);
      if (errs.length) {
        hasErrors = true;
        allErrors[f.name || f.id || '[unnamed]'] = errs;
      }
      showFieldErrors(f, errs);
    });
    return { valid: !hasErrors, errors: allErrors };
  }

  function handleFormSubmit(e) {
    var form = e.target;
    if (!(form && form.matches && form.matches('form'))) return;
    var result = validateForm(form);
    if (!result.valid) {
      e.preventDefault();
      App.emit('form:invalid', form, result);
      // focus first invalid field
      var first = form.querySelector('.has-error');
      if (first && typeof first.focus === 'function') first.focus();
      return false;
    }
    // Optionally, we can intercept and emit an event for AJAX forms with data-ajax="true"
    var ajax = form.getAttribute('data-ajax');
    if (ajax === 'true') {
      e.preventDefault();
      var data = new FormData(form);
      var action = form.getAttribute('action') || location.href;
      var method = (form.getAttribute('method') || 'GET').toUpperCase();
      var opts = { method: method };
      if (method !== 'GET') opts.body = data;
      App.emit('form:submit', form, data);
      fetch(action, opts).then(function (res) {
        if (!res.ok) throw new Error('Network response was not ok: ' + res.status);
        return res.json ? res.json() : res.text();
      }).then(function (payload) {
        App.emit('form:success', form, payload);
      }).catch(function (err) {
        App.emit('form:error', form, err);
      });
    }
    return true;
  }

  // App singleton
  var previousApp = global.App;
  var App = new Emitter();
  App._previous = previousApp;

  App.init = function (opts) {
    opts = opts || {};
    // apply persisted or preferred theme
    var theme = getStoredTheme();
    if (!theme) {
      // attempt to detect via prefers-color-scheme
      try {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) theme = 'dark';
      } catch (e) { /* ignore */ }
      theme = theme || 'light';
    }
    applyTheme(theme);

    // wire theme toggle elements
    qsa('[data-toggle-theme]').forEach(function (btn) {
      btn.addEventListener('click', function (ev) { ev.preventDefault(); toggleTheme(); });
    });

    // global form submit handler for client-side validation & ajax-submit support
    document.addEventListener('submit', handleFormSubmit, true);

    // basic input blur validation
    document.addEventListener('blur', function (ev) {
      var t = ev.target;
      if (t && t.getAttribute && t.getAttribute('data-validate')) {
        var errs = validateField(t);
        showFieldErrors(t, errs);
      }
    }, true);

    App.emit('init', { theme: theme });
  };

  App.destroy = function () {
    // remove listeners we attached via document (can't easily remove anonymous handlers here)
    // emit destroy
    App.emit('destroy');
    // restore previous global if present
    if (App._previous) global.App = App._previous;
  };

  // Export utilities
  App.fetchJson = fetchJson;
  App.toggleTheme = toggleTheme;
  App.applyTheme = applyTheme;
  App.validateForm = validateForm;

  // Auto-init on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { App.init(); });
  } else {
    // already ready
    setTimeout(function () { App.init(); }, 0);
  }

  // Expose
  global.App = App;

})(window);
