/*
  main.js - entry point script

  Responsibilities:
  - Wait for DOMContentLoaded
  - Create a simple UI: header, status area, button
  - Provide a small API for other modules via window.app (if available)
  - Demonstrate async initialization and error handling
  - Works in browsers supporting ES modules and older environments (attach to window)
*/

(function (global) {
  'use strict';

  // Minimal DOM helper
  function el(tag, props, children) {
    var node = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        if (k === 'class') node.className = props[k];
        else if (k === 'text') node.textContent = props[k];
        else node.setAttribute(k, props[k]);
      });
    }
    if (children) {
      children.forEach(function (c) {
        if (typeof c === 'string') node.appendChild(document.createTextNode(c));
        else if (c) node.appendChild(c);
      });
    }
    return node;
  }

  // Simple logger displayed in UI
  function createLogger(container) {
    var logArea = el('pre', { class: 'app-log', style: 'white-space:pre-wrap; background:#f7f7f7; padding:8px; border-radius:4px; max-height:200px; overflow:auto;' });
    container.appendChild(logArea);
    return {
      info: function (msg) {
        var line = document.createElement('div');
        line.textContent = '[info] ' + msg;
        logArea.appendChild(line);
        logArea.scrollTop = logArea.scrollHeight;
        console.log(msg);
      },
      error: function (msg) {
        var line = document.createElement('div');
        line.textContent = '[error] ' + msg;
        line.style.color = 'crimson';
        logArea.appendChild(line);
        logArea.scrollTop = logArea.scrollHeight;
        console.error(msg);
      }
    };
  }

  // Async initialization placeholder (e.g., fetch config)
  function initApp(container) {
    return new Promise(function (resolve, reject) {
      try {
        var header = el('h1', { text: 'App Main' });
        var description = el('p', { text: 'A small demo UI created by main.js.' });
        var btn = el('button', { text: 'Do work' });
        var status = el('div', { class: 'status', text: 'Ready' });

        container.appendChild(header);
        container.appendChild(description);
        container.appendChild(btn);
        container.appendChild(status);

        var logger = createLogger(container);
        logger.info('UI created. Awaiting interaction.');

        btn.addEventListener('click', function () {
          status.textContent = 'Working...';
          logger.info('Button clicked — starting async task.');

          // Simulate async work
          fakeWork().then(function (result) {
            status.textContent = 'Done';
            logger.info('Work finished: ' + result);
          }).catch(function (err) {
            status.textContent = 'Error';
            logger.error('Work failed: ' + (err && err.message ? err.message : String(err)));
          });
        });

        // Expose a small API
        var api = {
          doWork: function () {
            btn.click();
          },
          log: logger
        };

        resolve(api);
      } catch (err) {
        reject(err);
      }
    });
  }

  function fakeWork() {
    return new Promise(function (resolve) {
      setTimeout(function () {
        resolve('success');
      }, 500);
    });
  }

  // Auto-bootstrap when DOM is ready
  function boot() {
    var root = document.getElementById('app') || document.body;
    // Create container if not present
    if (!document.getElementById('app')) {
      var container = el('div', { id: 'app', style: 'font-family:Arial, Helvetica, sans-serif; margin:16px;' });
      document.body.insertBefore(container, document.body.firstChild);
      root = container;
    }

    initApp(root).then(function (api) {
      // attach to global for debugging or consumption
      try { global.app = api; } catch (e) { /* ignore */ }
      // dispatch a custom event for other modules
      var event;
      try {
        event = new CustomEvent('app:ready', { detail: {} });
      } catch (e) {
        event = document.createEvent('CustomEvent');
        event.initCustomEvent('app:ready', true, true, {});
      }
      document.dispatchEvent(event);
    }).catch(function (err) {
      console.error('Failed to initialize app:', err);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    // Already ready
    setTimeout(boot, 0);
  }

  // Export for module systems (UMD-light)
  try {
    if (typeof module !== 'undefined' && module.exports) {
      module.exports = { boot: boot };
    } else if (typeof define === 'function' && define.amd) {
      define(function () { return { boot: boot }; });
    } else {
      global.main = { boot: boot };
    }
  } catch (e) {
    // ignore export errors in strict environments
  }

})(typeof window !== 'undefined' ? window : (typeof global !== 'undefined' ? global : this));
