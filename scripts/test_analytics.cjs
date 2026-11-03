'use strict';

const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../static/js/portfolio-analytics.js'), 'utf8');

function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, callback, options = {}) {
      const entries = listeners.get(type) || [];
      entries.push({ callback, once: options.once });
      listeners.set(type, entries);
    },
    removeEventListener(type, callback) {
      listeners.set(type, (listeners.get(type) || []).filter(entry => entry.callback !== callback));
    },
    dispatchEvent(event) {
      for (const entry of [...(listeners.get(event.type) || [])]) {
        if (entry.once) this.removeEventListener(event.type, entry.callback);
        entry.callback(event);
      }
    }
  };
}

function element(tagName) {
  const attributes = new Map();
  return Object.assign(eventTarget(), {
    tagName,
    children: [],
    setAttribute(name, value) { attributes.set(name, value); },
    getAttribute(name) { return attributes.get(name) ?? null; },
    hasAttribute(name) { return attributes.has(name); },
    appendChild(child) { this.children.push(child); },
    querySelector(selector) {
      // Only the DOM subset used by this script is needed; attributes are matched generically.
      const [tag] = selector.split('[');
      const conditions = [...selector.matchAll(/\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]/g)];
      for (const child of this.children) {
        if (child.tagName === tag && conditions.every(([, name, value]) =>
          value === undefined ? child.hasAttribute(name) : child.getAttribute(name) === value)) return child;
        const nested = child.querySelector(selector);
        if (nested) return nested;
      }
      return null;
    }
  });
}

function page({ path = '/posts/example/', isPost = true, consent = 'accepted', height = 2000 } = {}) {
  const head = element('head');
  const body = element('body');
  // This marker is the contract emitted by Hugo for individual posts only.
  if (isPost) {
    const marker = element('meta');
    marker.setAttribute('name', 'portfolio-content-type');
    marker.setAttribute('content', 'post');
    head.appendChild(marker);
  }
  const window = Object.assign(eventTarget(), {
    scrollY: 0,
    innerHeight: 500,
    location: { origin: 'https://example.test', hostname: 'example.test', pathname: path }
  });
  const article = {
    offsetHeight: height,
    getBoundingClientRect() { return { top: 100 - window.scrollY }; }
  };
  const document = Object.assign(eventTarget(), {
    documentElement: { lang: 'en' },
    readyState: 'complete',
    head,
    body,
    createElement: element,
    getElementById(id) { return body.children.find(child => child.id === id) || null; },
    querySelector(selector) {
      if (selector === '.post-single .post-content') return article;
      return head.querySelector(selector) || body.querySelector(selector);
    }
  });
  const storage = new Map(consent === null ? [] : [['portfolio-analytics-consent-v1', consent]]);
  // No browser or network APIs are provided: appended scripts are recorded but never loaded.
  vm.runInNewContext(source, {
    window,
    document,
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    Event: class { constructor(type) { this.type = type; } }
  }, { filename: 'portfolio-analytics.js' });

  return {
    window,
    document,
    body,
    scripts: () => head.children.filter(child => child.tagName === 'script'),
    events: () => Array.from(window.dataLayer || [], args => Array.from(args)).filter(args => args[0] === 'event'),
    scroll(y) { window.scrollY = y; window.dispatchEvent({ type: 'scroll' }); }
  };
}

test('accepted posts emit article_read_75 exactly once at the reading threshold', () => {
  const browser = page();
  assert.equal(browser.scripts().length, 1);
  browser.scroll(1100); // Viewport bottom reaches 100 + 2000 * 0.75.
  browser.scroll(1500);
  browser.window.dispatchEvent({ type: 'resize' });
  browser.document.dispatchEvent({ type: 'analytics-enabled' });
  assert.deepEqual(browser.events().map(event => event[1]), ['article_read_75']);
  assert.equal(browser.events()[0][2].site_language, 'en');
});

for (const path of ['/about/', '/now/', '/privacy/', '/de/about/', '/de/now/', '/de/privacy/']) {
  test(`${path} does not emit article reads despite sharing the article layout`, () => {
    const browser = page({ path, isPost: false });
    browser.scroll(1500);
    browser.window.dispatchEvent({ type: 'resize' });
    assert.equal(browser.scripts().length, 1); // Consent still allows ordinary page analytics.
    assert.deepEqual(browser.events(), []);
  });
}

test('a post below 75 percent emits no reading event', () => {
  const browser = page();
  browser.scroll(1099);
  browser.window.dispatchEvent({ type: 'resize' });
  assert.deepEqual(browser.events(), []);
});

test('rejected consent loads no analytics script and sends no reading or click events', () => {
  const browser = page({ consent: 'rejected' });
  browser.scroll(1500);
  const link = element('a');
  link.setAttribute('href', 'mailto:example@example.test');
  link.closest = selector => selector === 'a[href]' ? link : null;
  browser.document.dispatchEvent({ type: 'click', target: link });
  assert.equal(browser.scripts().length, 0);
  assert.equal(browser.window.gtag, undefined);
  assert.deepEqual(browser.events(), []);
});

test('pending consent waits for acceptance before loading analytics or recording an eligible read', () => {
  const browser = page({ consent: null });
  browser.scroll(1500);
  assert.equal(browser.scripts().length, 0);
  assert.deepEqual(browser.events(), []);
  const panel = browser.body.children.find(child => child.id === 'analytics-choice');
  panel.remove = () => browser.body.children.splice(browser.body.children.indexOf(panel), 1);
  panel.querySelector('button').dispatchEvent({ type: 'click' });
  assert.equal(browser.scripts().length, 1);
  assert.deepEqual(browser.events().map(event => event[1]), ['article_read_75']);
});
