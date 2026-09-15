(function () {
  'use strict';

  var measurementId = 'G-8YC2E5MW2M';
  var consentKey = 'portfolio-analytics-consent-v1';
  var german = document.documentElement.lang.toLowerCase().indexOf('de') === 0;
  var enabled = false;
  var choice = null;

  try { choice = localStorage.getItem(consentKey); } catch (_) { /* Storage is optional. */ }

  function language() { return german ? 'de' : 'en'; }
  function send(name, details) {
    if (!enabled || typeof window.gtag !== 'function') return;
    window.gtag('event', name, Object.assign({ site_language: language(), transport_type: 'beacon' }, details || {}));
  }

  function start() {
    if (enabled) return;
    enabled = true;
    window.dataLayer = window.dataLayer || [];
    window.gtag = function () { window.dataLayer.push(arguments); };
    window.gtag('js', new Date());
    window.gtag('consent', 'default', {
      analytics_storage: 'denied', ad_storage: 'denied', ad_user_data: 'denied',
      ad_personalization: 'denied', functionality_storage: 'denied',
      personalization_storage: 'denied', security_storage: 'granted'
    });
    window.gtag('consent', 'update', { analytics_storage: 'granted' });
    window.gtag('config', measurementId, {
      anonymize_ip: true, allow_google_signals: false, allow_ad_personalization_signals: false,
      page_location: window.location.origin + window.location.pathname
    });
    var tag = document.createElement('script');
    tag.async = true;
    tag.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(measurementId);
    document.head.appendChild(tag);
    document.dispatchEvent(new Event('analytics-enabled'));
  }

  function save(value) {
    choice = value;
    try { localStorage.setItem(consentKey, value); } catch (_) { /* Applies to this page only. */ }
    if (value === 'accepted') start();
    else if (enabled) {
      window.gtag('consent', 'update', { analytics_storage: 'denied' });
      window.location.reload(); // Stop the loaded tag and subsequent collection.
    }
    render();
  }

  function button(label, action) {
    var element = document.createElement('button');
    element.type = 'button';
    element.textContent = label;
    element.addEventListener('click', action);
    return element;
  }

  function render() {
    var old = document.getElementById('analytics-choice');
    if (old) old.remove();
    if (choice === 'accepted' || choice === 'rejected') return;
    var panel = document.createElement('aside');
    panel.id = 'analytics-choice';
    panel.className = 'analytics-choice';
    panel.setAttribute('aria-label', german ? 'Analyse-Einwilligung' : 'Analytics choice');
    var message = document.createElement('p');
    message.textContent = german
      ? 'Darf ich anonyme Besuchs- und Klickstatistiken mit Google Analytics erheben? Erst nach Ihrer Zustimmung wird der Google-Tag geladen.'
      : 'May I use Google Analytics for visit and click statistics? The Google tag loads only if you agree.';
    panel.appendChild(message);
    var actions = document.createElement('div');
    actions.className = 'analytics-choice-actions';
    actions.appendChild(button(german ? 'Zustimmen' : 'Allow analytics', function () { save('accepted'); }));
    actions.appendChild(button(german ? 'Ablehnen' : 'Decline', function () { save('rejected'); }));
    var policy = document.createElement('a');
    policy.href = german ? '/de/privacy/' : '/privacy/';
    policy.textContent = german ? 'Datenschutz' : 'Privacy';
    actions.appendChild(policy);
    panel.appendChild(actions);
    document.body.appendChild(panel);
  }

  function classify(link) {
    var href = link.getAttribute('href') || '';
    if (href.indexOf('mailto:') === 0) return 'contact_click';
    if (href.indexOf('drive.google.com/file/') !== -1) return 'download_cv';
    if (link.hasAttribute('data-language-option')) return 'language_switch';
    if (href.indexOf('github.com/') !== -1) return 'github_click';
    if (href.indexOf('linkedin.com/') !== -1) return 'linkedin_click';
    if (link.closest('#work')) return 'view_project';
    if (link.hostname && link.hostname !== window.location.hostname) return 'outbound_link';
    return null;
  }

  function trackClick(event) {
    var link = event.target.closest && event.target.closest('a[href]');
    if (!link) return;
    var name = classify(link);
    if (!name) return;
    if (link.closest('#work') && name !== 'view_project') {
      var project = link.closest('.work-entry');
      if (project && project.querySelector('h3')) {
        send('view_project', { project_name: project.querySelector('h3').textContent.trim().slice(0, 100) });
      }
    }
    var details = {};
    if (name === 'language_switch') details.target_language = link.getAttribute('data-language-option');
    if (name === 'view_project') {
      var entry = link.closest('.work-entry');
      details.project_name = entry && entry.querySelector('h3') ? entry.querySelector('h3').textContent.trim().slice(0, 100) : 'project';
    }
    if (name === 'outbound_link') details.destination_host = link.hostname;
    if (name === 'download_cv') details.cv_action = 'external_view_link';
    // No email addresses, full URLs, query strings, free-form visitor input or PII in events.
    send(name, details);
  }

  function observeWork() {
    var work = document.getElementById('work');
    if (!work || !('IntersectionObserver' in window)) return;
    var visible = false;
    var observer = new IntersectionObserver(function (entries) {
      visible = entries.some(function (entry) { return entry.isIntersecting; });
      if (visible && enabled) {
        send('view_work');
        observer.disconnect();
      }
    }, { threshold: 0.1 });
    observer.observe(work);
    document.addEventListener('analytics-enabled', function () {
      if (visible) {
        send('view_work');
        observer.disconnect();
      }
    }, { once: true });
  }

  function observeArticle() {
    var article = document.querySelector('.post-single .post-content');
    if (!article) return;
    var fired = false;
    function check() {
      if (fired || !enabled) return;
      var start = article.getBoundingClientRect().top + window.scrollY;
      var length = article.offsetHeight;
      if (length > 0 && window.scrollY + window.innerHeight >= start + length * 0.75) {
        fired = true;
        send('article_read_75');
        window.removeEventListener('scroll', check);
      }
    }
    window.addEventListener('scroll', check, { passive: true });
    window.addEventListener('resize', check);
    document.addEventListener('analytics-enabled', check);
    check();
  }

  function ready() {
    if (choice === 'accepted') start();
    render();
    var footer = document.querySelector('.footer');
    if (footer) {
      var policy = document.createElement('a');
      policy.href = german ? '/de/privacy/' : '/privacy/';
      policy.textContent = german ? 'Datenschutz' : 'Privacy';
      policy.className = 'analytics-footer-link';
      footer.appendChild(policy);
      var settings = button(german ? 'Analyse-Einstellungen' : 'Analytics settings', function () {
        choice = null;
        render();
        document.getElementById('analytics-choice').querySelector('button').focus();
      });
      settings.className = 'analytics-settings';
      footer.appendChild(settings);
    }
    document.addEventListener('click', trackClick);
    observeWork();
    observeArticle();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();
}());
