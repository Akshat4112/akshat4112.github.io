# GA4 portfolio measurement handoff

The website uses GA4 `G-8YC2E5MW2M` on all English and German routes after explicit opt-in. The tag is blocked before consent (basic Consent Mode v2). It sends the consent default before the granted update and never grants advertising consent. No analytics traffic is intended from rejected or undecided visitors. This means GA4 counts **consenting visits only** and cannot identify individual visitors; avoid interpreting these counts as all traffic.

The browser emits `view_work` on the work section entering view, `view_project` on work-card link clicks, `download_cv` on clicks to the Drive CV *view* link (not an actual download), `contact_click` on email-link clicks, `github_click`, `linkedin_click`, `language_switch`, `article_read_75` once per article view, and `outbound_link` for remaining off-site links. Each includes `site_language`; some include `project_name`, `target_language`, `destination_host`, or `cv_action`. Do not add email addresses, visitor input, full URLs, query strings, or other PII to custom event parameters. GA4's automatic enhanced-measurement events and browser metadata should also be reviewed in the property settings.

Property-level work requiring Analytics Editor access:

1. In Admin → Data streams, confirm the measurement ID and inspect enhanced measurement. Disable automatic outbound click tracking if its full `link_url` parameter is not desired, and consider duplicate-click reporting.
2. In Admin → Data settings → Data retention, choose and document the appropriate retention (for example 2 months for this small portfolio), according to your own needs. This cannot be set from the static site.
3. In Admin → Data streams → Configure tag settings → Define internal traffic, enter your actual IP/range; create an internal-traffic data filter in Testing first, inspect results, then Activate. Never guess the owner's IP.
4. In Admin → Events / Key events, mark `contact_click` and/or `download_cv` as key events only if those proxies match your goals. A click does not confirm an email sent or a CV file downloaded.
5. In Explore → Funnel exploration, use `page_view` where page path is `/` or `/de/` → `view_work` → `download_cv` or `contact_click`. Break down by `site_language` where available. An open funnel may count users who skip the homepage; use a closed funnel for this sequence.
6. In Realtime and DebugView, test fresh opt-in sessions on `/`, `/de/`, `/posts/`, an English article, `/publications/`, `/about/`, `/privacy/`, and the German equivalents. Confirm one page view per consenting page and custom events, inspect parameters for PII, and verify **no** Google tag request before consent or after rejecting on a fresh page.
7. Compare page performance before/after on mobile and desktop; the GA script must not be requested before opt-in. Test with cookies/local storage blocked, a screen reader, and an ad blocker.

Only repository changes are included in this PR. Property settings and live Realtime/DebugView checks remain manual until a property Editor can verify them; do not mark the tracker Done before that.
