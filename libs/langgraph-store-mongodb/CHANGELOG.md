# Changelog

---

## Changes in version 0.4.0 (TBD)

- Fixes a NoSQL operator injection vulnerability (GHSA-533j-2v4q-mw5h) in `MongoDBStore.search()`
  and `asearch()`. Filter dict keys are now validated recursively; any key starting with `$` raises
  a `ValueError`, preventing callers from injecting MQL operators such as `$exists`, `$ne`, or
  `$where` into the query. Backported from upstream 14a6cc3 (INTPYTHON-961, upstream PR #384).
- Version set to 0.4.0 to match the upstream patched release for this advisory. This fork is not
  otherwise feature-equivalent to upstream 0.4.0; the number tracks the security patch level only.

## Changes in version 0.0.1 (2025/05/09)

- Initial release, added support for `MongoDBStore`.
