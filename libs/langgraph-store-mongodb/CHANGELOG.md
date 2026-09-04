# Changelog

---

## Changes in version 0.4.0 (TBD)

- Version bumped to 0.4.0, the first upstream release matching the patched range published for
  GHSA-533j-2v4q-mw5h. Upstream released 0.4.0 on 2026-09-04; before that the advisory's declared
  range (`< 0.4.0`) could not be satisfied by any existing version, and this package was pinned at
  0.3.0 instead. No functional change accompanies the bump - the security fix landed in 0.3.0
  below. This fork is not otherwise feature-equivalent to upstream 0.4.0, which additionally ships
  native reranking; the number tracks the security patch level only.

## Changes in version 0.3.0 (TBD)

- Fixes a NoSQL operator injection vulnerability (GHSA-533j-2v4q-mw5h) in `MongoDBStore.search()`
  and `asearch()`. Filter dict keys are now validated recursively; any key starting with `$` raises
  a `ValueError`, preventing callers from injecting MQL operators such as `$exists`, `$ne`, or
  `$where` into the query. Backported from upstream 14a6cc3 (INTPYTHON-961, upstream PR #384),
  which is the commit upstream shipped in its own v0.3.0.

## Changes in version 0.0.1 (2025/05/09)

- Initial release, added support for `MongoDBStore`.
