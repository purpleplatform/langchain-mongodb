# Changelog

---

## Changes in version 0.3.0 (TBD)

- Fixes a NoSQL operator injection vulnerability (GHSA-533j-2v4q-mw5h) in `MongoDBStore.search()`
  and `asearch()`. Filter dict keys are now validated recursively; any key starting with `$` raises
  a `ValueError`, preventing callers from injecting MQL operators such as `$exists`, `$ne`, or
  `$where` into the query. Backported from upstream 14a6cc3 (INTPYTHON-961, upstream PR #384).
- Version set to 0.3.0, the upstream release that actually carries this fix (`git tag --contains`
  on the upstream commit resolves to `libs/langgraph-store-mongodb/v0.3.0`, which is also the
  latest release on PyPI). This fork is not otherwise feature-equivalent to upstream 0.3.0; the
  number tracks the security patch level only.
- Note for scanners: GHSA-533j-2v4q-mw5h is published with an affected range of `< 0.4.0` for this
  package, but no 0.4.0 was ever released upstream or to PyPI, so that range is unsatisfiable. The
  advisory's own reference links point at the v0.3.0 release. Dependabot alerts derived from the
  bad range are dismissed as inaccurate rather than worked around with a fictional version.

## Changes in version 0.0.1 (2025/05/09)

- Initial release, added support for `MongoDBStore`.
