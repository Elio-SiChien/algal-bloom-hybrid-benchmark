# Public download verification

On 2026-10-30 the fixed v1.0.0-review source archive was downloaded without an Authorization header into a new directory. The public entry independently fetched the Release prediction asset without an Authorization header, checked the ZIP and 250 member hashes, and reran all checks in the validated isolated environment.

- Input-integrity and negative regression tests: passed (4 tests).
- All paper-result recomputation: passed (360 full-precision comparisons, max difference 4.44e-16).
- Selected-station raw-to-daily reconstruction: passed.
- Four-method training-year CPU smoke: passed, 80 origins × 7 forecast days per method.
- Full 40-run training replication: not executed or claimed.

Machine-readable proof: [PUBLICATION_VERIFICATION.json](https://github.com/Elio-SiChien/algal-bloom-hybrid-benchmark/releases/download/v1.0.0-review/PUBLICATION_VERIFICATION.json). The tested tag remains unchanged; this note records the later verification, rather than retroactively claiming it in the tagged source.

The optional GitHub Actions template was excluded because the active credential lacks workflow scope. Local tests and reproduction commands are included and verified; no additional account permissions were requested.
