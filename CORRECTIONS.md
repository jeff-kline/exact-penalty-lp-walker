# Corrections, withdrawals, and version history

This file is the public record of what changed and why. It is append-only:
entries are added, never rewritten or removed. If a claim in this repository
turns out to be wrong, the correction appears here rather than as a silent
edit.

## Policy

- **Corrections.** A material error in a claim, number, figure, or method
  description is recorded here with the date, what was wrong, what it is now,
  and how it was found. Immaterial fixes (typography, formatting, broken
  internal links) are not recorded individually.
- **Withdrawals.** If a claim cannot be supported, it is withdrawn here and
  struck in the reader-facing text, rather than deleted without trace.
- **Supersession.** A later version supersedes an earlier one. Earlier versions
  remain reachable through their tags and archives.
- **Immutable archives.** Once a version is tagged and archived under a DOI,
  that archive is never modified. Corrections after archiving appear in this
  file and in the living repository; the archived snapshot keeps whatever it
  said at the time, and the discrepancy is disclosed rather than papered over.
- **Prepublication wording.** A tag packaged before its DOI exists honestly
  says so. That wording is not retroactively edited once the DOI is active.

## Version history

### Unreleased — DRAFT

Initial preparation of the public release. No version has been tagged or
archived, so nothing here supersedes a prior public claim.

Corrections made during preparation, before any public version existed:

- **2026-08-17 — provisional Pinar panel counts (§5 of the technical note).**
  The note reported that the provisional reconstruction of Pinar's 1997
  algorithm certified four of the 27 Netlib models (`afiro`, `sc50a`, `sc50b`,
  `sc105`), with 19 stopping at a numerical gate and three reaching the time
  limit, over a time range of 19.9 to 117.9 ms.

  Re-running the shipped code on the shipped fixtures under the note's own
  documented protocol — 27 models, one thread, a 10-second cap — certifies
  **five** models. `stocfor1` also certifies, in roughly 307 ms. The correct
  partition is 5 certified, 18 numerical gate, 3 time limit, and `grow7` not
  measured, over a range of 15.7 to 306.9 ms.

  Two identical repetitions returned the same status on every model, so the
  discrepancy is not run-to-run variation. The root cause of the original
  count was not investigated. The note's §5 text, the caption and footnote of
  the Netlib figure, and the "5 of 27" qualifier were corrected to match the
  shipped code, and the backing run was frozen into
  `records/pinar1997/netlib_panel.json`.

  This does not affect the note's principal claims, which concern the t-walker
  and the staged Newton method; the Pinar reconstruction is a provisional
  reference point, explicitly excluded from the measured frontier.

- **2026-08-17 — figure footnote computed rather than hardcoded.**
  `experiments/pinar1997/render_provisional_netlib.py` printed a literal
  "4/27 certified" in the figure footnote regardless of the data it was
  rendering. It now derives both numbers from the record being plotted, so the
  figure cannot silently disagree with its own input again.

- **2026-08-17 — deterministic PDF build.**
  Two clean builds of the technical note differed in 60 of 513,258 bytes, all
  within the PDF trailer `/ID`, which pdfTeX derives from the output path.
  `\pdftrailerid{}` is now set in the preamble. Together with the pinned
  `SOURCE_DATE_EPOCH` in the build script, repeated clean builds are
  byte-identical.

- **2026-08-17 — private absolute paths removed (revised).**
  `cpp/twalker/Makefile` hardcoded an absolute path inside the author's
  virtualenv as the default location of the HiGHS shared library. It now
  discovers that directory from a configurable `PYTHON` interpreter, or accepts
  `HIGHS_LIBDIR` directly.

  **This entry originally claimed the build "no longer depends on a private
  path." That was an overclaim: only the Makefile had been fixed.** A
  subsequent audit found the same absolute path in six reader-facing commands
  in `cpp/twalker/README.md`, and a seventh baked into the frozen record
  `records/pinar1997/netlib_panel.json` as the `grow7` "fixture not found"
  detail string. All seven are now removed: the README uses `.venv/bin/python`,
  `experiments/pinar1997/run_netlib_panel.py` emits a repository-relative path
  instead of an absolute one, and the single string in the frozen record was
  rewritten to the relative form. No measurement was altered — `grow7` has no
  measurement, only a missing-fixture notice.

- **2026-08-17 — the terminal test was described incorrectly.**
  Section 3 of the note stated that the test `g'g <= tolerance` "serves as both
  a geometric stopping rule and the zero-curvature test." The zero-curvature
  reading is correct; the stopping-rule reading is false. Because
  `g = (I - P_W) b_W`, the vector `g` vanishes identically on any face whose
  active rows are independent — which includes every vertex of the dual
  feasible set that the path passes through en route to the optimum. A stop on
  `g'g` alone would therefore return a suboptimal point, and not only in
  contrived cases.

  The shipped C++ never used the bare test: `cpp/twalker/src/walker.cpp`
  requires a stationary face **and** an audited ratio scan showing no forward
  event **and** a passing original-data certificate, and its own comments say
  "vanishing motion on the active face is not a terminal proof." This was a
  defect in the prose, contradicted by the implementation. No code changed; the
  note and README now describe what the code does.

- **2026-08-17 — the value function was undefined and the piecewise claim was
  imprecise.** The note asserted that "the optimal penalty value is piecewise
  quadratic" while using the symbol `phi` without ever defining it. Read
  literally against the penalty program described one paragraph earlier, the
  claim is false: that program's value is `phi(t)/t`, which carries a `1/t`
  term and is not piecewise quadratic. The intended object,
  `phi(t) = max { t b'y - (1/2)||y||^2 : y in D }`, is now defined explicitly,
  and the distinction is stated.

- **2026-08-17 — the accuracy score was misnamed.**
  The note called its acceptance check a "componentwise backward-error
  certificate." The primal and dual terms are componentwise backward errors up
  to a guard term in the denominator, but the `y >= 0` violation and the
  primal-dual gap are neither backward errors nor componentwise, and no single
  perturbed linear program admits the pair as optimal. It is now described as
  an original-data KKT certificate with componentwise scaling on the primal and
  dual residuals — which is what the implementation's own schema string,
  `common-original-data-kkt-accuracy-v1`, has always called it. The score,
  the tolerance, and every reported measurement are unchanged.

- **2026-08-17 — a published number did not reproduce.**
  The note reported synthetic cell speedups "from 1.69 to 11.76" for ratios at
  most 2. Recomputing from the published record — per-cell median over the five
  repeats, envelope taken as the minimum over the three methods, restricted to
  the 24 cells with ratio above 1.0 — reproduces the median (2.8194), the
  maximum (11.7603), and the envelope picks (11/6/7) to four significant
  figures, but gives a minimum of **1.6292**, not 1.69. The value 1.69 appears
  to be the second-smallest cell, 1.6994, truncated. Corrected to 1.63.

- **2026-08-17 — "t-walker outpaces Newton" was true in 11 of 12 cells, not all.**
  At `m = 200`, ratio 1.1, the default-seed walker is 1.24 times *slower* than
  staged Newton. The triangular seed recovers that cell, so the envelope claim
  is unaffected, but the blanket phrasing overstated the raw comparison. Both
  the note and the README now give the count.

- **2026-08-17 — `grow7`'s exclusion was misdescribed.**
  The note said `grow7` "was not measured because its canonical fixture was
  missing." The canonical `netlib/grow7.mps` is shipped. What is missing is the
  compact panel re-encoding, and `cpp/twalker/fixtures_panel/manifest.json`
  records why: the export failed with `initialization did not produce an
  accepted face`. The original wording turned a substantive toolchain failure
  into a filing accident.

- **2026-08-17 — a second reproduction gap was disclosed.**
  `VERIFICATION.md` reported the timing-figure check as "Agrees" on the strength
  of the Netlib aggregates alone. The *synthetic* figure is not produced by that
  command: the renderer's default record is a different experiment (50/100/200
  variables, 2 repeats, 33 cells) than the one behind the published figure
  (25/50/200, 5 repeats, 24 cells). The intended bridge,
  `summarize_twalker_synth_nm.py`, cannot process the published record, because
  it demands every arm on every cell and the triangular-seeded arm fails closed
  on 3 of 27 cells. The numbers all reproduce by direct aggregation; the shipped
  command to do it does not exist. Recorded as a named residual risk.

- **2026-08-17 — the accuracy-score rename was incomplete.**
  The first correction updated the paper, the README, and one module docstring,
  but left the retired term in `ADMISSION.md`'s claim boundary and in the
  docstring of `certificate_pair` itself — the function the note cites as the
  authority. Both are now corrected, along with two further code comments.

- **2026-08-17 — the duplicate Markdown note was removed.**
  `paper/twalker_progress_note.md` was an abridged second copy of the technical
  note, kept as a working source for a planned HTML companion that does not
  exist. It drifted from the TeX source twice during preparation — carrying
  superseded Pinar counts and a wrong publication year — which is the failure
  mode a second claim surface invites. It has been deleted. The TeX source and
  the PDF built from it are now the only statement of the note's claims.

- **2026-08-17 — the headline claim counted two methods instead of three.**
  The note's opening and the README's claim box described taking "the faster
  certified result" of t-walker and Newton. The measured frontier is a lower
  envelope over **three** methods — the default-seed walker, the
  triangular-seeded walker, and staged Newton — as the frozen record's own
  frontier rule states and as both documents said correctly elsewhere. The
  comparison was also restated against staged Newton alone, since a lower
  envelope beats each of its members by construction and the "improves on
  either alone" phrasing was therefore unfalsifiable.

- **2026-08-17 — incomplete dependency pin.**
  The benchmark requirements file omitted `clarabel` and `pillow`, both of
  which are unconditional imports on the documented reproduction paths. The
  pinned environment is now `requirements.txt`, which lists them.

## Reporting an error

Open an issue on the repository, or contact Jeff Kline directly. Errors that
affect a published claim will be recorded here.
