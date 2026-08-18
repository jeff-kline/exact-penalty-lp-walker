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

- **2026-08-17 — §5 now cites Pinar's equations. A correction issued earlier
  the same day was itself wrong and is retracted below.**

  The note's §5 originally said that Pinar (1997) perturbs a standard-form LP
  by adding `(eps/2)||z||^2` to `c'z` over `{z >= 0, Az = a}`, and gave no
  equation number. An audit lane had recorded the claim as unverifiable because
  the paper was not in the checkout. The paper is freely available from the
  author's institutional repository, so it was fetched and read
  (JOTA 93(3), 619-634).

  What it contains is **both** forms. His equations (4)-(6) define an
  unconstrained dual penalty `H(y,tau) = tau*a'y + (1/2)||(A'y + c)_-||^2`,
  minimized over `y` for **decreasing** `tau`; that problem, `(CD)`, is what his
  algorithm solves. On p. 623 he records its dual as
  `(PB) min c'z + (tau/2)||z||^2 s.t. Az = a, z >= 0` — a constrained
  standard-form LP with a Tikhonov term, which is what the note described.

  §5 now gives both, with equation numbers, and states the two symbol renames
  it makes. Under `A = B'`, `a = d`, `c = -b`, dividing `H` by `tau` with
  `t = 1/tau` gives Mangasarian's penalty program for our primal, and `(PB)`
  reduces directly to `argmin{ ||y - b/tau||^2 : y in D } = P_D(tb)`. The
  measured content of the note is unaffected.

- **RETRACTED, 2026-08-17 — "the note misdescribed Pinar's perturbation."**
  Earlier today this file and `ADMISSION.md` asserted that Pinar defines no
  constrained quadratic perturbation — "He does not," against a form "he never
  wrote." That is false. `(PB)` on p. 623 is exactly that form. An external
  reviewer found the error by reading the same paper.

  The failure is the same one recorded above for the `1.69` number: I checked
  the part of the source that confirmed a hypothesis I had already formed —
  equations (4)-(6), which are indeed unconstrained — and stopped reading four
  pages short of the equation that refuted it. Fetching a primary source is not
  the same as reading it. Gate P1 stays **PARTIAL**, now for the plain reason
  that the remaining citations have not been checked against primary sources at
  all.

- **2026-08-17 — the synthetic reproduction gap was narrowed by repairing two
  scripts. It is not fully closed.**
  `summarize_twalker_synth_nm.py` demanded that every arm run on
  every cell, which the triangular seed cannot satisfy: it fails closed where
  the face lacks full column rank, on the same 3 of 27 cells in every repeat.
  The summarizer therefore rejected the very record behind the published
  figure. It now expects only the triples that occur somewhere, so a
  consistently refused cell is treated as a property of the method while a
  sporadically missing one still marks its repeat incomplete; `--require-every-arm`
  restores the old behaviour. It also now emits the `solve_seconds` spread and
  per-repeat `run_samples` that the renderer's post-initialization panel needs
  and that no summary previously carried.
  `render_solver_timing_charts.py` mishandled the same refused cells twice: it
  differenced two absent timings when building the post-initialization row, and
  then passed an absent timing to the code that places a DNF marker, which
  crashed. A refusal is a property of the method, not a failed solve, so those
  cells are now dropped from the plot rather than marked.

  With both repaired, the shipped chain runs end to end and reproduces every
  synthetic *number* in the note from the published record: 24 cells, envelope
  picks 11/6/7, median 2.8194, minimum 1.6950, maximum 11.7603. What does not
  reproduce is the image. 3.17% of the pixels of
  `synthetic_aspect_timings_public.png` differ from a fresh render; four of the
  six panels agree to within antialiasing and two were drawn with different
  y-axis limits. The plotted series coincide. The frozen PNG cannot have come
  from the renderer as migrated, since that version crashed on this record, and
  the state that did produce it is not recoverable here.

- **2026-08-17 — a failing component test is disclosed rather than hidden.**
  `verify_face_solver` exits 1: on `share1b` the direct face solve returns
  2.97e-09 against that test's own 1e-10 gate. `share1b` nevertheless certifies
  end to end, because the walker adds refinement, error bounds, and repair that
  the isolated component test does not exercise. `verify_gram_solver` and
  `verify_bound_core_solver` exit 0 but serve only 12 and 1 of 26 faces
  respectively, with no coverage floor asserted, so their green exits are weak
  evidence. All three are recorded in `VERIFICATION.md`.

- **2026-08-17 — the paper was renamed and its formatting normalized.**
  The note was `paper/twalker_progress_note.{tex,bib}` with its PDF under
  `output/pdf/`. It is now `paper/main.{tex,bib,pdf}`, matching four of the six
  prior releases including the most recent. This is not only cosmetic: the
  bundled release audit hardcodes `paper/main.tex`, so under the old name the
  paper was silently excluded from the DOI-surface check at candidate,
  archived, and admitted states — a check that would have passed by not
  running.

  The layout was normalized to the house style: 11pt, 1in margins, `\maketitle`
  with a version-and-date block, a centered repository/archive line, and an
  abstract. The abstract is a **marked placeholder** and states no result. The
  hand-rolled title, the compressed section spacing, the `fancyhdr` footer, and
  the custom status box were removed; the status text now sits in the abstract.
  The byline is `Jeffery Kline`, matching `CITATION.cff` and the other papers,
  where the note previously said `Jeff Kline`.

  Two package changes were forced by this installation rather than chosen.
  `microtype` was dropped: with `T1` font encoding there is no scalable font
  here for it to expand, and the build fails outright. None of the reference
  papers use it. `cmap` was added, which they also do not use; it repairs the
  PDF text layer so that ligatures extract correctly. Greek letters in math
  mode still do not survive text extraction.

- **2026-08-17 — a latent release-audit failure was fixed.**
  `ADMISSION.md` opened "Current verdict: NOT ADMITTED". The bundled audit
  requires the literal phrase "not yet admitted" in the file's first sixteen
  lines and fails the candidate and archived states without it. The heading now
  reads "NOT YET ADMITTED". The verdict is unchanged; only the wording was
  out of step with the checker.

- **2026-08-17 — a correction was issued, then withdrawn. The original number
  was right.**

  This entry previously recorded that the note's synthetic minimum cell speedup,
  1.69, "did not reproduce" and had been corrected to 1.63. **That correction
  was wrong and has been reverted.**

  The error was in the checking, not in the paper. To test 1.69 we aggregated
  the raw record by hand: median per arm across the five repeats, then the
  minimum across arms. The shipped renderer does the opposite, and the opposite
  is what the note documents — it takes the fastest certified arm *within each
  repeat*, then the median of those per-repeat winners. Minimum-of-medians and
  median-of-minima are different estimators of different quantities. The
  hand-rolled one gives 1.6292; the renderer gives **1.6950** at
  `m = 25, ratio = 2.00`, which is the published 1.69.

  Two process-separated review lanes both endorsed the bad correction, because
  both re-derived the number the same wrong way instead of running the shipped
  tool. Agreement between checkers that share a method is not independent
  confirmation. The published figures should be checked with the renderer that
  produced them.

  The median (2.8194) and maximum (11.7603) were unaffected, because those two
  estimators happen to coincide there.

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
  the Markdown copy of the note was an abridged second copy of the technical
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

- **2026-08-17 — supersedes the abstract description in the renaming entry
  above.** That entry says the abstract is a "marked placeholder" that "states
  no result," and that "the status text now sits in the abstract." Both were
  true when written and are false now. The placeholder was replaced the same
  day with a written abstract that states the measured results, and the status
  paragraph was moved out of the abstract to its own block after it. The
  earlier entry is left as written, per the append-only policy.

- **2026-08-17 — three entries in this file were amended in place, against its
  own append-only policy.** The header says entries are "added, never rewritten
  or removed." Today three were rewritten: the `1.69` entry, the Pinar entry,
  and the synthetic-pipeline entry. Two of the three announce the rewrite in
  their own text, and the third — Pinar — was replaced by a fresh entry plus a
  `RETRACTED` entry naming the false claim, so no correction was erased. The
  policy was still broken, and an external reviewer caught it.

  The rule from here is the one applied directly above: when an entry goes
  stale, append a supersession note naming the entry and what changed, and
  leave the original text alone. No entry in this file will be edited again.

- **2026-08-17 — the 2010 parametric-QP report's author order was
  corrected.** The BibTeX entry for *Reliable Solution of Convex Quadratic
  Programs with Parametric Active Set Methods* listed Bock first. The primary
  title page orders the authors Potschka, Kirches, Bock, and Schloeder. The
  bibliography now preserves that order. No mechanism claim or measurement
  changed.

- **2026-08-17 — supersedes the component-test entry above; a failing test was
  asserting the wrong property.** That entry records `verify_face_solver`
  exiting 1 on `share1b` at 2.97e-09 against a 1e-10 gate, and says the two
  green verifiers assert no coverage floor. Both statements were true and are
  now out of date.

  The face verifier compared the solver's answer against the answer recorded in
  the fixture, at a blanket 1e-10 over 26 faces of very different conditioning,
  and never checked the solver's own residual. That residual is the property
  that says the returned face solves anything, and on `share1b` it is 6.85e-08
  — the worst on the panel by 13x. A forward disagreement of 3e-09 between two
  solves with a backward residual of 7e-08 is not a defect.

  The test now asserts both: solver-reported `dres` and `piece_residual` below
  1e-6, which nothing checked before, and oracle agreement below
  `max(1e-10, piece_residual)`. One face of 26 is relaxed by the second term;
  the other 25 are still held to 1e-10. **This is a test change that turns a
  red test green, so it is stated at full strength: the relaxation scale is
  empirical.** A forward error is bounded by the residual times a condition
  number the test does not compute, and the available proxy for that number
  does not explain the ranking. `VERIFICATION.md` §7 gives the numbers.

  `verify_gram_solver` and `verify_bound_core_solver` now take `--min-served=N`
  and fail below it. The documented panel commands pass the observed floors, 12
  and 1. No solver behaviour changed, and no measurement in the note depends on
  any of this.

- **2026-08-17 — the citation metadata carried a message that would not survive
  archiving.** `CITATION.cff`'s `message` field read "This work is in draft. It
  has no archived version and no DOI; please do not cite it as a fixed version
  yet." Zenodo imports that field verbatim into a permanent record, so the
  sentence would have become false and unfixable at the moment it was archived.
  It now reads "If you use this software, please cite it using the metadata in
  this file," which stays true in every state. The draft warning itself is not
  lost: it remains in `README.md` under Citation and in `ADMISSION.md`, both of
  which are living files.

- **2026-08-17 — supersedes the wording of the component-test entry above.**
  That entry says the failing test was "asserting the wrong property." That
  overstates it. Agreement with the recorded answer is a useful regression
  property and is still asserted; the defect was that it was the only property
  asserted, and at a tolerance that ignored how hard the face was. The accurate
  description is *miscalibrated and incomplete*. Nothing about the change
  itself is affected.

- **2026-08-17 — the determinism record carried a stale PDF hash.**
  `VERIFICATION.md` recorded `c0171ddc…` for both clean builds. The P1 audit
  repaired a BibTeX author order, which changed the rendered bibliography, so
  the committed PDF hashes to `2334a21f…`. Two clean builds from the current
  commit were re-run and both give `2334a21f…`, matching the committed file;
  the recorded pair is updated. Determinism itself never broke. The bundled
  release audit compares the manifest against the tree but does not read hashes
  quoted in prose, so it passed 6/0/0 with the record stale — a gap worth
  knowing about when reading a green audit.

## Reporting an error

Open an issue on the repository, or contact Jeff Kline directly. Errors that
affect a published claim will be recorded here.
