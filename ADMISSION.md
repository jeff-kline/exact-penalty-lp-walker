# Admission record

**Current verdict: NOT YET ADMITTED — preparation in progress.**

**Release state:** DRAFT

**Version:** 0.1.0-draft

**Immutable release:** none. No tag, no archive, no DOI.

Admission is a project-defined release decision. It means the claim, evidence,
prior-work record, reproducibility materials, citation, and correction policy
passed the gates below against one frozen commit. **It is not peer review, a
correctness certificate, or proof of global novelty.** Nothing in this
repository has yet passed those gates.

## Standard applied

Jeff Kline, ["A Public Standard for This
Work"](https://jeff-kline.github.io/posts/research-program/index.html).

## Gate status

| Gate | Status | Evidence and disposition |
|---|---|---|
| P1 — prior work and credit | **PASS** | All 14 cited works were checked against primary material: ten full texts and four primary publisher/author records and abstracts where the full text was not freely obtainable. The audit compares the closest mechanisms, not just titles: the 1996 continuation papers already contain piecewise-linear paths, predictor–corrector/Newton machinery, and factor reuse; Pinar (1997) is the closest work because his `(CD)`/`(PB)` formulations map exactly to `P_D(tb)` under the paper's stated dualization and `t = 1/τ`; and general parametric-QP work already supplies ratio events, dependent-set exchanges, and factor updates. The release therefore claims no new path, homotopy, or generic active-set machinery and distinguishes its implementation/computation contribution. The corpus is bounded, four full-text access failures and the absence of independent specialist review remain explicit, and no global novelty claim is inferred. Full dispositions and exact checked-copy hashes are in `audit/reports/p1-primary-source-audit-20260817.md`. |
| A1 — claim and artifact consistency | **PASS** | The initial whole-release audits and later abstract-focused passes found and resolved the claim, credit, scope, and provenance defects recorded in `CORRECTIONS.md` and `AUDIT_LEDGER.md` entries 1–15. The final surface removes the unsupported priority and accuracy motivations; identifies the converted inequality encoding behind the Netlib ratio band; labels the synthetic panel, two walker seeds, hindsight frontier, and HiGHS reference; and reports frozen Netlib timings approximately because the rebuilt walker total differs by 1.2%. The remaining “about three orders” statement is supported by measured reductions of 3.23 and 2.63 orders. Its description of the algorithmic distinction from Pinar now matches §5: Pinar evaluates selected penalty values with predictor-corrector steps, whereas this method visits the path face by face and seeks the next breakpoint. Title, author, draft status, contribution boundary, limitations, evidence levels, and quantitative claims now agree across the paper, README, CFF, admission record, and frozen records. No claim of peer review, global novelty, competitive-solver performance, or automatic dispatch remains. Further material claim edits reopen A1. |
| R1 — release and stewardship | **PARTIAL** | Deterministic document build verified byte-for-byte across clean rebuilds, and independently from separate `git archive` checkouts. Five documented reproduction paths execute, including the Netlib walker panel, whose producer was reconstructed and now reproduces the published frontier total to 1.2%. Correction policy, citation metadata, third-party provenance, and `MANIFEST.sha256` over every tracked file exist and verify. Blocking: no tag, no archive, no DOI. Remaining qualification, not a blocker: the current build's internal counters differ from the frozen record on 13 of 24 models, and `beaconfd` has regressed 11x. |

## Claim boundary

- **Measured here:** on a 24-cell synthetic panel and the 27-model Netlib
  panel, the fastest certified result of the default-seed t-walker, the
  triangular-seeded t-walker, and the staged Newton method is faster than the
  staged Newton method alone, under one original-data KKT certificate, with
  componentwise scaling on the primal and dual residuals, at tolerance 1e-7.
- **Not claimed:** a new projection path, a new homotopy principle, a
  competitive LP solver, parity with HiGHS, an automatic dispatcher, or
  complete Netlib coverage.
- **Explicitly provisional:** the reconstruction of Pinar's 1997 algorithm is a
  correctness-first reference implementation. Its timings are directional and
  are excluded from the measured frontier. It is not a reproduction of Pinar's
  reported Fortran results.
- **Hindsight:** the reported frontier is a measured lower envelope over three
  methods, not a policy that selects among them without running them.

## Named residual risks

- **The Netlib headline reproduces to about 1%, on a rebuilt walker whose
  internal path has since changed on some models.** The producing harness was
  missing when this release was assembled; it now ships as
  `experiments/bench_twalker_netlib_panel.py`. Re-running it against a freshly
  built `verify_walker` and recomputing the frontier gives **9.712 s against
  the published 9.596 s, a 1.2% difference**.

  Two qualifications. First, the current build does not reproduce the frozen
  record's *internal* counters on 13 of 24 comparable models — pivot counts,
  seed iterations, and accepted supports differ, and `capri` now certifies
  where the record shows a settle-support cycle. Since two independent reruns
  agree with each other on all 26 models, and a batched invocation agrees too,
  the walker is deterministic on a fixed build; the divergence is against the
  older code state that produced the record, not run-to-run noise. Second, one
  model regressed sharply: `beaconfd` takes 11x longer than the record, tracking
  a seed-iteration count that rose from 11 to 300. It does not affect the
  headline, because Newton wins `beaconfd` on the frontier.

  The frozen records therefore remain the published evidence, and the note's
  existing instruction to read Netlib times at order-of-magnitude precision is
  the right one. What is now also true is that a reader can rebuild the walker
  and land within about 1% of the published total.
- **The synthetic figure now regenerates, after repairing two scripts, but
  not pixel-for-pixel.** `summarize_twalker_synth_nm.py` rejected the record
  behind the published figure because it demanded every arm on every cell, and
  the triangular seed fails closed on 3 of 27. It also never emitted two fields
  the renderer's post-initialization panel needs.
  `render_solver_timing_charts.py` then crashed while placing a DNF marker for
  a cell that carried no timing at all. Both are fixed, and the shipped
  chain now reproduces the published numbers exactly: 24 cells, envelope picks
  11/6/7, median 2.8194, minimum 1.6950, maximum 11.7603.

  What does not reproduce is the image. 3.17% of pixels differ from the shipped
  `synthetic_aspect_timings_public.png`. Four of six panels match to within
  antialiasing; two were drawn with different y-axis limits, and matching them
  needs a vertical rescale of 1.07 and 1.14. The plotted series coincide. Since
  the renderer as migrated crashed on this record, it cannot be the code that
  produced the frozen PNG, and the earlier renderer state is not recoverable
  from this repository. See `VERIFICATION.md` §1.
- **The C++ component verifiers exercise one face per model.** All three read
  the 26 panel fixtures, and each fixture carries exactly one face — 26 faces
  in total, against the thousands the walker visits on a full solve. They are
  smoke tests. Two of them additionally serve only part of that: the Gram route
  serves 12 of 26 and the bound-core route 1 of 26, both by fail-closed design.
  Coverage floors are now asserted at those observed values, so a silent drop
  fails the run, but a floor does not widen the panel. Separately, the face
  verifier's oracle-agreement gate is relaxed on one face to the solve's own
  backward residual; that scale is empirical rather than a proven bound. See
  `VERIFICATION.md` §7 and `CORRECTIONS.md`.
- The synthetic generator varies aspect ratio and planted support geometry
  together, so the panel establishes a regime rather than isolating a cause.
- The prototype is hybrid: an in-process Newton method builds the default
  `t = 0` seed, and difficult faces may invoke HiGHS-backed endpoint selection
  or terminal repair. That cost is charged to the walker, but the benchmark
  does not isolate purely native path algebra.
- Novelty language is bounded by a non-exhaustive literature review and should
  remain provisional until an optimization specialist checks the equation map
  and bibliography.
- Process-separated AI auditors share training data and blind spots. They are
  process evidence, not independent expert review.

## Execution boundary

Permission for one action never implies permission for the next.

| Action | Owner | Status |
|---|---|---|
| Local edits, builds, tests | agent | in progress |
| Create GitHub repository | Jeff Kline | not requested |
| Review-branch push | Jeff Kline | not requested |
| Default-branch update | Jeff Kline | not requested |
| Public tag | Jeff Kline | not requested |
| GitHub Release | Jeff Kline | not requested |
| Zenodo portal (enable repo, mint DOI) | Jeff Kline — authenticated portal, agent must not open it | not requested |
| Public-site listing | Jeff Kline | not requested |

## Archive route

**Zenodo GitHub integration.** No DOI exists before the GitHub Release, so
candidate metadata stays timeless and carries no DOI field. The repository must
be enabled in Zenodo *before* the Release is created. Before the Release, the
GitHub tag zipball is downloaded twice and pinned; the published Zenodo file
must equal those bytes. A `git archive <tag>` is generated twice as an
independent determinism check, and its hash is expected to differ from the
provider zipball.

## Remaining work before CANDIDATE

1. ~~Run the bounded audit lanes, record dispositions, and close A1 on the
   corrected surface.~~ Done. Four whole-release lanes and three later narrow
   abstract passes are recorded in `AUDIT_LEDGER.md`; entry 14 records the
   final dispositions. A1 is **PASS**. Further material claim edits reopen it.
2. ~~Resolve or formally accept the Netlib reproduction gap.~~ Resolved: the
   missing producer was reconstructed and the frontier reproduces to 1.2%.
   A separate question remains open for the author, and is a research matter
   rather than a release blocker: `beaconfd`'s seed now takes 300 iterations
   where the frozen record shows 11.
3. ~~Replace the `CITATION.cff` `message` with publication-safe, timeless
   wording.~~ Done: it now asks users to cite the supplied metadata without
   making a time-sensitive draft claim.
4. ~~Decide whether the Markdown copy of the note ships.~~ Done: removed.
   It duplicated every claim in the paper and had drifted from it twice. The
   TeX source is now the single claim surface for the note.
5. Run the bundled release audit at `--state candidate --require-clean`.

## State transitions

| State | Condition | Status |
|---|---|---|
| DRAFT → CANDIDATE | P1, A1, and pre-freeze R1 pass; prose and artifacts agree. | **not reached** |
| CANDIDATE → TAGGED | Freeze one clean commit; create one immutable tag. | not reached |
| TAGGED → ARCHIVED | Archive the tagged tree; verify the download byte-for-byte. | not reached |
| ARCHIVED → ADMITTED | Activate DOI, reconcile public surfaces, issue the verdict. | not reached |
