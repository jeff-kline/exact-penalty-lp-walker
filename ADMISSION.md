# Admission record

**Current verdict: NOT ADMITTED — preparation in progress.**

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
| P1 — prior work and credit | **PASS, bounded** | The note credits Mangasarian–Meyer (1979), Mangasarian (1984, 2004), Madsen–Nielsen–Pinar (1996), Pinar (1996, 1997), Best (1996), Bock et al. (2010), Ferreau et al. (2014), and Bartels (1980), and states explicitly that no new path or homotopy principle is claimed. A citation lane verified that all 14 citation keys resolve, no entry is orphaned or duplicated, and all 14 bibliography entries match authoritative records. A separate lane re-derived the change of variables showing Pinar's perturbed optimizer *is* the path used here, and confirmed the algebra is exact — so the novelty concession rests on a correct derivation. A bounded prior-art search over the four claimed contributions found no mechanism-level collision. **That is a source-negative result over a non-exhaustive search and is not a claim of global novelty.** One caveat could not be closed here: whether Pinar (1997) perturbs by exactly `(ε/2)‖z‖²` is a bibliographic fact requiring the original paper, which is not in this checkout. |
| A1 — claim and artifact consistency | **OPEN** | Four process-separated audit lanes ran against commit `40a4f58`. They found, and this tree corrects, the following material items: the §5 Pinar counts, a false description of the walker's terminal test, an undefined value function, a misnamed accuracy score, a headline that counted two methods where the frontier uses three, a status box asserting this repository does not exist, seven leaked absolute paths, and an overclaimed correction entry. All are recorded in `CORRECTIONS.md`. Every one was a prose or provenance defect; none changed a measurement. Because they were claim edits, A1 restarts from the corrected text and a confirming pass is required. |
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
- **The synthetic figure is frozen evidence that no documented command
  regenerates.** Figure 1 and the headline synthetic numbers (24 cells, median
  2.82x, envelope picks 11/6/7) come from
  `records/twalker_synth_nm/postinit_m25_50_200_r32_20260816/runs.jsonl`. The
  documented timing command reads a *different* default record
  (`ratio_1_to_512_20260816/summary.json`: 50/100/200 variables, 2 repeats,
  33 cells) and therefore produces a different figure — different panels, no
  post-initialization row, and no triangular-seeded series.

  The two-step path that would close this is blocked by a real tooling defect:
  `experiments/summarize_twalker_synth_nm.py` requires every arm present on
  every cell in a repeat, but the triangular-seeded arm fails closed on 3 of
  the 27 cells — behaviour the note itself documents. No repeat therefore
  counts as complete and the summarizer exits with "fewer than required
  complete repeats" on the very record the published figure came from.

  The underlying numbers do reproduce. Aggregating the raw record directly
  (per-cell median over 5 repeats, envelope = min over the three arms,
  restricted to the 24 cells with ratio > 1.0) gives median 2.8194x, max
  11.7603x, and envelope picks 11/6/7 — matching every published figure to four
  significant digits. What is missing is a shipped command that performs that
  aggregation. Until the summarizer accepts a fail-closed arm, R1's
  reproduction row stays PARTIAL on the synthetic panel.
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

1. ~~Run the bounded audit lanes and record dispositions.~~ Done: four lanes
   ran on commit `40a4f58`. Dispositions are recorded in `CORRECTIONS.md`.
   Because several were claim edits, A1 restarts from the corrected text and a
   confirming pass is still required.
2. ~~Resolve or formally accept the Netlib reproduction gap.~~ Resolved: the
   missing producer was reconstructed and the frontier reproduces to 1.2%.
   A separate question remains open for the author, and is a research matter
   rather than a release blocker: `beaconfd`'s seed now takes 300 iterations
   where the frozen record shows 11.
3. Replace the `CITATION.cff` `message` with publication-safe, timeless
   wording — Zenodo imports it verbatim into the permanent record.
4. ~~Decide whether `paper/twalker_progress_note.md` ships.~~ Done: removed.
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
