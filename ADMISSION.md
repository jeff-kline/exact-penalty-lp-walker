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
| P1 — prior work and credit | **OPEN** | The note credits Mangasarian–Meyer (1979), Mangasarian (1984, 2004), Madsen–Nielsen–Pinar (1996), Pinar (1996, 1997), Best (1996), Bock et al. (2010), Ferreau et al. (2014), and Bartels (1980), and states explicitly that no new path or homotopy principle is claimed. It also derives the exact change of variables showing Pinar's perturbed optimizer *is* the path used here. The literature review was bounded, not exhaustive. The citation and prior-art audit lane has not run. |
| A1 — claim and artifact consistency | **OPEN** | README written and aligned to the note. One material inconsistency was found and corrected during preparation: the note's §5 Pinar counts disagreed with what the shipped code produces (see `CORRECTIONS.md`). Because that was a claim edit, A1 restarts from the corrected text. The claim/public-prose audit lane has not run. |
| R1 — release and stewardship | **PARTIAL** | Deterministic document build verified. Four documented reproduction paths execute. Correction policy, citation metadata, and third-party provenance exist. Blocking: no manifest, no tag, no archive, no DOI, and one reproduction gap named below. |

## Claim boundary

- **Measured here:** on a 24-cell synthetic panel and the 27-model Netlib
  panel, the fastest certified result of the default-seed t-walker, the
  triangular-seeded t-walker, and the staged Newton method is faster than the
  staged Newton method alone, under one original-data backward-error
  certificate at tolerance 1e-7.
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

- **The Netlib headline is not re-derivable from this repository.** The record
  `records/twalker_cpp/native_seed_t0_netlib27_full_budgeted_20260816.json`
  supplies the 9.60 s frontier figure, and no script here regenerates it. The
  figures render from it faithfully, and the C++ walker that produced it is
  present, but the measuring harness itself is not. Until that harness is
  identified and shipped, the Netlib timings are *frozen evidence*, not a
  reproducible measurement, and R1's reproduction row cannot reach PASS.
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

1. Run the bounded audit lanes and record dispositions.
2. Resolve or formally accept the Netlib reproduction gap above.
3. Replace the `CITATION.cff` `message` with publication-safe, timeless
   wording — Zenodo imports it verbatim into the permanent record.
4. Generate `MANIFEST.sha256` over the frozen tree.
5. Run the bundled release audit at `--state candidate --require-clean`.

## State transitions

| State | Condition | Status |
|---|---|---|
| DRAFT → CANDIDATE | P1, A1, and pre-freeze R1 pass; prose and artifacts agree. | **not reached** |
| CANDIDATE → TAGGED | Freeze one clean commit; create one immutable tag. | not reached |
| TAGGED → ARCHIVED | Archive the tagged tree; verify the download byte-for-byte. | not reached |
| ARCHIVED → ADMITTED | Activate DOI, reconcile public surfaces, issue the verdict. | not reached |
