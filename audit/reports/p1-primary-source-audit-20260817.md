# P1 primary-source audit — 2026-08-17

## Scope and decision

This is a bounded, adversarial audit of every work cited by the public claim
surfaces at commit `b2d8ff2384d0a080764255d77eb323cb19dc76d0`. It asks whether
the repository credits the closest mechanisms, whether the claimed distinction
survives equation-level comparison, and whether any citation is carrying a
claim its source does not support. It is not a systematic literature review or
an expert novelty opinion.

**Decision: P1 PASS.** The path and homotopy are correctly disclaimed as prior
work. Pinar (1997) is the closest source: after the paper's stated dualization
and `t = 1/tau`, his regularized optimizer is the same projection path
`P_D(tb)`. The 1996 papers already contain piecewise-linear continuation,
predictor--corrector/Newton machinery, and factor reuse; the current paper
credits them and limits its contribution to a sparse direct breakpoint
implementation, certification and repairs, and bounded empirical results.
General parametric-QP work also predates the ratio tests, working-set
exchanges, and factor updates. No claim-prose change is required by this
audit. One credit defect was found and repaired: the BibTeX entry for the 2010
report had reordered its authors instead of preserving the primary text's
Potschka--Kirches--Bock--Schloeder order.

## Primary-source dispositions

| Key | Primary material checked | Disposition |
|---|---|---|
| `mangasarian-meyer-1979` | Full author-hosted paper, *SIAM J. Control Optim.* 17(6), 745--752; SHA-256 `2c71ea79a14b8d839fade835cb749f233d93b296e8823c28cc3c5616d4fab8a5` | Theorem 4 gives finite small-parameter exactness for convex/Lipschitz perturbations; Corollary 2 gives strict-convex selection within the LP solution set. Supports the historical exactness claim. |
| `mangasarian-1984` | Springer chapter record and source-provided abstract for DOI `10.1007/BFb0121017`; later primary papers checked below explicitly attribute the normal/minimum-norm projection result to this chapter | Supports the bounded claim that Mangasarian characterized normal LP solutions by projection. Full chapter text was not freely obtainable in the bounded search; this access limit remains visible. |
| `bartels-1980` | Elsevier title/metadata for DOI `10.1016/0024-3795(80)90227-X`; independently confirmed in the 1996 SIAM paper's prior-work discussion | The only public claim is title-level: Bartels used reduced-gradient basis-exchange techniques in a penalty LP method. The title states exactly that. Full text was not freely obtainable in the bounded search. |
| `madsen-nielsen-pinar-1996` | Full institutional-repository paper, *SIAM J. Optim.* 6(3), 600--616; SHA-256 `a0a57fd0f9309565fe0c8064b2991c1dabe97ca79c17679c7491921738c0e268` | Abstract and Sections 1, 4 and 5 establish piecewise-linear solution paths, finite continuation, predictor--corrector behavior, modified-Newton correction, factor updates/downdates, occasional refactorization, and iterative refinement. This is substantial prior mechanism and is credited. |
| `pinar-1996` | Full paper, *Math. Methods Oper. Res.* 44, 345--370; SHA-256 `3f874d26ccad9d1212dd52ff54f14e809aa2741801f0b22a8e626858dbd8007a` | Establishes a quadratic-penalty LP formulation, piecewise-linear minimizer paths, and a finite modified-Newton penalty algorithm. Supports the related-work statement; it narrows novelty but does not collide with the claimed implementation contribution. |
| `pinar-1997` | Full institutional-repository paper, *JOTA* 93(3), 619--634; SHA-256 `921700a62dac7a7bad4890723bdd625c874835fc91a499b5b9be0336792ef55b` | Equations (4)--(6) and `(PB)` on p. 623 verify the paper's equation map. Sections 3--4 verify path prediction, modified-Newton correction, retained `LDL^T` updates/downdates, and iterative refinement. This is the closest work and is described as such. |
| `best-1996` | Author-uploaded primary record and abstract for DOI `10.1007/978-3-642-99789-1_5` | The abstract explicitly gives piecewise-linear primal/dual solutions and active-set additions, deletions, and exchanges. Full author-uploaded text returned an access/rate-limit response in the bounded search. The same mechanism claims are independently supported in full by Bock et al. and Ferreau et al.; this source is not carrying them alone. |
| `mangasarian-2004` | Full author-hosted DMI 02-02 technical report / journal manuscript; SHA-256 `a6bfd0276e2addee6a8071bc71a12cfb8d1e965d1353acaae38b2fe68c7b74f4` | Proposition 2.1 gives a finite-parameter exact least-2-norm dual solution; Section 3 gives the modified Newton method. The abstract and introduction expressly target very tall systems and report the March 2002 technical report lineage. Supports the paper's method and chronology statements. |
| `kline-fung-2023` | Taylor & Francis primary publication record and source-provided abstract for DOI `10.1080/10556788.2022.2117356` | The abstract accurately restates Mangasarian's finite exactness result. The current chronology is independently fixed by Mangasarian's author page and DMI 02-02 manuscript. Full publisher text was not freely obtainable in the bounded search. |
| `bock-kirches-potschka-schloeder-2010` | Full Optimization Online report; SHA-256 `1d42c506a31f5be010eba7ad55ae426728fee673bdf200388f19f2701a1f8215` | Sections 1--4 verify affine homotopy, piecewise-affine optimizers, working-set exchanges, rank-dependence handling, and numerical safeguards. Supports the general parametric-QP setting. |
| `ferreau-etal-2014` | Full author-hosted paper, *Math. Program. Comput.* 6, 327--363; SHA-256 `6dc796e197a2749ff13a375fd8515347142b243573a9e8ec4c2d4a13a1aa988c` | Sections 2.2--2.3 explicitly give blocking-event ratio tests, dependent-constraint exchanges, factor reuse, and rank-one updates after active-set exchanges. Directly supports every mechanism in the general parametric-QP sentence. |
| `huangfu-hall-2018` | Full arXiv/publisher manuscript, *Math. Program. Comput.* 10, 119--142; SHA-256 `7c4f64ded589be566e2ac4e22443d3077fb936144bb70fa4e838bda64060e7af` | Documents the HiGHS dual revised-simplex implementation, sparse LU factorization, hyper-sparse solves, factor updates, and advanced variants developed over decades. The paper's “roughly eighty years” phrase is historical framing for the simplex-to-modern-solver lineage, not a claim that interior-point methods are eighty years old. |
| `gonzalez-sanz-nutz-2025` | Full author-hosted paper, *Appl. Math. Optim.* 91(3); SHA-256 `d4234add583e32e1bff2351140472b58223e5b772890b8016c55aed09ca72cea` | Lemma 2.4 and Theorem 2.5 give the piecewise-affine path, stationary finite threshold, and its exact value. Supports “sharpens finite-exactness thresholds.” |
| `gonzalez-sanz-nutz-riveros-2025` | Full arXiv v2 / journal manuscript, *SIAM J. Optim.* 35(2), 1419--1437; SHA-256 `9962c23367d732ec8b60f29b046b4b45a5732ae46530b30c1adc6d4c274dc35f` | Abstract, Theorem 3.2, and Theorem 4.4 show that face invariance/support monotonicity fails in general. Supports the paper's deliberately qualified statement. |

The checked PDFs are not redistributed because the repository does not assert
permission to do so. Hashes identify the exact copies read.

## Mechanism comparison

| Mechanism | Prior work | This release's bounded distinction |
|---|---|---|
| Finite quadratic regularization/exactness | Mangasarian--Meyer (1979), Mangasarian (1984, 2004) | No novelty claim. |
| Piecewise-linear LP penalty/continuation paths | Madsen--Nielsen--Pinar (1996), Pinar (1996, 1997) | No new path or homotopy claim. |
| Equivalent path `P_D(tb)` | Pinar (1997), after the stated dualization and reciprocal parameter | The correspondence is derived and credited; it is not presented as a new path. |
| Predictor/corrector, Newton correction, factor reuse | Madsen--Nielsen--Pinar (1996), Pinar (1997) | No novelty claim for these generic ingredients. |
| Ratio events, dependent-set exchanges, factor updates | Best (1996), Bock et al. (2010), Ferreau et al. (2014) | No novelty claim for the parametric-QP framework. |
| Sparse direct breakpoint prototype, original-data certificate, rank-deficient-face repairs | This repository's implementation and frozen experiments | Claimed as an implementation/computation contribution, not as a theorem or globally novel method. |

## Gate rows and residual limits

| P1 row | Status | Reason |
|---|---|---|
| Closest work | PASS | The exact Pinar path correspondence and the broader parametric-QP mechanisms are compared, not merely listed. |
| Original sources | PASS | Ten full primary texts were read. Four full texts were inaccessible; their primary publisher/author records and abstracts were checked, the access failures are named above, and no unique load-bearing claim rests on an abstract alone. |
| Contribution type | PASS | The release distinguishes prior theorem/path results from its implementation, certification, repairs, correspondence, and computation. |
| Novelty | PASS | The corpus is explicitly bounded and no global novelty claim is made from a negative search. |
| Residuals | PASS | Paywall/access limits, non-exhaustiveness, and absence of an independent optimization-specialist review remain visible here and in `ADMISSION.md`. |

Residual risk remains: the corpus is the release's 14-item bibliography plus a
bounded mechanism search, not a database-complete review; no independent
optimization specialist has reviewed the equation map; and four older full
texts were not available without additional access. These limits qualify the
strength of the novelty assessment, but they do not leave a false or uncredited
claim on the present public surfaces.
