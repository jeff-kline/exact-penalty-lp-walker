# Audit ledger

Append-only record of review passes over this release: what was audited, at
which commit, by what kind of reviewer, and what came of it. Dispositions live
in `CORRECTIONS.md`; this file records the process.

**All reviewers listed here are process-separated AI agents or the author.
None is an independent expert, and none of this is peer review.** Separated
agents share training data and blind spots — entry 6 below is a case where two
of them agreed on the same wrong answer.

| # | Date | Commit | Reviewer | Scope | Outcome |
|---|---|---|---|---|---|
| 1 | 2026-08-17 | `40a4f58` | cold agent, frontier tier | Claim and public prose across README, paper, PDF, CFF, admission, corrections | 9 must-fix. Status box asserting this repository did not exist; headline counting two methods where the frontier uses three; undefined symbol in the load-bearing derivation; stale counts in the Markdown copy |
| 2 | 2026-08-17 | `40a4f58` | cold agent, frontier tier | Mathematical re-derivation of the seven stated identities | 5 of 7 survived. Found the terminal-test claim false, with a counterexample, and the shipped C++ already contradicting the prose; found the accuracy score misnamed |
| 3 | 2026-08-17 | `40a4f58` | cold agent, mid tier | Citation mechanics and bounded prior art | PASS. 14/14 keys resolve, 14/14 entries verified against Crossref. Recorded Pinar's perturbation as unverifiable — **wrongly**, see entry 5 |
| 4 | 2026-08-17 | `40a4f58` | cold agent, mid tier | Reproducibility and release mechanics | PASS on manifest, determinism, aggregates. Found 7 leaked absolute paths and an overclaimed correction entry |
| 5 | 2026-08-17 | `ffa5cc8` | cold agent, frontier tier | Confirming pass over the applied corrections | 6 of 7 corrections landed; the certificate rename had leaked past 3 files. Found `grow7` misdescribed, an overstated "outpaces Newton", and an undisclosed second reproduction gap |
| 6 | 2026-08-17 | `363f212` | external reviewer | Whole-release review against the public standard | Scored the work 8/10 and withheld promotion to CANDIDATE. Correctly identified premature closure: the Pinar primary source was obtainable, a stale build reference survived the rename, and the skipped component tests contained a real failure |
| 7 | 2026-08-17 | working tree | author + agent | Acting on entry 6 | Pinar source obtained and read; §5's algebra rewritten from his equations (4)–(6); P1 reduced to PARTIAL. Component tests run: one fails. Synthetic pipeline repaired. **A prior correction was retracted**: the published 1.69 was right and the agent's 1.63 was wrong. Entry 8 shows the Pinar rewrite was itself wrong |
| 8 | 2026-08-17 | `0bb2a64` | external reviewer | Second whole-release review | Found the entry-7 Pinar correction **false**: `(PB)` on p. 623 of the same paper is the constrained regularized program the original text described. Also found closure overclaimed twice — the synthetic figure and a stale `VERIFICATION.md` section — and the dangling `agent_reports/` pointers still in place. All four disposed of in this pass |
| 9 | 2026-08-17 | `b2d8ff2` | primary agent, adversarial literature audit | P1 audit of all 14 cited works; equation/mechanism comparison of exactness, continuation paths, Pinar equivalence, parametric-QP events/exchanges, factor updates, and recent threshold/monotonicity claims | **P1 PASS.** Ten full primary texts read; four primary records/abstracts checked with full-text access failures named. The 1996 papers narrow the background more than the prior admission record showed, but the public text already credits them and bounds the contribution correctly. No claim-prose edit required; one credit repair corrected the 2010 report's BibTeX author order. Raw dispositions and checked-copy hashes preserved in `audit/reports/p1-primary-source-audit-20260817.md` |

| 10 | 2026-08-17 | `4942f81` | external reviewer | Component-test disposition, coverage floors, citation metadata | Reproduced all three verifiers, both deliberately failing floors, and the clean draft audit. Accepted the `share1b` disposition as a documented smoke-test policy and corrected its description from "wrong property" to *miscalibrated and incomplete*. Found the determinism record still quoting the pre-P1 PDF hash, which the mechanical audit cannot catch because it does not read hashes out of prose |

| 11 | 2026-08-17 | `dedc0b4` | primary agent, **self-check — weakest evidence in this table** | A1 confirming pass: every quantitative and scope claim in the abstract, paper, README, `CITATION.cff`, and `ADMISSION.md`, recomputed against the frozen records rather than cross-read | 24 of 25 checked claims reproduce exactly. **A1 does not close.** One high-severity finding: the abstract's "Mangasarian *first* built a Newton LP method from a quadratic penalty" is contradicted by §5 of the same paper and by entry 9's own report, which records Pinar (1996) as already giving a finite modified-Newton penalty algorithm. Three lower items on scope labels and qualifier placement. One fixed here; the rest touch author prose and are referred |

| 12 | 2026-08-17 | `6907c5a` | cold agent, frontier tier | Narrow A1 pass over the revised abstract alone: numbers re-aggregated with the shipped code, credit checked against §5 and entry 9's report, scope checked against README/CFF/ADMISSION/§1 | Every number reproduces and the priority claim is clean. **11 framing findings, A1 stays open.** Two are serious: the abstract states the Netlib result with no reference solver and none of the four caveats the rest of the release treats as governing; and the motivating claim that stopping tolerances cost accuracy is unmeasured and runs opposite to the release's own accuracy record, where Newton is the more accurate and better-covering method before crossover |

| 13 | 2026-08-17 | `8775dec` | cold agent, frontier tier | Second narrow A1 pass over the abstract and §1, aimed at the sentences rewritten the same day, plus an orphan check on the withdrawn accuracy claim | Withdrawal is clean: no orphan of the stopping-tolerance claim survives, and §4 reads consistently with §1. Every number reproduces. **Nine findings, A1 stays open.** The sharpest: "all of Netlib-27" is true only of the converted inequality encoding — in canonical MPS form those models have far fewer rows than columns — and the release's own `synth_nm.py` had documented that all along. Also caught that the reference solver added one commit earlier was quoted on a different estimator than the totals beside it |

| 14 | 2026-08-17 | `aaf36d1` | primary agent, final A1 disposition | Frozen-surface review of entry 13's remaining timing-precision, accuracy-rounding, and applicability questions; cross-check against the frozen records, `ADMISSION.md`, and the rebuilt walker result | **A1 PASS.** Netlib totals and the geometric-mean HiGHS comparison are now explicitly approximate; the frozen values remain exact in the verification record. “About three orders” is retained because the measured reductions are 3.23 and 2.63 orders. No “practical linear programs” claim remains. All earlier material findings are resolved or carried as named limitations. The claim surface is frozen; any later material claim edit reopens A1 |

| 15 | 2026-08-18 | `c6d1db5` | primary agent, narrow post-closure check | User-directed abstract clarification of the algorithmic distinction from Pinar; checked against §5 and the P1 record | **A1 PASS.** The abstract now says directly that the method differs by following $y(t)$ face by face and seeking breakpoints. That is the same bounded distinction already stated in §5; it does not claim a new path, homotopy, predictor-corrector principle, or generic active-set machinery. The incidental “we run it from two seeds” clause is removed. No other claim changes |

| 16 | 2026-08-18 | `30fe475` | author-directed correction + primary-agent check | MPS constraint-count interpretation across the paper, README, CFF, admission record, and converter | **A1 PASS after correction.** Standard MPS default variable bounds are genuine LP restrictions. The repository converter makes finite bounds explicit as inequality rows and represents equalities as paired inequalities. Earlier prose wrongly treated the 1.5–2.9 band as merely manufactured by an encoding. Active claim surfaces now state that the ratio describes the explicit inequality system solved here, not MPS `ROWS` records alone. |

| 17 | 2026-08-18 | `b260912` | primary agent, narrow README consistency pass | README title, headline claim, prior-work distinction, limitations, and reproduction links checked against the frozen paper and verification record | **A1 PASS after repair.** The README claim box now carries the paper's 11-of-12 scope, 2.82x median, hindsight qualification, and approximate 17x HiGHS gap instead of a broader portfolio claim. The Pinar distinction now matches §5, the title matches the release metadata, and the Netlib rerun link points to `VERIFICATION.md` §5 rather than the document-build section. No paper claim changed. |

## Standing limits on this ledger

- Entries 1–5 were launched by the same root agent that then applied their
  findings. That agent is not neutral about its own work.
- Entry 6 is the only review not initiated by the agent doing the writing.
- Raw lane reports were returned through the agent interface and not preserved
  as files. Later lanes should write to `audit/reports/` so the raw text
  survives independently of any summary of it.
- The `1.69` episode in entry 7 is the sharpest lesson available here: two
  separated reviewers endorsed a wrong correction because both re-derived the
  quantity the same wrong way instead of running the shipped renderer.
  **Check a published figure with the tool that produced it.**
- Entry 8 is the same failure in a second dress. Entry 7 fetched the Pinar
  paper, read the four pages that confirmed a hypothesis already formed, and
  declared the note wrong on the strength of them — while the equation that
  refuted the correction sat on p. 623 of the file already open. **Two of the
  three corrections issued under agent review have been wrong.** Weight this
  ledger accordingly: it records that reviews happened, not that they were
  right.
- A green run of the bundled release audit does not mean the prose agrees with
  the tree. It verifies the manifest against the files; it does not read the
  hashes, counts, or timings quoted in `VERIFICATION.md`. Entry 10 found a
  stale PDF hash sitting behind a 6-pass audit. Numbers written into prose have
  to be re-checked by hand whenever the artifact they describe is rebuilt.
