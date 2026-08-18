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
