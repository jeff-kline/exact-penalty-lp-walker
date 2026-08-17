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
| 7 | 2026-08-17 | working tree | author + agent | Acting on entry 6 | Pinar source obtained and read; §5's algebra found **wrong** and rewritten from his equations (4)–(6); P1 reduced to PARTIAL. Component tests run: one fails. Synthetic pipeline repaired end to end. **A prior correction was retracted**: the published 1.69 was right and the agent's 1.63 was wrong |

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
