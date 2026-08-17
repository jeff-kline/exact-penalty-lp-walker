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

- **2026-08-17 — build no longer depends on a private path.**
  `cpp/twalker/Makefile` hardcoded an absolute path inside the author's
  virtualenv as the default location of the HiGHS shared library. It now
  discovers that directory from a configurable `PYTHON` interpreter, or accepts
  `HIGHS_LIBDIR` directly.

- **2026-08-17 — incomplete dependency pin.**
  The benchmark requirements file omitted `clarabel` and `pillow`, both of
  which are unconditional imports on the documented reproduction paths. The
  pinned environment is now `requirements.txt`, which lists them.

## Reporting an error

Open an issue on the repository, or contact Jeff Kline directly. Errors that
affect a published claim will be recorded here.
