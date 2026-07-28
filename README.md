# ATLAS NeurIPS 2026 anonymous submission package

This package contains an anonymous theory submission prepared with the official NeurIPS 2026 style.

## Files

- `main.tex`: main paper, references call, appendix call, broader impact, and checklist call.
- `proofs.tex`: complete proofs, supporting lemmas, and an auditable protocol for future empirical evaluation.
- `references.bib`: bibliography.
- `checklist.tex`: mandatory NeurIPS 2026 paper checklist.
- `neurips_2026.sty`: unmodified official style file distributed in the NeurIPS 2026 author kit.
- `revision_notes_ja.md`: Japanese explanation of the reconstruction and remaining reviewer risks.

## Compilation

Use a current TeX Live installation and run:

```bash
latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex
```

The submission build uses the default `neurips_2026` option, which produces an anonymous Main Track manuscript with line numbers. Do not add `final` or `preprint` for the initial submission.

The build log emits three audit markers:

```text
MAIN_END_PAGE=...
APPENDIX_START_PAGE=...
CHECKLIST_START_PAGE=...
```

`MAIN_END_PAGE` must not exceed 9. References, technical appendices, broader impact, and checklist follow the main paper in the same PDF. The build package also contains `page_audit.txt` and `compile_audit.txt` generated from the log.

## Scope of reproducibility

The paper is a theory contribution and reports no empirical result. The estimator is defined by Equations (7)--(10) in `main.tex`; all assumptions and proofs are included. Appendix `Auditable evaluation protocol for future empirical work` specifies the strict one-step information clock required for any later experimental paper.

## Submission checks

Before submission, independently verify:

1. US Letter page size.
2. Embedded Type 1 or TrueType fonts, for example with `pdffonts`.
3. `MAIN_END_PAGE <= 9` in `page_audit.txt`.
4. No undefined citations or references in `compile_audit.txt`.
5. Anonymous metadata, source, supplementary files, and external links.
6. The mathematical constants and all proofs, especially the minimax lower bound, through independent coauthor review.

Build audit trigger: 2026-07-28 final clean ready-for-review branch.
