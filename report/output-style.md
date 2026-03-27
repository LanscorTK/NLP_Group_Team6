# Academic Style Guide

## Core Principle
Write in clear, concrete, evidence-based academic prose.
Prefer precision over ornament.

## Prohibited Practices
- Do not fabricate citations, quotations, or paraphrases.
- Do not use empty filler phrases such as:
  - "It is worth doing that"
  - "It should be noted that"
  - "It is important to mention that"
  unless they add real analytical value.
- Avoid inflated or generic AI-sounding verbs where simpler alternatives are better, including:
  - utilize
  - leverage
  - delve into
  - underscore
  - facilitate
  - robust (unless technically justified)
- Do not use vague evaluative wording without evidence, such as:
  - very effective
  - highly significant
  - strong performance
  unless supported by results.

## Sentence Design
- Prefer sentences under 30 words.
- Avoid more than two levels of subordination.
- Prefer one claim per sentence where possible.
- Prefer concrete nouns and active verbs.
- Use transitions sparingly and only when they clarify logic.

## Terminology
- All technical and project-specific terms must follow `glossary.md`.
- Do not introduce alternative labels for an already defined concept unless explicitly approved.

## Output Format
- **Canonical source:** `report/main.tex` (LaTeX, ACL 2023 template).
- **Modular structure:** each section lives in `report/sections/<name>.tex`, included via `\input{}` from `main.tex`.
- **`report/draft.md`** is kept as a readable reference but is **not** the submission artifact. All edits should go into the `.tex` files.
- **Compile chain:** `pdflatex main && bibtex main && pdflatex main && pdflatex main` (run from `report/`).
- **Figures:** stored in `../figures/`, referenced via `\includegraphics{}` (graphicspath set in preamble).
- **Bibliography:** `report/references.bib` (BibTeX). Managed by `acl_natbib.bst` (loaded automatically by `acl.sty`).
- **Placeholders:** use `\textcolor{red}{[MISSING --- description]}` for any data not yet available.

## Referencing
- Use natbib citation commands: `\citet{}` for textual (e.g. "Devlin et al. (2019) showed..."), `\citep{}` for parenthetical (e.g. "...as shown previously (Devlin et al., 2019)").
- All BibTeX entries live in `report/references.bib`. Do not add manual `\bibitem` entries.
- Check author names, year, title, and venue carefully.
- Do not insert a citation unless the source has been verified.