# Project: Advanced Methods in SciPy and Statsmodels (QMD Book)

## Authoritative Documents

1. **`STEERING.md`** -- style, process, audit checklists, scope boundaries. Read in full before any chapter work.
2. **`CHAPTERS.md`** -- the 23-chapter plan (7 parts + prerequisites appendix). Scope, APIs, theory depth, and dependencies per chapter.

These two documents govern all content decisions. This file does not duplicate them.

## Project Structure

```
.
├── _quarto.yml              # Quarto book config (7 parts, 23 chapters + appendix)
├── index.qmd                # Preface
├── chapters/                # One .qmd per chapter
│   ├── 01-ml-pipeline.qmd
│   ├── 02-smooth-optimization.qmd
│   ├── ...
│   ├── 23-hte-double-ml.qmd
│   └── appendix-prerequisites.qmd
├── assets/
│   ├── custom.scss           # HTML theme overrides
│   └── book.mplstyle         # Matplotlib style for all figures
├── references.bib            # Bibliography
├── requirements.txt          # Pinned Python dependencies
├── STEERING.md               # Style & process (the law)
├── CHAPTERS.md               # Chapter plan with scope and depth signals
├── NOTATION.md               # Symbol ledger (empty until chapters draft)
├── GLOSSARY.md               # Term ledger (empty until chapters draft)
├── TOC.md                    # Chapter status tracker
├── FORWARD_REFS.md           # Unresolved forward references
├── RUNNING_EXAMPLES.md       # Reusable datasets and test problems
└── STYLE_DEVIATIONS.md       # Approved departures from STEERING.md
```

## Key Rules (defer to STEERING.md for full detail)

- **Matplotlib only** for plots; use `assets/book.mplstyle`
- **No mixing** Python code and pseudocode in the same block
- **Pseudocode notation** must match NOTATION.md / STEERING.md §2
- **All code must run** in the pinned environment. Verify before committing.
- **No unilateral decisions** on adding/removing chapters, changing notation, or adding libraries. Flag and ask.
- **Self-audit checklist** (STEERING.md §8) must pass before any chapter is marked complete.
- **Primary libraries**: scipy, statsmodels, linearmodels. scikit-learn used in Ch. 1 (pipeline chapter) and for datasets elsewhere.

## Quarto Commands

```bash
quarto preview          # Live preview
quarto render           # Full build (HTML + PDF)
quarto render --to html # HTML only
quarto render --to pdf  # PDF only
```

## Python Environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Teaming (not subagents)

This project uses **Claude Teams** (`TeamCreate` + `SendMessage` + `TaskList`) for chapter production. Teams are a distinct tool set from subagents (`Agent` tool). Do not conflate them:

| | Teams | Subagents |
|---|---|---|
| Tools | `TeamCreate`, `SendMessage`, `TaskList`, `TaskCreate`, `TaskUpdate` | `Agent` tool |
| Persistence | Teammates persist across tasks; reuse via `SendMessage` | Each `Agent` call is a fresh, stateless spawn |
| Coordination | Shared task list, message passing, idle/wake cycle | Fire-and-forget; result returned once |
| When to use | Sustained multi-chapter production work | One-off research, search, or mechanical tasks |

**Team roles for this project:**
- **Lead (you):** Writes chapter content. Owns the narrative voice, theory, and code.
- **Editor (`~/.claude/agents/editor.md`):** Three hats:
  1. **Research** — verify API behavior, read library source, find citations, test edge cases
  2. **Review** — audit drafts against STEERING.md §8 checklist (notation, proofs, code, voice, consistency)
  3. **Edit** — tighten prose, improve transitions, flag pedagogical gaps, suggest restructuring

Follow the workflow-guardrails skill §13b for team discipline: spawn once, assign explicitly, review before reporting, never shutdown unless the user says so.

## Matplotlib Style

Every chapter should load the book style:
```python
import matplotlib.pyplot as plt
plt.style.use('assets/book.mplstyle')
```
