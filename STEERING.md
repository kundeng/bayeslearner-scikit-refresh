## 1. Scope and Promise

### 1.1 What This Book Is

SciPy and statsmodels are the computational backbone of scientific Python. Their documentation tells you *how* to call a function. This book tells you *what that function does* and *why it works*.

Every chapter takes a method implemented in scipy or statsmodels, develops the mathematics that justify it, shows how the library translates that mathematics into code, re-implements the core algorithm from scratch so the reader can see every moving part, and then covers the diagnostics and failure modes that the documentation never mentions. The goal is not to replace the library documentation but to provide the layer underneath it — the layer a practitioner needs when defaults stop working, when convergence fails, when standard errors look wrong, or when a method's assumptions are violated and the output is silently garbage.

The book covers 23 chapters in 7 parts: an ML-pipeline introduction that serves as a roadmap (Part I), optimization (Part II), Monte Carlo methods (Part III), likelihood inference (Part IV), time series (Part V), specialized and multivariate models (Part VI), and causal inference (Part VII). A 30-page prerequisites appendix gathers assumed background material for reference.

### 1.2 Prerequisites

The reader is assumed to have:

- **Linear algebra.** Matrix operations, eigendecompositions, positive definiteness, SVD. At the level of a one-semester undergraduate course or Strang's *Introduction to Linear Algebra*.
- **Multivariate calculus.** Gradients, Hessians, Taylor expansions, the chain rule in several variables. Lagrange multipliers at the cookbook level (the book re-derives them properly in Chapter 2).
- **Probability and mathematical statistics.** Random variables, expectation, variance, covariance, joint and conditional distributions, the law of large numbers, the central limit theorem, maximum likelihood estimation. At the level of Casella and Berger chapters 1–7, or Rice's *Mathematical Statistics and Data Analysis*.
- **Basic regression.** The reader has fit a linear regression, knows what a $p$-value is, and has a working intuition for what "the model doesn't fit" means. The book does not teach regression from zero; it dissects regression under misspecification (Chapter 10).
- **Python fluency.** The reader writes Python comfortably, uses NumPy for array computation, and has called scipy or statsmodels functions before. The book does not teach Python syntax, NumPy broadcasting, or pandas indexing.
- **Floating-point arithmetic.** The reader knows that floating-point numbers are finite-precision and that $a + (b + c) \neq (a + b) + c$ in general. The prerequisites appendix provides a concise refresher; the book uses this knowledge without re-deriving it.

Material below this line is *not* assumed: measure-theoretic probability, functional analysis, abstract algebra, or systems programming. When a chapter needs a result from outside the prerequisites (e.g., the functional central limit theorem in Chapter 14), it states the result precisely, gives a reference, and builds on it without requiring the reader to verify the proof.

### 1.3 The Reader Promise

A reader who completes this book will be able to:

1. **Read the source.** Given any function in scipy.optimize, scipy.stats, or statsmodels, the reader can open the source code, identify the algorithm, and understand why it was implemented that way.
2. **Diagnose failures.** When an optimizer does not converge, when standard errors are wrong, when a bootstrap interval has poor coverage, or when a time-series model produces nonsense forecasts, the reader knows what to check and why.
3. **Choose defaults deliberately.** The reader understands the theoretical content of every default — why `scipy.optimize.minimize` defaults to BFGS, why statsmodels' OLS uses HC1 when asked for heteroscedasticity-consistent standard errors, why `scipy.stats.bootstrap` uses BCa — and can override them with justification.
4. **Extend and combine.** The reader can implement a variant not in the library (a new robust loss, a custom state-space model, a doubly robust estimator) by building on the from-scratch implementations in the book.
5. **Read the literature.** The bibliographic notes in each chapter point to the primary sources. A reader who has worked through the chapter can read the original papers — the notation has been aligned, the context has been set, and the key ideas have been developed.

---

## 2. Notation Conventions

### 2.1 Guiding Principle

Mathematical notation in the book stays as close as possible to scipy's parameter naming conventions. Where scipy names a parameter `x0`, the mathematics uses $x_0$. Where scipy names a function argument `fun`, the mathematics uses $f$. The reader should never have to maintain two unrelated symbol systems in their head.

When a chapter's primary library deviates substantially from scipy conventions — most notably statsmodels, which uses `endog`/`exog` instead of $y$/$X$ — the chapter provides an explicit mapping table in its "Library Implementation" section. The book's mathematical exposition uses the base notation defined here; the mapping table bridges to the library.

`NOTATION.md` at the project root is the authoritative symbol-by-symbol ledger. It records every symbol introduced in any chapter and is governed by the conventions in this section. The conventions here are the law; `NOTATION.md` is the census.

### 2.2 Scalar, Vector, and Matrix Conventions

| Convention | Meaning | Example |
|-----------|---------|---------|
| Lowercase italic | Scalar | $x$, $\alpha$, $\lambda$ |
| Lowercase bold | Column vector | $\mathbf{x}$, $\boldsymbol{\beta}$, $\boldsymbol{\theta}$ |
| Uppercase bold | Matrix | $\mathbf{X}$, $\mathbf{A}$, $\mathbf{I}$ |
| $\mathbf{x}_i$ | The $i$-th observation (a row of $\mathbf{X}$, transposed to a column) | |
| $x_{ij}$ | Element $(i,j)$ of $\mathbf{X}$ | |
| $\mathbf{X}^\top$ | Matrix transpose | |
| $\mathbf{X}^{-1}$ | Matrix inverse | |
| $\|\mathbf{x}\|$ | Euclidean norm unless otherwise stated | |
| $\|\mathbf{x}\|_p$ | $\ell_p$ norm, stated explicitly | |

### 2.3 Optimization Notation

These names are chosen to match `scipy.optimize.minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints)` and `scipy.optimize.least_squares(fun, x0, jac, bounds, method)`.

| Symbol | Meaning | scipy parameter |
|--------|---------|-----------------|
| $f(\mathbf{x})$ or $f$ | Objective function | `fun` |
| $\mathbf{x}_0$ | Initial point | `x0` |
| $\mathbf{x}^*$ | Solution / minimizer | result `.x` |
| $\nabla f(\mathbf{x})$ or $\mathbf{g}(\mathbf{x})$ | Gradient | `jac` |
| $\nabla^2 f(\mathbf{x})$ or $\mathbf{H}(\mathbf{x})$ | Hessian | `hess` |
| $\mathbf{H}(\mathbf{x})\mathbf{p}$ | Hessian-vector product | `hessp` |
| $\mathbf{r}(\mathbf{x})$ | Residual vector (least squares) | `fun` in `least_squares` |
| $\mathbf{J}(\mathbf{x})$ | Jacobian of the residual | `jac` in `least_squares` |
| $\Delta$ | Trust-region radius | |
| $\alpha_k$ | Step size at iteration $k$ | |
| $\mathbf{p}_k$ | Search direction at iteration $k$ | |
| $\mathbf{B}_k$ | Quasi-Newton Hessian approximation at iteration $k$ | |

$\mathbf{H}$ is reserved for the Hessian in optimization contexts (Parts I and III–VI). Chapters that need $\mathbf{H}$ for another purpose (e.g., the hat matrix in regression) must use a distinct symbol and declare it in the chapter's local notation table.

### 2.4 Statistical Model Notation

| Symbol | Meaning | Library name |
|--------|---------|--------------|
| $n$ | Number of observations | `nobs` |
| $p$ | Number of predictors (excluding intercept) | `df_model` (statsmodels, noting this counts rank) |
| $\mathbf{y}$ | Response vector ($n \times 1$) | `endog` (statsmodels) |
| $\mathbf{X}$ | Design matrix ($n \times (p+1)$ with intercept) | `exog` (statsmodels) |
| $\boldsymbol{\beta}$ | Coefficient vector | `params` (result attribute) |
| $\hat{\boldsymbol{\beta}}$ | Estimated coefficient vector | `.params` |
| $\boldsymbol{\varepsilon}$ | Error / disturbance vector | `.resid` |
| $\hat{\sigma}^2$ | Estimated error variance | `.scale` (statsmodels OLS) |
| $\ell(\boldsymbol{\theta})$ | Log-likelihood | `.llf` |
| $\mathcal{I}(\boldsymbol{\theta})$ | Fisher information matrix | |
| $\hat{\mathcal{I}}(\boldsymbol{\theta})$ | Observed information matrix | |

### 2.5 Distribution Notation

scipy.stats parameterizes all continuous distributions with shape parameters followed by `loc` and `scale`. The book follows this convention in mathematical notation.

| Notation | scipy call | Shape parameters |
|----------|-----------|------------------|
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | `norm(loc=mu, scale=sigma)` | none |
| $X \sim \text{Gamma}(a, \text{scale}=s)$ | `gamma(a, scale=s)` | `a` |
| $X \sim t_\nu$ | `t(df=nu)` | `df` |
| $X \sim \chi^2_\nu$ | `chi2(df=nu)` | `df` |
| $X \sim F_{\nu_1, \nu_2}$ | `f(dfn=nu1, dfd=nu2)` | `dfn`, `dfd` |
| $X \sim \text{Beta}(a, b)$ | `beta(a, b)` | `a`, `b` |

When a scipy shape parameter name is opaque (e.g., `s` for the lognormal shape), the chapter's mathematical development uses a standard name (e.g., $\sigma$ for lognormal) and the Library Implementation section provides the mapping.

### 2.6 Probability and Expectation

| Symbol | Meaning |
|--------|---------|
| $\Pr(\cdot)$ | Probability |
| $\mathbb{E}[\cdot]$ | Expectation |
| $\text{Var}(\cdot)$ | Variance |
| $\text{Cov}(\cdot, \cdot)$ | Covariance |
| $\xrightarrow{d}$ | Convergence in distribution |
| $\xrightarrow{p}$ | Convergence in probability |
| $\xrightarrow{a.s.}$ | Almost sure convergence |
| $O_p(\cdot)$, $o_p(\cdot)$ | Stochastic order notation |
| $\overset{d}{=}$ | Equal in distribution |

### 2.7 Indexing Conventions

- Observations are indexed by $i = 1, \dots, n$.
- Predictors are indexed by $j = 1, \dots, p$ (or $j = 0, \dots, p$ when the intercept is explicit).
- Iterations are indexed by $k = 0, 1, 2, \dots$ (matching the zero-indexed convention in scipy's callback).
- Time indices in Part IV use $t = 1, \dots, T$.
- Groups or clusters are indexed by $g = 1, \dots, G$.

### 2.8 Pseudocode

Pseudocode blocks use mathematical notation from this section, not Python syntax. A pseudocode block is typeset distinctly from a code block and is never mixed with Python in the same listing. The correspondence between pseudocode and the Python implementation that follows it should be line-for-line obvious; where it is not, the text explains the gap.

Pseudocode conventions:

- Assignment: $\leftarrow$
- Loops: **for**, **while**, **repeat … until**
- Conditionals: **if**, **else if**, **else**
- Return: **return**
- Functions: named in small caps, e.g., $\textsc{BacktrackingLineSearch}$
- Comments: preceded by $\triangleright$

### 2.9 Chapter-Level Notation Tables

When a chapter's primary library uses parameter names that differ substantially from the base notation above, the chapter must include a mapping table at the start of its Library Implementation section. The table has three columns: **Book notation**, **Library name**, **Notes**. This is mandatory for any chapter using statsmodels (where `endog`/`exog` replace $\mathbf{y}$/$\mathbf{X}$) and for any chapter whose library introduces domain-specific naming (e.g., `groups` in MixedLM and GEE, `order` in ARIMA, `exog_re` for random-effects design matrices).

---

## 3. Chapter Template

Every chapter follows this 11-section template. No section may be omitted. If a section has nothing to say for a particular chapter (rare — flag it if you think so), it still appears with a brief note explaining why it is empty.

### Section 1: Motivation

Why the reader should care about this method. What problem does it solve? What goes wrong without it? This section is concrete: it opens with a specific scenario, not a historical survey. It ends by stating what the chapter will deliver and what it will not cover.

Length: 1–2 pages.

### Section 2: Mathematical Foundation

The mathematical framework the method rests on. Definitions, key assumptions, and the core results stated precisely. The emphasis is on **intuition**: why does this method work? What happens when the assumptions fail? How does the theory explain the API design and the default parameter choices?

This section is *not* a theory textbook. Proofs are included only when they build intuition that the reader will reuse — typically when the proof *is* the algorithm (e.g., the BFGS update derivation leads directly to the code) or when the proof explains a failure mode the reader will encounter. Most results are stated precisely, given 2–3 sentences of intuition, and cited.

This section may contain short code demonstrations when a theoretical claim is best understood by seeing what happens.

Length: 3–8 pages. Shorter is better if the intuition is clear.

### Section 3: The Algorithm

The method translated into a step-by-step procedure. Presented as pseudocode using the conventions in §2.8. The pseudocode is complete enough that a reader could implement it from this section alone. Where the algorithm has variants, the main variant is presented first and the alternatives discussed after.

This section contains pseudocode blocks but no Python code.

Length: 2–5 pages.

### Section 4: Statistical Properties

What the estimator or algorithm converges to, at what rate, under what conditions. Consistency, asymptotic normality, efficiency, breakdown. This section connects the algorithm from Section 3 to the theory from Section 2: "The algorithm converges to the quantity we defined, at the rate we proved, under the assumptions we stated."

This section contains no code.

Length: 2–5 pages.

### Section 5: Library Implementation

How scipy or statsmodels implements the method. This section shows the library call, explains every non-obvious parameter, identifies which defaults carry theoretical content (and what that content is), and maps the library's naming to the book's notation via a mapping table (see §2.9).

This section uses the library as a black box. The reader learns what buttons to push and what each button does.

Length: 3–6 pages.

### Section 6: From-Scratch Implementation

A clean, pedagogical Python implementation of the core algorithm. This implementation is not production code — it is readable code that corresponds line-by-line to the pseudocode in Section 3. The section ends by verifying the from-scratch implementation against the library output on a test problem.

The purpose is understanding, not performance. If the from-scratch version is 100x slower, that is fine; the text notes it and explains what the library does differently.

Length: 3–8 pages.

### Section 7: Diagnostics

How to tell whether the method worked. Convergence checks, residual plots, sensitivity analysis, assumption tests. This section is prescriptive: "Check X. If X fails, it means Y. Do Z." Every diagnostic shown is connected to a specific assumption from Section 2.

Length: 2–5 pages.

### Section 8: Computational Considerations

Time and space complexity. Numerical stability issues. What scales and what does not. When to use an approximation and what you lose. This section is where the finite-difference Hessian's $O(p^2)$ cost is stated, where the $O(n^3)$ cost of GLS is noted, where the stochastic-gradient alternative is mentioned for problems too large for exact methods.

Length: 1–3 pages.

### Section 9: Worked Example

An end-to-end analysis that uses the method on a realistic problem. The example goes from raw data to final interpretation, using the library implementation from Section 5 and the diagnostics from Section 7. Every code block is shown with its output. The example should be substantive enough that it could appear in a methods paper's simulation section.

Where a running example from `RUNNING_EXAMPLES.md` fits, it is used. Where a new dataset is needed, it is described fully and added to the running-examples ledger.

Length: 4–8 pages.

### Section 10: Exercises

Exactly four exercises per chapter. One must be a diagnostic-failure exercise. The remaining three are drawn from:

- **Derivation.** Complete a proof step that the chapter left as "verify that…"
- **Implementation.** Extend the from-scratch code to handle a variant.
- **Application.** Apply the method to a new dataset.
- **Diagnostic failure.** Given a scenario where the method's assumptions are violated, identify the failure from diagnostic output. (Mandatory — every chapter has one.)
- **Comparison.** Compare two methods or two sets of defaults on the same problem.
- **Conceptual.** Explain, without code, why a particular behavior occurs.

Every exercise includes a full worked solution in a `%% ... %%` delimited block. Solutions are QMD: prose explanation, math where needed, and fully annotated Python code with outputs. A solution without annotation is not a solution — the reader should learn from reading it, not just check their answer.

Exercises are numbered and, where appropriate, rated by difficulty: ($\star$), ($\star\star$), ($\star\star\star$).

Length: 2–4 pages.

### Section 11: Bibliographic Notes

Where the method comes from, who developed it, and what to read next. References are specific: "The BFGS update was introduced independently by Broyden (1970), Fletcher (1970), Goldfarb (1970), and Shanno (1970); see Nocedal and Wright (2006), Ch. 6 for a textbook treatment" — not "see the literature on quasi-Newton methods." This section may contain remarks, historical context, and pointers to advanced topics that the chapter does not cover.

Length: 1–2 pages.

---

## 4. Proof Standard

### 4.1 Guiding Principle

This is a coding book, not a theory book. Proofs are included only when they serve one or more of:

- **Intuition** — the proof explains *why* the method works in a way that changes how the reader thinks about it
- **Algorithm** — the proof *is* the algorithm (e.g., deriving the BFGS update leads directly to the NumPy code)
- **API design** — the proof explains *why* the interface is designed the way it is (e.g., why `method='BFGS'` needs a gradient but not a Hessian)
- **Diagnostics** — the proof explains *what goes wrong* when assumptions fail and *how to detect it*
- **Practical tuning** — the proof explains *why* a parameter choice matters (e.g., why `c2=0.9` for BFGS but `c2=0.1` for CG)

If a proof does none of these, it should be a statement with intuition and a citation. The from-scratch implementation is the proof for most readers: if the code matches scipy's output, the theory is validated.

### 4.2 Three Levels

1. **Proof that builds intuition.** The proof is given because it teaches something the reader will use — either directly in code or in understanding why code behaves as it does. These proofs are typically short (under 1 page) and lead directly to an implementation or a diagnostic.

2. **Result with intuition and citation.** The result is stated precisely. Two to three sentences explain *why* it is true — the key idea, the geometric picture, what would go wrong without it. A specific reference is given. Code may demonstrate the result. This is the **default level** for most results.

3. **Statement with reference.** The result is stated precisely and cited. Used for technical results that the reader needs to trust but not internalize (e.g., convergence rates, regularity conditions).

### 4.3 Assignment Policy

The default is level 2 (result with intuition). The agent *upgrades* to level 1 only when the proof serves one of the purposes in §4.1. The agent does not need approval to *downgrade* from the proof levels specified in `CHAPTERS.md` — the CHAPTERS.md entries were written before this rebalancing and many of their "full proof" designations should now be level 2.

### 4.4 General Principles

- **No "clearly" without backing.** One sentence of intuition replaces the word "clearly."
- **No "it can be shown."** State the result, give intuition, cite the reference, or show code.
- **Asymptotic regime stated.** Every asymptotic result specifies what is growing.
- **The code is the proof.** When a from-scratch implementation matches scipy's output to 6 digits, that verification carries more weight for this book's audience than a convergence theorem.
- **Don't get stuck in proofs.** If a proof is taking more than a page and doesn't lead to code or a diagnostic, it's too long. State the result, give the key idea, cite, and move on.

### 4.5 Cross-Chapter Reuse

When a later chapter needs a result from an earlier chapter, it cites by number and does not re-derive.

---

## 5. Code Conventions

### 5.1 Language and Environment

All code in the book is Python, executed in the environment defined by `requirements.txt` at the project root. No code block in the book may use a library not listed in `requirements.txt` unless the addition has been approved under §7.3 and `requirements.txt` has been updated. An import that fails in the pinned environment is a failed chapter.

### 5.2 Reproducibility

Every code block that involves randomness must use `numpy.random.Generator` via `numpy.random.default_rng` with an explicit integer seed. The legacy API (`numpy.random.seed`, `numpy.random.randn`, etc.) is never used.

The canonical pattern is:

```python
import numpy as np
rng = np.random.default_rng(42)
```

The `rng` object is created once per chapter (in the first code block that needs it) and passed explicitly to every function that requires randomness. Functions that accept a `random_state` or `seed` parameter from scipy or statsmodels receive the `rng` object or its underlying `BitGenerator` as appropriate.

Seeds must be deterministic: every code block in the book, run in sequence in a fresh Python session with the pinned environment, must reproduce the outputs shown. If a code block depends on output from a previous block, that dependency is explicit.

### 5.3 Code Style

- **PEP 8** governs formatting. Lines do not exceed 79 characters (the Quarto rendering column is narrower than a terminal, and long lines wrap badly in print).
- **Imports at the top.** Each chapter's first code block contains all imports for the chapter. Subsequent blocks do not re-import.
- **No star imports.** `from scipy.stats import *` is never used.
- **Explicit over implicit.** Named keyword arguments are preferred over positional arguments whenever the function has more than two parameters. Library defaults that carry theoretical content (e.g., `method='BFGS'`, `cov_type='nonrobust'`, `alpha=0.05`) are always spelled out, even when the code would produce the same result without them.
- **Variable names match notation.** Where the mathematics uses $\mathbf{X}$, the code uses `X`. Where the mathematics uses $\hat{\boldsymbol{\beta}}$, the code uses `beta_hat`. Mapping between book notation and code variable names must be immediately obvious; where it is not, a comment explains.
- **Comments explain why, not what.** A comment like `# Compute the gradient` above a line that obviously computes the gradient adds nothing. A comment like `# Gradient uses the Woodbury identity to avoid forming the full Hessian` adds information the reader needs.

### 5.4 From-Scratch Code

From-scratch implementations (Section 6 of each chapter) follow these rules:

- **Pure NumPy/SciPy.** From-scratch code uses only NumPy for array operations and, where necessary, SciPy for low-level numerical primitives (linear algebra solves, special functions). It does not call the high-level scipy or statsmodels function it is reimplementing.
- **Correspondence to pseudocode.** Each from-scratch implementation must correspond line-by-line to the pseudocode in Section 3 of the same chapter. Variable names in the code match the pseudocode symbols (transliterated per §2 conventions). Where the code must diverge from the pseudocode for practical reasons (e.g., vectorizing a loop), the divergence is noted in a comment.
- **Verification against library.** Every from-scratch implementation ends with a verification block that runs both the from-scratch version and the library version on the same input and asserts that the results agree to a stated tolerance. The tolerance is chosen to reflect the algorithm, not floating-point noise: if the from-scratch and library versions use different convergence criteria, the tolerance reflects that.
- **No cleverness for its own sake.** The code is meant to teach, not to impress. Broadcasting tricks, one-liners, and micro-optimizations are avoided if they obscure the algorithmic structure. A readable three-line version is preferred over a clever one-line version.
- **Docstrings.** Every from-scratch function has a docstring stating what it computes, what its parameters are (with types), and what it returns. The docstring references the relevant section of the chapter ("Implements Algorithm 3.1").

### 5.5 Library Defaults

Many scipy and statsmodels functions ship with defaults that encode substantive statistical or algorithmic choices. The book treats these defaults as part of the theory, not as incidental implementation decisions. Specifically:

- **Every default that carries theoretical content must be named explicitly** in the Library Implementation section (Section 5 of each chapter). "Theoretical content" means the default affects the statistical properties of the output (e.g., convergence rate, consistency, coverage) or selects among meaningfully different algorithms (e.g., `method='BFGS'` vs `method='trust-ncg'`).
- **The theoretical meaning of the default must be stated.** It is not enough to say "`scipy.optimize.minimize` defaults to `method='BFGS'`." The text must say *why* BFGS is the default (good general-purpose performance for smooth unconstrained problems; superlinear convergence under standard conditions) and *when the default is inappropriate* (large-scale problems where L-BFGS-B or trust-region methods are preferable; nonsmooth objectives where BFGS will fail).
- **Code in the book spells out defaults rather than relying on them.** Even when the default is the desired choice, the code includes the keyword argument so the reader sees it:

  ```python
  result = minimize(
      fun=rosenbrock,
      x0=x0,
      method='BFGS',       # default, but stated for clarity
      jac=rosenbrock_grad,
  )
  ```

  This rule applies to all parameters whose default carries theoretical content. It does not apply to genuinely incidental parameters (e.g., `disp=False`, `full_output=True`) unless their values matter for the chapter's narrative.

- **Statsmodels covariance-type defaults receive particular attention.** The default `cov_type='nonrobust'` in statsmodels assumes homoscedastic, independent errors. Every chapter that uses a statsmodels model must state which `cov_type` is being used and why. If the chapter's worked example involves heteroscedastic or clustered data, the code must use the appropriate robust variant (`HC0`–`HC3`, `HAC`, `cluster`) and the text must explain the choice.

Mathematical notation in pseudocode matches §2 of this document. No mixing of Python and pseudocode in the same block.

### 5.6 Plotting
- Matplotlib only; no seaborn, no plotly
- Every figure has a caption explaining what to look for
- Color is functional, not decorative; figures must remain interpretable in grayscale
- No 3D plots unless the geometry genuinely is three-dimensional
- Figure-generating code is shown in the book, not hidden

### 5.7 Datasets
- Synthetic data: generated in-chapter with shown code and seed
- Real data: drawn from statsmodels' built-in datasets, scikit-learn datasets, or a small set of clean public sources cited in the bibliography
- No proprietary, gated, or url-fragile sources
- Every dataset used is described: what it is, how many rows, what the variables mean, why it's appropriate for the method

### 5.8 Library Versions
The book targets specific versions, pinned in a `requirements.txt` at the project root. The agent verifies the running environment matches before generating output. Version drift is a release blocker; a chapter using deprecated APIs is a failed chapter.

---

## 6. Cross-Chapter Consistency Rules

### 6.1 Re-Reading Discipline
Before drafting any chapter, the agent must:
1. Read this steering document in full
2. Read `NOTATION.md`
3. Read the table of contents and any chapter the current chapter explicitly depends on
4. Read at least the §1 (Motivation) of the immediately preceding chapter

This is non-negotiable. Most consistency failures come from skipping this.

### 6.2 Forward References
Allowed, but must be specific: "covered in §17.4" not "covered later." If the referenced section does not yet exist, the reference is logged in `FORWARD_REFS.md` and resolved when that section is written.

### 6.3 Backward References
Required when reusing a result. "By Theorem 4.2…" not "as we showed earlier." The agent verifies the cited theorem exists and says what is claimed before committing.

### 6.4 Terminology Lock
Once a term is introduced with a definition, that definition is used everywhere. A `GLOSSARY.md` is maintained and re-read at chapter start. If a chapter wants to use a term differently, this is a steering-document-level decision, not a chapter-level one — flag it.

### 6.5 Example Reuse
A small set of running examples (a regression dataset, a time series, an optimization test problem suite) recurs across chapters. These are defined in an early chapter and referenced thereafter. The agent does not invent new datasets when an existing running example fits.

---

## 7. Working Process

### 7.1 Per-Chapter Workflow
1. **Outline pass.** Agent produces a section-by-section outline against the template, with key theorems named and key code listings sketched. Human reviews and approves before drafting.
2. **Theory pass.** Sections 2–4 drafted. Code-free. Reviewed for correctness and notation compliance.
3. **Implementation pass.** Sections 5–7 drafted, with all code executed and outputs captured.
4. **Worked example and exercises.** Sections 8–10 drafted.
5. **Integration pass.** Cross-references resolved, notation ledger updated, glossary updated, any forward references logged.
6. **Self-audit.** Agent runs the audit checklist (§8 below) and reports findings before declaring the chapter done.

### 7.2 Files Maintained at Project Root
- `STEERING.md` — this document
- `NOTATION.md` — symbol ledger
- `GLOSSARY.md` — term ledger
- `TOC.md` — table of contents with status per chapter (planned / outlined / drafted / reviewed / final)
- `FORWARD_REFS.md` — unresolved forward references
- `requirements.txt` — pinned dependencies
- `RUNNING_EXAMPLES.md` — datasets and test problems reused across chapters
- `STYLE_DEVIATIONS.md` — log of any approved departures from this document, with rationale

### 7.3 What the Agent Does Not Decide Unilaterally
- Adding a chapter
- Removing a chapter
- Changing notation conventions in §2
- Changing the chapter template in §3
- Introducing a new external library
- Departing from the proof standard in §4

For any of the above, the agent stops and surfaces the question.

---

## 8. Self-Audit Checklist (Run Before Marking Any Chapter Complete)

The agent must affirmatively answer each of these and produce the audit as a comment on the chapter draft.

**Notation**
- [ ] Every symbol used appears in `NOTATION.md` or in the chapter's local mini-table
- [ ] No symbol is used with two meanings in the same chapter
- [ ] Conventions from §2 are followed without exception, or deviations are logged

**Theory**
- [ ] Every theorem has a proof, sketch with citations, or explicit reference
- [ ] Every assumption used is numbered and referenced when invoked
- [ ] No "it can be shown" or "clearly" without backing
- [ ] Asymptotic regime is stated for every asymptotic claim

**Code**
- [ ] All code blocks have been executed in the pinned environment
- [ ] Outputs in the book match the outputs produced
- [ ] From-scratch implementation is verified against the library version
- [ ] Library defaults that carry theoretical content are explicitly named
- [ ] Random seeds shown and reproducible

**Structure**
- [ ] All eleven sections of the template are present
- [ ] Section 9 worked example goes end to end
- [ ] Exactly four exercises, including one diagnostic-failure exercise, each with a full worked solution
- [ ] Bibliographic notes present and specific (not "see the literature")

**Cross-Chapter**
- [ ] Forward references logged
- [ ] Backward references verified
- [ ] Terminology matches `GLOSSARY.md`
- [ ] Running examples reused where appropriate

**Scope**
- [ ] No drift into Bayesian, causal, RL, interpretability, or LLM territory
- [ ] Chapter scope matches what was approved in the outline pass

If any box is unchecked, the chapter is not done and the agent must say so explicitly rather than declaring completion.

---

## 9. Things the Agent Should Flag, Not Decide

When the agent encounters any of the following, it stops and asks rather than guessing:

- A theorem it is reconstructing from memory rather than from a reference it has actually consulted
- A library API whose current behavior it cannot verify by running code
- A notation collision not resolved by §2
- A topic that seems to require pulling in an out-of-scope area
- An exercise it cannot itself solve to confirm difficulty calibration
- A claim about a recent paper or version-specific behavior

Flagging is preferred to fabrication. A flagged question is cheap; a wrong derivation in a graduate textbook is expensive.

---

## 10. Tone and Voice

- Direct, second person where addressing the reader, third person otherwise
- Assume intelligence and prior background; do not over-explain undergraduate material
- Where a result is genuinely surprising or a default is genuinely dangerous, say so plainly
- Avoid hedging language ("it might be useful to note that perhaps…"); commit to claims or do not make them
- Humor is allowed in remarks and bibliographic notes; not in theorem statements

---

## 11. Definition of Done for the Book

The book is complete when:
- All planned chapters are at "final" in `TOC.md`
- `NOTATION.md` has been audited end to end with no orphaned or duplicated symbols
- Every code block in the book runs cleanly in the pinned environment in a fresh checkout
- Every theorem is provable from references the reader can obtain
- A reader following the prerequisites in §1.2 can read any chapter and meet the §1.3 reader promise

Anything less is a draft.
