# Book Chapter Plan (Focused Edition)

This document supersedes the previous `CHAPTERS.md`. The agent does not add, remove, or reorder chapters without escalation per `STEERING.md` §7.3.

The book has 23 chapters in 7 parts. Foundational material (floating point, dense linear algebra, basic distributions, parametric tests, correlation, power) is assumed and is not given chapter-level treatment. A 30-page **Prerequisites Appendix** gathers the omitted material as reference, not pedagogy.

Each entry specifies: scope, primary APIs, theoretical core, depth signals (what specifically gets a full proof, what gets a sketch with citation), and dependencies.

---

## Part I — Motivation and Overview

### Chapter 1. The ML Pipeline as a Window into SciPy
**Scope.** A quick ML refresher that doubles as the book's roadmap. Build a complete scikit-learn pipeline — preprocessing, model fitting, cross-validation, hyperparameter search — then systematically unpack what each stage does under the hood. When `LogisticRegression.fit()` runs, it calls an optimizer (→ Ch. 2–6). When `cross_val_score` splits and scores, it makes statistical assumptions about the loss surface (→ Ch. 10) and about what the score estimates (→ Ch. 9). When `GridSearchCV` searches, it solves a global optimization problem with noisy evaluations (→ Ch. 5). AutoML frameworks (FLAML, auto-sklearn) compose these pieces and add their own assumptions. This chapter names the questions; the rest of the book answers them.
**Library.** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`); `scipy.optimize.minimize` shown briefly as the engine beneath `.fit()`; `statsmodels.api.OLS` shown briefly for coefficient-level inference that scikit-learn deliberately omits.
**Theory — light.** This is a survey chapter. It introduces the bias-variance decomposition informally, defines the train/validation/test split as a statistical protocol, and states (without proof) that cross-validation estimates expected loss. Every claim made here is proved rigorously in a later chapter; forward references are specific.
**AutoML.** The CASH (Combined Algorithm Selection and Hyperparameter optimization) problem stated as a mathematical optimization problem. Connection to Ch. 5 (derivative-free/global optimization) made explicit. Honest treatment of what AutoML does and does not automate — feature engineering, problem formulation, and model diagnostics remain human work.
**Depends on.** Prerequisites Appendix only.

---

## Part II — Optimization for Statistical Estimation

### Chapter 2. Smooth Optimization: Line Search, Quasi-Newton, and Trust Regions
**Scope.** Newton, BFGS, L-BFGS-B, Newton-CG, trust-region methods. Line search vs trust region as a design axis.
**Library.** `scipy.optimize.minimize` with `BFGS`, `L-BFGS-B`, `Newton-CG`, `trust-ncg`, `trust-krylov`; user-supplied gradients/Hessians vs finite differences vs automatic differentiation (pointer to `jax`/`autograd`, not used).
**Theory — full proofs.** Wolfe and Armijo conditions; Zoutendijk's theorem; secant equation and BFGS update derivation; Dennis–Moré characterization of superlinear convergence; trust-region subproblem and the Cauchy point.
**Theory — sketched with citation.** Convergence of L-BFGS under standard assumptions; Steihaug–Toint truncated CG.
**Depth additions over previous plan.** Full Dennis–Moré proof; explicit treatment of why finite-difference Hessians degrade Newton's method; backtracking strategies under nonconvexity.
**Depends on.** Prerequisites Appendix.

### Chapter 3. Constrained and Equality-Constrained Optimization
**Scope.** SQP and interior-point methods for nonlinearly constrained problems. KKT theory developed in earnest.
**Library.** `scipy.optimize.minimize` with `SLSQP`, `trust-constr`; `LinearConstraint`, `NonlinearConstraint`. From-scratch SQP on small problems.
**Theory — full proofs.** First-order KKT conditions under LICQ; second-order sufficient conditions; constraint qualifications (LICQ, MFCQ, CRCQ) and counterexamples for each; SQP as Newton's method on the KKT system; primal-dual interior-point logarithmic barrier.
**Theory — sketched.** Active-set strategies; Maratos effect and second-order corrections.
**Depth additions.** Worked counterexamples showing each constraint qualification failing; full derivation of why SQP equals Newton on KKT.
**Depends on.** Ch. 2.

### Chapter 4. Large-Scale and Composite Optimization
**Scope.** First-order methods at scale; nonsmooth composite objectives $f(x) + g(x)$; coordinate methods. SciPy coverage is partial; the chapter is candid about the boundary.
**Library.** From-scratch implementations of gradient descent with momentum, Nesterov acceleration, ISTA/FISTA, proximal gradient, ADMM, coordinate descent. Cross-reference statsmodels' `fit_regularized` and the elastic-net path. `cvxpy` mentioned once as the prototyping tool of choice for convex composite problems.
**Theory — full proofs.** $O(1/k)$ rate for gradient descent under Lipschitz gradient; $O(1/k^2)$ for Nesterov via the standard estimating-sequence argument; convergence of proximal gradient; proximal operator as resolvent of subdifferential; ADMM convergence under convexity (Boyd-style proof); coordinate descent for separable regularizers.
**Theory — sketched.** Acceleration under strong convexity; nonconvex extensions; stochastic variants (SGD, SVRG, SAGA) at the level of statement and intuition.
**Depth additions.** Full Nesterov derivation; full Moreau envelope and proximal operator chapter section; ADMM convergence; coverage of when first-order methods beat second-order at scale and when they don't.
**Depends on.** Ch. 2.

### Chapter 5. Global, Stochastic, and Derivative-Free Optimization
**Scope.** Nonconvex landscapes; derivative-free methods; stochastic global search.
**Library.** `scipy.optimize.differential_evolution`, `dual_annealing`, `basinhopping`, `shgo`, `direct`; `Nelder-Mead`, `Powell`, `COBYLA`.
**Theory — full proofs.** Simulated annealing convergence in probability under logarithmic cooling; Lipschitz-based DIRECT correctness on Lipschitz functions.
**Theory — sketched with care.** Convergence (and lack thereof) for Nelder–Mead — McKinnon's counterexample shown explicitly; differential evolution heuristics and their lack of global guarantees; basin-hopping.
**Depth additions.** McKinnon counterexample worked through; honest treatment of why most global optimizers have weak guarantees and what to do about it in practice.
**Depends on.** Ch. 2.

### Chapter 6. Nonlinear Least Squares and Implicit Estimation
**Scope.** Sum-of-squares objectives with structure; root finding for nonlinear systems; M-estimation as a unifying lens for "find the parameter that solves an estimating equation."
**Library.** `scipy.optimize.least_squares` (`lm`, `trf`, `dogbox`); `curve_fit`; `scipy.optimize.root` with `hybr`, `broyden1/2`, `krylov`.
**Theory — full proofs.** Gauss–Newton as Newton with omitted second-order term; Levenberg–Marquardt as trust-region least squares; quadratic convergence of Newton; Broyden secant update.
**Theory — sketched.** Jacobian-free Newton–Krylov; robust loss functions and their implicit weighting (full treatment deferred to Ch. 12).
**Depth additions.** Connection between nonlinear least squares and quasi-likelihood foreshadowed; merge of root-finding and least-squares into a single chapter as both are zero-finding for $\nabla f$ or $F$.
**Depends on.** Ch. 2, 4.

---

## Part III — Probability, Sampling, and Monte Carlo

### Chapter 7. Monte Carlo Integration and Variance Reduction
**Scope.** Monte Carlo as a numerical method; the full variance-reduction toolkit, treated with the rigor it deserves.
**Library.** Hand-built MC estimators using `numpy.random.Generator` and `scipy.stats`; `scipy.stats.qmc` (Sobol, Halton, LatinHypercube).
**Theory — full proofs.** $\sqrt{n}$ rate of plain MC; optimal importance sampling proposal and the zero-variance limit; control variates as projection (regression interpretation); antithetic variates variance reduction under monotonicity; stratified sampling variance bound; Rao–Blackwellization.
**Theory — sketched.** Star discrepancy and the Koksma–Hlawka inequality; effective sample size in self-normalized importance sampling; adaptive importance sampling.
**Depth additions over previous plan.** Three former chapters (sampling methods, MC integration, simulation studies) compressed into one focused chapter; new explicit treatment of self-normalized importance sampling and its bias; QMC absorbed in via Koksma–Hlawka.
**Depends on.** Prerequisites Appendix.

### Chapter 8. MCMC: Theory and Diagnostics
**Scope.** Markov chain Monte Carlo at the depth of a graduate methods chapter, not a tour. From-scratch implementations to ground intuition; an honest pointer to PyMC/numpyro for production use.
**Library.** From-scratch Metropolis–Hastings, Gibbs, slice sampling, Hamiltonian Monte Carlo (small-scale, no autodiff). Diagnostics with `arviz`-compatible output where useful.
**Theory — full proofs.** Detailed balance implies stationarity; Metropolis–Hastings acceptance ratio derivation; Gibbs as a special case of M-H; geometric ergodicity for M-H on log-concave targets (statement with sketch); $\hat R$ definition and use; effective sample size from autocorrelation; asymptotic variance of MCMC estimators (CLT for Markov chains, statement of Kipnis–Varadhan).
**Theory — sketched.** Hamiltonian Monte Carlo via leapfrog and detailed-balance preservation; NUTS at conceptual level.
**Depth additions.** This is now a full chapter rather than the appendix-style treatment in the previous plan. Justification: MCMC theory is reused implicitly throughout Bayesian-adjacent diagnostics (e.g., posterior predictive checks, simulation-based calibration), and the math is too important to gesture at.
**Depends on.** Ch. 7.

### Chapter 9. Resampling Inference and the Bootstrap
**Scope.** Frequentist inference via simulation. The chapter that confronts both the power and the failure modes of the bootstrap.
**Library.** `scipy.stats.bootstrap`, `permutation_test`, `monte_carlo_test`; from-scratch jackknife, BCa, and block bootstrap for time series.
**Theory — full proofs.** Bootstrap consistency for the sample mean (Hall-style argument); BCa interval construction; permutation test exactness under exchangeability; jackknife bias estimation.
**Theory — sketched with explicit failure cases.** Bootstrap failure for the maximum, for non-smooth functionals, for the sample variance under heavy tails. Subsampling as a partial remedy. Block bootstrap consistency for stationary processes (Künsch).
**Depth additions.** Explicit catalog of bootstrap failure modes with code demonstrating each; subsampling as the "what to do when bootstrap fails" toolkit; honest framing.
**Depends on.** Ch. 7.

---

## Part IV — Inference in Likelihood Models

### Chapter 10. The Wald, Score, and Likelihood-Ratio Trinity
**Scope.** The unifying inferential framework for likelihood-based models. Cited and reused in every subsequent regression chapter.
**Library.** `model.wald_test`, `wald_test_terms`, `score_test`, `compare_lr_test`; `aic`, `bic`; `linearmodels` for non-nested testing.
**Theory — full proofs.** Asymptotic normality of MLE under standard regularity conditions (Cramér's conditions stated and used); information matrix equality; asymptotic equivalence of Wald, score, and LR tests under the null; Chernoff's theorem for boundary cases; Vuong's test for non-nested comparison.
**Theory — sketched.** Robust ("sandwich") form of Wald tests under misspecification; behavior of the trinity under weak identification (forward-referenced to Ch. 20); higher-order corrections (Bartlett); information criteria as approximations to predictive risk (AIC as Kullback–Leibler estimator, BIC as Laplace approximation to marginal likelihood).
**Depth additions.** This chapter now leads Part IV rather than appearing midway through regression. Every later regression chapter cites it. Full proof of asymptotic equivalence; Chernoff boundary case; explicit treatment of the sandwich form.
**Depends on.** Ch. 2; Prerequisites Appendix.

### Chapter 11. Linear Regression Under Misspecification
**Scope.** OLS treated honestly: as an estimator of a best linear predictor whose properties degrade gracefully as Gauss–Markov assumptions fail. The chapter is structured around what breaks and how to fix it.
**Library.** `statsmodels.api.OLS`, `WLS`, `GLS`, `GLSAR`; `cov_type` in `HC0`–`HC3`, `HAC`, `cluster`; diagnostics (`outlier_test`, `OLSInfluence`, `het_breuschpagan`, `het_white`, `linear_reset`, `acorr_ljungbox`, `variance_inflation_factor`).
**Theory — full proofs.** Gauss–Markov; finite-sample distribution under normality; asymptotic normality under weaker assumptions; sandwich variance estimator derivation; HC0–HC3 as small-sample corrections; HAC (Newey–West) variance estimator with kernel and bandwidth; cluster-robust variance and its asymptotics in $G \to \infty$; influence function and influence diagnostics.
**Theory — sketched.** Behavior under endogeneity (forward-referenced to Ch. 20); high-dimensional regimes (forward-referenced to Ch. 12).
**Depth additions.** Full sandwich derivation; explicit treatment of HC variants and their finite-sample properties; cluster-robust variance with the few-clusters problem; linkage to influence-function theory (preview of M-estimation in Ch. 12).
**Depends on.** Ch. 10.

### Chapter 12. Generalized Linear Models and M-Estimation
**Scope.** GLMs as a unified family, then a generalization to the M-estimator framework that subsumes GLMs, robust regression, and quantile regression as instances. Chapter delivers what was previously three chapters (GLMs, robust regression, quantile regression) with a single theoretical spine.
**Library.** `statsmodels.api.GLM` with `Gaussian`, `Binomial`, `Poisson`, `NegativeBinomial`, `Gamma`, `InverseGaussian`, `Tweedie`; `RLM` with `HuberT`, `Hampel`, `TukeyBiweight`, `RamsayE`, `AndrewWave`; `QuantReg`.
**Theory — full proofs.** Exponential dispersion family; canonical and noncanonical links; score and information for GLMs; IRLS as Fisher scoring (full derivation); deviance and its $\chi^2$ asymptotics; M-estimator consistency under standard conditions; M-estimator asymptotic normality and the "bread and meat" sandwich form; influence function as Gâteaux derivative; breakdown point of the median and of $M$-estimators with bounded $\psi$; quantile regression as $M$-estimator with check loss; subgradient optimality conditions for QR.
**Theory — sketched.** Quasi-likelihood and dispersion estimation; semiparametric efficiency bound for M-estimators (statement, with reference to Bickel–Klaassen–Ritov–Wellner).
**Depth additions.** This is the chapter that previously did not exist as a unified treatment. The M-estimator framing makes Wald/score/LR from Ch. 10 immediately reusable; influence-function theory connects to causal-inference robust-estimator theory in Ch. 23.
**Depends on.** Ch. 10, 11; Ch. 4 for QR computation.

### Chapter 13. Discrete Choice and Limited Dependent Variables
**Scope.** Models for binary, ordinal, multinomial, and count outcomes, with zero-modification and selection.
**Library.** `Logit`, `Probit`, `MNLogit`, `OrderedModel`, `Poisson`, `NegativeBinomial`, `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomialP`, `HurdleCountModel`, `ConditionalLogit`, `ConditionalPoisson`; selection models (Heckman two-step) from `linearmodels` or hand-rolled.
**Theory — full proofs.** Latent-utility derivation of probit and logit; identification (scale and location normalizations); IIA in MNL and the Hausman–McFadden test; marginal effects (average vs at means) and their asymptotic variances via the delta method; zero-inflation as a finite mixture and identification thereof.
**Theory — sketched.** Nested logit and mixed logit (statement and pointer to specialized packages); Heckman selection model identification via exclusion restriction.
**Depth additions.** Full delta-method treatment of marginal effects (a chronic point of confusion); IIA failure modes with explicit examples; identification arguments throughout, not just estimation.
**Depends on.** Ch. 10, 12.

### Chapter 14. Mixed Effects, GEE, and Clustered Data
**Scope.** Within-cluster dependence handled via random effects (conditional models) or working correlation (marginal models). Direct comparison of the two paradigms.
**Library.** `statsmodels.regression.mixed_linear_model.MixedLM`; `statsmodels.genmod.generalized_estimating_equations.GEE` with `Exchangeable`, `AR`, `Independence`, `Unstructured`; cross-reference `pymer4` for nlme-style fits.
**Theory — full proofs.** Variance components in linear mixed models; profile likelihood for variance parameters; REML as a likelihood for residual contrasts; BLUP derivation; GEE as quasi-likelihood with specified working correlation; consistency of GEE under misspecified working correlation; sandwich variance for GEE.
**Theory — sketched.** Generalized linear mixed models (Laplace approximation, adaptive Gauss–Hermite quadrature, pseudo-likelihood) — statement of methods, full derivation deferred to references.
**Depth additions.** REML derivation in full (a standard "skip" in less rigorous treatments); explicit comparison of marginal vs conditional interpretation with worked example showing where they disagree.
**Depends on.** Ch. 11, 12.

---

## Part V — Time Series

### Chapter 15. ARIMA, Exponential Smoothing, and the Box–Jenkins Methodology
**Scope.** Univariate stochastic models for stationary and integrated series.
**Library.** `statsmodels.tsa.arima.model.ARIMA`, `SARIMAX`; `ExponentialSmoothing`, `ETSModel`; `adfuller`, `kpss`, `pp`; `acf`, `pacf`, `acorr_ljungbox`.
**Theory — full proofs.** Wold decomposition; stationarity and invertibility conditions for ARMA in terms of root location; ACF and PACF derivations for AR(1), MA(1), ARMA(1,1); Yule–Walker equations; conditional vs exact MLE for ARMA; unit root distribution (Dickey–Fuller asymptotic theory) at the level of Phillips' functional CLT argument with full statement; KPSS as a stationarity test rather than unit-root test.
**Theory — sketched.** Innovations algorithm; spectral representation of stationary processes (statement, used to motivate frequency-domain methods).
**Depth additions.** Full Dickey–Fuller asymptotic treatment (functional CLT, integrated Brownian motion limits, why standard $t$-tables are wrong) — this is the chapter section that justifies the existence of the chapter at this level.
**Depends on.** Ch. 10; Prerequisites Appendix.

### Chapter 16. State Space Models, Kalman Filtering, and Regime Switching
**Scope.** State-space form as a general framework subsuming ARIMA, exponential smoothing, structural time series, dynamic factor models. Hidden Markov regime switching as a discrete-state cousin.
**Library.** `statsmodels.tsa.statespace.MLEModel`, `SARIMAX`, `UnobservedComponents`, `DynamicFactor`, `VARMAX`; `MarkovRegression`, `MarkovAutoregression`.
**Theory — full proofs.** Kalman filter recursions derived from joint Gaussian conditioning (full derivation); Kalman smoother (RTS); innovations form of the likelihood; EM for state-space models; Hamilton filter for Markov regime switching.
**Theory — sketched.** Particle filtering for non-Gaussian/nonlinear state-space models (statement, with pointer); square-root and information-form filters for numerical stability.
**Depth additions.** Kalman recursions derived end-to-end rather than presented as a fait accompli; Hamilton filter recursions in full.
**Depends on.** Ch. 15.

### Chapter 17. Multivariate Time Series and Cointegration
**Scope.** Vector models with shared dynamics; integrated systems; structural identification.
**Library.** `statsmodels.tsa.api.VAR`, `VECM`, `coint_johansen`, `coint`; impulse response functions, FEVD, Granger causality.
**Theory — full proofs.** VAR stability via companion-matrix eigenvalues; Wold representation in VAR; Granger causality as restriction on VAR coefficients; Engle–Granger two-step procedure; Johansen reduced-rank regression and trace/eigenvalue statistics; cointegrating rank inference under Johansen.
**Theory — sketched.** Structural VAR identification (Cholesky, long-run, sign restrictions); local projections as an alternative to VAR-IRFs (Jordà); structural breaks.
**Depth additions.** Johansen procedure derived rather than recipe-style; explicit treatment of identification problem in SVARs.
**Depends on.** Ch. 15, 16.

---

## Part VI — Specialized and Multivariate Models

### Chapter 18. Survival Analysis
**Scope.** Time-to-event modeling under right censoring (and an honest paragraph on left-truncation and competing risks).
**Library.** `statsmodels.duration.hazard_regression.PHReg`; `statsmodels.duration.survfunc`; `lifelines` for parametric AFT and time-varying covariates (cross-referenced, not a dependency).
**Theory — full proofs.** Hazard, cumulative hazard, survival function relationships; Kaplan–Meier as a nonparametric MLE (Greenwood's variance derivation in full); log-rank test as a score test from a Cox model; Cox partial likelihood derivation and its consistency under the proportional hazards assumption; Schoenfeld residuals and their use in PH-assumption diagnostics.
**Theory — sketched.** Counting-process martingale formulation (statement only, with reference to Andersen–Gill); accelerated failure time models; competing risks via cause-specific hazards and the Fine–Gray subdistribution hazard.
**Depth additions.** Greenwood derivation; partial likelihood derived; honest treatment of why partial likelihood works (profile likelihood interpretation).
**Depends on.** Ch. 12.

### Chapter 19. Multivariate Methods and Nonparametric Smoothing
**Scope.** Dimension reduction and multivariate inference, then nonparametric regression with full asymptotic theory. The two halves are linked by the "what does this estimator converge to and at what rate" question.
**Library.** `statsmodels.multivariate.pca.PCA`, `factor.Factor`, `cancorr.CanCorr`, `manova.MANOVA`; `statsmodels.nonparametric` (`lowess`, `KernelReg`, `KDEMultivariate`, `KDEUnivariate`).
**Theory — full proofs.** PCA via SVD; factor model identifiability (rotation indeterminacy); Wilks' lambda and its $F$ approximation; CCA as generalized eigenvalue problem; kernel density estimator AMISE and optimal bandwidth derivation; bias-variance trade-off in kernel regression; local linear regression and its design-adaptivity (boundary bias result).
**Theory — sketched.** Pillai, Hotelling, Roy MANOVA statistics; cross-validation for bandwidth selection; sieve methods (statement and pointer).
**Depth additions.** Full AMISE derivation and bandwidth-selection theory; rate $n^{-2/(d+4)}$ explicitly derived and its curse-of-dimensionality consequences; local linear regression bias result (Fan's theorem).
**Depends on.** Ch. 10, 11.

### Chapter 20. Instrumental Variables and Weak Identification
**Scope.** Endogeneity, instruments, and the inferential pathologies that arise when instruments are weak. Full chapter rather than a section because the theory is rich and the practical importance is high.
**Library.** `linearmodels.iv` (`IV2SLS`, `IVGMM`, `IVLIML`); GMM via `statsmodels.sandbox.regression.gmm`; weak-IV diagnostics from `linearmodels`.
**Theory — full proofs.** IV identification (relevance, exclusion, monotonicity); 2SLS as a method-of-moments estimator and its consistency; LIML derivation; GMM consistency and asymptotic normality; efficient GMM and the optimal weighting matrix; over-identification ($J$) test; LATE theorem (Imbens–Angrist) under monotonicity.
**Theory — sketched but with code demonstrations.** Weak-instrument asymptotics: bias of 2SLS toward OLS, non-normal limiting distribution, Stock–Yogo critical values, Anderson–Rubin and conditional likelihood-ratio tests as weak-IV-robust inference.
**Depth additions.** The previous plan rolled IV/RDD/DiD/synthetic control into one chapter. IV alone now gets a chapter, with weak-instrument theory treated rather than mentioned. LATE proven, not asserted.
**Depends on.** Ch. 10, 11.

---

## Part VII — Causal Inference

### Chapter 21. Potential Outcomes, Identification, and Estimation Under Unconfoundedness
**Scope.** The potential-outcomes framework and the classical estimator catalog. Identification arguments are foregrounded; estimation is downstream of identification.
**Library.** `statsmodels` for outcome and propensity models; from-scratch implementations of IPW, regression adjustment, AIPW, matching with Mahalanobis and propensity-score distances. `DoWhy` cross-referenced for SCM-style identification.
**Theory — full proofs.** Potential outcomes, SUTVA, ignorability, positivity; propensity score theorem (Rosenbaum–Rubin) — full proof; IPW identification and asymptotic variance; regression adjustment identification; AIPW double-robustness theorem (estimator is consistent if either outcome model or propensity model is correct) — full proof; balance diagnostics under matching.
**Theory — sketched.** Pearl's do-calculus and back-door/front-door criteria (statement and worked example); covariate-adjustment validity (Shpitser–VanderWeele).
**Depth additions over previous plan.** Full Rosenbaum–Rubin proof; full AIPW double-robustness proof; explicit treatment of when AIPW achieves the semiparametric efficiency bound (Hahn 1998 — statement).
**Depends on.** Ch. 11, 12, 13.

### Chapter 22. Quasi-Experimental Designs: RDD, DiD, and Synthetic Control
**Scope.** Three identification strategies that exploit design rather than ignorability.
**Library.** `linearmodels` for two-way fixed effects; from-scratch local-polynomial RDD; `pysyncon` or hand-rolled synthetic control; Callaway–Sant'Anna estimator via `differences` package (cross-reference) or hand-rolled.
**Theory — full proofs.** Sharp RDD identification (continuity at the cutoff) and local polynomial estimation with optimal bandwidth (Imbens–Kalyanaraman); fuzzy RDD as IV; DiD identification under parallel trends; the failure of two-way fixed effects under staggered adoption with heterogeneous treatment effects (Goodman-Bacon decomposition statement and intuition); synthetic control as a constrained convex optimization problem.
**Theory — sketched.** Callaway–Sant'Anna and Sun–Abraham estimators as solutions to the staggered-adoption problem; permutation inference for synthetic control.
**Depth additions.** Sharp RDD with full IK bandwidth; Goodman-Bacon decomposition discussed (this is where DiD practice has been quietly broken for a decade — the chapter cannot ignore it); synthetic control as constrained optimization linked back to Ch. 4.
**Depends on.** Ch. 11, 20, 21.

### Chapter 23. Heterogeneous Treatment Effects and Double Machine Learning
**Scope.** Estimating CATE and related heterogeneous-effect functionals; the Neyman-orthogonality / double-ML framework.
**Library.** `EconML` (DML, causal forests, meta-learners — T-, S-, X-, R-, DR-learners); cross-reference `causalml`. Integration with statsmodels nuisance estimators.
**Theory — full proofs.** Neyman orthogonality definition and why it removes first-order bias from nuisance estimation; cross-fitting and its role in achieving $\sqrt{n}$ rates with slow nuisance estimators; DML for partially linear model and for the average treatment effect; influence-function representation of efficient estimators; semiparametric efficiency bound (statement, with reference back to Ch. 21).
**Theory — sketched with code.** Causal forests (Wager–Athey statement and intuition); meta-learner taxonomy and when each is appropriate.
**Depth additions.** Full Neyman-orthogonality treatment with derivation of why naive plug-in fails; explicit linkage to influence-function theory introduced in Ch. 12. This was a single section in the previous plan; it is now the closing chapter and gets the depth the topic demands.
**Depends on.** Ch. 12, 20, 21.

---

## What Was Cut and Why

To recover space for depth, the following were removed from chapter status. Most live in the **Prerequisites Appendix**; a few are mentioned in bibliographic notes only.

- **Floating point and conditioning** — appendix.
- **Dense linear algebra decompositions** — appendix; pointers to Trefethen–Bau and Golub–Van Loan.
- **Sparse and large-scale linear algebra** — used as black box; appendix sketch only.
- **Numerical integration and ODEs** — appendix; out of scope as a focus area (the book is statistics, not numerical analysis).
- **Interpolation and FFT/spectral methods** — appendix sketch; Lomb–Scargle and spectral density estimation referenced in Ch. 15.
- **Linear and mixed-integer programming** — bibliographic notes in Ch. 3; not a focus.
- **Probability distributions in SciPy** (the API tour) — appendix.
- **Multivariate distributions in SciPy** (the API tour) — appendix; KDE absorbed into Ch. 19.
- **Pseudorandom and quasi-random sampling** — absorbed into Ch. 7.
- **Parametric hypothesis testing** (basic $t$/$F$/$\chi^2$) — appendix; the framework lives in Ch. 10.
- **Multiple testing** — moved to a section in Ch. 10; full FDR treatment compressed.
- **Nonparametric tests** — appendix.
- **Correlation and dependence** — appendix.
- **Power and sample size** — appendix.
- **Structural equation models** — removed entirely. Justification: `semopy` is the only viable Python option and its ecosystem is shallow; the topic's center of gravity remains in R/Mplus and treating it shallowly would mislead. A two-page bibliographic note in Ch. 19 points readers to `semopy` and to Bollen.

---

## Depth Signals (Cross-Cutting)

What "deeper" means concretely in this revision:

- **Asymptotic theory** is derived where it's load-bearing (MLE in Ch. 10; sandwich variance in Ch. 11; Dickey–Fuller in Ch. 15; AMISE in Ch. 19; Neyman orthogonality in Ch. 23) rather than cited.
- **Influence-function theory** appears as a recurring thread: introduced in Ch. 11 (regression diagnostics), formalized in Ch. 12 (M-estimation), reused in Ch. 18 (survival), Ch. 21 (AIPW), and Ch. 23 (DML).
- **Identification** is treated as a separate concern from estimation, with worked counterexamples, in Ch. 13 (latent utility), Ch. 17 (SVARs), Ch. 20 (IV), and Ch. 21–23 (causal).
- **Failure modes** get explicit treatment: bootstrap failure (Ch. 9), Nelder–Mead McKinnon counterexample (Ch. 5), constraint qualification failures (Ch. 3), weak instruments (Ch. 20), TWFE under staggered adoption (Ch. 22).
- **Connections across chapters** are stated explicitly. M-estimation in Ch. 12 connects to AIPW in Ch. 21 connects to DML in Ch. 23. The reader sees the spine.

---

## Topics Deliberately Excluded

Unchanged from previous plan: Bayesian computation beyond Ch. 8 MCMC; deep learning and autodiff frameworks; reinforcement learning; interpretability; LLMs/NLP; spatial statistics; GARCH (pointer in Ch. 15 bibliographic notes); SEM (pointer in Ch. 19 bibliographic notes).
