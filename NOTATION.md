# Notation Ledger

> Canonical symbol definitions for the book. Every symbol used in any chapter must
> appear here or in that chapter's local mini-table. No symbol may carry two meanings.
>
> Governed by STEERING.md §2. This is the census; §2 is the law.

## Core (STEERING.md §2)

| Symbol | Meaning | First Introduced |
|--------|---------|-----------------|
| $n$ | Number of observations | Ch. 1 |
| $p$ | Number of features / parameters | Ch. 1 |
| $\mathbf{X}$ | Design matrix ($n \times p$) | Ch. 1 |
| $\mathbf{y}$ | Response vector | Ch. 1 |
| $\boldsymbol{\beta}$ | Coefficient vector | Ch. 1 |
| $\hat{\boldsymbol{\beta}}$ | Estimated coefficients | Ch. 1 |
| $f(\mathbf{x})$ | Objective function | Ch. 2 |
| $\nabla f(\mathbf{x})$, $\mathbf{g}_k$ | Gradient | Ch. 2 |
| $\nabla^2 f(\mathbf{x}_k)$ | Hessian (exact) | Ch. 2 |
| $\mathbf{x}_0$ | Starting point (optimization) | Ch. 2 |
| $\mathbf{x}^*$ | Minimizer | Ch. 2 |

## Optimization (Ch. 2)

| Symbol | Meaning | First Introduced |
|--------|---------|-----------------|
| $\mathbf{B}_k$ | Quasi-Newton Hessian approximation at iteration $k$ | Ch. 2 |
| $\mathbf{H}_k$ | Inverse Hessian approximation $\mathbf{B}_k^{-1}$ (BFGS) | Ch. 2 |
| $\mathbf{p}_k$ | Search direction at iteration $k$ | Ch. 2 |
| $\alpha_k$ | Step size at iteration $k$ | Ch. 2 |
| $\Delta_k$ | Trust-region radius at iteration $k$ | Ch. 2 |
| $\mathbf{s}_k$ | Step: $\mathbf{x}_{k+1} - \mathbf{x}_k$ | Ch. 2 |
| $\mathbf{y}_k$ | Gradient change: $\nabla f_{k+1} - \nabla f_k$ | Ch. 2 |
| $\gamma_k$ | $1/(\mathbf{y}_k^\top \mathbf{s}_k)$ — BFGS update scalar | Ch. 2 |
| $\rho_k$ | Trust-region actual/predicted reduction ratio | Ch. 2 |
| $m_k(\mathbf{p})$ | Quadratic model at iteration $k$ | Ch. 2 |
| $\tau_k$ | Cauchy point scaling factor | Ch. 2 |
| $L$ | Lipschitz constant of $\nabla f$ | Ch. 2 |
| $\kappa$ | Condition number of $\nabla^2 f(\mathbf{x}^*)$ | Ch. 2 |
| $c_1, c_2$ | Wolfe condition parameters | Ch. 2 |
| $\theta_k$ | Angle between $\mathbf{p}_k$ and $-\nabla f_k$ | Ch. 2 |

## ML Pipeline / Cross-Validation (Ch. 1)

| Symbol | Meaning | First Introduced |
|--------|---------|-----------------|
| $\hat{f}$ | Fitted regression estimator | Ch. 1 |
| $\mathcal{D}$ | Training dataset | Ch. 1 |
| $K$ | Number of CV folds | Ch. 1 |
| $D_k$ | Indices in fold $k$ | Ch. 1 |
| $L(y, \hat{y})$ | Loss function | Ch. 1 |
| $\hat{f}^{(-k)}$ | Estimator trained without fold $k$ | Ch. 1 |
| $\mathcal{A}$ | Set of learning algorithms (AutoML) | Ch. 1 |
| $\boldsymbol{\Lambda}_r$ | Hyperparameter space for algorithm $A_r$ | Ch. 1 |
| $\mathbf{P}$ | Hat matrix $\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ | Ch. 1 |
| $P_{ii}$ | Leverage: diagonal of hat matrix | Ch. 1 |

## Probability (STEERING.md §2.6)

| Symbol | Meaning | First Introduced |
|--------|---------|-----------------|
| $\mathbb{E}[\cdot]$ | Expectation | Ch. 1 |
| $\text{Var}(\cdot)$ | Variance | Ch. 1 |
| $\sigma^2$ | Irreducible noise variance | Ch. 1 |

---

*Update this ledger during the Integration pass of every chapter.*
