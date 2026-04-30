# Style Deviations Log

> Any approved departures from `STEERING.md` are recorded here with rationale.
> Unapproved deviations are bugs.

| Chapter | Section | Deviation | Rationale | Approved By | Date |
|---------|---------|-----------|-----------|-------------|------|
| All | §3 Sec 10, §8 | Changed from "at least six exercises" to "exactly four exercises, each with full worked solution in %% delimited QMD" | Four deep exercises with solutions teach more than six shallow ones without. Solutions are the pedagogical instrument. | User | 2026-04-30 |
| Ch. 1 | §3 | Extra section "The Roadmap" added beyond the 11-section template; sections reorganized around pipeline narrative | Ch. 1 is a survey/roadmap chapter (CHAPTERS.md: "Theory — light"). The roadmap table is the chapter's unique contribution and cannot be folded into another section without losing its function. | Lead | 2026-04-30 |
| Ch. 8 | §2.7 | MCMC iterations indexed by $t$ instead of $k$ | $t$ is universal MCMC convention (Gelman, Robert & Casella, Neal). §2.7's $k$ rule is for scipy.optimize callbacks. Using $k$ for chain indices would confuse readers familiar with the MCMC literature. | Editor | 2026-04-30 |

---

*A deviation without an entry here is not approved.*
