# Clarifications recorded before computing residuals

- C1 is an additive task calibration model. It cannot reorder policies within a task for a positive simulation coefficient. Rankings use unclipped predictions; scores are clipped only for MAE.
- Add B2_ridge with the same 0.10 penalty as C1 to distinguish task-feature effects from shrinkage. The specified C1 penalizes the simulation coefficient as well as attributes. The unpenalized B2 remains the baseline requested by the spec.
- The source MMRV uses disagreements in strict greater-than comparisons. This includes predicted ties in a directional way; it is not a strict-inversions-only metric. Tests reproduce that exact convention.
- Taxonomy has articulation and precision perfectly collinear among the five Google tasks. Separate effects cannot be identified there. The remaining columns are descriptive; no extra fitted attributes are added.
- All slopes and coefficients are unconstrained. A negative slope, if produced, is reported rather than silently corrected.
- No trial counts are assigned to SIMPLER cells until correspondence to the rounded constants is established. Protocol-level counts in the paper do not justify fabricated Bernoulli episodes.
