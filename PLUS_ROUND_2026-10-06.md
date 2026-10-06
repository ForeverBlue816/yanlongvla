# October 6 — Plus selects variants; LIBERO measures final deployment

This protocol supersedes previous scheduling requirements. H4 and H4b stop. The task9 gallery and completed FP task8 rerun remain archived evidence, without further labelling. Historical H1/H3b task8/9 observations are mechanism probes only. No new standalone task8/9 evaluation or selection is permitted.

All method and intervention comparisons use the original pre-registered **500 camera + 500 initial-state LIBERO-Plus instances**, paired, seed7. Preserve the subset, including its existing base-task composition. Report combined and per-dimension paired differences against FP and the original2-bit backbone + uniform-M2 expert. Reuse completed FP, original2-bit and expert-only uniform-M2 rows. Partial runs have no published success-rate row.

## S1 and S2

On the original256 training calibration observations, estimate one scalar per backbone layer and token type (image/language), jointly pooling K and V: `alpha = <KV_FP, deltaKV> / ||KV_FP||^2`. Report separate K/V diagnostics. No discrete state tokens exist in this model. Preserve native cache dtypes and padding.

The additive energy partition must be orthogonal: first remove the pooled token-vector error mean; then project centered errors onto centered FP KV; the remainder is orthogonal residual. Report the centered projection slope separately from the uncentered intervention alpha, so mean and gain energies are not double counted. Report all per-layer/type energy fractions.

S2(a) multiplies FP cached KV by `1+alpha`. S2(b) divides the original2-bit cached KV by `1+alpha`. Both receive Plus1000. Nonpositive or nonfinite gains fail validation without clipping. Require exact graph/eager actions on256 training observations and exact unity-gain recovery of the underlying model on256 observations. Both models retain their original expert; S2(b) uses uniformM2.

Report KV error and mean prefix attention entropy for FP, original2-bit and S2(b) on the fixed200 saved Plus observations. Entropy is the expert's attention conditioned on valid prefix keys, averaged over10 Euler steps,18 layers, heads and10 queries, in nats. Also report probability mass assigned to prefix keys. Scaling occurs after backbone prefill, so measuring only backbone prefill attention would not measure the correction. Instrumentation must leave all200 action outputs identical.

## Method candidates and calibration coverage

H5a, H5b, H5c4 and H5c16 retain their accepted fitted checkpoints and receive Plus1000, without the former H4 gate. H5d is not included in this revised evaluation round. Put S2(b) and H5 rows in one dual-reference table. TV remains an appendix diagnostic, not a selection metric.

Repair calibration generation without changing the original state/image consistency tolerances. Record the root cause and validated repair. Main coverage budget remains256 observations from the original60 training episodes:128 original,64 independently perturbed camera and64 independently perturbed initial state. Perturbation parameters and evaluation seed remain7. Never source calibration from Plus test instances. Fit both backbone+expert and backbone-only variants. Report image/language KV error and bias fractions, plus exact FP32 trace(J_D,9)/70 on the128 perturbed observations and matched clean observations. Evaluate both variants on Plus1000 and LIBERO medium1000.

## Final protocol and promotion

The user explicitly retained all40 tasks for the final deployable rows and coverage experiment: **medium1000/full2000**, with excluding-LIBERO10-tasks8/9 columns **950/1900** alongside. Tasks8/9 are present only within that unchanged benchmark. Neither their standalone scores nor LIBERO medium selects variants.

Before new results, freeze seven deployable candidates: H5a/b/c4/c16, S2(b), coverage backbone-only, coverage both. Once all seven have complete validated Plus1000 results, choose the highest combined success count (lexicographic model-name tie break). Promote once only if the improvement over original2-bit is at least30/1000 and the lower bound of the paired95% CI is above0. S2(a), a causal FP intervention, is not a deployment candidate. Intervals use the existing2000-draw suite-stratified base-task/instance paired bootstrap; they are pointwise and do not adjust for selecting the best variant.

The promoted configuration receives LIBERO full2000 and an **expanded Plus1000camera+1000initial-state** evaluation. Per explicit user clarification, retain and reuse the original500+500 instances; independently select500 additional instances per dimension without replacement from the existing manifest, fixed before outcomes. Report expanded2000 and added1000 separately. The expanded result overlaps the selection data and must not be described as independent confirmation. Reuse accepted original rows with source hashes; do not rerun them.

Server-side Slurm dispatch caps all requested+running GPUs at8, resumes interrupted valid outputs, and publishes complete validated milestones to the report-only GitHub repository. Older dispatchers remain stopped. New implementation and checkpoints stay private.
