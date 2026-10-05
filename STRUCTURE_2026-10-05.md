# Structure, not magnitude — evening experiment protocol, 2026-10-05

User addendum and both accepted clarifications are registered in `runs/structure_20261005/preregistration.json`. Seed7, at most8 GPUs requested/running. H3b has scheduling priority; all H5 evaluation requires completed H3b and H4 review. Existing running H3/H4 jobs and frozen inference contracts are preserved. Old dispatcher184775 was stopped; pending H5 reservations184783/184801/184802 were withdrawn for H3b and fitting will resume under the new dispatcher.

H3b:10 conditions ×2 tasks ×200 paired episodes, IDs50–249, plus500 mixed episodes for1× =4500. Conditions: actual per-observation deltaKV at0.25/0.5/1/2×; per-layer/type mean-only and residual-only; per-token random signs shared acrossK/V and all layers; token-direction permutation retaining receiving-position jointK/V norm; layer permutation; −1×. Report per-layer/type bias-energy fractions and mean error relative to KV norm. PersistentBF16 donors;256 exact zero/FP and unit/H1c guards precede rollouts.

H4b: separate stalled-holding and cup-in-door-open50-step rules; exclude last50 steps; report0.5/1/2cm sensitivity and undetected counts. Search starts on saved5-step observation grid; both images are original saved frames, state is derived from the corresponding saved simulator state by forward kinematics. No action replay or re-rendering. Include successfulFP andM2 task9 episodes as phase controls. Fifteen historical “other” cases are candidates; only8 have verified final containment, so do not relabel all15 as cup-in.

Plus-J:200Plus observations,100 per dimension, and200 matched baseline observations. ExactFP32 trace(J_D,9)/70, paired distributions, fixed manifest order and matching original task/state0. Disclose repeated baseline observations and environment differences. Expert-only uniformM2 additionally runs fixed500+500Plus.

H5d: calibration-only per-layer/image-language mean deltaK/V cancellation in cachedKV. Expert remains uniformM2. EvaluateKVerrors, diagnosticTV,200each tasks8/9,medium1000,Plus1000 afterH3b+H4 gates. H5a/b/c fitting remains planned. H2TV table stays in the appendix as diagnostic evidence only.

Registration is not a measured result. Branch status is reported only after implementation, validation and actual scheduler evidence.
