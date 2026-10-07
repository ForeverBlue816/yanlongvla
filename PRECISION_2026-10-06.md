# October6 evening — projection protection, deep precision and decision matching

This round supersedes the prior dispatcher. Preserve all completed historical results and runtime contracts. All new variants first use the original paired seed7 LIBERO-Plus500camera+500initial-state subset. No standalone LIBERO10tasks8/9 evaluations, H4/H4b labelling or new weighting schemes.

## Candidates

| Candidate | Definition | Training budget |
|---|---|---|
| C1 | Restore all36 Gemma language k_proj/v_proj matrices,18 layers×2×256×2048, to exact originalBF16 in the native basis. All other stored tensors match original2bit. | Reuse original native weights; no refitting needed |
| C2 | All28 eligible Gemma Linear layers in blocks14–17 use3bit; blocks0–13 and vision remain2bit. Expert stays uniformM2. | Reuse accepted M3 layer fits from the same original calibration/recipe |
| C3 | Original2bit backbone codebook-only decision matching, all10 velocity steps and16 latents. |2304 optimizer updates=9passes over256training observations; starts from original2bit |
| C4 | Original H5b fine-tune on C2. |768updates,8latents,steps0–5 |
| C5 backbone | Mixed-calibration backbone refit, original-calibration M2 expert. |Original quantization recipe,256mixed observations |
| C5 both | Mixed-calibration backbone and M2 expert refits. |Same mixed256, same bits |

The actual C1 export has **3.000463596439 whole-model bpw**, **+0.077873695432**, adding **32,643,036 tensor bytes** over original2bit2.922589901006. C2 has **3.054192145406 whole-model bpw**, **+0.131602244399**, adding **55,164,928 tensor bytes**. The denominator remains3,353,433,872 original active elements. Count all stored tensors, including codebooks, codes, scales, signs, masks, embedding and fixed tables; checkpoint header bytes are reported separately in checkpoint metadata. BF16 decode caches are runtime memory, not additional serialized compression payload. All changed and retained tensors undergo exact export/reload checks.

C3/C4 preserve originalH5b Adam learning rate3e−4 and gradient clipping1.0, fixed assignments, scales, norms and expert. FP32 optimizer masters roundtrip throughFP16 codebooks on every forward. The frozen uniformM2 expert supplies both sides of the velocity-direction comparison, with FP versus candidate prefixKV and shared latent/path/time. The comparison remains on native10×32 velocities. C3 expands the final fixed budget; the former23hour truncation is removed. Preemption/time limits resume from saved optimizer state, never from a selected best checkpoint.

## Coverage repair and conditional follow-up

Earlier generation failed at observation159 after74 accepted samples. Preserve those artifacts. The new pilot tests executing the originalFP64 actions rather than FP32 conversion outputs, retaining the previously audited source seed0/demo-reset alignment. This matches the action execution in the [pinned OpenVLA training-data regeneration code](https://github.com/openvla/openvla/blob/0ebc5e333ee7a916813bf50db240f1af7ba1d295/experiments/robot/libero/regenerate_libero_dataset.py). New camera/state perturbations and all evaluations retain seed7. No tolerances are relaxed: clean state max error≤0.005 and both image MAEs≤15/255, for all256observations before fitting. Regenerate into a separate artifact directory; keep prior failures intact.

The mixed set remains128standard+64camera+64initial-state from the original60training episodes, independently perturbed and never drawn from Plus test instances. Both coverage variants receive Plus first and the separately authorized LIBEROmedium1000/950. If either achieves positive paired gain over original2bit with95%CI lower bound>0, run originalH5b budget on that variant using the same mixed256. This significant-positive-gain trigger was explicitly approved by the user; it is distinct from promotion.

## Diagnostics and curves

For every candidate, report original training256 and fixed savedPlus200 per-layer image/language K/V relative error energy, S1 scalar alpha and orthogonal mean/centered-proportional/residual energy fractions. Explicitly retain deep-layer language-V diagnostics. Report expert attention conditional on valid prefix keys and prefix attention mass on the same200Plus observations; instrumentation must preserve all200actions exactly. No Plus observations enter fitting or checkpoint choice.

For every training update, save cosine loss and current-observation pre-update image/languageKV error. At update0, every64updates and the final update, measure the same16training observations (indices0,16,…,240) with latent0. Plot all raw update losses and the fixed monitor, without smoothing or omitting observations. Training curves are descriptive single-seed evidence, not a rollout-quality proxy. Export raw values and reproducible Python PDF/SVG/PNG plots, with alignment/text/collision checks and separate visual review.

## Selection and deployment

Wait for C1–C4 and both coverage rows, plus any triggered coverageH5b rows. Choose the largest combined Plus success count, lexicographic model-name tie break. Promote only if improvement over original2bit is≥30/1000 and the paired95%CI lower bound is>0. Preserve the original suite-stratified task/instance paired bootstrap and disclose that its intervals are pointwise, not multiplicity-adjusted. Freeze one winner; no targeted task8/9 selection.

The winner receives LIBERO full2000 with the excluding8/9 column1900 alongside, and the existing frozen expandedPlus1000camera+1000initial-state set. Reuse the original500+500 instances; separately report the added1000 and expanded2000. The expanded score overlaps selection and is not independent confirmation. Coverage retains its explicit LIBEROmedium1000/950 exception.

C1/C2 composition proceeds in parallel, followed by their diagnostics/Plus evaluation. C3 runs concurrently; C4 starts after C2 is exported. Coverage continues when its data guard passes. All requested+running GPUs count toward the eight-GPU cap. Server-side Slurm dispatch and milestone-only report publication continue independently of the user's computer.

## Subsequent D3 amendment

The [complete October7 addendum](ADDENDUM_2026-10-07.md) now incorporates the revised D3: three offline FP-prefix interventions over all40 tasks and three pairedPlus500+500 runs, with TV, E_R and D_M diagnostics. D1, three separate A8 feasibility rows and the promotion threshold are retained. Numerical targets are predictions. C1–C5 continue, while final deployment now requires the A8 feasibility decision and validated winning-backbone + uniformM2-expert + A8 composition before full/expanded evaluation. The previous direct-to-final dispatch is superseded.


## October7 execution update — C3 latent parallelism

The user requested multiple GPUs for C3 training. A four-GPU continuation now splits the same observation's 16 latents into four shards of four, sums gradients with the original global denominator, clips once and takes one Adam update. Rank zero owns optimizer state and broadcasts updated codebooks. Globally unused parameters keep grad=None, preserving Adam skip semantics. The fixed2304updates,256training observations,10velocity steps, seed namespace and final-iterate rule are unchanged.

The original one-L40S worker continues while the four-RTX6000Ada reservation queues and runs its startup checks. Before cancelling that specific worker, the replacement must pass exact action checks for FP and the original2bit model on256held-out observations collectively across ranks, the initial differentiable-forward256check, and serial-versus-parallel gradient/Adam checks on two fixed training observations. The latter tests numerical agreement within preregistered execution tolerances, not bitwise equality of complete training trajectories. CPU mathematical, handoff and scheduler tests passed; GPU validation and measured speedup remain pending allocation.

After all guards pass, acquire the training lock and restore the latest atomic checkpoint, including all Adam moments and completed updates. Preserve the original frozen fit manifest and sources; record the new execution code and hardware in separate per-attempt proofs. Restart/preemption uses checkpoint continuation. If the original training finishes first, the replacement exits without refitting. Pending and running reservations together stay within eight GPUs; no C4 evaluation is interrupted. Background dispatch and report publication continue independently of the user's computer.
