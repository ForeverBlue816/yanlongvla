# October 8 round — C5 replacement (Plus-train calibration), RAQ-FT, A8 feasibility

Priority order from the user's 2026-10-08 instruction. All comparisons use the original paired seed7 LIBERO-Plus 500 camera + 500 initial-state subset; the gate is unchanged (≥ +3 pp over 704/1000 with paired 95% CI lower bound > 0). FP has not seen Plus-train data.

## P0 — C3

Job 188637 (one L40S) continues untouched to its fixed 2304 updates; its Plus evaluation and curves are dispatched by the existing precision controller. The queued four-GPU continuation (189890) cannot start before C3 finishes and exits without refitting once `complete.json` exists; it is cancelled only after that point.

## P1 — Calibration on the official LIBERO-Plus training set

The demo-replay perturbation pipeline (C5) and its diagnostic are cancelled. Data: `lerobot/libero_plus` (LeRobot v3, 14,347 episodes, 40 tasks, AV1 videos), pinned revision `f3f49f426d75030177b18778374005bc12ccd588`. Frames are decoded from the videos at the sampled timestamps; the state comes from the data parquet.

**Leakage check.** The release carries task text only (no per-episode perturbation labels). The first observation of every Plus test instance (selection 500+500 and the 1000+1000 expansion, 2000 instances; seed7, init state0; both the raw reset frame and the frame after ten dummy steps) is rendered and compared with the first frame and initial state of every same-task training episode:

- same instance: ≥ 90% near-identical pixels (|Δ| ≤ 6/255 on all channels) and MAE ≤ 6/255;
- same camera parameters: near-identical fraction above a threshold calibrated on pairs of *different* camera test instances of the same task (99th percentile + 0.05, floor 0.60);
- same robot initial state: end-effector pose L∞ ≤ 2e-3 with ≥ 50% near-identical scene pixels.

Task-level overlap is by benchmark design (Plus perturbs the 40 training tasks). Flagged episodes are excluded before sampling; all distributions are logged in `results/plustrain_round.json`.

**Banks** (seed 7, fixed before outcomes): (a) LIBERO 256 (existing); (b) 64 Plus-train episodes × 4 frames, stratified over the 40 tasks; (c) 128 LIBERO (2 per original calibration episode + 8) + 128 Plus-train (32 of the 64 episodes × 4). **Validation**: 75 Plus-train episodes × 4 frames = 300 observations, episode-disjoint from (b)/(c); all fine-tune hyperparameters and early stopping are chosen here. The Plus test subset is never used for selection.

**Runs.** Original-metric 2-bit backbone refit (input moments + prefix action-Fisher, RVQ planes 2, original uniform-M2 expert) on (b) and (c); H5b-style codebook fine-tune (steps 0–5, 8 latents, Adam 3e-4, clip 1) on (b) and (c) with the best-validation iterate (evaluation every 64 updates, patience 4, ≤ 3 epochs). Each row: Plus 500+500 paired vs FP and original 2-bit, LIBERO medium 1000, image/language KV error on training256, own bank and savedPlus200, and the D1 propagated-error fraction.

## P2 — RAQ-FT (readout-consistency fine-tune)

L = Σ_l w_l · mean_{t,h} ‖W_{O,h}(o^q_{l,t,h} − o^fp_{l,t,h})‖² + β·L_H5b, β = 0.1, w_l = 1/18. The queries, suffix K/V and masks are the FP expert's teacher-forced values along its own trajectory (all 10 steps × 8 heads × 10 action tokens); o^fp reads FP prefix KV, o^q reads the student's prefix KV. Trainable: backbone codebooks, row scales and backbone norms (FP32 masters, stored-dtype roundtrip); codes, signs, masks, expert, embedding and tables frozen. 768 updates (3 epochs × 256, 8 latents), ≤ 1 GPU-day, final iterate. Rows: RAQ-FT on original 2-bit (bank a); control with the same loss/budget/data on C2 (3-bit deep layers); coverage row on bank (c); the existing H5b row (732/1000) is the loss-function control. Submitted after C3 reports.

## P3 — A8 feasibility (excluded from method selection)

A8 = per-token symmetric INT8 fake quantization of every quantized-Linear input (288 backbone + 126 expert sites) immediately after the folded Hadamard; softmax, QKᵀ and norms stay BF16. The FP row folds the same deterministic RHT into its 414 Linear sites (fold-only control must reproduce FP actions up to BF16 rounding). Rows: (i) FP + A8; (ii) original 2-bit backbone + 2-bit expert + A8; (iii) row (ii) + per-token INT8 KV cache. Reported: held-out action MSE (256 held-out observations, noise seed 0), KV error, Plus 500+500 paired vs the non-A8 model. Decision only: whether the final deployable row is W2A8.

## P4 — Reference and diagnostics

3-bit backbone + uniform-M2 expert on the Plus subset (one run); D1 local vs propagated KV error (per layer, K/V, token type; raw and reader-weighted with Σ_q from FP expert queries and W_Oᵀ W_O for V); D3 reader-subspace vs orthogonal-complement injection (offline TV over the 1600-observation bank, then three Plus runs) when GPUs are free.

## Execution

Fitting, fine-tuning and offline diagnostics run on the best available cards (RTX Pro 6000 / RTX 6000 Ada under the rose QoS, L40S otherwise) with the device recorded in every artifact; paired policy rows (Plus, LIBERO medium) keep the validated L40S runtime of all existing rows. Dispatch is server-side (`plustrain_controller.py`, ledger `runs/plustrain_20261008/jobs.json`) within the eight-GPU cap shared with the precision controller; complete validated milestones publish to the report-only GitHub repository.
