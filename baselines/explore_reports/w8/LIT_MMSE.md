# W8: why a weight shrink helps (literature) and is the shrink MMSE-expected (CPU)

Note: /home/d250009/yanlongvla-clean/baselines/RELATED_2026-11-06.md does not exist. I could not check the log for duplicates by that file. I grepped baselines/*.md: arXiv ids 2609.25376, 2609.22895, 2605.28803, 2605.24011, 2510.13626 appear there. Check for overlap by hand.
Web search returned summaries only. I did not read the full papers. Treat claims as "search summary".

## Part 2: c versus -rel^2 (CPU, 432 tensors each)
Setup: E = Q - W, rel = |E|/|W|, c = <E,W>/<W,W>.
Exact identity (no assumption): <E,W> = <E,Q> - |E|^2, so c = -rel^2 + <E,Q>/<W,W>.
MMSE (Lloyd-Max) condition: <E,Q> = 0 (E orthogonal to Q), which gives c = -rel^2.
The identity that holds here is the exact one: c + rel^2 = <E,Q>/<W,W>. It is > 0 in all tensors, so <E,Q> > 0. The MMSE orthogonality fails.
(Check: E[(W-Q)Q]=0 gives <Q-W,Q>=0, not <Q-W,W>=0. The MMSE value is -rel^2 only for <E,W>.)

| file | median c | median -rel^2 | c/(-rel^2) median | min | max | p10-p90 |
|---|---|---|---|---|---|---|
| quip_shrink.json | -0.0516 | -0.1588 | 0.321 | 0.240 | 0.461 | 0.274-0.364 |
| qtip_w2_libero10_shrink.json | -0.0580 | -0.1140 | 0.509 | 0.264 | 0.843 | 0.387-0.602 |
| quip_w2_libero10_shrink.json | -0.0513 | -0.1572 | 0.329 | 0.241 | 0.461 | 0.278-0.365 |

- c < 0 in 100% of tensors; c/(-rel^2) is in (0,1) in 100%.
- Median rel: QuIP# 0.398, QTIP 0.338. So -rel^2 is 0.11 to 0.16, and the measured c is only 1/3 (QuIP#) to 1/2 (QTIP) of it.
- corr(c, -rel^2) across tensors: QuIP# 0.45, QTIP -0.13. So c is nearly constant (about -0.05) and not tied to rel^2. An MMSE quantizer would give c that follows -rel^2 tensor by tensor.
- Reading: the quantizers are not MMSE in the plain-weight (Frobenius) metric. A shrink exists (c<0, a norm-aware gain), but it is about 3x to 2x smaller than the plain-MMSE value. QuIP# and QTIP rotate W (incoherence processing), quantize in a rotated basis with a codebook fitted to Gaussian data, and use a Hessian-weighted objective (LDLQ / trellis). In that metric the orthogonality holds for the rotated and weighted error, not for <E,W> in the plain metric. A global per-tensor scale also enters (QuIP# fits scale to the Gaussian codebook). A plain-metric shrink of about -0.05 for all tensors is hence plausible as an MMSE-like side effect of the codebook, but it is not the plain MMSE amount.
- Caveat: norm_ratio in quip_shrink.json is 1.029 (|Q|>|W|) while c<0. So Q is not a plain "shrunk W". The shrink is a component along W (-5%), plus extra error energy orthogonal to W. Only 1.7% of error energy is along W (c^2/rel^2 ~ 0.0027/0.159).
- Untested: whether the along-W shrink is the Hessian-metric MMSE shrink. A cheap check: compute <E,W>_H / <W,W>_H with H from the saved Hessians.

## Part 1: literature (best first)
1. RETAIN. Yadav, Zhou, Wagenmaker, Pertsch, Levine, "Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging", ICLR 2026. Search found: https://arxiv.org/html/2512.08333v1. Claim: interpolating pi_pre and pi_ft weights (1-a)*pre + a*ft beats both on OOD task variants; the language backbone matters most. Relation: closest VLA precedent that moving fine-tuned weights toward another point helps; our 0.95 shrink is interpolation toward 0 (or toward a point near 0), not toward pre-trained.
2. WiSE-FT. Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022. https://arxiv.org/abs/2109.01903. Claim: linear interpolation between zero-shot and fine-tuned weights raises OOD accuracy by 2-10 pp and ID by about 1 pp at no cost. Relation: Bridge eval in SimplerEnv is a sim-to-real-style shift from the fine-tune data; weights pulled back from the fine-tuned optimum behave like WiSE-FT at a small alpha.
3. Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse", ICML 2023. https://arxiv.org/abs/2303.06296. Claim: attention entropy collapse is linked to large q/k spectral norms; limiting q/k norms fixes it. Relation: scaling all W by 0.95 scales q and k logits by 0.95^2 = 0.9025 (softer attention, an attention temperature of 1.108), if the q/k inputs come after a scale-invariant norm. This is a concrete mechanism for a shrink.
4. Model soups. Wortsman et al., ICML 2022. https://arxiv.org/abs/2203.05482. Claim: averaging fine-tuned weights improves accuracy and robustness; linked to flatness and confidence. Relation: averaging reduces the norm of the weights (the mean of non-aligned vectors is shorter than each), a shrink toward the pre-trained mean; same direction as ours.
5. Momentum-based Weight Interpolation (MCL). Stojanovski, Roth, Akata, NeurIPS 2022. https://arxiv.org/abs/2211.03186. Claim: slow interpolated weights of zero-shot models keep generality (+4% on CL benchmarks). Relation: second, weaker precedent for interpolation helping a fine-tuned model.
6. RobustVLA. https://arxiv.org/abs/2510.00037. Claim: pi0 flow-matching head trained with bounded output-noise (TRADES-like, a label-smoothing view) is more robust to perturbation. Relation: noise on action output acts like a regulariser for a flow-matching head; quantisation error might act in the same way (an indirect link).
7. Saliency-Aware Quantized Imitation Learning. Park et al., ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.html. Claim: QAT policies recover full-precision performance. Relation: "recover", not "beat"; shows a quantised policy at parity is normal, a gain of 25 points is not. Our effect is thus not a known generic effect of quantisation.
8. QVLA (2026 preprint). https://www.alphaxiv.org/overview/2602.03782v1. Claim (self-reported, via search summary): 4-bit OpenVLA at or above full precision. Relation: only precedent I found for quantised >= FP in a VLA; no mechanism given; I did not verify.
9. Self-Improving VLA Policies: Selected Diffusion Noise (SDN). https://arxiv.org/abs/2606.14084. Claim: action smoothing helps most on high-precision WidowX tasks in SimplerEnv. Relation: a shrink acting as smoothing or damping of the actions (smaller steps) would help the same precise tasks. Candidate mechanism, not shown.
10. LQER (ICML 2024). https://arxiv.org/abs/2402.02446. Claim: quantisation error is structured (low-rank in the activation-scaled space). Relation: background that E is not white noise; supports a systematic component of E. Weak.
11. Lloyd (1982) and Max (1960), classical quantiser theory. Claim: at the optimum the reconstruction is a centroid, so E[(W-Q)Q]=0 and the reconstruction has less energy than the source (variance shrink). Relation: the base of Part 2. Not a web-checked source here.

Gaps (searched, nothing found): post-hoc scaling of fine-tuned weights toward zero at test time; weight-noise regularisation in flow-matching action heads; documented action-magnitude overshoot or overconfidence of fine-tuned VLAs in SimplerEnv (only run-to-run noise of about +-6% on Bridge and per-platform action-scale differences). Claim in the log that the shrink is a "known" effect has no direct citation.

## Mechanisms to test cheaply
A. Attention temperature: scale only the q and k projections by s (logit factor s^2), and only v/o, only MLP, only the action-head output layer. Compare with a global 0.95. (Likely overlaps X14 "where the shrink acts"; check.)
B. Action damping: multiply the predicted action (or the flow velocity) by 0.95 at inference, with FP weights. If this gives the gain, the cause is action magnitude (SDN, overshoot); no weight change needed.
C. Interpolate FP toward a different point (pre-trained GR00T N1.7 base, WiSE-FT style) at alpha = 0.05 to 0.2. If it matches the 0.95 shrink, the effect is "fine-tuned model is slightly over-fitted", and not specific to zero.
D. Hessian-metric shrink: <E,W>_H/<W,W>_H per tensor, to test if the quantisers are MMSE in their own metric (Part 2 caveat).
