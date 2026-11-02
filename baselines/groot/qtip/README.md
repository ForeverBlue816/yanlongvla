# QTIP row of the GR00T N1.7 2-bit baselines: everything as run (staged 2 Nov 2026, not cleaned up)

Row `qtip_w2`: Plus 1000 = **573/1000** (camera 294, robot-init 279), NVIDIA RTX PRO 6000 Blackwell, run seed 7, paired against our FP 618. Pure post-training quantization: no layer-wise or end-to-end fine-tuning was run. This README says at each point what we have, what we do not have, and what we deleted.

## 1. What is in this folder
| file | what it is |
|---|---|
| `quantize_qtip.py`, `common.py` | the quantizer and its helper, identical to the versions that produced the row (one line differs from `baselines/quantize_qtip.py` on `main`: the layer-count assert also allows the 216 action-head layers) |
| `job_script_groot.pbs` | the job script as it was when the row ran (commit `e8a56c4`, 31 Oct): the exact command lines, with our scratch paths |
| `calibration_manifest.json` | the 256 calibration observations (which LIBERO episodes and frames) |
| `accounting_backbone.json`, `accounting_head_libero_{spatial,object,goal,10}.json` | bit accounting per key, with `exact_command` |
| `../results/groot_2bit_ours_*.plus_1000.jsonl`, `groot_fp_ourgpu.plus_1000.jsonl` | per-unit Plus 1000 files (see 5) |

## 2. Quantization, as run
- QTIP toolkit commit `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` (the job script refuses to run on any other commit).
- `SETTINGS` in `quantize_qtip.py`: codebook `bitshift`, L 16, K 2, V 2, td_x 16, td_y 16, decode_mode `quantlut_sym`, tlut_bits 9, scale_override 0.9, sigma_reg 1e-2.
- **LDLQ** (read from the toolkit at that commit): `LDLQ(Wr, L, cb, args, buf_cols=128, for_kernel=True)`. We call `ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=has_kernel)`, so **buf_cols = 128** (the default) and `for_kernel = utils.has_kernel(quantlut_sym, 16, 2, 2, 9, 16, 16)`. This LDLQ has no "rounds" argument.
- **Lookup table (tlut) initialisation** (`lib/codebook/bitshift.py`, `quantlut_sym`, V = 2): 512 centres from `scipy.cluster.vq.kmeans` on 2^20 random Gaussian 2-vectors, rescaled to std 0.9682458365518543, **cached in `/tmp/kmeans_9_2.pt`** and reloaded from there when it exists. We build the codebook once per quantizer process and do not seed the random state before that call, so the table is **not reproducible from our settings alone**: it depends on the state of `/tmp` on the node. We checked 6 layers of each of the five keys (backbone and four heads) in our fit records: all use the **same** table (identical FP16 values, hash 9ebc1255 on the raw bytes), so one table served the whole run. The table is stored in our fit records (the per-layer 512 x 2 FP16 table in the payload, see 6). We did not train it (no `ft_train_lut`).
- Per layer: `torch.manual_seed(0)` before each layer for the random sign vectors SU, SV; `regularize_H(H, 1e-2)`; randomized Hadamard on both sides (`matmul_hadUt`); Wscale = rms(Wr) / (rms(codebook lut) x 0.9); block LDL with block size 16; LDLQ onto the bitshift trellis; undo the transform; crop.
- Padding: SigLIP layers whose dimension is 4304 (no Hadamard size) are zero-padded to 4480 = 140 x 32 on that dimension, quantized, cropped back. Every other dimension uses the toolkit's Hadamard sizes without padding.
- No outlier handling, no fine-tuning, no end-to-end step.

## 3. Calibration and Hessians
- 256 observations, unperturbed frames only (`calibration_manifest.json`: 60 LIBERO episodes of `physical-intelligence/libero`, 4 to 5 frames each); the same 256 for every method. Backbone Hessians come from the libero_spatial checkpoint (the backbone is identical in all four suites); each suite's action head has its own.
- H = uncentered E[x x^T] accumulated in float64 over every forward call of every quantizable Linear (432 per suite: backbone 216 = vision 104 + language 112, head 216) during full policy inference on the 256 observations; flow-matching noise seed `SeedSequence([7, observation_index])`. H is not normalised; the quantizer applies `regularize_H` (sigma_reg 0.01).

## 4. Command lines (from `job_script_groot.pbs`; `<run>` is our scratch run folder)
```
python prepare_checkpoints.py --out <run>/checkpoints --hf-cache <run>/hf
python calibrate.py     --checkpoints <run>/checkpoints --out <run>/calibration --hf-cache <cache>
python export_layers.py --checkpoints <run>/checkpoints --calibration <run>/calibration --out <run>/layers
python quantize_qtip.py --layers <run>/layers/<key> --calibration <run>/calibration/<key> \
       --checkpoint <run>/checkpoints/<libero_spatial or the suite> --out <run>/q/qtip_w2/<key> --qtip-repo <qtip> --bits 2
         # key in: backbone, head_libero_spatial, head_libero_object, head_libero_goal, head_libero_10
python import_external.py --checkpoints <run>/checkpoints --quantized <run>/q/qtip_w2 --out <run>/ckpt/qtip_w2
python run_plus.py --checkpoints <run>/ckpt/qtip_w2 --out <run>/eval/qtip_w2/plus_1000.jsonl \
       --groot-python <env python> --plus-python groot/jobs/plus_python.sh      # all 1000 units
```
`import_external.py` writes the dequantized weights back into the checkpoint in its stored dtype (fake-quant: a 2-bit fit evaluated through dense weights).

## 5. Evaluation harness and per-unit files
- **The harness is the one already on `main`.** We compared our `baselines/groot/{serve,run_plus,eval_libero_plus,calibrate,export_layers,groot_common,import_external,quantize}.py` at the commit where the QTIP row was produced with the files on `main`: identical, except one comment line in `run_plus.py` and a line in `prepare_checkpoints.py` that skips DeepSpeed training state. (Since then we added options to `serve.py`, `calibrate.py` and `quantize.py` for later experiments; none was used for this row.)
- 1000 units of `baselines/data/plus_units_selection_1000.json`, run seed 7; the flow-matching noise is seeded per episode from the unit's seed (`noise_seed` in each line), so two models see identical noise on identical units. SDPA attention with the default torch kernel choice.
- **We never logged which attention kernel "default" resolved to** on our GPU. Logging was added on 2 Nov (`serve.py` prints the kernels of the first 3 attention calls); the QTIP row itself is unlogged. This matters: with weights unchanged, forcing the `math` or `efficient` kernel flips 11.5% / 14% of the 200-unit test (our note E0), while rerunning on the same GPU and kernel flips 0 (FP) and 1 (QTIP) of 1000 units.
- Per-unit files, the same 14 fields as the files in `../results/` on `main` and the same 1000 unit ids: `groot_2bit_ours_qtip_w2` (573), `groot_2bit_ours_quip10_w2` (official QuIP#, 10 sweeps), `groot_2bit_ours_quip_w2` (QuIP#, 0 sweeps), `groot_2bit_ours_vptq_w2`, `groot_2bit_ours_aqlm_w2`, and `groot_fp_ourgpu` (our FP, 618). The prefix `groot_2bit_ours_` follows your naming; these are our runs, not yours.

## 6. Bit accounting
- Linear payload per layer: packed trellis (2 bits per weight, padded shape counted) + FP16 SU and SV sign vectors + FP32 Wscale + the layer's FP16 lookup table (512 x 2). Quantized linear: **2.018 bpw** (432 layers per suite).
- Whole model **5.591 bpw**: everything outside the 432 Linear layers (embeddings, norms, AdaLN linears, patch embeddings, other tensors) is retained at the stored BF16 and counted in full. Per-key records: `accounting_*.json`.

## 7. What we do not have, what we deleted, and large files
- **Not in this folder and not in git:** the fit records (`q/qtip_w2`, **13 GB**: trellis codes, tables, scales per layer for all five keys) and the Hessians (`calibration`, tens of GB). They are on our cluster scratch and still exist; they are too large for a normal git push.
- **Deleted:** the fake-quant checkpoint of this row (about 26 GB per suite; regenerable with `import_external.py` from the fit records). Over the last days we also deleted, because of repeated scratch quota failures, the fit records of two VPTQ runs, the fake-quant checkpoints of the finished rows, and temporary test checkpoints; each is logged. The QTIP fit records were not deleted.
- **Not recorded:** the attention kernel (above); the `/tmp` k-means cache file itself (its content lives in the fit records).
- **Not known to us:** which file you call "the LIBERO medium file". It is not part of our GR00T Plus run.

## 8. Open questions for you
1. How do you want the 13 GB of fit records (and, if useful, the Hessians)? Options: a private download link we create, a shared path on the cluster if you have access, or only a few layers you name.
2. Which file is "the LIBERO medium file"?
3. Which GPU, PyTorch version and attention kernel did your 618 (FP) and 486 / 529 runs use? We suggest pinning one kernel for the final rows so that the two sets are comparable.
4. Fairness: should every method be reported at its official settings only, or an official row plus one tuned row per method (we propose the second, tuned by calibration error, never by Plus 1000)? GPTQ got 9 variants and VPTQ 8, while AQLM, QTIP and QuIP# got one each.
5. VPTQ at true 2 bits: the official README's "~2 bit" model (v8 k65536) is 3.50 bits once its codebook is counted. Keep our main row v4 k256 (2.003 bits, 495), or use the closest-to-official structure that fits 2 bits (v8 k12288, 1.988 bits, 426)?
