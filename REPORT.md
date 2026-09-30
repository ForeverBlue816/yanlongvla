# 当前实验报告 — 2026-09-30

**当前阶段：优先验证 HD-SR-VQ 自身的效果。G0 已通过；完整存储核算正在验证。尚无 (a)–(d) 的方法收益结论。**

## 当前进展

- FP 复现：6,000 次 rollout，四套件平均成功率 **96.8667%**，三种子均值的 95% t 区间 **[95.6862%, 98.0472%]**；完整分套件结果见下文。
- Step 1 小样本对照：两个 lm_head 的调用次数均为 **0**；移除后 **32** 个配对动作块逐元素一致。VLM lm_head 与 embedding 共享权重，embedding 保留一份（E009）。
- int8 逐行 absmax embedding：小样本执行窗口相对 MSE **9.66268e-5**，门槛 **1e-4**；最大单样本相对 MSE **0.00350129**，最大绝对动作差 **0.122290**。这是接近门槛且存在尾部误差的初测，不能代替完整验证（E009）。
- 完整 **256 观测 x 8 seeds** 核算作业 **177438** 已提交；通过前不发布新的 bpw 结论、不把 int8 embedding 设为已验证默认。模型和数据仍放在 `/projects/yanlongvla`。
- 方法离线验证集已冻结：与校准数据的 **60** 条轨迹完全分离的 **40** 条训练轨迹、**256** 个观测 x **8** seeds；不参与诊断或拟合（E008）。

## 核算与诊断决定

量化 Linear bpw 使用实际保存的索引、码本及相关元数据；whole-model bpw 计全部存储张量、别名只计一次，分母移除已验证无用的头。每次推理的读取量单列，按实际模块调用和 embedding 行查找统计逻辑读取；它不是 GPU DRAM 测量。按步少读码平面不自动降低模型存储。

| 配置 | quantizable-Linear bpw | whole-model bpw | bytes read / inference |
|---|---:|---:|---:|
| 去无用头 + int8 embedding | 完整核算待完成 | 完整核算待完成 | 完整核算待完成 |

| 诊断 | 当前决定 | 证据状态 |
|---|---|---|
| D1 相干/非相干分数 | pending | 实现已准备，未运行 |
| D2 subset decoding | pending | 实现已准备，未运行 |
| D3 容忍曲线及 MSE/MMD 代理 | pending | 注入与度量已准备，未运行 |

全部使用 **10** 个 flow steps、每步 **10** 个 suffix tokens、执行窗口 **K=5**、有效维度 **7**。每层输入缓存包括全部 token；同值 Q/K/V、gate/up 输入先验证再共享存储。三个诊断决定写入本地 CONFIG 后才进入方法比较。D4 延后。

## HD-SR-VQ 方法表

下表的目标为 action-expert quantizable-Linear **3.0 / 2.0 bpw**，backbone 保持 BF16（上游规定保留 FP32 的张量继续保留）。表中尚无测量值；实际开销和读取量必须匹配后才比较效果。

| Variant | 组成 | 实测 Linear bpw | whole-model bpw | bytes/read inference | Held-out proxy | Screening success | Full success / CI |
|---|---|---:|---:|---:|---:|---:|---|
| (a) | uniform depth / prefix / shared codebooks | pending | pending | pending | pending | pending | pending |
| (b) | D3-weighted per-step depth | pending | pending | pending | pending | pending | pending |
| (c) | (b) + conditional centroids + affine | pending | pending | pending | pending | pending | pending |
| (d) | (c) + subset swap + heterogeneous objective | pending | pending | pending | pending | pending | 仅 D2 启用时运行 |
| whole model | 最佳 expert + SigLIP/Gemma 量化 | pending | pending | pending | pending | pending | pending |

(c) 内部会分别测 centroids only、affine only、both；若 (c) 对 (b) 的代理改善不足 **5%**，如实报告并停止该组件的细化。先完成 PTQ 表，再考虑冻结 codes 的 KD 独立行。筛选协议为 spatial + libero_10，各 **10 eps/task、1 seed**；晋级配置使用四套件各 **50 eps/task、3 seeds**。完整门槛与实验命令记录在 EXPERIMENTS。

## 当前支持的结论

已经支持的结论是 FP 复现通过，且无用输出头在小样本真实校准推理中可以无误差移除。int8 embedding 仍须通过完整误差门槛；相干分数、按步深度、条件质心、affine 和 subset 解码是否改善量化质量均未得到验证。因此目前不能声称 HD-SR-VQ 提高成功率或优于任何外部方法。

外部 GPTQ/AWQ/ActQuant 基线全部延后。此前 AQLM 在 10 tokens 的两个形状上约 **4x** 慢于 BF16 的 CUDA Graph 测量保留于 E005；这只是内核微测，不是本方法端到端结果。方法表完成前不继续内核工作。

## G0 evaluation update — 2026-09-29T18:31:50.996876+00:00

Status: **passed**; observed **6000 / 6000** planned episodes.

Report generator code: `21c919bf482d6ed8d19c7e96b07ca85236008acb`. Commands: `python scripts/launch_g0.py` (two workers), then `python scripts/publish_g0_summary.py`. Exact evaluation code and worker job IDs are in the launch ledger entry.

Seeds: [7, 17, 27]; 50 episodes/task. two-sided 95% Student t across the three seed-level means; df=2; clipped to [0,100]. Describes seed variation on fixed tasks, not independent rollout uncertainty.

| Suite | Per-seed success % | Mean % | 95% CI % | Absolute gap vs quoted (pp) |
|---|---|---:|---|---:|
| libero_spatial | [98.2, 98.0, 98.6] | 98.266667 | [97.50775005991878, 99.02558327341455] | 0.533333 |
| libero_object | [98.8, 99.2, 98.2] | 98.733333 | [97.48301143741116, 99.9836552292555] | 0.533333 |
| libero_goal | [96.8, 98.8, 97.2] | 97.600000 | [94.97103575680975, 100] | 0.400000 |
| libero_10 | [91.8, 93.4, 93.4] | 92.866667 | [90.57191854404732, 95.16141478928603] | 0.466667 |
| overall | [96.4, 97.35, 96.85] | 96.866667 | [95.68615661219496, 98.04717672113836] | 0.016667 |

G0 passed the predeclared coverage and tolerance check. Diagnostics may begin; no quantization or method gain is claimed here.
