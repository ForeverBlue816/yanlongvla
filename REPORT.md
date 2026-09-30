# 当前实验报告 — 2026-09-30

**当前阶段：优先验证 HD-SR-VQ 自身的效果。所有新实验均采用单 seed。G0、单 seed 核算与 checkpoint 重载均已通过；D1–D3 单 seed 诊断均已完成，尚无 (a)–(d) 的方法收益结论。**

## 当前进展

- FP 复现：6,000 次 rollout，四套件平均成功率 **96.8667%**，三种子均值的 95% t 区间 **[95.6862%, 98.0472%]**；完整分套件结果见下文。
- **Step 1 完整单 seed 对照已通过**：256 个观测上，两个 lm_head 的调用次数均为 **0**，移除后的动作逐元素一致；VLM embedding 保留一份，active denominator 为 **3,353,433,872** 个原始权重元素（E011）。
- int8 逐行 absmax embedding：执行窗口相对 MSE **1.99510e-5**，低于门槛 **1e-4**；最大单观测相对 MSE **0.00350129**，最大绝对动作差 **0.122290**。平均门槛通过，尾部误差如实保留（E011）。
- 核算 **177454** 与重载 **177463** 均通过；D1 **177465**、D2 **177481** 已完成。D3 六档离线幅度共 **60** 个条件、**2000** 个扰动 episode 和复用的 **200** 个干净参考已全部完成，完整配对核验通过（E026）。四卡工作分为 **177565 / 177582**，均正常完成；全部新实验为单 seed。模型和数据位于 `/projects/yanlongvla`。
- 方法离线验证集已冻结：与校准数据的 **60** 条轨迹完全分离的 **40** 条训练轨迹、**256** 个观测 x **1** seed；不参与诊断或拟合（E008）。

## 核算与诊断决定

量化 Linear bpw 使用实际保存的索引、码本及相关元数据；whole-model bpw 计全部存储张量、别名只计一次，分母移除已验证无用的头。每次推理的读取量单列，按实际模块调用和 embedding 行查找统计逻辑读取；它不是 GPU DRAM 测量。按步少读码平面不自动降低模型存储。

| 配置 | quantizable-Linear bpw | whole-model bpw | bytes read / inference |
|---|---:|---:|---:|
| 去无用头，native embedding | 16.000000 | 16.571214 | 17,434,050,720 |
| 去无用头 + int8 embedding | 16.000000 | 15.317291 | 17,434,020,060 |

| 诊断 | 当前决定 | 证据状态 |
|---|---|---|
| D1 相干/非相干分数 | **采用非相干分数** | Spearman **0.99194**，top-10% overlap **92.02%**；触发放弃相干贡献的门槛（E013） |
| D2 subset decoding | **开启，保留 (d)** | median κ **0.23349**，有效 top-32 子空间的 pair-max 主角度中位数 **81.05°**（E014） |
| D3 容忍曲线及 MSE/MMD 代理 | **选 MSE；保留逐步深度** | MSE 与失败率 Spearman **0.37661**，95% 区间 **[−0.14776, 0.77667]**；容忍幅度比 **5.23688**。代理关联不稳健（E026） |

D2 对单 seed 导致的低秩做了显式审计：**627/5670** 个 layer/pair 单元的 top-32 角度不可稳定识别，排除后报告 **81.05°**；κ 仍使用全部配对，独立支持保留 subset 试验（E014）。

![D1/D2 诊断决定](figures/diagnostic_selection/diagnostic_selection.png)

图 a：全部 **126** 个 Linear 层的排名统计，橙色菱形为全部 **38,928,384** 个 group 的合并统计。图 b：**5043** 个秩有效 layer/pair 单元的 κ 与最大主角度；虚线为预设门槛。这些是固定校准集的描述性结果，无跨 seed 置信区间。可下载 [PDF](figures/diagnostic_selection/diagnostic_selection.pdf)、[D1 源数据](figures/diagnostic_selection/d1_layer_source.csv)、[D2 源数据](figures/diagnostic_selection/d2_pair_source.csv)；数据覆盖与图像质检见 [QA](figures/diagnostic_selection/QA.md)。

D3 完成 **0.01 / 0.03 / 0.1 / 0.3 / 1.0 / 3.0** 六档、十个步骤的 **60** 个离线条件。闭环按离线预定规则选择幅度 **1.0**，完成十个步骤各 **200** 个配对 episode，另复用 **200** 个干净参考。扰动条件成功率有 **6** 个不同值，范围 **93.5–98.5%**，恰好达到预定的 **5 个百分点** 门槛，因此无需追加更强噪声筛选（E026）。

后续代理按预设规则选择**相对动作 MSE**：与失败率的 Spearman 为 **0.37661**，条件 95% 区间 **[−0.14776, 0.77667]**；MMD 为 **−0.27783**，区间 **[−0.48781, 0.50724]**。两种相关性的配对差异区间为 **[−0.19467, 0.84933]**。这些区间均跨零，证据不足以声称 MSE 稳定预测成功率或显著优于 MMD。该选择遵守事先固定的操作规则，后续方法效果仍须由 LIBERO 验证。

MSE 容忍幅度全部有测量括区间，最大/最小比 **5.23688**，保留逐步深度。按逆容忍能量归一到均值 1 的最终权重（step 0–9）为 **[0.16165, 0.16395, 0.17041, 1.10332, 1.11461, 1.12582, 1.11747, 0.24157, 0.36795, 4.43324]**。MMD 阈值仍未达到，但它未被选中，因此无需为此继续补测。

![D3 离线扰动及候选容忍区间](figures/d3_offline/d3_offline.png)

图 a/b 包含六档幅度、十个步骤的全部 **60** 个离线条件。图 c 橙点为 MSE=1e-3 的首次穿越插值，灰线是**已测幅度的括区间，不是置信区间**。1.0 档的步骤 3–6 有 98–99.7% 的平方误差来自同一个观测；全部样本保留，因此不能仅凭 MSE 曲线宣布代理可靠。该图展示离线证据；最终 MSE 选择依据上述完整闭环筛选，相关性不确定性已单独报告。下载 [PDF](figures/d3_offline/d3_offline.pdf)、[六档源数据](figures/d3_offline/offline_source.csv)、[候选容忍区间](figures/d3_offline/mse_tolerance_source.csv)；说明见 [QA](figures/d3_offline/QA.md)（E017/E018/E020）。

| 扰动步骤 | spatial 成功 | libero_10 成功 | 合计成功率 |
|---|---:|---:|---:|
| clean | 100/100 | 91/100 | 95.5% |
| 0 | 100/100 | 93/100 | 96.5% |
| 1 | 100/100 | 92/100 | 96.0% |
| 2 | 99/100 | 96/100 | 97.5% |
| 3 | 98/100 | 94/100 | 96.0% |
| 4 | 99/100 | 94/100 | 96.5% |
| 5 | 99/100 | 94/100 | 96.5% |
| 6 | 99/100 | 95/100 | 97.0% |
| 7 | 100/100 | 97/100 | 98.5% |
| 8 | 99/100 | 95/100 | 97.0% |
| 9 | 98/100 | 89/100 | 93.5% |

逐步 MSE/MMD 原值、区间和来源哈希见 [D3 完整证据](results/d3_screen_1_evidence/EVIDENCE.md)及 [CSV](results/d3_screen_1_evidence/screening_source.csv)。每行两套件各 10 tasks ×10 eps，只有 seed7；干净参考不参与十个扰动条件的相关性计算（E026）。

D3 的 MMD 现在对单 seed 下的观测/动作样本集合计算，衡量跨观测的边缘分布，不再估计每个观测的多噪声条件分布（E010）。

全部使用 **10** 个 flow steps、每步 **10** 个 suffix tokens、执行窗口 **K=5**、有效维度 **7**。每层输入缓存包括全部 token；同值 Q/K/V、gate/up 输入先验证再共享存储。三个诊断决定现已写入本地 CONFIG，核算与诊断入口门槛通过，可以进入方法实现和比较。D4 延后。

读取量是第一个固定观测的逻辑权重读取，包含三次图像编码和十次 expert 调用；该观测的 200 个语言 token 含 15 个不同索引，换 prompt 会略有变化。张量存储为去头模型 **6,946,308,960 bytes**、再加 int8 embedding **6,420,690,272 bytes**。这些是未做 VQ 的核算参考，不是低位宽方法结果（E011）。

整模型后续目标还有一项已核实的预算约束：当前保留张量占 **2.41443 whole-model bpw**；若每个 eligible Linear 至少保存一层 8-bit 索引/8 个权重，忽略码本等开销时也已达 **3.22086 bpw**。因此 **3.0/2.5 whole-model bpw** 需要另行验证保留张量或分配方式的调整；不能把 nominal index bits 当成整个模型的实际位宽（E011）。

## 单 seed 配对参考筛选

复用历史 FP seed7 中相同的 episode0–9，与新 compact/int8 embedding 参考严格配对；每套件 **10 tasks ×10 eps**（E018）。

| 参考策略 | spatial | libero_10 | 合计 |
|---|---:|---:|---:|
| 原始 FP，复用已有结果 | 97/100 | 96/100 | **96.5%** |
| 去头 + int8 embedding | 100/100 | 91/100 | **95.5%** |

共有 **4** 个 episode 由失败转成功、**6** 个由成功转失败，净变化 **−1.0 个百分点**。这是单 seed 的描述性配对结果，不是成功率等价证明。后续 G2 保留原始 FP 对照；不能把 embedding 的平均动作误差门槛通过当作闭环无损。

## HD-SR-VQ 方法表

下表的目标为 action-expert quantizable-Linear **3.0 / 2.0 bpw**，backbone 保持 BF16（上游规定保留 FP32 的张量继续保留）。表中尚无测量值；按用户明确选择，主表配平**实际存储 bpw**，读取量单独报告。平均只读两层的三层码平面 checkpoint 仍按实际三层存储计费（E024）。

| Variant | 组成 | 实测 Linear bpw | whole-model bpw | bytes/read inference | Held-out proxy | Screening success | Full success / CI |
|---|---|---:|---:|---:|---:|---:|---|
| (a) | uniform depth / prefix / shared codebooks | pending | pending | pending | pending | pending | pending |
| (b) | D3-weighted per-step depth | pending | pending | pending | pending | pending | pending |
| (c) | (b) + conditional centroids + affine | pending | pending | pending | pending | pending | pending |
| (d) | (c) + subset swap + heterogeneous objective | pending | pending | pending | pending | pending | 仅 D2 启用时运行 |
| whole model | 最佳 expert + SigLIP/Gemma 量化 | pending | pending | pending | pending | pending | pending |

晋级结果的置信区间将使用配对、按 suite 分层的 task/episode bootstrap，明确限定为单 seed 下的区间。作为预先固定的全协议参考，已有 FP seed7 的 **1928/2000=96.4%**，其该口径 95% 区间为 **[93.79875%, 98.30125%]**；这与下文历史“三 seed 均值的 t 区间”含义不同。没有新增 FP rollout（E021）。

(c) 内部会分别测 centroids only、affine only、both；若 (c) 对 (b) 的代理改善不足 **5%**，如实报告并停止该组件的细化。先完成 PTQ 表，再考虑冻结 codes 的 KD 独立行。筛选协议为 spatial + libero_10，各 **10 eps/task、1 seed**；晋级配置使用四套件各 **50 eps/task、1 seed**。完整门槛与实验命令记录在 EXPERIMENTS。

## 当前支持的结论

已经支持的结论是 FP 复现通过，无用输出头在完整单 seed 校准对照中可以无误差移除，int8 embedding 的平均动作误差通过设定门槛；D1 显示两种评分排名高度一致，按预设规则放弃相干评分贡献；D2 的非交换性和有效主角度支持继续测试 subset；D3 按预定规则选择 MSE 并保留逐步深度，但其成功率关联区间跨零。按步深度、条件质心、affine 和 subset 解码的实际量化收益仍待方法表验证。因此目前不能声称 HD-SR-VQ 提高成功率或优于任何外部方法。

外部 GPTQ/AWQ/ActQuant 基线全部延后。此前 AQLM 在 10 tokens 的两个形状上约 **4x** 慢于 BF16 的 CUDA Graph 测量保留于 E005；这只是内核微测，不是本方法端到端结果。方法表完成前不继续内核工作。

## Historical completed G0 evaluation update — 2026-09-29T18:31:50.996876+00:00

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
