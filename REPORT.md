# 当前实验报告 — 2026-09-30

**当前阶段：优先验证 HD-SR-VQ 自身的效果。所有新实验均采用单 seed。G0、单 seed 核算与 checkpoint 重载均已通过；原 D1–D3 诊断已完成；稳健 D3 已完成并明确保留删失界，D3b screening 正在进行，尚无量化方法收益结论。**

## D3 后补充：当前执行方案与已完成重算（E027）

先验证(a)的 M=3/2/1；这里是索引的名义3/2/1bpw，实际储存必须包含码本、scale、mask、RHT符号。若(a)在3bpw已达到原始FP−1.5个百分点，跳过该档(b)–(d)，把消融放到退化≥3个百分点的最低实测位宽及上一档。加入(b*)前9步只读一个码本/最后一步全读，以及(b**)前8步读一个/第8步读两个/第9步全读；M=1时退化为同一配置并复用结果。主要排序指标为40条独立轨迹上的执行窗口5×7动作MSE；成功率是门槛，拐点处再比较成功率。

**稳健 D3 已完成全部预定补测，精确权重仍有删失；保守操作权重已冻结（E031）。** 先计算256个单观测相对MSE，再报告中位数及每侧裁剪10%（25个值）的均值。原有60条件中，步骤0–7在最大幅度3.0仍未达到1e-3阈值，因此不能外推完整权重；步骤8、9的中位数容忍幅度分别为 **2.02161、0.450722**。原mean-based结果保留作对照。后续10/30/100/300均已补测，总计100条件；步骤0–7仍未达到阈值，报告容忍度>300，停止继续扩噪，原单步骤LIBERO不重跑。

为避免把删失值冒充实测容忍度，分配采用逆容忍能量的保守上界，再归一到均值1：步骤0–7各 **2.15030e-5**，步骤8 **0.473527**，步骤9 **9.52630**。这是明确的操作性选择；**不是完整识别出的精确权重**，也不是置信区间。历史mean-based权重仍保留。已测曲线上的容忍幅度比下界超过665.60，支持继续逐步深度试验。[完整100点曲线](results/d3_robust_final_curves.csv)和[删失界、操作权重及原始对照](results/d3_robust_final.json)。

单一主导观测为 **index186，episode1271/frame91**，任务为“pick up the black bowl on the ramekin and place it on the plate”。它占幅度1.0下步骤3/4/5/6总动作误差的 **99.7211% / 99.5560% / 99.1785% / 98.0572%**。仅对此观测的32个初始噪声干净动作采样现已完成（E029，结果见下），不增加LIBERO实验seed。

D3b 同时扰动全部10步，所有步骤复用同一固定权重噪声方向，幅度 **0.03/0.1/0.3/0.6/1.0**。每档200个screening episode，seed7；held-out MSE分别对compact和原始FP报告。原始FP筛选为193/200，因此 **≥190/200通过FP−1.5pp门槛，≤189/200失败**。只报告首次实测失败对应的MSE及邻近括区间；不把噪声曲线当成已验证的量化成功率预测。

D3b尚无完整成功率。任务177749的RHT已完成并通过，离线D3b继续；初始screening worker启动失败后，worker2已通过全部干净/扰动参考检查并重启，其余两个worker由替代任务177765续跑。初次任务177750失败，保留日志，不计入结果。用户已允许耗时任务使用八卡；账户当前最多四张L40，八张A40/pro6000虽在配额内，但尚无八卡同类已验证资源可用。**真实RHT验证已通过（E028）**：全部126层、256观测/noise0，执行窗口相对MSE **8.65817e-7**，小于预设1e-4；最大单观测相对MSE **2.33284e-6**，保留张量逐字节一致。[完整测量](results/rht_validation.json)。联合beam、GPTQ反馈、固定码WLS及部署流程已实现；39项相关测试通过，现已开始实际VQ拟合（E030）。

**D3b离线五档已全部完成（E029）；成功率门槛待完整screening。** 256观测来自40条独立轨迹，噪声seed0，以下为全部10步同时扰动的执行窗口相对MSE：

| 相对权重噪声幅度 | 相对原始FP的动作MSE | 相对compact的动作MSE |
|---|---:|---:|
| 0.03 | 1.55841e-5 | 5.93038e-6 |
| 0.1 | 6.21539e-5 | 5.16691e-5 |
| 0.3 | 5.75154e-4 | 5.53313e-4 |
| 0.6 | 3.67268e-3 | 3.61295e-3 |
| 1.0 | 2.53731e-2 | 2.52233e-2 |

[完整离线CSV](results/d3b_offline/curves.csv)及[来源哈希](results/d3b_offline/summary.json)。每个量化位宽的D3b拐点预测会在该位宽screening之前写入记录，避免事后解释。

异常观测的32次干净采样也已完成：执行窗口夹爪符号27次为`-----`，4次`+----`，1次`++---`；第一主成分解释82.73%的样本方差，在0.5/1倍Scott带宽下有双峰，2倍带宽下合并为单峰。有两类夹爪命令时序和带宽敏感的双峰迹象，**尚不能稳健确认多模态，也不能据此断言单模态**。[逐次采样摘要](results/observation186_noise/draw_summary.csv)及[分析说明](results/observation186_noise/summary.json)。该检查仅针对一个观测，不增加LIBERO实验seed。

**量化实现进展（E030）**：一个真实2048×1024的q_proj层已完成M3拟合，耗时18.20秒，部署payload803850bytes；保存重载和全部10个mask的独立重构逐元素一致。这是实现检查，不是策略效果结论。**全部126层×M3/M2/M1，共378份层级拟合已完成**，四个拟合作业均正常结束；完整checkpoint正在组装，整策略MSE/成功率尚未测量。运行期间四张L40S负责诊断/screening，两张A6000和两张5090拟合，合计八卡；拟合完成后已释放对应GPU。新增5090先通过CUDA数值检查和真实已拟合层重构检查，所有策略动作/成功率比较仍在L40S。

可下载 [全部60个稳健曲线点](results/d3_robust/curves.csv)、[稳健/裁剪/mean-based容忍度与来源哈希](results/d3_robust/summary.json)。

## 保留张量与整模型位宽下界（E027）

论文主位宽为 **quantizable-Linear bpw**，whole-model bpw作为辅助核算。已逐项列出全部 **41个大于100万元素的保留张量**：[完整大小、类型、作用清单](results/reserved_floor/large_reserved_tensors.csv)。其中37个AdaRMS矩阵各3,145,728元素、两个timeMLP矩阵各1,048,576元素，均维持native FP32；其余为526,647,296元素的tokenembedding和2,359,296元素的视觉语言projector。norm、bias、timeMLP、AdaRMS均不量化。

| 核算场景 | 保留张量bytes | 仅保留张量的whole-model bpw下界 | 再加所有Linear至少一层索引的下界 |
|---|---:|---:|---:|
| (i) int8 embedding，其余native | 1,012,077,920 | 2.414428 | 3.220857 |
| (ii) int4 embedding，其余native | 748,754,272 | 1.786239 | 2.592668 |
| (iii) int8 embedding及允许的大保留张量 | 1,009,726,816 | 2.408819 | 3.215248 |

场景(iii)新增可量化的大张量只有视觉语言projector，主要保留开销来自受保护的AdaRMS。(ii)/(iii)为**解析核算、未做精度验证**；右列尚未计VQ码本等开销，不是已实现checkpoint位宽。int4使用两元素一字节及FP32逐行scale；新增int8也计FP32逐行scale。[详细核算及来源](results/reserved_floor/summary.json)。

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
| D3 容忍曲线及 MSE/MMD 代理 | **选 MSE；稳健操作权重已冻结，含删失界** | MSE 与失败率 Spearman **0.37661**，95% 区间 **[−0.14776, 0.77667]**；容忍幅度比 **5.23688**。代理关联不稳健（E026） |

D2 对单 seed 导致的低秩做了显式审计：**627/5670** 个 layer/pair 单元的 top-32 角度不可稳定识别，排除后报告 **81.05°**；κ 仍使用全部配对，独立支持保留 subset 试验（E014）。

![D1/D2 诊断决定](figures/diagnostic_selection/diagnostic_selection.png)

图 a：全部 **126** 个 Linear 层的排名统计，橙色菱形为全部 **38,928,384** 个 group 的合并统计。图 b：**5043** 个秩有效 layer/pair 单元的 κ 与最大主角度；虚线为预设门槛。这些是固定校准集的描述性结果，无跨 seed 置信区间。可下载 [PDF](figures/diagnostic_selection/diagnostic_selection.pdf)、[D1 源数据](figures/diagnostic_selection/d1_layer_source.csv)、[D2 源数据](figures/diagnostic_selection/d2_pair_source.csv)；数据覆盖与图像质检见 [QA](figures/diagnostic_selection/QA.md)。

D3 完成 **0.01 / 0.03 / 0.1 / 0.3 / 1.0 / 3.0** 六档、十个步骤的 **60** 个离线条件。闭环按离线预定规则选择幅度 **1.0**，完成十个步骤各 **200** 个配对 episode，另复用 **200** 个干净参考。扰动条件成功率有 **6** 个不同值，范围 **93.5–98.5%**，恰好达到预定的 **5 个百分点** 门槛，因此无需追加更强噪声筛选（E026）。

后续代理按预设规则选择**相对动作 MSE**：与失败率的 Spearman 为 **0.37661**，条件 95% 区间 **[−0.14776, 0.77667]**；MMD 为 **−0.27783**，区间 **[−0.48781, 0.50724]**。两种相关性的配对差异区间为 **[−0.19467, 0.84933]**。这些区间均跨零，证据不足以声称 MSE 稳定预测成功率或显著优于 MMD。该选择遵守事先固定的操作规则，后续方法效果仍须由 LIBERO 验证。

MSE 容忍幅度全部有测量括区间，最大/最小比 **5.23688**，保留逐步深度。以下为历史 **mean-based** 权重，已不作为新方案的最终权重；按逆容忍能量归一到均值 1 的权重（step 0–9）为 **[0.16165, 0.16395, 0.17041, 1.10332, 1.11461, 1.12582, 1.11747, 0.24157, 0.36795, 4.43324]**。MMD 阈值仍未达到，但它未被选中，因此无需为此继续补测。

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
