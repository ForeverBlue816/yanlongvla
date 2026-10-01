# 当前实验报告 — 2026-10-01

**当前阶段：优先验证 HD-SR-VQ 自身的效果。所有新实验均采用单 seed。G0、单 seed 核算与 checkpoint 重载均已通过；原 D1–D3 诊断已完成；稳健 D3 已完成并明确保留删失界，D3b 全部1000个扰动回合已完成，方法评估入口已通过；(a)三档完整动作验证已完成；3/2/1 bpw screening分别为193/200、191/200、193/200，全部通过FP−1.5pp门槛；未找到预定的成功率拐点。尚无(b)–(d)收益结论。**


## 2026-10-01 新阶段（E041，执行中，尚无新增完整质量结论）

进度快照 **2026-10-01T05:59:33.564275+00:00**：M1已记录903/2000回合（含复用200），M2为200/2000；骨干M3/M2分别保存199/199层，各须288层。当前两张5090及一张A6000继续拟合、两张L40S继续M1；余下一张5090及额外L40S分片排队/续排。尚无新的完整策略MSE或成功率。[可下载进度快照](results/phase_oct1_progress.json)。

按用户新优先级启动 P1/P2，全部新实验仍为单 seed。P1：M1先于M2，各补1800个episode，复用已验收200个screening episode；完整结果将对原始FP seed7报告四套件逐项与整体的配对bootstrap区间，libero_10单列。双L40S作业178947已开始M1；原排队的第二组178948已拆为两个单L40S作业178999/179000，以利用单张空闲卡；四个逻辑worker的覆盖与配对协议不变；不会把部分回合成功率作为完整结果。

P2关键路径：实际骨干为288个Linear、2,392,879,104个原始权重元素，准备M3/M2，expert固定为已验证M1。两张5090上的真实观测试跑178955通过：可微十步采样与同设备M1前向逐元素一致、重复action-Fisher一致，且视觉和语言KV均有非零信号。这是校准实现验证，不是新量化模型的效果。完整256观测校准178957因重复检查使用累计差分而引入约4e-28的浮点相消误差，在第一观测检查处停止；改为两个独立零初始化累加器直接比较，完全一致的门槛未变。替代校准 **178962已完整通过256观测验收**，覆盖288个Linear和180组共享输入矩。两张5090已实际开始M3/M2拟合（178961）。A6000因预计排队约15小时，其尚未启动的两个分片已按速度优先改交额外两张已验证的5090（179014），179014随后也已获配，一度四张5090同时拟合。05:46UTC其中一组178961被集群抢占，已保存结果保留；剩余worker2/3已拆成单A6000续跑179056和单5090续跑179057，以利用零散资源。卡数以带时间戳的进度快照为准。额外两张L40S仍排队；所有held-out/成功率评估仍限L40S。 [完整骨干校准验收](results/backbone_calibration.json)记录了两分片前向/梯度重复检查均精确一致；Gemma最后一层的q/o/MLP共五个prefix输出分支没有动作梯度，单独用H重建，其余按action-Fisher拟合。这仍是校准和实现结果，尚无骨干量化后的策略MSE或成功率。

P2的导出、双L40S held-out验证、M3先行screening及门槛通过后的M2/full评估已接为依赖任务。P3的固定十步AdaRMS常量表、实际打包int4 embedding和200集配对筛选入口也已接好；当前尚无这些整策略验证结果，不把FP16表默认视为精确。旧下界表为原布局历史核算，新增FP16 embedding scale及常量折叠后的下界将另报。P4的三个均匀配置已接通拟合、打包导出、独立L40S离线动作验证和配对screening，尚未启动GPU实验；P1/P3完整结束并释放卡后，自动申请四张拟合卡，为P2保留至多四张评估卡，总量不超过八卡。实测拐点处的匹配读取量消融仍待后续。含码本、scale、mask、RHT符号的解析expert位宽依次为0.543593、0.517906、1.070108bpw（对应名义0.5/0.5/1.0）；16项码本的索引须真实按4bit打包。实际结果与读取量将另列。P1–P3完成前不做外部基线或内核。

P3条件性核算（尚未验证精度）：固定表替代39个矩阵及其bias，可移除474,419,200bytes；FP32表连同实际十步schedule共4,546,600bytes。保留int8 embedding时，仅保留张量的whole-model下界预计为 **1.293493bpw**；int4/FP16逐行scale若通过，预计为 **0.664077bpw**。若FP16表也通过1e-6门槛，相应下界为1.288070/0.658654。以上保留原始训练权重分母，**不含尚需加上的VQ权重，不是已测整模型位宽或精度结论**。[完整解析核算](results/reserved_floor/phase_oct1_projection.json)。


论文段落草案（单seed，P1待完成）：

On the 200-episode, seed-7 LIBERO screening protocol, uniform vector quantization of the flow action expert at approximately 1 bpw (1.0303 actual stored quantizable-Linear bpw) matches the full-precision reference point estimate, 193/200 successes, with a paired difference of 0 pp and a conditional 95% interval of [-5.0, +5.5] pp. Thus no loss is observed in this screening sample, while equivalence and full-protocol robustness remain unestablished pending P1. Under the specified minimum-read budget, calibration-optimal prefix-depth allocation assigns the full depth to the final Euler step in all 126 action-expert Linear layers at M2 and M3. This agrees with the single-step tolerance curve and the final-step projection argument; because this budget permits only one full-depth step, the finding concerns its placement, not optimality among unrestricted schedules. These are single-seed observations with a BF16 backbone and int8 embeddings (13.9271 whole-model bpw for expert M1), not whole-model 1 bpw compression.

## D3 后补充：当前执行方案与已完成重算（E027）

先验证(a)的 M=3/2/1；这里是索引的名义3/2/1bpw，实际储存必须包含码本、scale、mask、RHT符号。若(a)在3bpw已达到原始FP−1.5个百分点，跳过该档(b)–(d)，把消融放到退化≥3个百分点的最低实测位宽及上一档。加入(b*)前9步只读一个码本/最后一步全读，以及(b**)前8步读一个/第8步读两个/第9步全读；M=1时退化为同一配置并复用结果。主要排序指标为40条独立轨迹上的执行窗口5×7动作MSE；成功率是门槛，拐点处再比较成功率。

**稳健 D3 已完成全部预定补测，精确权重仍有删失；保守操作权重已冻结（E031）。** 先计算256个单观测相对MSE，再报告中位数及每侧裁剪10%（25个值）的均值。原有60条件中，步骤0–7在最大幅度3.0仍未达到1e-3阈值，因此不能外推完整权重；步骤8、9的中位数容忍幅度分别为 **2.02161、0.450722**。原mean-based结果保留作对照。后续10/30/100/300均已补测，总计100条件；步骤0–7仍未达到阈值，报告容忍度>300，停止继续扩噪，原单步骤LIBERO不重跑。

为避免把删失值冒充实测容忍度，分配采用逆容忍能量的保守上界，再归一到均值1：步骤0–7各 **2.15030e-5**，步骤8 **0.473527**，步骤9 **9.52630**。这是明确的操作性选择；**不是完整识别出的精确权重**，也不是置信区间。历史mean-based权重仍保留。已测曲线上的容忍幅度比下界超过665.60，支持继续逐步深度试验。[完整100点曲线](results/d3_robust_final_curves.csv)和[删失界、操作权重及原始对照](results/d3_robust_final.json)。

单一主导观测为 **index186，episode1271/frame91**，任务为“pick up the black bowl on the ramekin and place it on the plate”。它占幅度1.0下步骤3/4/5/6总动作误差的 **99.7211% / 99.5560% / 99.1785% / 98.0572%**。仅对此观测的32个初始噪声干净动作采样现已完成（E029，结果见下），不增加LIBERO实验seed。

D3b 同时扰动全部10步，所有步骤复用同一固定权重噪声方向，幅度 **0.03/0.1/0.3/0.6/1.0**。每档200个screening episode，seed7；held-out MSE分别对compact和原始FP报告。原始FP筛选为193/200，因此 **≥190/200通过FP−1.5pp门槛，≤189/200失败**。只报告首次实测失败对应的MSE及邻近括区间；不把噪声曲线当成已验证的量化成功率预测。

**D3b screening 已完整完成（E032）**：五档共1000个扰动episode，加上复用的200个干净参考，全部通过任务/episode/噪声配对与无基础设施错误检查。初始失败任务保留日志，替代worker完成缺口；未降低验证标准。用户授权的大任务八卡并行已实际使用：四张L40S做策略诊断，两张A6000和两张5090做独立层拟合；所有策略比较保持L40S。**真实RHT验证已通过（E028）**：全部126层、256观测/noise0，执行窗口相对MSE **8.65817e-7**，小于预设1e-4；最大单观测相对MSE **2.33284e-6**，保留张量逐字节一致。[完整测量](results/rht_validation.json)。联合beam、GPTQ反馈、固定码WLS及部署流程已实现；相关数值检查已通过，全部378份实际VQ层拟合完成（E030）。

**D3b离线与完整screening均已完成（E029/E032）。** 256观测来自40条独立轨迹，噪声seed0，以下为全部10步同时扰动的执行窗口相对MSE：

| 相对权重噪声幅度 | 相对原始FP动作MSE | 相对compact动作MSE | 成功回合 / 200 | 成功率 |
|---|---:|---:|---:|---:|
| 0（compact参考） | 1.09443e-05 | 0 | 191 | 95.5% |
| 0.03 | 1.55841e-05 | 5.93038e-06 | 197 | 98.5% |
| 0.1 | 6.21539e-05 | 5.16691e-05 | 194 | 97.0% |
| 0.3 | 0.000575154 | 0.000553313 | 194 | 97.0% |
| 0.6 | 0.00367268 | 0.00361295 | 187 | 93.5% |
| 1.0 | 0.0253731 | 0.0252233 | 174 | 87.0% |

**首次实测失败在幅度0.6：187/200=93.5%，对应MSE 0.00367268**；前一档幅度0.3仍为194/200=97%，MSE 0.000575154。只确定这两个观测端点，不插值出精确成功率阈值。包含干净参考时成功率并不单调，轻噪声下的提高也不据此宣称收益。[完整结果及来源哈希](results/d3b/summary.json)、[成功率/MSE表](results/d3b/table.csv)。

[完整离线CSV](results/d3b_offline/curves.csv)及[来源哈希](results/d3b_offline/summary.json)。每个量化位宽的D3b拐点预测会在该位宽screening之前写入记录，避免事后解释。

异常观测的32次干净采样也已完成：执行窗口夹爪符号27次为`-----`，4次`+----`，1次`++---`；第一主成分解释82.73%的样本方差，在0.5/1倍Scott带宽下有双峰，2倍带宽下合并为单峰。有两类夹爪命令时序和带宽敏感的双峰迹象，**尚不能稳健确认多模态，也不能据此断言单模态**。[逐次采样摘要](results/observation186_noise/draw_summary.csv)及[分析说明](results/observation186_noise/summary.json)。该检查仅针对一个观测，不增加LIBERO实验seed。

**量化实现进展（E030）**：一个真实2048×1024的q_proj层已完成M3拟合，耗时18.20秒，部署payload803850bytes；保存重载和全部10个mask的独立重构逐元素一致。这是实现检查，不是策略效果结论。**全部126层×M3/M2/M1，共378份层级拟合已完成**，四个拟合作业均正常结束；三个完整checkpoint已组装并通过严格策略重载；整策略MSE已完成；M3/M2/M1成功率均已完整验收。运行期间四张L40S负责诊断/screening，两张A6000和两张5090拟合，合计八卡；拟合完成后已释放对应GPU。新增5090先通过CUDA数值检查和真实已拟合层重构检查，所有策略动作/成功率比较仍在L40S。

**(a) 部署存储已实测核算并通过严格策略重载（E032/E034）**。下表骨干仍为native BF16/FP32，所以全体Linear及whole-model位宽明显高于仅action expert位宽。读取量见下面的离线方法表；明确区分压缩表示逻辑读取和当前实际质量评估所用的BF16解码缓存路径。

| 名义索引bpw | expert quantizable-Linear bpw | 全体quantizable-Linear bpw | whole-model bpw | 全部张量payload bytes |
|---|---:|---:|---:|---:|
| 3 | 3.056851 | 14.509471 | 14.115285 | 5,916,834,380 |
| 2 | 2.043593 | 14.392784 | 14.021186 | 5,877,389,900 |
| 1 | 1.030335 | 14.276098 | 13.927086 | 5,837,945,420 |

[实际checkpoint核算与哈希](results/uniform_checkpoint_accounting.json)。D3b入口验收177820已通过；实际离线任务177845/177846/177853全部完成。筛选采用177854/177855两组独立绑定的双卡worker（最多四张L40S），每档200个episode、seed7。历史仅排队的布局按集群规则重提，未重复执行或丢弃任何结果。

## 首批方法参考(a)：完整离线结果（E034）

256个held-out观测、40条与校准完全分离的轨迹、noise0，评估执行窗口前5×7。所有张量严格重载、全部保留张量一致，四个固定观测重复推理逐元素一致。以下为相对**原始FP**的全局动作MSE，包含默认int8 embedding的固定误差。

| (a)名义索引bpw | expert Linear实际bpw | whole-model bpw | 动作相对MSE | 压缩表示逻辑读取 bytes/inference | 当前BF16缓存路径逻辑读取 bytes/inference |
|---|---:|---:|---:|---:|---:|
| 3 | 3.056851 | 14.115285 | 3.10745e-05 | 12,395,449,800 | 17,436,048,840 |
| 2 | 2.043593 | 14.021186 | 0.000161955 | 12,001,005,000 | 17,436,048,840 |
| 1 | 1.030335 | 13.927086 | 0.0106768 | 11,606,560,200 | 17,436,048,840 |

[全部数值、绝对MSE、compact参考MSE及来源哈希](results/method_offline/summary.json)、[可下载CSV](results/method_offline/table.csv)。三档额外BF16解码缓存均为622,854,144bytes，不包含在部署checkpoint中，但计入运行时内存；上表均为参数逻辑读取，**不是DRAM测量或已实现的压缩内核加速**。保留模块的读取用同一固定核算观测，未将其误称为整个LIBERO平均。

D3b规则在screening前记录的预测是：名义3/2bpw通过，1bpw失败；1bpw的MSE约0.01068已高于D3b首次掉点的0.00367。**完整screening推翻了1bpw会失败的预测：1bpw仍为193/200。** 这说明D3b的高斯权重噪声MSE阈值不能直接迁移成VQ的成功率阈值；不能据此外推位宽拐点。历史预测完整保留，未作事后改写。

**全部参考筛选完成（E038）**：

| 名义索引bpw | spatial | libero_10 | 合计 | 相对原始FP | 门槛 |
|---|---:|---:|---:|---:|---|
| 3 | 99/100 | 94/100 | 193/200 =96.5% | 0pp | 通过 |
| 2 | 98/100 | 93/100 | 191/200 =95.5% | −1pp | 通过 |
| 1 | 98/100 | 95/100 | 193/200 =96.5% | 0pp | 通过 |

每档均核验四个worker的checkpoint/离线哈希、全部200个配对episode与无错误记录；两组GPU作业正常结束，CPU177859验收通过。[完整参考表及配对统计](results/method_reference/summary.json)、[CSV](results/method_reference/table.csv)、[M1单项证据](results/reference_M1_screen.json)。1bpw与FP有7个episode改善、7个退步；净差0pp的单seed配对条件95%区间为**[−5.0,+5.5]pp**。相同总成功数不等于逐episode等价，也不能证明没有性能损失；2bpw略低也不证明1bpw优于2bpw。

**3/2/1均未下降≥3pp，预设网格内没有成功率拐点。** 按原补充规则，自适应消融网格为空，3bpw继续跳过。已撤下尚未开始的依赖作业；一组worker发现空网格后正常退出，没有开始候选拟合。2026-10-01新阶段已明确调整为：先完成M1/M2四套件验证，同时量化骨干，再向低于1bpw探索；原有空网格不再阻塞该新阶段。当前没有把任何档位事后改称实测拐点，(b)–(d)的收益仍未测。


校准-only结构检查（E033）：固定的(b*)最低码平面读取预算，加上每步至少读一层且所有存储平面都须使用，迫使每层只有一步能读全M。126层的校准最优选择在M2/M3均为最后一步，因此候选(b)与(b*)完全相同；若对应档位进入后续网格，将共用结果。这限制了该预算下“学习逐步分配”的独立贡献，不能把同配置的重复运行当作不同方法证据。(b**)及条件码本/affine仍需单独验证；(c)/(d)尚无实测策略收益。

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

按最新addendum，先完整筛选(a)的名义M3/M2/M1，再选实测退化档及上一档；若(a)3bpw已达标，跳过该档(b)–(d)。骨干保持native BF16/FP32，默认int8 embedding。主表配平**实际expert存储bpw**，读取量单独报告。完整参考筛选没有满足退化≥3pp的档位，原定自适应网格为空；后续方向待调整。

| Variant | 组成 | expert Linear实际bpw | whole-model bpw | 压缩逻辑 / 当前缓存逻辑 bytes/inference | Held-out相对MSE | Screening success | Full success / CI |
|---|---|---:|---:|---:|---:|---:|---|
| (a) M3 | uniform / prefix / shared | 3.056851 | 14.115285 | 12,395,449,800 / 17,436,048,840 | 3.10745e-05 | **193/200, 96.5%** | pending |
| (a) M2 | uniform / prefix / shared | 2.043593 | 14.021186 | 12,001,005,000 / 17,436,048,840 | 0.000161955 | **191/200, 95.5%** | pending |
| (a) M1 | uniform / prefix / shared | 1.030335 | 13.927086 | 11,606,560,200 / 17,436,048,840 | 0.0106768 | **193/200, 96.5%** | pending |
| (b) / (b*) | 当前候选分配相同，选中后共用结果 | pending | pending | pending | pending | pending | pending |
| (b**) | 前8步1层，第8步2层，末步全读 | pending | pending | pending | pending | pending | pending |
| (c) | 条件centroids + affine | pending | pending | pending | pending | pending | pending |
| (d) | (c) + subset swap + heterogeneous objective | pending | pending | pending | pending | pending | pending |
| whole model | 最佳expert + SigLIP/Gemma量化 | pending | pending | pending | pending | pending | pending |

原始M1预算40,109,292bytes（1.030335bpw）无法容纳完整(c)/(d)的额外元数据。此前为可能选中这一档，预先固定了最小共同可行预算41,030,892bytes（**1.054010bpw**）用于潜在主比较，并重新分配一个逐步均匀(a)参考；原始M1保留为原始参考，现无成功率退化证据。这一预算方案尚未进入策略实验。所有实际预算和离散分配余量均公开，不加padding、不隐藏metadata。[解析预算预检](results/adaptive_budget_preflight.json)。该调整在任何自适应策略结果之前固定。

原始全M1配置的mask确实相同，但提高到共同预算后(a)/(b)可能在部分层保存两层索引；不能把整个调整后网格当成重复配置。即使(d)仅用M1，异质目标的code重拟合仍可能改变结果。内部(c)的centroid-only、affine-only、both固定相同codes、masks和存储深度，逐项报告不同的元数据开销。读取是固定核算观测下的逻辑张量模型，非DRAM/速度测量；当前使用额外BF16缓存的质量评估路径。

**全评估复用准备已通过（E039）**：用现有M1结果做CPU检查，200个screening回合可原样接入四套件全协议；重复初始化不重复回合，回合内容或控制协议变更会被拒绝。检查固定了控制/预处理/噪声来源及初始状态资产，已有结果未修改。每个晋级配置仍需新跑1800个回合，目前尚未启动。[复用验证摘要](results/full_reuse_summary.json)。

晋级结果的置信区间将使用配对、按 suite 分层的 task/episode bootstrap，明确限定为单 seed 下的区间。作为预先固定的全协议参考，已有 FP seed7 的 **1928/2000=96.4%**，其该口径 95% 区间为 **[93.79875%, 98.30125%]**；这与下文历史“三 seed 均值的 t 区间”含义不同。没有新增 FP rollout（E021）。

(c) 内部会分别测 centroids only、affine only、both；若 (c) 对 (b) 的代理改善不足 **5%**，如实报告并停止该组件的细化。先完成 PTQ 表，再考虑冻结 codes 的 KD 独立行。筛选协议为 spatial + libero_10，各 **10 eps/task、1 seed**；晋级配置使用四套件各 **50 eps/task、1 seed**。完整门槛与实验命令记录在 EXPERIMENTS。

## 当前支持的结论

当前最强的量化证据是：骨干保持native精度、默认int8 embedding时，均匀参考(a)在action-expert实际**1.0303bpw**达到两套件单seed的193/200，与原始FP总成功数相同；此时**whole-model仍为13.9271bpw**。作为较高精度对照，expert2.0436bpw在256个独立held-out观测上相对原始FP动作MSE为1.62e-4，单seed两套件screening为191/200=95.5%，比FP低1个百分点并通过门槛；3.0569bpw为193/200=96.5%。压至1.0303bpw后MSE上升到0.01068，但完整screening仍为193/200=96.5%。原定成功率拐点不存在，D3b预测没有迁移成功。**这是参考量化结果，HD-SR-VQ自提组件的额外收益仍未验证；四套件晋级评估也尚未完成。** D1使相干评分贡献被放弃；D2保留subset试验；稳健D3的早期步骤容忍度仍右删失，采用的是公开说明的保守操作权重。当前读取约束下(b)候选与(b*)相同，学习分配的独立贡献因而受限；条件质心、affine和subset是否有效须等后续匹配存储表。不能声称提高成功率或优于外部方法。

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
