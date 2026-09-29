# 当前实验报告 — 2026-09-29

**G0 最新状态：passed；完整实测结果与置信区间见文末更新。尚无量化收益结论。**

## 主要发现

- 锁定版本的 `pi05_libero` 实际动作块为 **10×32**，LIBERO 有效 **7** 维，执行窗口 **K=5**；通用默认值 **50×32** 不适用于此配置（E000）。
- 模型构造器含 **3,616,757,520** 个独立参数，其中 **263,323,648** 个属于动作推理未调用的 expert `lm_head`。这部分目前仍计入存储，移除前需要验证（E002）。
- **预算约束：**当前全部构造参数中，规定保留的张量按理想 **16 bit** 计，单独就占 **4.03655 bpw**；可量化 Linear 用 **3 bpw** 时，合计是 **6.27970 whole-model bpw**，且还没算码本等开销。这是基于参数枚举的算术下界，不是实测显存。全模型低位宽目标需要先处理实际存储方案，按步减少解码深度不能自动减少已存储的码平面（E002）。
- ActQuant 官方代码已公开。其 **3.0 backbone bpw** 对应 **5.6 whole-model bpw**，必须区分（**quoted**，E000）。

完整逐层 Linear、AdaRMS 关系和字节模型见 [MODEL_MAP.md](MODEL_MAP.md)。

## 已测量的基础结果

L40S，torch **2.7.1+cu126**，batch **1**；以下为 CUDA event 测得的 BF16 Linear 时间，单位 **ms**。**100** 次预热后，取 **20** 组、每组 **100** 次调用的组均值中位数。合成输入、热缓存测试，非策略端到端延迟（E003）。

| Tokens | 1024→4096 | 4096→1024 |
|---:|---:|---:|
| 1 | 0.007884 | 0.009206 |
| 10 | 0.007864 | 0.010762 |
| 64 | 0.009943 | 0.012328 |

AQLM 已编译并通过正确性检查。补充 CUDA Graph 对比以减少 Python 调用开销（E005）：

| Tokens | 形状 | BF16 graph ms | AQLM graph ms | AQLM/BF16 |
|---:|---|---:|---:|---:|
| 1 | 1024→4096 | 0.005832 | 0.007507 | 1.287× |
| 1 | 4096→1024 | 0.007990 | 0.006536 | 0.818× |
| 10 | 1024→4096 | 0.005897 | 0.023546 | 3.993× |
| 10 | 4096→1024 | 0.006004 | 0.023869 | 3.975× |
| 64 | 1024→4096 | 0.009234 | 0.027812 | 3.012× |
| 64 | 4096→1024 | 0.010795 | 0.029051 | 2.691× |

**当前 AQLM 路径在 64 tokens 下仍慢于 BF16，不能宣称这一场景加速。**单 token 的一个形状有收益，不能推广为端到端收益。VPTQ、W4A16 尚未测量；量化模型的实际存储及成功率也尚未验证（E005）。

LIBERO EGL 仿真已通过：两路 **256×256×3** 图像、**10** 步 dummy action，状态有限。该检查没有运行策略，不是任务成功率（E001b）。

## 资产和门槛

| 项目 | 当前状态 | 证据 |
|---|---|---|
| 官方 checkpoint | 16 个对象，12,439,085,481 bytes，已校验 | E000 |
| 校准数据 | 60 条训练轨迹，覆盖 40 个任务，1,205,270,267 parquet bytes | E000 |
| 校准观测 | 固定 256 个观测 × 8 个噪声种子；尚未缓存激活 | E001b |
| PyTorch 转换 | 96 GiB 重跑完成，实际使用的参数与归一化资产检查通过 | E004b |
| 策略冒烟 | 有限动作，固定噪声重复差异 0；单任务真实闭环 2/2 成功，仅为冒烟 | E006 |
| G0 | passed；完整结果见文末 | G0 evaluation update |
| G1 / G1b / G2 | not started | 实验记录 |

官方平均 **96.85%** 是 **quoted**，不是本项目实测（E000）。

## 当前支持的结论与下一步

当前证据支持：实验资产与 GPU 仿真可用，并已发现必须处理的配置、转换和全模型位宽预算问题；尚无 HD-SR-VQ 效果结论。

下一步：检查完整 BF16 G0 结果并在失败时停留调试；补齐 VPTQ/W4 内核对比；G0 通过后再推进诊断及含保留参数开销的量化基线。G0 结束后会自动追加统计到此仓库（E007）。

命令、代码版本、失败原因与完整数字见 [EXPERIMENTS.md](EXPERIMENTS.md)；模型权重、数据、实现代码和原始日志保存在本地 `/projects/yanlongvla`。

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
