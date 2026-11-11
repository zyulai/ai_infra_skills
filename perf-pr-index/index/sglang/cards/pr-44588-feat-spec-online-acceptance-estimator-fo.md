# PR #44588: [Feat] spec: Online acceptance estimator for DSpark confidence-scheduled verification

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Qiaolin-Yu |
| 合并 | 2026-11-11 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, deepseek, speculative-decoding, blackwell, npu, run-ci, jit-kernel, bypass-fail-fast |
| Diff | +607/-62, 11 files（pack diffstat 缺失且 diff 在 40k 截断、注明另有 10 file(s) dropped；计数按 pack 保留的 diff 文本） |
| 链接 | https://github.com/sgl-project/sglang/pull/44588 |

## 一句话结论

在线拟合的 logistic 接受率估计器替代训练 confidence head 驱动 DSpark compact 验证调度，c=256 时较 static TPOT 快 8%。

## 动机

DSpark 的 confidence-scheduled（compact / cap-accept）验证按每位置接受置信度在 target verify 前裁剪低存活 draft token，但此前置信度只能来自训练好的 confidence head；没有 head 的 draft（以及将来的 DFlash 等其他 drafter）无法使用该调度器。本 PR 加入服务中在线拟合 draft 分布的 logistic 模型作为替代置信源，合并了 #44866（binned MAP 拟合、DSpark 接线、padding 修复）与 #46182（无相消的 logit(q_max) kernel、drafter 无关估计器 API）的工作。首版只接 DSpark，在 DeepSeek-V4.1-Flash 上测试。

## 优化点

估计器按 `sigmoid(slope·logit(q_max_k) + intercept[k])` 为 draft 位置 k 打分；特征只读取位置 k 的 draft 分布（依赖之前的 draft、与被采样 token 无关），在其上裁剪保持无损，与 head 相同。三个 Triton kernel（均有 torch 参考、全部 device tensor 状态、不与 host 同步、可写 caller buffer 以进 CUDA graph）：predict 分词表块算 logit(q_max) 并把非最大和与最大值分开，避免 confident 行相消；accumulate 按 (position, feature bin) 统计 label；refit 单 program 完成带高斯先验的罚项 MLE（Newton 迭代 + line search，历史指数衰减），每 100 observed 步一次。DSpark 在 draft graph tail（或 eager proposal）预测、worker 在 accept 后 grading；label 覆盖 verify_lens−1 内的 draft（window 之后的 draft 读 padding，不得计为 rejection，否则接受率被低估、预算塌缩）；compact padding 行改为点质量分布。启用估计器时与 sharded accept、NVIDIA megakernel draft sampler、`SGLANG_DSPARK_USE_GREEDY_DRAFT`、int8 Markov walk 互斥（它们不保留全词表 step-logits 行）。`SGLANG_DSPARK_USE_ONLINE_ESTIMATOR=1` 可在 checkpoint 有 head 时仍选估计器；static verification 两者都不用。

## 关键改动

```text
python/sglang/srt/speculative/online_acceptance_estimator.py: drafter 无关的估计器（predict/observe/refit）
python/sglang/kernels/ops/speculative/acceptance_estimator.py: 三个 Triton kernel + torch 参考
python/sglang/srt/speculative/dspark_components/dspark_draft.py / dspark_draft_sampler.py: 提议链路接入
python/sglang/srt/speculative/dspark_components/dspark_config.py: builds_confidence_head 选择逻辑
python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py: point_mass_fill（padding 行变点质量）
```

```python
feature = torch.where(rest > 0, -torch.log(rest), MAX_LOG_ODDS)   # logit(q_max)
predictions_out.copy_(torch.sigmoid(slope * features + intercepts))
```

## 性能声明

DeepSeek-V4.1-Flash、4×B300、TP4/EP4、DSpark block 5，SPS 表按此设置 profile。步长（每请求验 6 token、只验 20%）：bs64 15.9 vs 全验 23.4 ms、bs128 22.2 vs 30.1、bs256 29.1 vs 42.1——裁剪省 25–30% 步时。AIME 2026（`bench_serving --backend sglang-oai-chat`，temp 1.0 top-p 0.95，reasoning max，8192 输出上限，单次运行）：c=256 时 compact+online TPOT 8.09 ms / 25063 tok/s 对 static 8.81 / 23089（+8% TPOT、+8.6% 吞吐，与 compact+head 相当）；c=64 时两个 compact 臂都落后 static ~2.5%（裁剪省的少于损失的接受 token）。高接受率 workload（accept ~5.1/6）各臂差 ~3% 以内。估计器开销：B300、vocab 129280、5 位置、bf16 下 predict 3.2 µs（bs1）/ 10.8（bs16）/ 34（bs64），含 observe 约 50 µs/step（bs64）。GSM8K 200 题（temp 0）：static 0.975 / compact+head 0.98 / compact+online 0.985；aime26 0.967。

## 验证与回归风险

新增 CPU 单测（特征/预测、label 规则、向生成模型收敛、饱和校准、DSpark grading、head-or-estimator 选择）与 1-GPU kernel 测试（Triton 对 torch 参考、CUDA 上收敛、CUDA graph 捕获下 refit 后预测）；复用 #44866 的 `test_ragged_cutoff_preserves_distribution` 与 kernel parity 点质量用例；B300 上现有 DSpark/ragged-verify/server-args 套件 97 通过。每次 refit 后各位置平均预测与观测接受率差 <0.001。风险：与 sharded accept、megakernel sampler、greedy draft、int8 markov walk 互斥；仅接 DSpark、仅在 DeepSeek-V4.1-Flash 验证；compact+head 的 gsm8k 分数来自 rebase 前。

## 关联

合并 #44866 与 #46182 两条前序线；同分类 DFlash host 路径优化见 #47685。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
