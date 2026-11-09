# PR #32370: Optimize FP32 LM head for bf16/fp16

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ilyasher-harmonic |
| 合并 | 2026-08-11 |
| 分类 | Other(tier: major) |
| Label | run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/32370 |

## 一句话结论
`use_fp32_lm_head` 在 bf16/fp16 输入下改用 bf16xbf16→fp32 的 `torch.mm(out_dtype=fp32)`,LM head matmul 提速 10x。

## 动机
此前 `use_fp32_lm_head` 的实现是把 hidden states 与 LM head 权重各自 cast 成新的 fp32 张量,再做 fp32×fp32→fp32 matmul。fp32 cast 开销大,且 fp32 GEMM 本身比低精度 GEMM 慢;该 PR 在保留 fp32 累加精度的前提下去掉这两处浪费。

## 优化点
在 `logits_processor.py` 的 `_compute_lm_head` 中增加分派:当 hidden states 在 CUDA 上、且与 lm_head.weight 同为 fp16 或 bf16 时,直接调用 `torch.mm(hidden_states, lm_head.weight.T, out_dtype=torch.float32)`,以 bf16×bf16→fp32(带 fp32 accumulator)替代显式 cast + fp32 matmul,不物化 fp32 副本;不满足条件的设备或 dtype 组合保留原显式 fp32 cast 路径。测试侧在 `test_fp32_lm_head.py` 的 probe 机制中补上对 `torch.mm` 的拦截与 `out_dtype` 断言,新增 fp32 输入回退到显式 fp32 matmul 的用例。

## 关键改动

```text
python/sglang/srt/layers/logits_processor.py: _compute_lm_head 中 use_fp32_lm_head 分支改用 torch.mm(out_dtype=fp32),条件不满足时回退
test/registered/rl/test_fp32_lm_head.py: probe 增加 torch.mm 与 out_dtype 校验,新增 fp32 回退用例
```

## 性能声明
H200 上 10x speedup:Before 5.7ms,After 0.56ms。benchmark 代码形状为 M=40, K=4096, N=248320 的 bf16 matmul(WARMUP=3, REPS=10,`torch.cuda.Event` 计时),对比 `explicit_cast_matmul` 与 `torch.mm(out_dtype=fp32)`。除形状与硬件外,PR 未注明对应的具体模型/workload。

## 验证与回归风险
依赖更新后的 `test/registered/rl/test_fp32_lm_head.py`:校验在 CUDA 上 fp16/bf16 输入走 `mm` 且 `out_dtype=fp32`,非 CUDA 或 fp32 输入回退显式 fp32 matmul,flag 关闭时保持原 dtype 路径。风险:仅在 CUDA + 同 dtype(fp16/bf16)时启用新路径,数值上依赖 `out_dtype=fp32` 的 fp32 累加精度,PR 未附端到端精度评测;CI 状态显示 PR Test (Base) 与 (Extra) 均为失败标记。

## 关联
无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
