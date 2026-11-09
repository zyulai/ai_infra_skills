# PR #7635: Apply dsv3_fused_a_gemm kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2025-07-02 |
| 分类 | Attention / Kernel(major) |
| Label | (none) |
| Diff | diffstat 缺失(素材未提供统计);diff 实际仅 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/7635 |

## 一句话结论

DeepSeek MLA 的 fused_qkv_a_proj 小批量路径换用 dsv3_fused_a_gemm 专用 kernel,bs1 吞吐 +6%。

## 动机

前序 PR #7630 已实现并剖析了 `dsv3_fused_a_gemm` kernel(profile 与 accuracy 数据在其中),本 PR 把它接入 DeepSeek 模型前向。DeepSeek-V3/R1 的 MLA 吸收路径中 `fused_qkv_a_proj_with_mqa` 是一个 2112×7168 的 BF16 GEMM,小 batch(≤16 行)时属于低延迟敏感的小 GEMM,通用 linear 路径不是最优。PR 依赖发新版 sgl-kernel 提供 `dsv3_fused_a_gemm` 入口。

## 优化点

在 `DeepseekV2AttentionMLA.__init__` 增加形状门控 `use_min_latency_fused_a_gemm`:仅当模型有 `fused_qkv_a_proj_with_mqa`、权重为 BF16、shape 为 [2112, 7168](即 DSv3 系列)、CUDA 且 SM ≥ 90 时启用。`forward_absorb_prepare` 中,当 `hidden_states.shape[0] <= 16`(小批量/decode 低延迟场景)时,把原来的 `nn.Linear` 前向替换为对 `weight.T` 直接调用 sgl-kernel 的 `dsv3_fused_a_gemm`;大批量仍走原路径。kernel 本体的改动在 #7630,本 PR 是纯接入。

## 关键改动

```text
python/sglang/srt/models/deepseek_v2.py: 形状门控 + forward_absorb_prepare 小批量切换到 dsv3_fused_a_gemm
```

```python
self.use_min_latency_fused_a_gemm = (
    hasattr(self, "fused_qkv_a_proj_with_mqa")
    and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.bfloat16
    and self.fused_qkv_a_proj_with_mqa.weight.shape[0] == 2112
    and self.fused_qkv_a_proj_with_mqa.weight.shape[1] == 7168
    and is_cuda
    and get_device_sm() >= 90
)
...
if hidden_states.shape[0] <= 16 and self.use_min_latency_fused_a_gemm:
    fused_qkv_a_proj_out = dsv3_fused_a_gemm(
        hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T
    )
```

## 性能声明

E2E bench_serving,DeepSeek-R1-FP4(modelopt_fp4),TP8,random 输入/输出各 1000 token,range-ratio 0.98:

| max_concurrency | 输出吞吐 main → 本 PR | mean TPOT main → 本 PR |
| ---: | ---: | ---: |
| 1 | 82.011 → 86.853 tok/s | 12.026 → 11.342 ms |
| 4 | 273.061 → 283.627 tok/s | 14.269 → 13.720 ms |
| 16 | 784.243 → 791.398 tok/s | 19.982 → 19.784 ms |

作者结论:"6% improvement on bs1"。kernel 级 profile 数据在 #7630,本 PR 未重复给出。

## 验证与回归风险

PR 正文声明 profile 与 accuracy 测试已在 #7635 引用的前序 PR #7630 列出,本卡素材未包含其细节。回归风险:门控条件把改动严格限定在 DSv3 形状 + BF16 + SM≥90 + ≤16 行的组合上,其他模型/精度/批量仍走原 linear;若 `dsv3_fused_a_gemm` 对非连续或特殊 stride 输入行为不同,受影响的是 DeepSeek 系列 decode 前向。依赖新版 sgl-kernel 发布。

## 关联

前序:#7630(kernel 引入与 profile)。同区域无其他编号被本 PR 正文提及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
