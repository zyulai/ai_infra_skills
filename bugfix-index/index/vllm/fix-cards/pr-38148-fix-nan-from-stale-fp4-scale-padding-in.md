# PR #38148: Fix NaN from stale FP4 scale padding in create_fp4_scale_tensor

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38148](https://github.com/vllm-project/vllm/pull/38148) |
| 作者 | elvircrn |
| 合并日期 | 2026-04-01 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
FP4 scale 张量用 torch.empty 留下脏 padding,被 mm_fp4 kernel 读入致 NaN;改 torch.zeros 修复。

## 问题现象
Blackwell(GB200)上 NVFP4 量化的 MoE 模型出现 NaN 污染:prefill 期间 MoE 层输出 sporadic NaN——收到 ≤32 个 token 的 expert(256 expert 时很常见)会走 `use_8x4_sf_layout` 路径触发;NaN 随后级联污染后续所有层、污染 KV cache 条目,并经 KV 传输(NIXL)传播到 decode server。作者在 GB200 集群用 DeepSeek-R1-0528-NVFP4-v2 复现,定位 NaN 起源于 prefill 时 MoE 第 41 层(输入干净、过 MoE 后变 NaN)。

## 根因
正文披露:`create_fp4_scale_tensor` 用 `torch.empty` 分配 swizzled scale 张量,当行数 `m` 小于 `rounded_m`(按 128 对齐 tile 边界)时,padding 行的 scale 是陈旧 GPU 内存;若其中恰为 FP8 NaN(`float8_e4m3fn` 下的 `0x7F`),TRT-LLM `mm_fp4` kernel 在 `use_8x4_sf_layout=True`(`m <= 32` 时触发)下会读取这些 padding scale 并应用到真实行,产生 NaN 输出。

## 修复方案
在 `vllm/_custom_ops.py` 的 `create_fp4_scale_tensor` 中,把两处 `torch.empty` 改为 `torch.zeros`(swizzled 分支与普通分支)。零 scale 保证 padding 参与计算时贡献 `0 × data = 0`,中和 kernel 读取 padding 的行为:

```diff
-        return torch.empty(
+        return torch.zeros(
             (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
         )
```

## 影响与回归风险
- 影响窗口:正文未给出引入问题的提交/版本范围;可确认受影响的是走 `create_fp4_scale_tensor` 且 `m` 非零填充路径的 NVFP4 推理。
- 回归面:仅 FP4 scale 张量的分配方式;`torch.zeros` 比 `torch.empty` 多一次清零,在分配上有轻微开销,数值路径上 padding 贡献归零、真实行不受影响。作者已验证修复中和了 padding scale 污染;现有 FP4 单测在测试计划中尚未勾选跑完。

## 关联
- FlashInfer issue: https://github.com/flashinfer-ai/flashinfer/issues/2861

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
