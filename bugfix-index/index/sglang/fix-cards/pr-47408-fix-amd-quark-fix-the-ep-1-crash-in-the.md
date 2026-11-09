# PR #47408: [Fix] [AMD] quark: Fix the EP>1 crash in the Quark MXFP4 W4A16 MoE path and restrict it to gfx1151

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47408](https://github.com/sgl-project/sglang/pull/47408) |
| 作者 | yichiche |
| 合并日期 | 2026-10-31 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 5 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
Quark MXFP4 MoE W4A16 路径 EP>1 时 -1 专家 id 直接 scatter_add 越界致 GPU 崩溃;映射额外桶并置零,路径收窄到 gfx1151。

## 问题现象
用 `--ep-size` > 1 在 W4A16 路径(#41389 引入)上服务 Quark MXFP4 MoE checkpoint 时,GPU 直接 abort,报 `HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016`。所有走 W4A16 路径且 EP>1 的 GPU 都会触发:#41389 中 gfx1151 与 gfx942 均有报告,本 PR 在 gfx1151 上复现。此外该路径原本在所有非 gfx95x/gfx1250 的 HIP GPU(含 gfx942/MI300X)上都会启用,但它只为 gfx1151 编写和验证。附带问题:带 clamp/alpha 参数的 SwiGLU 变体(如 gpt-oss、DeepSeek-V4、MiniMax-M3)能通过 `activation == "silu"` 检查,跑出静默错误结果。

## 根因
正文披露:专家并行下标准 dispatcher 发给每个 rank 的 `topk_ids` 中,其他 rank 持有的专家 id 为 `-1`,而 `_apply_triton_w4a16` 把这些 id 直接喂给 `counts.scatter_add_(...)`,越界写入导致 GPU 硬件异常。正文同时披露另外两处问题:启用门槛过宽(「HIP 且非 gfx95x/gfx1250」而非仅 gfx1151);不支持的 SwiGLU 变体未被拒绝而静默算错。

## 修复方案
改动集中在 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` 及 `environ.py`:
- EP 修复:`-1` id 映射到额外桶 `num_experts`(排序在最后),`counts` 多分配一槽并在 `make_ragged_tensor_metadata` 前丢弃,非本 rank 的散布行在 top_k 求和前用 `masked_fill` 置零;全部在 device 上完成,CUDA graph 捕获不受影响。
```diff
+        valid = flat_ids >= 0
+        flat_ids = torch.where(valid, flat_ids, self.num_experts)
-        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=x.device)
+        counts = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=x.device)
-        ragged = make_ragged_tensor_metadata(counts, num_tokens * top_k)
+        ragged = make_ragged_tensor_metadata(counts[:-1], num_tokens * top_k)
+        out = out.masked_fill(~valid.view(num_tokens, top_k, 1), 0).sum(dim=1)
```
- 门槛:`_use_triton_w4a16` 改为仅 `gfx1151` 且 `SGLANG_ENABLE_QUARK_MXFP4_MOE_W4A16`(新注册于 `environ.py` 的 `EnvBool(True)`),gfx942 等回退 AITER 路径。
- `_process_weights_triton_w4a16` 对 `gemm1_alpha`/`gemm1_beta`/`gemm1_clamp_limit`/`swiglu_limit` 非空即抛 `NotImplementedError`;并 `logger.info_once` 提示 W4A4 checkpoint 以 W4A16 运行及关闭方式。
- 新增单元测试 `test/registered/unit/layers/quantization/test_quark_w4a4_mxfp4_moe_w4a16.py`。

## 影响与回归风险
- 影响窗口:环境变量旧名 `SGLANG_QUARK_MXFP4_MOE_W4A16` 只存在于 #41389 分支,从未进入 release(正文说明)。
- 回归面:gfx1151 上每次 MoE 调用新增 `torch.where`、一个额外桶和一次 `masked_fill`(ep=1 时为恒等操作);正文基准显示输出吞吐 -1.3%(并发 1)到 -3.0%(并发 16)、TTFT +7.1%~+8.1%,单次运行未能与噪声区分。gfx942/MI300X 等回退 AITER 路径。ep=1 下 MoE 输出与修复前 bit-identical;GSM8K 63.0%→60.0% 的 3 点差距被作者归因于温度 1.0 采样噪声(200 题标准误约 3.4 点)。

## 关联
- [#41389](https://github.com/sgl-project/sglang/pull/41389)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
