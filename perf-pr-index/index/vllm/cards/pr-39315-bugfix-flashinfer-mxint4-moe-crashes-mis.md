# PR #39315: [Bugfix] FlashInfer MXINT4 MoE crashes, missing do_finalize

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2026-04-09 |
| 分类 | MoE \| tier: major |
| Label | bug, ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/39315 |

## 一句话结论

补齐 do_finalize 修复 MXINT4 MoE 崩溃，恢复 VLLM_USE_FLASHINFER_MOE_INT4；启用后 1k1k 约快 4%。

## 动机

FlashInfer 0.6.4 变更了 fused MoE 后端接口，MXINT4 路径没有同步更新，导致 `VLLM_USE_FLASHINFER_MOE_INT4` 自 vLLM v0.15.1 起不可用（崩溃）。本 PR 修复 issue #39245，让该量化路径重新可启用。属于接口适配型 bugfix，而非新增算法。

## 优化点

改动集中在 `flashinfer_trtllm_mxint4_moe` wrapper：调用底层 `trtllm_mxint4_block_scale_moe` 时补传 `do_finalize=True` 以匹配 FlashInfer 0.6.4 新接口；同时兼容返回值可能为 tuple/list 的情况（取 `out[0]`）后再做 `.to(x.dtype)` 类型转换。此前代码直接对新接口返回值调用 `.to()`，缺少 do_finalize 引发崩溃。另新增一个对比 wrapper 与裸 kernel 的单元测试。

## 关键改动

- `vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py`：核心修复，补齐 `do_finalize=True` 并解包返回值后再转 dtype。
- `tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py`：新增 `test_flashinfer_trtllm_mxint4_moe_wrapper`，对比 wrapper 与直接调用 `trtllm_mxint4_block_scale_moe`（bf16，m∈{1,33}、n=7168、k=512、e=384、topk=8，标注 Skip for non SM100）。

```text
         routed_scaling_factor=None,
         routing_method_type=routing_method_type,
         enable_pdl=None,
+        do_finalize=True,
         output=None,
         tune_max_num_tokens=8192,
-    ).to(x.dtype)
+    )
+    if isinstance(out, (tuple, list)):
+        out = out[0]
+    out = out.to(x.dtype)
```

## 性能声明

PR 正文 Test Plan 称：启用该 flag 后服务正常启动、GSM8k 通过，"1k1k runs ~4% faster with this flag enabled"。Test Result 给出 vLLM 0.19.0 下 `VLLM_USE_FLASHINFER_MOE_INT4=0` 与 `=1` 两组对照：GSM8K（1319 题、5-shot）accuracy 0.936 → 0.939，output tokens/s 908.814 → 888.617；serving benchmark（10 请求、并发 1）output token throughput 129.02 → 134.03 tok/s、Mean TTFT 120.90 → 127.76 ms、Mean TPOT 7.65 → 7.35 ms；（320 请求、并发 32）output token throughput 1512.62 → 1574.30 tok/s、total token throughput 3021.29 → 3144.48 tok/s、Mean TPOT 20.28 → 19.52 ms。除版本与 workload 外，硬件与精度 PR 未注明测试条件。

## 验证与回归风险

验证：新增 kernel 单测对比 wrapper 与底层 kernel（SM100）、GSM8K 精度对照、两组 serving benchmark。可能受影响路径：FlashInfer MXINT4 MoE 路径、`VLLM_USE_FLASHINFER_MOE_INT4` 开关、SM100（Blackwell）设备；正文未说明对其它量化后端或其它硬件的回归测试。两组 GSM8K accuracy 接近（0.936 / 0.939）。

## 关联

PR 正文仅提及修复 issue #39245 及 FlashInfer 0.6.4 / vLLM v0.15.1 版本，未给出同区域其他 PR 编号，无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
