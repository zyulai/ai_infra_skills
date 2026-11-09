# PR #39315: [Bugfix] FlashInfer MXINT4 MoE crashes, missing do_finalize

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39315](https://github.com/vllm-project/vllm/pull/39315) |
| 作者 | benchislett |
| 合并日期 | 2026-04-09 |
| labels | bug, nvidia, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
FlashInfer 0.6.4 更换 fused MoE 接口后 MXINT4 路径未跟进(缺 do_finalize、返回值处理),`VLLM_USE_FLASHINFER_MOE_INT4` 自 v0.15.1 起不可用;补齐调用修复。

## 问题现象
启用 `VLLM_USE_FLASHINFER_MOE_INT4` 时 FlashInfer MXINT4 MoE 崩溃,该 flag 自 vLLM v0.15.1 起一直不可用(修复前)。修复后(vLLM 0.19.0)flag 开启可正常启动并跑通评测:GSM8k 1319 题 5-shot 精度 0.939(关闭 flag 时 0.936),1k1k serving benchmark 320 请求、并发 32,总时长 210.17 s(关闭时 218.74 s)——作者报告 1k1k 约提速 4%。

## 根因
正文披露:FlashInfer 0.6.4 更改了 fused MoE 后端的接口,MXINT4 路径没有随之更新,导致调用即崩。具体缺失(由 diff 推断):调用 `trtllm_mxint4_block_scale_moe` 时未传 `do_finalize=True`,且新接口可能返回 tuple/list,代码直接对返回值 `.to(x.dtype)`,形状/类型不匹配即崩溃。

## 修复方案
改 `vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py` 的 `flashinfer_trtllm_mxint4_moe`:给 `trtllm_mxint4_block_scale_moe` 调用补 `do_finalize=True`;返回值若为 tuple/list 则取第一个元素;再统一 `.to(x.dtype)`。新增测试 `tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py::test_flashinfer_trtllm_mxint4_moe_wrapper`,在多组 (m,n,k,e,topk) 参数下断言 wrapper 输出与裸 kernel 调用完全一致(atol=0, rtol=0)。

```diff
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

## 影响与回归风险
- 影响窗口:正文说明自 v0.15.1(FlashInfer 0.6.4 接口变更之后)该路径即不可用,直至本修复。
- 回归面:仅 FlashInfer MXINT4 MoE wrapper(`flashinfer_mxint4_moe.py`);显式 `do_finalize=True` 恢复 finalize 步骤,输出语义与裸 kernel 一致(测试以零容差对齐);其他量化路径(NVFP4、FP8、Marlin)未动。

## 关联
- [#39245](https://github.com/vllm-project/vllm/issues/39245)(正文 "FIX" 指向的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
