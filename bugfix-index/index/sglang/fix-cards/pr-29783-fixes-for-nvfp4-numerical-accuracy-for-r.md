# PR #29783: Fixes for NVFP4 numerical accuracy for router GEMM output and wrong correction bias cast

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29783](https://github.com/sgl-project/sglang/pull/29783) |
| 作者 | b8zhong |
| 合并日期 | 2026-07-06 |
| labels | deepseek, run-ci |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
NVFP4 DeepSeek 路由两处精度缺陷:correction bias 误转 bf16、router GEMM 输出缺 fp32;修复后 accept len 3.413→3.894。

## 问题现象
正文以效果数据描述现象:NVFP4(modelopt_fp4)量化 DeepSeek 模型接受长度(accept len)偏低。quick testing 中 accept len 从 3.413 提升到 3.894;OSL 2048 用 generate 测试:修复前 len = 3.459, tps = 304.65;修复后 accept len = 3.842, tps = 334.12;再加本 PR 第二个 commit 后 len = 3.879, tps = 335.94(基本持平)。即路由精度差导致投机解码接受率低、吞吐受损。

## 根因
正文披露两点根因:
1. correction bias 从 fp32 cast 成 bf16 是错误的——加载 checkpoint 数值时即可确认这些值的精度发生了变化(此前 diff 中 modelopt_fp4 + flashinfer_trtllm 后端分支把 `correction_bias_dtype` 设为 `torch.bfloat16`)。
2. router GEMM 输出需要 fp32 以保证数值稳定性,trtllm 就是这么做的;计算 dtype 仍为 bf16,因此性能损失应当极小。此前 CUDA 非 deepseek-v4 路径用 `F.linear` 做 bf16 GEMM,输出仍是 bf16。

## 修复方案
`python/sglang/srt/models/deepseek_v2.py` 两处:
1. `__init__`:删除 modelopt_fp4 + flashinfer_trtllm 后端时 `correction_bias_dtype = torch.bfloat16` 的分支,使该场景 correction bias 保持 fp32(与 noaux_tc 的默认一致)。
2. `forward` 的 router GEMM 分支:删除 `_is_npu` 走 `F.linear` 的分支,改为 `elif not _is_cuda` 才走 `F.linear`;CUDA 路径统一改用 `sglang.jit_kernel.dsv4.linear_bf16_fp32`(cuBLAS bf16 x bf16 -> fp32 GEMM,注释注明 torch.mm 的 out_dtype kwarg 仅 CUDA 支持),不再区分是否 deepseek v4——原代码仅 v4 用 fp32 输出、其余走 bf16 `F.linear`。

```diff
-                if (
-                    quant_config.get_name() == "modelopt_fp4"
-                    and get_moe_runner_backend().is_flashinfer_trtllm()
-                ):
-                    correction_bias_dtype = torch.bfloat16
-                elif _use_aiter and quant_config.get_name() in (
+                if _use_aiter and quant_config.get_name() in (
...
-                else:
-                    # After testing, we may use the faster code in `if deepseek v4` branch
-                    logits = F.linear(hidden_states, self.weight, None)
+                logits = linear_bf16_fp32(hidden_states, self.weight)
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:DeepSeek-V2/V3 系列路由(topk_method=noaux_tc 的 MoE router)路径:modelopt_fp4 + flashinfer_trtllm 场景 correction bias dtype 变回 fp32;所有 CUDA 非 aiter 路由 GEMM 改经 `linear_bf16_fp32`(JIT kernel),性能与数值与原 `F.linear` 不同;`_is_npu` 路径语义变为 `not _is_cuda`(含 NPU 之外的其他非 CUDA 设备)。

## 关联
正文未列出(仅含 CI 运行记录链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
