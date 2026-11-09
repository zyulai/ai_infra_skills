# PR #11081: Fix DSR1 accuracy for flashinfer_trtllm MoE with FP8 quantization

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11081](https://github.com/sgl-project/sglang/pull/11081) |
| 作者 | trevor-m |
| 合并日期 | 2025-09-30 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

flashinfer_trtllm MoE 的 w13→w31 权重映射此前只对 NVFP4 生效,FP8 量化也需该映射,补上后 GSM8K 从 0.025 恢复到 0.959。

## 问题现象

DeepSeek-R1-0528 以 `--moe-runner-backend flashinfer_trtllm --quantization fp8 --tp 8` 服务时,GSM8K(8-shot,1319 题)精度崩塌:修前 Accuracy 0.025、Invalid 0.067;修后 Accuracy 0.959、Invalid 0.000(吞吐 9207→5488 token/s,时延 73.3→26.0 s)。

## 根因

正文披露:#10758 引入的改动把 w13→w31 权重映射(flashinfer 假定 `w13_weight` 实为 w31 格式,scales 同理)限定为仅在 NVFP4 量化(`ModelOptNvFp4FusedMoEMethod`)下应用;但 flashinfer_trtllm 同样支持 FP8 量化,也需要这一映射,否则权重布局错误导致 GSM8K 精度归零。

## 修复方案

改 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的 `_weight_loader_impl`:w13→w31 映射条件由「flashinfer_trtllm 且量化方法类名 == ModelOptNvFp4FusedMoEMethod」放宽为「flashinfer_trtllm 且(quant_method 是 ModelOptNvFp4FusedMoEMethod **或** Fp8MoEMethod)」:

```diff
-        if (
-            should_use_flashinfer_trtllm_moe()
-            and self.quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoEMethod"
+        if should_use_flashinfer_trtllm_moe() and (
+            isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
+            or isinstance(self.quant_method, Fp8MoEMethod)
         ):
             shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]
```

(顺带由类名字符串比较改为 isinstance 判断。)另在 `python/sglang/srt/server_args.py` 更新断言信息:`modelopt_fp4 quantization is required for Flashinfer TRTLLM MoE` → `modelopt_fp4 or fp8 quantization is required for Flashinfer TRTLLM MoE`。

## 影响与回归风险

- 回归面:`moe_runner_backend=flashinfer_trtllm` 下的 MoE 权重加载——FP8 MoE checkpoint 的 w13/w31 权重与 scales 加载顺序翻转;NVFP4 路径行为不变(isinstance 与原类名判断等价)。其他 runner backend 不受影响。

## 关联

- #10758(引入该限定、造成回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
