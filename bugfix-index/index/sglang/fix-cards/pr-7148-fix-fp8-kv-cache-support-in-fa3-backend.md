# PR #7148: Fix FP8 KV Cache Support in FA3 Backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7148](https://github.com/sgl-project/sglang/pull/7148) |
| 作者 | guoyuhong |
| 合并日期 | 2025-06-25 |
| labels | (none) |
| 热度 | 👍 5 · 💬 12 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
FA3 后端 + FP8 KV cache 跑 DeepSeek R1 报类型不匹配等错误,修复后可正常启用 FP8 KV cache,KV 容量翻倍。

## 问题现象
作者尝试用 FA3 后端部署 DeepSeek R1(R1 model)并开启 FP8 KV cache 量化,先后遇到两类错误:启动时先出现类型不匹配(type mismatch)相关报错;解决类型不匹配后,运行期间又出现新报错(正文以截图展示,未给出文字版日志)。该问题导致 FP8 KV cache 在 FA3 后端完全不可用,无法获得 KV 容量翻倍的收益。此外正文提到该 PR 顺带修了一个设置 `SGLANG_LOG_EXPERT_LOCATION_METADATA=1` 时的小 bug(同样以截图展示报错)。

## 根因
正文只说明存在类型不匹配错误及随后的运行期错误,未给出逐步根因分析。从 diff 看,机制是:FA3 后端在 `kv_cache_dtype != "auto"` 时无条件把 q 转成 FP8 并做 KV descale,但 FA3 kernel 在 head_dim > 256 时要求 fp16/bf16 数据类型(diff 注释写明「fa3 kernel require fp16 and bf16 data type in this case」);同时 MLA absorbed 路径下从 KV pool 取出的 FP8 k/v/rope 张量未转回 q 的数据类型,导致传入 flash_attn_varlen_func 的 q/k/v 数据类型不一致(由 diff 推断)。

## 修复方案
改 `python/sglang/srt/layers/attention/flashattention_backend.py` 的 `forward_extend`/`forward_decode`:
1. KV scaling 条件增加 `layer.head_dim <= 256`,并在该分支内对 `q_rope`/`k_rope` 也做 `.to(self.kv_cache_dtype)` 转换:
```diff
-        if self.kv_cache_dtype_str != "auto":
+        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
             if layer.k_scale is not None:
                 descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                 k_descale = layer.k_scale.expand(descale_shape)
                 v_descale = layer.v_scale.expand(descale_shape)
             q = q.to(self.kv_cache_dtype)
+            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
+            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
```
2. MLA absorbed 路径将 `kv_cache`(及 MHA 分支的 k/v)在传入 kernel 前 `.to(q.dtype)` 转回 q 的数据类型。
另改 `model_runner.py`:`fp8_e4m3` 在 HIP 上使用原生支持的 `torch.float8_e4m3fnuz`,CUDA 上仍用 `torch.float8_e4m3fn`;并把 expert location metadata 日志从调用 `debug_str()` 改为直接打印对象。测试方面在 `test/srt/test_mla_deepseek_v3.py` 新增 `TestMLADeepseekV3Fa3Fp8Kvcache`(开 `--kv-cache-dtype fp8_e4m3` + `--attention-backend fa3` 跑 GSM8K,断言 accuracy > 0.62)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:FA3 后端所有使用 FP8 KV cache 的路径(forward_extend / forward_decode 的 MHA 与 MLA absorbed 分支)的条件判断与数据类型转换被改动;head_dim > 256 时不再做 FP8 descale/q 转换,行为与之前不同;HIP 平台的 fp8_e4m3 KV cache dtype 选择发生变化;expert location metadata 日志格式变化。GSM8K 测试类原有 CUDA 判断从 `torch.cuda.is_available() and torch.version.cuda` 改为 `is_cuda()`。

## 关联
- https://github.com/sgl-project/sglang/issues/5651

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
