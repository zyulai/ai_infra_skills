# PR #42224: [Perf] [NPU] dsa: Add INT8 LightningIndexer to accelerate indexing on A2/A3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @konst42r |
| 合并 | 2026-11-16 |
| 分类 | Quantization (tier: major) |
| Label | documentation, npu, run-ci |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 14 个文件,无新文件,含文档与 3 个测试文件修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/42224 |

## 一句话结论

为 Ascend A2/A3 NPU 新增 `--indexer-kv-cache-dtype int8`:把 DSA lightning-indexer 的 K cache 存为 INT8(FP16 scale)并走 NPU 原生 `npu_quant_lightning_indexer`,indexer 提速 2x,长上下文 GSP 负载平均 TTFT 从 67402 降到 49125(约 −25%)。

## 动机

A2 与 A3 NPU 不支持 FP8 LightningIndexer,FP16 是唯一实用精度,indexer 因此成为长上下文负载的瓶颈。本 PR 为 NPU LightningIndexer 增加 int8 量化选项,通过 `--indexer-kv-cache-dtype int8` 显式开启、默认关闭,保持既有行为不变。

## 优化点

- 新 server 参数 `--indexer-kv-cache-dtype`(choices 仅 `int8`,默认 None):DSA lightning-indexer KV cache 的量化配方(仅 Ascend NPU)。
- 新 resolution-pipeline hook `handle_indexer_kv_cache_compatibility`:显式请求配方时快速失败而非静默降级——与 `--enable-hierarchical-cache` 组合直接拒绝;仅限 NPU(`npu_quant_lightning_indexer` 是 CANN op);仅限实际带 lightning indexer 的 DSA 模型。
- `resolve_indexer_quant_mode()`:把参数映射为 indexer quant mode;INT8 配方只能建在非 FP8 主 KV cache 之上(FP8 主 cache 的 indexer cache 已经量化,`int8` 叠加 `--kv-cache-dtype fp8_e4m3` 被拒)。
- `NPUMLATokenToKVPool` 接受 `indexer_quant_mode`:INT8 时 `index_k_store_dtype = torch.int8`、`index_k_scale_dtype = torch.float16`(FP8 模式保持 FP32 scale),两种量化模式都创建 scale buffer,index-K setter 断言期望 dtype。
- `dsa_npu_indexer`:INT8 key cache 时 indexer 权重转 FP16(INT8 kernel 要求 FP16 权重),op 经 `torch.ops.npu.npu_quant_lightning_indexer` 调用(`torch_npu` 绑定在导入 custom_ops 时对 INT8 cache 失常)。
- `pool_configurator` 修正 INT8 配方的 indexer cache 内存 sizing:每元素 1 字节加每 token 一个 FP16 scale(`index_head_dim + 2` 字节/token 而非 BF16 大小,`index_head_dim=128` 时 130 vs 256 字节/token)。

## 关键改动

```text
python/sglang/srt/mem_cache/kv_cache_dtype.py:
+QUANT_MODE_TOKEN_INT8 = 0
+_INDEXER_KV_CACHE_DTYPE_TO_QUANT_MODE = {"int8": QUANT_MODE_TOKEN_INT8}
python/sglang/srt/hardware_backend/npu/memory_pool_npu.py:
+        self.indexer_kv_store_int8 = (
+            indexer_quant_mode == QUANT_MODE_TOKEN_INT8
+            and not self.dsa_kv_cache_store_fp8
+            and index_head_dim is not None
+        )
+        if self.indexer_kv_store_int8:
+            self.index_k_store_dtype = torch.int8
+            self.index_k_scale_dtype = torch.float16
python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py:
+                if pool.index_k_store_dtype == torch.int8:
+                    # INT8 keys require FP16 weights
+                    op_weights = weights.to(torch.float16)
+                ...
+                topk_indices = torch.ops.npu.npu_quant_lightning_indexer(
python/sglang/srt/model_executor/pool_configurator.py:
+                if indexer_quant_mode == QUANT_MODE_TOKEN_INT8:
+                    indexer_size_per_token = (
+                        index_head_dim + torch._utils._element_size(torch.float16)
+                    )
+                    element_size = 1
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实;端到端硬件型号正文未标注,代码路径为 Ascend NPU A2/A3。)

- "The int8 lighting indexer itself is 2x faster than fp16"。
- Accuracy(GSM8K,5 runs,GLM5.2-w4a8-int):INT8 93.9 / 94.1 / 94.2 / 94.5 / 94.6(%);FP16 baseline 94.4 / 94.5 / 94.5 / 94.5 / 94.7(附截图)。
- 端到端:`python3 -m sglang.benchmark.serving --gsp-prompts-per-group 17 --host 127.0.0.1 --port 30000 --dataset-name generated-shared-prefix --gsp-system-prompt-len 180000 --gsp-question-len 10828 --gsp-output-len 1 --gsp-num-groups 1 --backend sglang-oai-chat`;"average TTFT reduces from 67402 to 49125 (~-25%)"(FP16 与 INT8 的结果以截图给出)。

## 验证与回归风险

单测(test_kv_cache_configurator_npu_indexer.py 扩充):TestNPUIndexerMemorySizing 校验 cell size(None+bf16 → 256、int8+bf16 → 130、None+fp8 → 132);TestNPUIndexerInt8Cache 覆盖 hierarchical cache 双重拒绝(args 层与 pool 层)、FP8 主 cache 即使直接传入 quant mode 也保持 FP8 indexer、resolve_indexer_quant_mode 的接受/拒绝矩阵、空输入 scale 保持 FP16;另更新 test_npu_index_k_elision.py 与 test_npu_flash_mla.py 的 server_args stub。CI 状态:Base 通过,Extra 失败,AMD 通过。回归风险:(1) INT8 与 hierarchical cache 不兼容,两处拒绝;(2) `do_npu_cp_balance_indexer` 对量化 indexer cache 未实现,触发即 RuntimeError;(3) `torch.ops.npu` 与 `torch_npu` 别名的差异依赖 custom_ops 行为,注释明言绑定 "is not working for int8 cache in case of custom_ops imported";(4) 每次 forward 的权重 FP16 转换开销与 scale dtype 断言(set_index_k_scale_buffer)。

## 关联

同区域:`QUANT_MODE_TOKEN_INT8` 注释提到 v2 `quant_lightning_indexer` 的 FP8/MXFP8/FP4 quant 模式(INT8 走 v1 op,quant_mode 参数保持 0);`--indexer-kv-cache-dtype` 已写入 docs 的 server_arguments.mdx。本批次内无直接依赖项。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
