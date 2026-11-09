# PR #15904: [NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @OrangeRedeng |
| 合并 | 2026-01-28 |
| 分类 | MoE (tier: major) |
| Label | documentation, quant, deepseek, npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/15904 |

## 一句话结论

Ascend NPU 上给非量化 MoE 权重加 ND→FRACTAL_NZ 转换加速 GroupedMatmul(平均约 11%),并把 transpose 移到加载后、修掉 Qwen3 MoE 双倍内存消耗。

## 动机

NPU 的 GroupedMatmul kernel 对 FRACTAL_NZ 格式权重更快,此前只有量化 MoE 走了 NZ 转换(#11984 的遗留部分),非量化 MoE 模型仍以 ND 布局逐次 `transpose(1,2)` 计算。同时 Qwen3 MoE 在 ascend 上有双倍内存消耗 bug:`load_weights` 末尾立即初始化 `routed_experts_weights_of_layer`(逐层 `get_moe_weights()` 引用专家权重),叠加 `process_weights_after_loading` 中 `.contiguous()` 新副本后旧存储未及时释放的 workaround(`resize_(0)` 手工释放),权重副本在加载窗口内翻倍。

## 优化点

三件事。(1) 非量化 MoE 的 NZ 化:`UnquantizedEmbeddingMethod`/unquant MoE 路径在 `process_weights_after_loading` 中对 `w13_weight`/`w2_weight` 先 `transpose(1,2)` 再 `npu_format_cast` 成 FRACTAL_NZ,`forward_npu` 里的 grouped_matmul 直接用转换后的权重,删除每次 forward 的 `transpose(1,2)`;同时把 `npu_format_cast` 封装进 `hardware_backend/npu/utils`,W8A8 Int8 dynamic MoE 方法与 EP MoE layer 的加载后处理统一改走它,并删掉 `release_weight_cache`/`resize_(0)` workaround。(2) `qwen3_moe.py` 的 `routed_experts_weights_of_layer` 改为 `LazyValue` 惰性初始化,消除加载期的额外引用导致的权重驻留翻倍。(3) 文档更新:DeepSeek 与 Qwen3 的 NPU 示例不再需要 `ENABLE_MOE_NZ`/`ENABLE_ASCEND_MOE_NZ` 环境变量(NZ 成为默认行为)。新增 `test_ascend_memory_consumption.py` 固定 Qwen3-30B-A3B.w8a8 TP2 启动后占用 ≤16 GB 的回归线。

## 关键改动

```text
python/sglang/srt/layers/quantization/unquant.py: 非量化 MoE 权重加载后 transpose+npu_format_cast(NZ)
python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py: W8A8 MoE 改用 npu_format_cast,删 resize_(0) workaround
python/sglang/srt/models/qwen3_moe.py: routed_experts_weights_of_layer 改 LazyValue 惰性化
test/registered/ascend/test_ascend_memory_consumption.py: 新增 NPU 内存消耗回归测试
docs/platforms/ascend_npu_*.md: 移除 MOE NZ 环境变量
```

```python
+        if _is_npu:
+            for weight_name in ["w13_weight", "w2_weight"]:
+                weight = getattr(layer, weight_name)
+                weight.data = weight.data.transpose(1, 2)
+                weight.data = npu_format_cast(weight.data)
```

## 性能声明

- PR 正文:非量化 MoE 模型 NZ 转换带来"11% speedup an average";具体数字在截图中,pack 正文无文本数字。
- 测试条件:服务端 Qwen3-30B-A3B,`--tp-size 4 --attention-backend ascend --device npu --quantization modelslim --cuda-graph-bs 64 --disable-radix-cache`;客户端 `bench_serving`(ShareGPT random、`--flush-cache`,prompts/并发/输入输出长度以 `*` 占位,正文未填具体值)。硬件为 Ascend NPU,具体型号未注明。
- 内存:新增测试断言 Qwen3-30B-A3B.w8a8、TP2、`--max-total-tokens 1024` 下服务器启动后占用 ≤16.00 GB(双倍内存修复的量化护栏)。

## 验证与回归风险

- 新增 NPU nightly 回归测试(`register_npu_ci`,est_time 400s)验证内存上限;精度验证以截图给出,pack 正文无文本结果。
- 回归面:仅 NPU 后端路径;`forward_npu` 不再做条件 transpose,依赖加载后权重已为 NZ 格式,若其他 NPU 模型复用同一 unquant 方法将一并转换;文档确认 NZ 不再需要环境变量开关。CI 结论 pack 中不可见。

## 关联

- 前序:#11984(closed,本 PR 为其一部分的重新落地)。
- 同区域:NPU MoE/量化方法系列(fused_moe_method_npu)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
