# PR #33743: [Fix] moe: Fix flashinfer TRT-LLM BF16 expert weight reload on refit

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33743](https://github.com/sgl-project/sglang/pull/33743) |
| 作者 | Kh4L |
| 合并日期 | 2026-10-27 |
| labels | bug, documentation, quant, run-ci |
| 热度 | 👍 0 · 💬 39 |
| 分类 | MoE |
| bug 类型 | general |

> 注:本卡素材中的 diff 在 40000 字符处截断(文件粒度),截断后的改动(如新增单测文件全文)未包含在素材中,未据此编造。

## 一句话结论

BF16 MoE 专家权重的 TRT-LLM BlockMajorK 逆变换只对 routed backend 生效,另一 backend 的 refit 加载因 blocked 形状直接报错、重复 refit 还会静默打乱数值;改为两种 backend 统一做形状+数值的 in-place 还原。

## 问题现象

BF16 MoE expert weights 在冷加载时被转换为 FlashInfer TRT-LLM BlockMajorK 布局;逆还原(restore)此前只对 routed backend 执行,而 `flashinfer_trtllm` 与 `flashinfer_trtllm_routed` 都使用该变换。因此非 routed backend 上,后续把 checkpoint 格式的张量拷入 blocked 参数会失败,报错:

```
The size of tensor a (64) must match the size of tensor b (2048) at
non-singleton dimension 2
```

正文并指出第二个问题:仅恢复形状(shape-only restore)是不够的——未变化的 expert 值必须在下一次 pack 前回到 checkpoint 顺序,否则反复 refit 会**静默打乱**它们的数值。本 PR fixes #27787。

## 根因

正文披露两点:其一,逆还原的触发条件错误——旧 `maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load` 仅在 runner backend 为 `flashinfer_trtllm_routed` 时执行,而正向变换对两种 backend 都发生,不对称导致非 routed 路径带着 blocked 布局接收 checkpoint 数据。其二,旧还原只做 `param.data.reshape(expected_shape)`,只纠正形状不还原行置换后的数值顺序;repack 循环进行时,数值会在反复 pack/reshape 中错位。此外,直接张量更新与 P2P/RDMA 写入绕过 model weight loaders,完全不经过 loader 侧还原钩子,这也是 refit 会拿到 blocked 参数的原因之一。

## 修复方案

1. 逆还原改由 `use_flashinfer_trtllm_moe` 门控,覆盖 routed 与非 routed 两种 BF16 backend。
2. 新增 `_restore_trtllm_bf16_canonical_layout`:in place 恢复 canonical 形状**与数值**(求逆行置换),保留参数 storage;用 argsort 求逆前断言 flashinfer 行排列是双射,并拒绝元素数不符——「fail loudly rather than silently scrambling rows」。
3. 新增 `restore_weights_before_loading`(实现于 `UnquantizedFusedMoEMethod`),`begin_weight_update` 路径经 `weight_updater.py` 的 `model_load_weights` 在任何线程写入前统一还原两个专家参数(`w13_weight`、`w2_weight`),覆盖 direct tensor 与 P2P/RDMA 写;loader 侧还原保留给磁盘 reload(后者不开 weight-update session)。
4. post-load pack 加形状 guard:`_maybe_apply_flashinfer_trtllm_bf16_block_layout` 只对 canonical 形状的参数做 block 转换,already-blocked 保持不变;`end_weight_update` 每 session finalize 一次而非每 bucket,复用 LoRA-aware model-loader traversal,不新增 updater/base hook 或 per-bucket finally 回调。

关键摘录(unquant.py):

```diff
-        runner_backend = self.get_moe_runner_backend(layer)
-        if runner_backend is None or not runner_backend.is_flashinfer_trtllm_routed():
+    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(...):
+        """Restore checkpoint layout for both routed and non-routed TRT-LLM backends."""
+        if not self.use_flashinfer_trtllm_moe:
             return
```

改动文件:`python/sglang/srt/layers/quantization/unquant.py`、`model_executor/model_runner_components/weight_updater.py`、`docs/.../sglang_for_rl.mdx`,新增单测 `test/registered/unit/layers/quantization/test_flashinfer_trtllm_bf16_moe_reload.py`(10 个测试方法)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,`flashinfer_trtllm` / `flashinfer_trtllm_routed` 的 BF16 MoE 权重 reload 与 RL refit(张量/分布式/P2P-RDMA 更新、磁盘 reload)路径全部重排;forward kernels 不变,canonical restore 与 pack 每 weight-update session 各付一次。验证为 CPU 单测(loader-free 断言冷加载字节相等、storage 指针不变、禁止调用 model weight loader;control 复现修复前三处 TP-rank 失败);正文明确 FlashInfer layout 入口在测试中是确定性 CPU stand-in,不覆盖 on-device kernel 精度、真实 RDMA 传输与端到端 RL 验证,GPU CI 亦未跑通(run-ci gate 待维护者恢复)。

## 关联

- issue [#27787](https://github.com/sgl-project/sglang/issues/27787)(Fixes #27787)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
