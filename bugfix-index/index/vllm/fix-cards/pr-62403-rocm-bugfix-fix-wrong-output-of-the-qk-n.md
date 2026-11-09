# PR #62403: [ROCm][Bugfix] Fix wrong output of the QK-norm + RoPE + KV-cache fusion on ROCM_ATTN

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62403](https://github.com/vllm-project/vllm/pull/62403) |
| 作者 | jiaqiang-dot-liu |
| 合并日期 | 2026-10-21 |
| labels | bug, rocm, torch.compile, verified |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
ROCM_ATTN 融合把 NHD 写进按 interleaved 读的 V cache,致 Qwen3 解码乱码;让每层都切换 interleaved 并对 interleaved 层改走非融合写入。

## 问题现象
开 `fuse_qk_norm_rope_kvcache` 后:Qwen3 模型解码输出乱码(Qwen3-8B GSM8K 0.87 → 0.005),prefill 看似正确,其后的每个 decode 步读 cache 出错;同时开 `fuse_rope_kvcache` 时,无 QK-norm 的模型 V cache 损坏(Llama-3.1-8B GSM8K 0.766 → 0.718/0.706)。现有融合测试只用单个 attention 层,覆盖不到第二层。仅 `ROCM_ATTN` 受影响;该融合默认关闭。

## 根因
正文披露两处破坏「融合 kernel 写 interleaved V、层内 kernel 按 interleaved 读」契约的缺陷:
1. torch >= 2.11 下 `layer_name` 提升为一个 pattern 服务全模型,`QkNormRopeKvCacheFusionPass` 循环在第一个 eligible layer 后 `break`,只有一层调用 `set_interleaved_v_cache()`,其余层 `_use_interleaved_v_cache=False`,融合 kernel 写 NHD 布局而这些层按 paged 布局读。运行时在 Qwen3-0.6B(28 层)确认:main 上只有 1 层被 set,该层融合写用 `use_shuffle_layout=True`,其余 27 层为 False。
2. `fuse_rope_kvcache` 也开时,pass 在任何 pattern 匹配前就把层切到 interleaved V;QK-norm pattern 不匹配的层由 `do_rope_and_kv_cache_update` 走 Triton writer(`triton_rope_and_cache`),只存常规 V 布局。只修第一处会让第二处打到每一层(Llama GSM8K 0.012)。

## 修复方案
两个 commit 各修一处:① `qk_norm_rope_kvcache_fusion.py`:每个 eligible layer 都执行 `set_interleaved_v_cache()`,仅用 `patterns_registered` 标志跳过重复的 pattern 注册(移除循环尾部的 `break`);② `rocm_attn.py` `do_rope_and_kv_cache_update`:`_use_interleaved_v_cache` 时用 `ops.rotary_embedding` 就地做 RoPE,再走已写 interleaved 布局的非融合 `do_kv_cache_update`,常规 V 层保留融合 Triton writer:

```diff
+            if _USE_LAYERNAME and patterns_registered:
+                continue
+            patterns_registered = True
```

```diff
+        if self._use_interleaved_v_cache:
+            ops.rotary_embedding(positions, query, key, layer.head_size,
+                                 cos_sin_cache, is_neox)
+            self.do_kv_cache_update(layer, key, value, kv_cache,
+                                    layer_slot_mapping)
+            return
```

## 影响与回归风险
- 影响窗口:正文说明问题存在于 `main`(验证基线 main @ 6af1e0ae27,torch 2.11 的 layer_name 行为);`fuse_qk_norm_rope_kvcache` 默认关闭,仅 ROCM_ATTN。
- 回归面:`QkNormRopeKvCacheFusionPass` 对所有 eligible 层切 interleaved V 并去重 pattern 注册;`RocmAttentionImpl.do_rope_and_kv_cache_update` 在 interleaved V 层由单融合 kernel 拆成两个 kernel(RoPE + cache 写),正文提到融合写 interleaved 布局需 AITER 侧改动才能恢复单 kernel;两个 pass 测试文件新增用例(三层全切换、interleaved_v 对照)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
