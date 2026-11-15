# PR #54820: [ROCm] Avoid DSpark placeholder shared vocab allocations

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @keneoneth |
| 合并 | 2026-11-11 |
| 分类 | Speculative Decoding(tier: major) |
| Label | rocm, speculative-decoding, ready, deepseek, mrv2, DSv4, dflash |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/54820 |

## 一句话结论

DeepSeek-V4 DSpark 在 ROCm 上不再为 draft 侧分配 placeholder `embed_tokens`/`lm_head`(steady state 由 target 模型 alias),每 rank 省约 1.73 GiB 瞬态启动内存。

## 动机

DeepSeek-V4 DSpark 在 ROCm 上即使 AMD DSpark draft 模型在 steady state 并不持有共享词表权重,也在启动时分配 draft 侧 placeholder `embed_tokens` 与 `lm_head`;`load_dspark_model()` 会把 target 模型的共享 vocab 模块 alias 进 draft 模型,placeholder 只是瞬态启动开销。在 Strix Halo TP=2 bring-up 上该开销大到足以致命:每个 placeholder 每 rank 约 0.863 GiB,合计约 1.73 GiB/rank。

## 优化点

把修复限定在 ROCm 实现 `vllm/models/deepseek_v4/amd/dspark.py`:

- `embed_tokens`、`lm_head` 保持 `None`,直到 `load_dspark_model()` alias target 模块;
- 在两模块被使用前加 defensive assertions;
- 删除不再使用的 placeholder 分配相关 import(`ParallelLMHead`/`VocabParallelEmbedding`)。

与既有 in-tree DSpark 模式一致:`vllm/models/kimi_k3/nvidia/dspark_mla.py` 即设 `has_own_embed_tokens = False`、`has_own_lm_head = False` 并在 draft 加载后依赖 target alias。

## 关键改动

```text
vllm/models/deepseek_v4/amd/dspark.py: embed_tokens/lm_head 置 None + 断言 + 清理 placeholder import
```

```diff
-        # Shared with the target (aliased by the speculator's loading utility).
-        self.embed_tokens = VocabParallelEmbedding(
-            config.vocab_size,
-            config.hidden_size,
-            prefix=maybe_prefix(prefix, "embed_tokens"),
-        )
+        # Assigned by load_dspark_model from the target. Avoiding a placeholder
+        # here reduces transient startup memory for DSpark bring-up.
+        self.embed_tokens: nn.Module | None = None
```

## 性能声明

- 启动内存:移除两个固定尺寸 placeholder,合计每 rank 省约 1.73 GiB 瞬态 draft 侧启动分配(各约 0.863 GiB); savings 与 `gpu_memory_utilization` 取值无关。
- 在 2-node Strix Halo TP=2 ROCm 上以 `gpu_memory_utilization` = 0.30/0.34/0.38/0.42/0.46/0.50/0.60/0.70/0.80/0.90/0.93/0.95/0.97/0.99 反复重启验证:打补丁后旧的 DSpark 启动即失败(内存不足)模式不再复现。
- Offline TP=2 DSpark-vs-baseline 每 request E2E 延迟对比(正文原表):

| Input -> Output | Baseline E2E latency | DSpark E2E latency | Speedup |
| --- | ---: | ---: | ---: |
| `1024 -> 128` | `178.63s` | `44.39s` | `~4.0x` |
| `4096 -> 256` | `608.98s` | `85.91s` | `~7.1x` |
| `8192 -> 256` | `375.22s` | `86.00s` | `~4.4x` |

(注:该表是 DSpark 投机解码相对 baseline 的对比,正文未声明这些倍数由本 PR 单独带来;本 PR 自身的直接收益是启动内存。)

## 验证与回归风险

- 验证平台:2-node Strix Halo TP=2 ROCm;测试手段为 14 档 `gpu_memory_utilization` 的 DSpark 重复启动测试 + offline DSpark-vs-baseline 对比。
- 回归风险:改动只限 ROCm 路径,NVIDIA DSpark 实现不变;`embed_tokens`/`lm_head` 在 alias 前为 `None`,依赖新增的 `assert self.embed_tokens is not None` / `assert self.lm_head is not None` 在误用时尽早报错。
- diffstat 缺失(pack 未提供 +/− 统计)。

## 关联

- 与 in-tree 模式对齐:`vllm/models/kimi_k3/nvidia/dspark_mla.py`(kimi_k3 NVIDIA DSpark MLA,`has_own_embed_tokens=False` 模式)。
- Label 关联:DSv4、dflash、mrv2、deepseek、rocm、speculative-decoding。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
