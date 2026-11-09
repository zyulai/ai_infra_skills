# PR #49302: [Bugfix] Fix DSA crash under breakable piecewise cudagraphs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49302](https://github.com/vllm-project/vllm/pull/49302) |
| 作者 | njhill |
| 合并日期 | 2026-07-21 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 3 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
breakable piecewise cudagraph replay 对 batch 元数据按捕获尺寸填充,导致 DSA 索引器/sparse MLA 崩溃,恢复 pre-#48822 的「replay 不填充请求」约定修复。

## 问题现象
#48822 使 breakable piecewise graph replay 时把 batch 元数据填充到捕获形状(`num_reqs` 填充到 `min(num_tokens, max_num_reqs)`,尾部是零长度请求)。这会破坏「attention 以 graph break 点形式 eager 运行、从 forward context 读取真实 batch」的后端:DSA indexer decode 路径把填充后的请求数乘以 next_n 展开(deepgemm 断言 `batch_size * next_n == weights rows` 失败),sparse MLA prefill 路径无法为零长度填充请求构建 TMA descriptor。后果:V2 runner 上 DeepSeek-V4-Flash 崩溃(gsm8k 得分 0.0000/engine-crash),MiniMax-M3 Sparse 同样受影响。

## 根因
正文披露根因:#48822 引入的 replay 侧 batch 元数据填充对 breakable piecewise 图是不必要的——图内 kernel 并不需要这些填充:runner 总是把 `slot_mapping` 填充到捕获的 token 数,图内 kernel 自行从它填充(Inkling 的 `sconv_seq_metadata` 按 `slot_mapping.shape[0]` 填尾行、`fused_sconv` 跳过 `slot == -1` 行,与 batch 元数据形状无关);而 eager 运行的 break 点 kernel 读到的却是填充后的假 batch,触发 deepgemm 断言与 TMA descriptor 构建失败。

## 修复方案
在两个 replay 侧填充点恢复 pre-#48822 的不变量:
1. `vllm/v1/worker/gpu/cudagraph_utils.py` `_init_candidates`:piecewise 图描述符恢复 `num_reqs=None`(不再因 `use_breakable_cg` 填充到 `min(num_tokens, max_num_reqs)`),并更新注释说明 break-point kernel 从 forward context 读真实 batch、图内 kernel 自行处理 token 填充。
2. `vllm/v1/worker/gpu/model_states/default.py` `prepare_attn`:replay 元数据改回仅 `CUDAGraphMode.FULL` 时使用填充尺寸(`num_reqs_after_padding`/`num_tokens_after_padding`),piecewise(含 breakable)用真实(未填充)尺寸;捕获期元数据仍保持填充形状(捕获 dummy batch 恰为捕获尺寸,天然成立)。
3. 顺带硬化 `tests/v1/cudagraph/test_breakable_cudagraph.py`:模块级 `os.environ` 改为 autouse fixture(monkeypatch 恢复环境变量、`envs.disable_envs_cache()`),并修复 `cuda_capture_stream` fixture 泄漏流缓存污染后续同进程测试的问题。

关键 diff 摘录:
```diff
-                if mixed_mode == CUDAGraphMode.FULL or (
-                    mixed_mode == CUDAGraphMode.PIECEWISE and self.use_breakable_cg
-                ):
+                if mixed_mode == CUDAGraphMode.FULL:
                     num_reqs = min(num_tokens, self.max_num_reqs)
...
-        if cudagraph_mode == CUDAGraphMode.FULL or (
-            cudagraph_mode == CUDAGraphMode.PIECEWISE
-            and is_breakable_cudagraph_enabled()
-        ):
+        if cudagraph_mode == CUDAGraphMode.FULL:
```

## 影响与回归风险
- 影响窗口:#48822 合并后至本修复之间,使用 breakable piecewise cudagraph(`FULL_AND_PIECEWISE`)且 attention 后端带 eager graph break 点(DSA / sparse MLA)的模型,如 DeepSeek-V4-Flash、MiniMax-M3 Sparse。
- 回归面:影响 piecewise/breakable cudagraph 的 descriptor 构建与 replay 元数据尺寸两条路径。作者在 GB200 上验证 DeepSeek-V4-Flash gsm8k(TP2+EP、MTP、fp4 indexer cache、`FULL_AND_PIECEWISE`)修复后 0.9492;breakable cudagraph 与 dispatch 单测不变。普通 FULL 图路径保持填充行为不变。

## 关联
- [#48822](https://github.com/vllm-project/vllm/pull/48822)(引入 replay 侧填充的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
