# PR #60762: [Bugfix][DSv4.1] Declare 64-token kernel blocks for SM120 sparse backends

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xzwgit |
| 合并 | 2026-11-05 |
| 分类 | Attention / Kernel(tier: major) |
| Label | bug, documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, torch.compile, ci/build, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, mrv2, mistral, DSv4, dflash, rust, kimi, k3, scheduler, kv-cache-manager, glm, minimax, cohere, DSv4.1, mooncake, pooling, beam, hisparse, recipe, nixl |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/60762 |

## 一句话结论

SM120 的 DSv4.1 稀疏 MLA/indexer 后端补声明 64-token kernel blocks，修复 `--block-size 64` 协商失败。

## 动机

在 SM120 上，DSV41 的 sparse-MLA 后端只声明 `[128]`-token kernel blocks（`get_supported_kernel_block_sizes` 返回 `[64 if family(90) else 128]` / `[128]`），因此用 `--block-size 64` 启动会在 block-size 协商处报错：`ValueError: No common block size for 64 (FLASHINFER_MLA_SPARSE_DSV41: [128]; DEEPSEEK_V41_INDEXER: [128])`。正文说明方案来自 @aagallo（见 #59203），本 PR 补齐验证矩阵。

## 优化点

(1) 仿照既有 SM90 分支，在 SM12x 上对三处站点（indexer backend、sparse MLA backend、FlashInfer sparse backend）声明 `[64]`：diff 中在 `DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes` 与 `vllm/v1/attention/backends/mla/indexer.py` 的同名方法里加入 `current_platform.is_device_capability_family(120)` 判断并 `return [64]`，SWA / backend selection 不变。(2) 每处 SM12x `[64]` 带行内注释指名其规避的失败（#59203）。(3) 正文强调依赖 FlashInfer `>= 0.7.1rc5`（runtime page sizes）：pin 的 `0.7.0.post1` + 本 PR 仍不服务（约 110 s 后 `page_block_size=32 is unsupported`）；并按 review 讨论把落地顺序排在 FlashInfer bump #60948 之后或与 #57292 守卫一起。(4) 按 review 移除早先的 `tests/v1/attention/test_dsv41_indexer_block_sizes.py`（其固定值会漂移，且 block size 在 C++ kernel launch 处断言），不加单测。

## 关键改动

```text
vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py: import current_platform；FlashInfer sparse MLA backend 的 get_supported_kernel_block_sizes 对 SM12x 返回 [64]
vllm/v1/attention/backends/mla/indexer.py: indexer backend 的 get_supported_kernel_block_sizes 对 SM12x 返回 [64]
```

关键片段：

```diff
     @staticmethod
     def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
+        # SM12x kernels only instantiate 64-token sparse-MLA blocks (128
+        # trips the block_kv assert, #59203).
+        if current_platform.is_device_capability_family(120):
+            return [64]
         return [128]
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据（perf claims 检测为 none），未注明性能 benchmark 测试条件。正文的量化结果均为正确性与可服务性验证：8x RTX PRO 6000（SM120, TP8）、vLLM `0.31.1rc1.dev158+gab905a885` + FlashInfer `0.7.1rc5`，四个 patched cell 均在约 90-160 s 内启动且答题正确、zero failures；GSM8K n=100 为 89-91%，四种可服务配置 decode 吞吐在 c1/c4 相差 <1%。最终形态复核（8x RTX PRO 6000 96GB、vLLM 0.33.0、fp8 KV、无显式 `--block-size`）：冷 JIT READY 约 7.5 min、KV cache 2,397,337 tokens、4/4 短答探针正确且 greedy 输出 3 次逐字节一致。

## 验证与回归风险

5-cell / stock·A=`[64]`·C=`[64,128]` × default·64·128 的 GSM8K n=100 矩阵：stock 两种都失败，A/C 在 default 与 64 可服务，`128` 在两种声明下均失败；另有 pre-commit / CI。最终形态另做 4/4 探针 + 3 次 greedy 一致复核。风险：强依赖 FlashInfer `>= 0.7.1rc5`（`0.7.0.post1` 不支持 runtime page sizes）；该 stack 下 `--block-size 128` 在两种声明下都不服务；落地顺序依赖 #60948 或 #57292；review 决定不加单测，靠 kernel launch 处 C++ 断言兜底。

## 关联

#59203（方案与 SM120 serving failure，@aagallo 描述）；#57292（FlashInfer 缺 runtime page sizes 时 fail-fast，正文称可配对）；#60948（FlashInfer bump）；#56509（同一失败、声明有重叠）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
