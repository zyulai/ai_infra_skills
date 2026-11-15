# PR #64087: [Bugfix][Kernel] Fix reshape & cache dropping KV heads past 512 chunks

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @giuseppegrossi |
| 合并 | 2026-11-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | bug, rocm, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64087 |

## 一句话结论

修复 ROCm `reshape_and_cache` 在每 token 超过 512 个 16-byte chunk 时丢弃 KV heads 的静默写丢 bug(影响 #62697 的 Pattern B);修复后三个 failing case 与 `TRITON_ATTN` 对齐(max error ≤ 0.0006),≤512 chunk 形状性能不变,超限形状按实际写入字节归一后保留 88.5–92.2% 的 broken baseline 部分写吞吐。

## 动机

`reshape_and_cache` 在 `num_kv_heads * head_size * cache_element_size > 8192` 字节(即每 token > 512 chunks)时,chunks past the 512th per token 没有线程写入,KV cache 里残留旧数据,`ROCM_ATTN` 后端静默读错。根因:#25955 把 kernel 的 thread-stride loop 换成 one thread per (KV head, 16-byte chunk),但保留了 launch cap `dim3 block(std::min(num_heads * head_div_x, 512))`;如 bf16 下 32 KV heads × head 256 = 1024 chunks,heads 16–31 永不写入。CI 的 `test_reshape_and_cache` 只用 `NUM_HEADS = [8]`(至多 256 chunks),从未触发该上限。未写的 slot 保留 cache block 原值,输出静默错误而非崩溃。

## 优化点(bugfix)

- Kernel 侧把 per-chunk body 包回 thread-stride loop:`for (h_block_idx = threadIdx.x; h_block_idx < num_heads * h_block_count; h_block_idx += blockDim.x)`,scale setup 移到 loop 上方;launch 配置不变,保持 one block per token(正文注明大部分 diff 是重缩进,隐藏 whitespace 后仅 10 行改动;kernel 文件本身的 diff 未在 pack 展示)。`reshape_and_cache_flash_kernel` 已按同样方式 stride。
- Python 侧 `do_kv_cache_update` 增加守卫:native HIP kernel 仅在 `num_kv_heads * (head_size // x) <= 512` 时选用,超限 shape 回退非 native 路径。
- 设计取舍:strided loop 是最小正确性修复;2D launch(多 block per token)可暴露更多并行但引入调度开销、需跨 token 数与架构调优,留作后续性能优化单独评估。

## 关键改动

```text
vllm/v1/attention/backends/rocm_attn.py: do_kv_cache_update 加 chunks≤512 守卫,超限不走 native write_to_paged_cache
tests/kernels/attention/test_cache.py(新增): test_rocm_attn_kv_cache_update_writes_all_heads 回归测试(32 KV heads × 256, fp16)
(kernel 内部 thread-stride loop 改动见正文描述;pack 的 diff 未展示该文件)
```

```python
-        if block_size in (16, 32) and has_native_layout:
+        # The native kernel writes at most 512 chunks of x elements per token.
+        if (
+            block_size in (16, 32)
+            and has_native_layout
+            and self.num_kv_heads * (self.head_size // key_cache.shape[4]) <= 512
+        ):
             # Normal 16, 32 with contiguous blocks: use vLLM native HIP C++ logic.
             PagedAttention.write_to_paged_cache(
```

## 性能声明

- 正确性(reporter 的 reproducer 脚本,#62697 原样):`ROCM_ATTN` 对 fp32 reference 的 max error——`p2-d256-bs16-gqa1`(32/32 heads, head 256, decode)0.1935 → 0.0006;`p2-d192-bs16-gqa1`(head 192)0.1955 → 0.0006;`p2-d256-bs16-gqa1-prefill` 0.0967 → 0.0003;`ok-d256` fresh-prefill 0.0085 不变且 cache read-back 精确(原最多偏 5.31)。
- 性能(`benchmarks/kernels/benchmark_reshape_and_cache.py --dtype bfloat16 --block-size 16 --mode cudagraph --iters 200`,2–65,536 tokens,每配置 3 轮 baseline/fix 交替,取中位数,1× MI355X):≤512 chunks/token 形状 fix/base 0.969–1.005×(性能不变);超限形状 raw fix/base 为 32×256 auto 1.604×、64×256 fp8 1.650×、128×128 auto 2.693×(all sizes),≥1024 tokens 时 2.260× / 2.169× / 4.487×——但 baseline 分别只完成了 50%/50%/25% 的写入工作;按实际写入字节归一,修复 kernel 保留 88.5%、92.2%、89.1% 的 broken baseline partial-write 吞吐。
- GSM8K(`tests/evals/gsm8k/gsm8k_eval.py`,1,319 题,5-shot,MI355X,`main` dd4e40142a,Qwen/Qwen1.5-14B-Chat):`ROCM_ATTN` main 0.411 → 本 PR 0.691(`TRITON_ATTN` main 参考 0.688)。

## 验证与回归风险

- 测试(全部 1× MI355X, gfx950):`pytest -v tests/kernels/attention/test_cache.py -k test_reshape_and_cache_many_heads tests/compile/passes/test_rope_kvcache_fusion.py -k ROCM_ATTN tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py`。新增回归测试直接验证 32 heads × 256 在 fp16 下每个 KV head 写入(atol=0, rtol=0)。
- 直接检查 `ops.reshape_and_cache`:baseline 恰好漏写根因预测的 heads(32×256 时 heads 16–31;32×192 时 head 21 部分写入、heads 22–31 未写),修复后无;其余六个 case(GQA-4 与 block-528)在两个 build 上均通过。
- 风险:仅 MI355X 实测;strided loop 是最小修复,超限 shape 的多 block 并行布局未做(留作后续);Pattern A(#62697 的 decode reads at block size 528 on gfx1201)不在本 PR 范围。影响面:KV heads × head size per GPU 超过 4096 elements(bf16)的模型,如 13B+ MHA 模型(Qwen1.5-14B、Llama-2-13B、OLMo-2-13B)TP=1;MI300/MI355 上仅显式选 `ROCM_ATTN` 时触发,RDNA 上是默认 backend。

## 关联

- 修复 #62697 的 Pattern B(Pattern A 是另一问题,未在此处理);根因出自 #25955。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
