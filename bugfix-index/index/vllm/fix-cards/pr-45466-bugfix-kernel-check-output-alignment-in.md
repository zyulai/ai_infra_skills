# PR #45466: [Bugfix][Kernel] Check output alignment in vectorize_with_alignment (fixes misaligned-address crash for non-multiple-of-8 head sizes)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45466](https://github.com/vllm-project/vllm/pull/45466) |
| 作者 | HumphreySun98 |
| 合并日期 | 2026-06-18 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
`vectorize_with_alignment` 只检查输入指针对齐,导致 `reshape_and_cache_flash` 对非 16B 对齐 KV-cache 行做向量写而崩溃,现同时校验输出对齐、必要时退化为纯标量拷贝。

## 问题现象
`FLEX_ATTENTION` 搭配 `head_size=46`(如 `inferno-project/vllm-mixtral-2` fuzz 模型)报 `CUDA error: misaligned address`,崩溃点实为 `reshape_and_cache_flash_kernel`。故障是确定性的、精确跟随 dst 行对齐:NHD 布局下 slot 16(行字节偏移 1472,mod 16 = 0)通过,slot 17(偏移 1564,mod 16 = 12)必崩;head_size 48/64 通过、46 崩溃。FlexAttention 只是唯一接受任意 head_size 的后端,因此成了"报信者"。

## 根因
正文披露(含 issue 评论的根因分析):`vectorize_with_alignment`(csrc/libtorch_stable/quantization/vectorization_utils.cuh)只检查**输入**指针的对齐,注释假定"输出与输入保证相同"。该假设对 `reshape_and_cache_flash_kernel` 不成立:src 是新鲜的 K/V 投影,dst 是位于 `block_idx * block_stride + block_offset * page_stride` 的 KV-cache 行。`head_size=46`(bf16)时行距不是 16 字节的倍数,部分行落在 16B 边界之外,而 16 字节向量存取(`vec_n_t<bf16, 8>` 带 `__align__(16)`)触发 fault。

## 修复方案
改 `vectorization_utils.cuh`:
1. 向量化快路径同时要求输出指针按自身向量宽度(`VEC_SIZE * sizeof(OutT)`)对齐:
```cpp
bool can_vec = ((addr & (WIDTH - 1)) == 0) &&
               ((out_addr & (OUT_WIDTH - 1)) == 0) &&
               ((len & (VEC_SIZE - 1)) == 0);
```
2. 若对齐 input 的 prefix 阶段无法同时对齐 output(两者相位不同,向量化不可能),整体退化为纯标量循环。
3. `tests/kernels/attention/test_cache.py` 新增 `test_reshape_and_cache_flash_unaligned_rows` 回归测试(head_size=46、num_heads=13,NHD/HND × auto/fp8 × bf16/fp32 × cuda/triton);既有 `HEAD_SIZES`(64/80/256)全是 16 字节倍数,故此前未暴露。

## 影响与回归风险
- 回归面:所有 `vectorize_with_alignment` 调用方(fp8/int8 量化 kernel、reshape_and_cache/flash 等)。正文说明:当前可正常工作的调用方 in/out 行相位一致,仍走原路径,行为不变;新增检查只是拷贝循环外的少量 ALU 操作。修复与 attention 后端无关,移除了每个调用方的"无保护向量写"隐患。

## 关联
- #41257(fixes)
- [根因分析 issue 评论](https://github.com/vllm-project/vllm/issues/41257#issuecomment-4695960083)
- #41454(被关闭的替代方案,非本 PR 重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
