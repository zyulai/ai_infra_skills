# PR #64138: [Perf][DSA] Read BF16 logits directly in persistent top-k

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-10-30 |
| 分类 | Sampler / Detokenization (tier: major) |
| Label | bug, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),另有 5 个文件因超长从 diff 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/64138 |

## 一句话结论

persistent top-k 直接读 BF16 indexer logits 并在寄存器中拓宽为 FP32,长行场景 selection 阶段延迟最多降 44%,并省掉整个 FP32 logits 临时分配。

## 动机

DSA(deepseek sparse attention)indexer 用 DeepGEMM 产出 BF16 logits 后,原路径要再跑一个转换 kernel 把全量 logits 物化为 FP32 张量,persistent top-k 再按 FP32 读取。这条外置转换既增加一次全量读/写(2 字节读 + 4 字节写/分数),又需要一个保留满 capacity 的 FP32 大缓冲(如 Q=128、K=100000 时约 128.248 MiB 的 graph 私有分配)。Q/K 投影与打分本身不变,可省的正是这段下游工作。

## 优化点

机制:把 persistent/sampled/filtered 三条 top-k 路径的输入模板化为 `InT`(float 或 `__nv_bfloat16`),内核在加载处做无损拓宽——BF16 载入 4/8 字节的 packed 整数,通过移位掩码把每个 BF16 位型放进 FP32 高 16 位(`load_score4`/`load_score2`/`to_float` helper),寄存器中的分数数组与后续 radix 比较仍全部是 FP32,选择算法本身未改。入口 `persistent_topk` 接受 FP32/BF16 两种 dtype(BF16 要求列连续,BF16 路径要求 128 KiB shared memory),并在 caller(`sparse_attn_indexer.py`)通过 `takes_bf16_logits()` capability 接口判断:支持的 persistent decode 调用(含默认 `auto` 解析到 persistent)跳过整张 FP32 logits 的分配与转换 kernel;cooperative、DCP、prefill 保持原行为。该 capability 接口设计与 PR #62172(cooperative top-k 的 BF16 输入)一致,两边互补。

## 关键改动

```text
csrc/topk_histogram_4096.cuh: 新增 to_float/load_score2/load_score4 的 BF16/FP32 重载与 predicated tail load(BF16 无效元素填 -inf)
csrc/persistent_topk.cuh / persistent_topk_overflow.cuh / sampled_topk.cuh: 三条 top-k 路径输入参数模板化 InT,所有读取经 to_float 拓宽
csrc/topk.cu: launch_persistent_topk 按 logits dtype 分发,__nv_bfloat16/float 双实例化,入口校验放宽
vllm/model_executor/layers/sparse_attn_indexer.py(在省略文件中): supported persistent 调用跳过 FP32 全量转换
tests/kernels/test_top_k_per_row.py: 新增 32 个 BF16 kernel/dispatch 用例
```

关键加载代码(8 字节输入扩为 4 个 FP32):

```cpp
__device__ __forceinline__ float4 load_score4(const __nv_bfloat16* ptr) {
  const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
  return make_float4(
      __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xFFFF0000u),
      __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xFFFF0000u));
}
```

## 性能声明

GB200 独占节点,A/B/B/A 新进程、CUPTI cold-L2 CUDA graph replay、30 samples/run,测 pre-PR `98fc6da9` vs PR `293880733f`,只覆盖 selection 阶段(不含 DeepGEMM producer),非端到端:Q65/4096/512 persistent 7.616 → 6.784 µs(-10.92%);Q256/8192/1024 15.648 → 15.456(-1.23%);Q128/100000/512 47.072 → 31.392(-33.31%);Q128/200000/1024 78.240 → 43.488(-44.42%);cooperative 对照组 -0.21%(未声明为收益)。所有 persistent 用例的额外 FP32 分配变为 0。成分拆解:外置转换 2.848-35.264 µs 被整体移除是主因;native top-k 本身长行仅 -3.18%/-5.46%,短行(Q65)反而 +16.56% 变慢(拓宽增加寄存器指令)。另有基于 DeepSeek-V4-Flash-DSpark / V4.1-Flash / V3.2-Exp checkpoint 配置的 18 个补充用例,提升 5.85%-39.66%。作者明确声明:无端到端模型提速、无未测 shape 的普适结论。

## 验证与回归风险

B200 CI `tests/kernels/test_top_k_per_row.py` 全量 806 passed / 44 skipped / 0 failed(含全部 32 个新增 BF16 用例);65,536 个 BF16 位型的标量与 packed 拓宽逐一比对 FP32 位(含 NaN 与带符号零);每个计时 shape 均通过 selected-score 精确一致、有效唯一索引、输入保留检查。风险点:历史模型评测(collocated V4-Flash-DSpark,TP4+EP,8 条约 9.9K 上下文 GSM8K 派生 prompt)中 FP32 与 BF16 均 8/8 答案正确,但仅 2/8 的完整 token 序列一致,差异原因未定位,作者未声明生成 token 等价;短行 native kernel 变慢约 16.56%,依赖 caller 层面以省掉转换来抵消;BF16 路径有 128 KiB smem 与列连续约束。

## 关联

同区域:PR #62172(cooperative top-k BF16 输入,共享 capability 接口设计)、PR #64580(FP32 producer 配置,两者均未混入本次计时基线);同一 persistent_topk 内核的健壮性修复见 pr-51538(负长度 clamp)。同分类其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
