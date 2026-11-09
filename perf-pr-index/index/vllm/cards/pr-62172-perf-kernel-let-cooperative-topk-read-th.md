# PR #62172: [Perf][Kernel] Let cooperative_topk read the BF16 indexer logits

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @haosdent |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 6 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/62172 |

## 一句话结论

cooperative_topk 直接读 BF16 logits 免去 FP32 upcast,B300 上 decode top-k 快 22%-57%。

## 动机

SM100 上 DSA(sparse attention indexer)的 DeepGEMM logits 内核本就输出 BF16,但 decode 阶段的 top-k 前会把 paged MQA logits 先 upcast 成 FP32(`as_fp32_logits`)再送进 `cooperative_topk`。main 上 `deep_select` 与(自 #64138 起)`persistent` 已直接读 BF16,唯独 `auto` 在 ≤64 行时选择的 `cooperative` 路径(下述测试中每次 decode 都命中,最多 16 requests x 4 tokens)仍走 upcast,白白多一次转换与 2 倍显存流量。

## 优化点

把 `cooperative_topk` 全链路模板化到输入类型 `InT`(FP32/BF16):`CooperativeTopKParams`、TMA staging buffer、直方图 load 均按元素类型参数化;TMA 对齐单位按 16 字节换算成 8 个 BF16 或 4 个 FP32 元素,stride/每块对齐随元素尺寸调整;进直方图 bin 时统一 widen 成 FP32,两种类型使用同一套 radix key,保证选择结果与 FP32 widening 完全一致(#59645 的精确溢出恢复路径也经 `to_float`/`load_score4` 以同样的 widening 语义读取)。BF16 专用优化:每 32-bit load 读两个 staged score 并用一次 packed FP16 转换同时求出两个 coarse bin(`extract_coarse_bins_bf16x2`);同容量共享内存驻留两倍的 BF16 元素使部分 shape 单 pass 完成;single-CTA register 路径把 BF16 打包进 `uint2`;tie refinement 只对 tie 的 key 做 radix-256 轮。Python 侧 `SparseIndexerTopk.takes_bf16_logits` 对 `cooperative`(显式或 `auto` 选中)在满足约束(≤64 行、行连续且 16 字节对齐)时也返回 true。

## 关键改动

```text
csrc/cooperative_topk.cuh: InT 模板化 params/smem/TMA 流水线,BF16 x2 packed binning、双倍 stage 驻留
csrc/cooperative_topk.cu: 按 logits.scalar_type() 分发 BF16/FP32,kTmaAlign<InT> 校验
csrc/topk_histogram_4096.cuh: extract_coarse_bins_bf16x2 packed 转换与 ScoreVec4
csrc/cooperative_topk_overflow.cuh: 溢出恢复路径经 widening helper 读取 BF16
vllm/model_executor/layers/sparse_attn_indexer.py: takes_bf16_logits 覆盖 cooperative
```

关键片段(BF16 双 score packed binning):

```cpp
template <uint32_t kBits>
__device__ __forceinline__ uint2 extract_coarse_bins_bf16x2(uint32_t w) {
  const __half2 h = __floats2half2_rn(__uint_as_float(w << 16),
                                      __uint_as_float(w & 0xFFFF0000u));
  const uint32_t bits = *reinterpret_cast<const uint32_t*>(&h);
  const uint32_t sign = bits & 0x80008000u;
  const uint32_t key = bits ^ ((sign - (sign >> 15)) | 0x80008000u);
  return make_uint2((key & 0xFFFFu) >> (16 - kBits), key >> (32 - kBits));
}
```

## 性能声明

- Decode top-k,B300(SM103),main(FP32 upcast + FP32 top-k)vs 本 PR(BF16 top-k):rows {1,4,8,16,33,64} x row length {16K,64K,128K,256K,1M} x k {512,2048} 全组合 22%-57% 更快(例:k=512、64 行 1M:221.0 → 106.6 µs,-52%)。BF16 top-k 内核单独对比 main 的 FP32 top-k 快 8%-38%;FP32 logits 输入时最快 +16%、最多慢 0.2%。
- `persistent_topk` BF16(共享 single-CTA 路径)较 #64138 基线快 24%-31%(rows ≤ 32K)。
- 端到端:InferenceX AgentX,GLM-5.3-NVFP4,4x GB200,与 #62170、#62171 合并评测,decode time per turn 在并发 1/4/8/16 降 2.8%/2.5%/1.5%/1.5%,本 PR 约占其中 1/3。
- 精度:BF16→FP32 widen 精确,selection 与 torch.topk 逐值一致;32 条 GSM8K greedy 输出与 acceptance length(3.4904)与 main 相同;MRCR 594 样本总体差异 -0.013(CI 含 0)。

## 验证与回归风险

`tests/kernels/test_top_k_per_row.py` 于 B300:870 passed / 44 skipped(main 为 806 passed);新增/扩展的用例覆盖 FP32+BF16 全 cluster size、16-bit grid overflow、streaming overflow、NaN/mixed-lengths overflow 等;`test_sparse_indexer_topk_backend_resolution` 与 `test_sparse_indexer_resolved_persistent_bf16` 校验 `auto` 下 cooperative 也接受 BF16 且选择精确。基准用 `triton.testing.do_bench_cudagraph` 中位数、main/PR/PR/main 交叉顺序跑两遍(同 build 中位数差 0.1%、最大 3.3%)。风险:仅限满足 64 行、连续且 16 字节对齐行的配置;DCP、仅接受 FP32 的 top-k backend、非对齐 BF16 行仍走 upcast;DeepSeek-V4.1 的 candidate-block selection 保持 FP32 block score(作者未在该模型上实测);kernel 负载(如 kSmemSize、stage 数)随 dtype 变化,极端 shape 上单 pass 判定改变可能影响占用。

## 关联

同系列:#64138(persistent 读 BF16,本 PR 复用其 widening helpers)、#64580(FP32 logits 默认)、#59645(精确溢出恢复)、#63118(overflow NaN 测试);端到端评测与 #62170、#62171、#62173、#62174 一组。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
