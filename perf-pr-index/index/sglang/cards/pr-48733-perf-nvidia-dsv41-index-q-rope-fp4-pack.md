# PR #48733: [Perf] [NVIDIA] dsv41: Index-Q RoPE + fp4 pack runs one thread per fp4 block for prefill-sized calls

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @dmtri35 |
| 合并 | 2026-11-06 |
| 分类 | Quantization |
| Label | jit-kernel |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 2 个文件(fp4_indexer_rope.cuh、test_fp4_indexer.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/48733 |

## 一句话结论

DeepSeek-V4.1 FP4 indexer 的 Index-Q RoPE+打包 kernel 为 prefill 尺寸新增一 thread 一 fp4 block 的实现,32K-token 调用 251.1→55.4 µs(4.5x),输出逐位不变。

## 动机

`index_q_rope_pack_weights`(`fp4_indexer_rope.cuh` 的 `index_q_kernel`)对 DSV4.1 indexer query 做 RoPE 与两段式 fp4 打包并写 head weights,旧实现给每个 (token, head) 的 128 bf16 行分配一个 warp:每行 4 字节 loads、四次 16-lane max 归约加两次 shuffle、每 lane 两次 1 字节 payload store、4-warp CTA。该 shape 按 decode 测量选型,适合 decode 但在 prefill 尺寸远离带宽上限:VR200 上 32768-token chunk × 32 index heads(1M 行,约 340 MB 流量,HBM 速度下约 26 µs)实测每个 indexer layer 耗时 251 µs。

## 优化点

新增 `index_q_block_kernel`:每个 32 元素 fp4 block 给一个线程(每行 4 线程,256 线程 CTA 处理 64 行),CTA 内 warp 0-3 管 head blocks、warp 4-7 管 RoPE blocks 使 RoPE 分支 warp-uniform。每线程先完成全部 load(input block 与 RoPE 半区的 `freqs_cis` 切片,Blackwell 上 32 字节向量),再做 tail pairs RoPE、树形 block absmax(seed 0,NaN 丢弃语义与 fmaxf 链一致)、fake quant 到 e2m1 grid 值,packer 指数直接从 grid amax 的 biased exponent 读取(无除法,`index_pack_exponent_grid`),两个 2 的幂 scale 折叠成一次乘法,最后一次 16 字节 payload store 加 scale word 的 1 字节;行的首个线程写 head weight。逐位等价:除 block absmax 外所有 fp4 阶段均逐元素、max 与顺序无关,且 RoPE 乘加显式写成 `__fmaf_rn`/`__fmul_rn` 的 `fma(x, fx, -(y*fy))` 形式与 warp kernel 的 nvcc 收缩一致。Dispatch:`launch` 在 (token, head) 行数 ≥ 8192 且 SM100 族(CC 10.x)时选 block kernel(按行数计,4-token speculative verify × 32 heads × 64 请求也达阈值),更小 batch、其他 GPU 与 ROCm 保持 warp kernel;并对 block kernel 路径检查 32/16 字节对齐,错误视图报清晰错误而非 fault。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh: 新增 index_q_block_kernel、index_block_absmax、index_pack_exponent_grid;IndexQKernel::launch 增加行数+CC 分派与对齐校验;rope 收缩显式化
test/registered/kernels/ops/attention/test_fp4_indexer.py: 新增逐位一致性测试并注册 4-gpu-b200 runner
```

关键机制(指数免除法 + 折叠 scale):

```python
SGL_DEVICE uint32_t index_pack_exponent_grid(float amax) {
  const auto biased = static_cast<int32_t>((__float_as_uint(amax) >> 23) & 0xFF);
  const auto exponent = biased == 255 ? 255 : biased - 2;
  return static_cast<uint32_t>(min(max(exponent, 114), 254));
}
```

## 性能声明

VR200(SM107)单卡,32 heads × 128,bf16 head weights,CUDA graph 20 calls,µs/call:

| tokens | rows | main | 本 PR |
| --- | --- | --- | --- |
| 256 | 8,192 | 2.5 | 1.6-1.7 |
| 1,024 | 32,768 | 6.4 | 2.7 |
| 8,192 | 262,144 | 52.2 | 13.7 |
| 32,768 | 1,048,576 | 251.1 | **55.4(4.5x)** |

≤255 token 行走 warp kernel 不变。E2E:4x VR200,DeepSeek-V4.1-Flash,DEP4 prefill(每 rank 32K chunk),本 kernel 首版 + 两个 DeepGEMM MQA-logits epilogue 改动(DeepGEMM #138/#140)合计 **+1.0% prefill throughput**;index-Q 部分单独约每 192 ms forward 省 0.6 ms(3 个 indexer layers,各约 190 µs)。B300 数据在 review 线程中。当前版 kernel 来自 @zcnrex 的 review,比首版再快约 10-23%(同字节数)。

## 验证与回归风险

新增 `test_index_q_rope_pack_prefill_matches_decode_kernel`:4100×32 heads 一次调用(block kernel)与 128-token 分次调用(warp kernel)要求 payload/scale/weights 逐位相等,输入覆盖 7 个族(normal、极端 row scale、稀疏含 signed zero、低于 amax floor、e2m1 舍入平局、全指数域、inf/nan),int32/int64 positions,VR200 上 14/14 通过,非 CC 10.x 自动跳过;`test_fp4_indexer.py` 全文件 VR200 108 passed / 2 skipped / 6 failed(6 例为既有 `topk_v2` JIT 环境问题,B300 上 110 passed)。CI PR Test Base/Extra/AMD 当时均失败(x)。风险点:分派 gate 覆盖整个 SM100 族,B200/GB300 时序未经本 PR 正文验证(作者欢迎补充、必要时收窄 gate);对齐校验是新增前置条件。

## 关联

同区域(DSV4.1 indexer / FP4 kernel):#26788(DSA indexer faste kernel)、#26722(AMD triton sparse)、#32779(CUDA fused triton sparse MLA)、#43736(SWA decoder replay);配套 DeepGEMM #138/#140(MQA-logits epilogue)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
