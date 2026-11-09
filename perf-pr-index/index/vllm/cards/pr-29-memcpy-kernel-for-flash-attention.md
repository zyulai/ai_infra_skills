# PR #29: Memcpy kernel for flash attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @suquark |
| 合并 | 2023-04-11 |
| 分类 | Attention / Kernel(major) |
| Label | 无(pack 未记录 label) |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 kernel、pybind 与测试/基准文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/29 |

## 一句话结论

新增 `gather_cached_kv` CUDA kernel,把分页 KV cache gather 成 flash attention 所需的连续 K/V,小 token 数场景较初版提速约 20%。

## 动机

flash attention 需要连续布局的 K/V 张量,而 vLLM(cacheflow 时期)的 KV cache 按 `[num_blocks, ...]` 分页存储,key 还有 `[head_size/x, block_size, x]` 的特殊重排布局。要在 flash attention 前使用缓存,需要一个专门的 gather/memcpy kernel 把对应 slot 的 K/V 抽到连续张量;初版实现性能尚可但小 batch 下有明显优化空间。

## 优化点

新增 `gather_cached_kv_kernel`:每个 token 一个 CUDA block,按 `slot_mapping` 定位 block/offset,手工计算 key cache 重排布局与 value cache 布局的源地址,用 `__ldg` 只读缓存加载写入连续输出。优化版 `gather_cached_kv_kernel_optimized` 在此基础上做 4 路循环展开(要求 `num_heads*head_size % 4 == 0`,已知用例均满足),把源/目标索引计算与加载/存储分组流水化,减少每线程的串行访存延迟。入口按 dtype dispatch,grid 为 `(num_tokens)`,block 大小 min(num_heads*head_size, 512)。

## 关键改动

```text
csrc/cache_kernels.cu: 新增 gather_cached_kv_kernel 与 4 路展开的 optimized 版本及 host 封装
csrc/cache.cpp: pybind 暴露 gather_cached_kv
benchmark/benchmark_cache.py: 新增 gather_cached_kv 吞吐/延迟基准
tests/kernels/cache.py: 新增 test_gather_cached_kv 对照参考实现
```

关键片段(cache_kernels.cu):

```cuda
const int unroll_factor = 4;
for (int i = threadIdx.x; i < dim / unroll_factor; i += blockDim.x) {
  #pragma unroll
  for (int j = 0; j < unroll_factor; ++j) {
    /* 计算 src/tgt 索引并 __ldg 加载 */
  }
  #pragma unroll
  for (int j = 0; j < unroll_factor; ++j) { /* 写回 key/value */ }
}
```

## 性能声明

A100-40GB(正文注明理论最优带宽 1.6 TB/s),num_heads=40、head_size=128、block_size=8、fp16、num_blocks=1024,优化 kernel 对比未优化版本:
- num_tokens=64:0.008 ms、156.5 GB/s 对 0.010 ms、125.9 GB/s;num_tokens=128:216.2 对 160.7 GB/s —— 小 token 数下约 +20% speedup。
- num_tokens=2048 时两者接近(191.6 对 189.8 GB/s),大 token 数收益消失。
- 实测吞吐 152–216 GB/s,远低于 1.6 TB/s 理论值,作者归因于 cache 内存布局不理想。PR 未注明测试用 GPU 型号以外的具体机器配置。

## 验证与回归风险

`tests/kernels/cache.py` 新增 `test_gather_cached_kv`:随机 slot_mapping 下与逐 token 的参考实现(直接索引 key_cache/value_cache)做 `torch.allclose` 对照,纳入 `test_cache` 套件。风险面:属于早期 cacheflow 基础 kernel,影响所有 flash attention 路径的 KV gather;`assert(dim % 4 == 0)` 对 head 维度有隐含假设,新增非常规 head 配置需复核。

## 关联

无直接前序/后续 PR 记录;同期基础 cache kernel(reshape_and_cache、swap_blocks)出自同文件。KV 数据搬运方向后续有 #38460(批量 KV cache swap 用 cumem 拷贝)等优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
