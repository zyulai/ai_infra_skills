# PR #57978: [ROCm][Perf] Parallelise AITER MLA page-index expansion over token chunks

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fululi12 |
| 合并 | 2026-10-01 |
| 分类 | Attention / Kernel |
| Label | ready, rocm |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/57978 |

## 一句话结论

AITER MLA page-index 展开 kernel 增加 token chunk 网格维度,单请求 kernel 快 6.8×。

## 动机

`_expand_page_indices_kernel` 原以 `(num_reqs,)` 网格启动,每个 program 用串行 `for` 循环遍历该请求的全部 token,并行度完全来自 batch 维度。低并发时退化:num_reqs==1 时单个 workgroup 串行展开整条序列,GPU 其余部分空闲。MI355X TP8、~97k 上下文、conc=1 实测 61.6 µs/次、每步 7 次共 0.431 ms,占 34.9 ms step 的 1.23%;而 97k 个 int32 store 仅 ~388 KB,远低于一微秒量级的 HBM 带宽极限——kernel 不是带宽瓶颈而是被串行化。影响所有 ROCm MLA 模型的低并发场景,并非某一模型专属。

## 优化点

给 kernel 增加第二个网格维度并按 token chunk 划分,每个 program 恰好处理一个 `BLOCK_SIZE` chunk,去掉串行循环;算术、mask 与 store 逻辑不变,只把迭代空间映射到 GPU 上。chunk 数在 host 侧从 block table 宽度推导:`num_chunks = max(1, cdiv(block_table.shape[1] * kernel_block_size, 1024))`,这是每请求 token 数的上界,因此不引入 device 同步;ragged batch 中落到短请求末尾之后的 program 直接 early return,不发masked 访存。同时把 block table 的两个 stride 都显式传入 kernel,不再假设列 stride 为 1(非连续视图也要正确)。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: launch 站点增加 num_chunks 维度;kernel 改为 (req, chunk) 二维并行
tests/v1/attention/test_expand_page_indices.py: 新增与 PyTorch 参考实现的逐位等价性测试
tests/v1/attention/test_rocm_aiter_mla_mtp_split.py: grid 断言改为二维并校验 stride 传参
```

```python
req_idx = tl.program_id(0)
chunk_idx = tl.program_id(1)
...
chunk_start = chunk_idx * BLOCK_SIZE
if chunk_start >= num_tokens:
    return
```

## 性能声明

PR 明确这是 kernel 级 microbenchmark,不是端到端吞吐声明。torch-profiler trace(MI355X,ISL 131072,conc=1,TP8,MTP depth 6,同一 build 与 serve config,各 6 次迭代):22.4 µs/call → 3.3 µs/call,−85.4%,即 **6.8×**。并发扫描无 shape 回退(最低 0.98×,约 10.7 µs 噪声地板),384k 上下文单请求达 10.59×;ragged batch(1×384k+7×1k)10.64×、(1×384k+63×1k)8.38×。端到端:当前 main 上 kernel 每步 3 次,可省约 0.24% 的 23.40 ms step,低于 1200 s conc=1 arm 约 1% 的 run-to-run 波动,PR 声明"无可测量的 serving 吞吐提升,不应据此推断"。GSM8k 全量 1319 题 0.9659,不变。

## 验证与回归风险

新增 `test_expand_page_indices.py`,与朴素 PyTorch 参考逐位比对(atol=0, rtol=0),覆盖 chunk 边界、1024±1、ragged 多请求、非连续 block table 视图、零长度请求、过度 provision 的 chunk;另在设备上对 8 组真实形状逐字节比对(97000/4096/1025/1023/38011/4212/1 个 index 全部 identical)。输出逐位不变,精度按构造不受影响。回归风险低:kernel 只改迭代空间;注意上游已把每步调用数从 7 降到 3,收益绝对值比早期测量缩小。

## 关联

该 kernel 由 #39616 引入,此后未改动过;PR 排查过相邻 open PR(#57085、#53492、#57923、#46172、#54899)均无重叠。同批 ROCm attention 方向:#51794。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
