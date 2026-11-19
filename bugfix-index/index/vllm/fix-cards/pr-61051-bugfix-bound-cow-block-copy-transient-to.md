# PR #61051: [Bugfix] Bound CoW block-copy transient to avoid OOM near the memory limit

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61051](https://github.com/vllm-project/vllm/pull/61051) |
| 作者 | michael-gpu |
| 合并日期 | 2026-11-17 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
GPU 上的 KV cache 块拷贝 `blocks[dst] = blocks[src]` 会先把所有源块 gather 成一个设备临时张量,部分前缀命中触发的一波 copy-on-write 拷贝可能一次申请数 GiB,在贴近 `gpu_memory_utilization` 的部署上直接 OOM;本 PR 用一个 Triton 内核让每个连续字节段从源直接搬到目标,不再产生块大小的临时内存。

## 问题现象
`copy_kv_cache_blocks_inplace`(`vllm/v1/worker/utils.py`)原先执行 `blocks[dst] = blocks[src]`,右侧是对 `len(src)` 个块的整体化 gather。KV cache 块可达数十 MiB(Mamba state,或跨所有层的块),部分前缀缓存命中引发的 copy-on-write 拷贝波在接近内存上限的部署上 OOM。新增测试 `test_copy_kv_cache_blocks_inplace_cuda_skips_gather_temporary` 在 `main` 上失败:`assert (20972032 - 16777216) < 1048576` 不成立,即拷贝产生了块大小级别的额外分配。

## 根因
正文披露根因:`blocks[dst] = blocks[src]` 的右侧是一个物化 gather,会把所有源块先汇聚成一个设备临时张量再 scatter;拷贝量本身就是块大小总和。而 profiling 和 warmup 从不触发 copy-on-write,内存预留中没有任何一部分为它留出,因此这个瞬时峰值在可扩展 KV cache 的精确预留模型下无处安放。

## 修复方案
GPU 路径改为一次 Triton 内核直拷:

- 每个缓存视图的拷贝拆成连续字节段(尾部连续维度折叠成一段),覆盖 padded、strided、overlaid、kernel-block-split 视图与 ReplaySSM tracker;`_block_copy_runs` 产出 `[3, num_runs]` 的源地址/目标地址/字节数。
- 所有段合并为一次 Triton launch(`_copy_runs_kernel`),按能整除所有地址与长度的最宽字长拷贝;仅描述符表(每段 24 字节)上设备。
- 在当前 stream 上发射,保持与被替换的 indexed copy 相同的顺序语义(块清零之后、前向之前)。
- 启用前缀缓存时在 kernel warmup 中预编译内核(`kernel_warmup.py`),避免服务中首次 copy-on-write 触发 JIT。
- 链式拷贝(同一块既是源又是目标)回退 indexed copy;CPU/XPU 缓存维持 indexed copy。

效果:拷贝比索引拷贝快 21-28x(约 3.2 TB/s vs 117 GiB/s);Qwen3.5-35B-A3B 异步调度下 CoW 波次中位耗时 1148 ms vs `main` 1383 ms(-17%);GSM8K 20-shot 精度 0.7877 vs 0.7885(基本持平)。

## 影响与回归风险
- 内存:GPU 块拷贝不再分配块大小临时内存,峰值瞬时可从 GiB 级降到 0;这是在 `gpu_memory_utilization` 贴边部署下的关键收益。
- 已知局限(正文):同一 storage 的 overlaid 视图可能以相同数据写同一目标字节两次;块级不相交假设同一 storage 的两个视图不会错位(当前没有 allocator 会产生这种情况)。ROCm 跑同一 Triton 内核但未本地测试(由 AMD 镜像 CI 步骤覆盖);XPU 和 CPU 保持 indexed copy。
- 演进过程:该 PR 先把 gather 限制在 256 MiB 分块(仍需未预留内存且慢),再试过 `cuMemcpyBatchAsync`(2 MiB 段只达约 476 GiB/s,且约 1000 个描述符排队后会阻塞 host 达 488-524 ms,破坏异步调度),最终换为 Triton 方案。
- 测试:`tests/v1/worker/test_attn_utils.py`、`tests/v1/worker/test_utils.py` 在 GB200/CUDA 13.0 上 134 passed;新增非 8 字节对齐用例。

## 关联
- PR 描述与当前方案(commits 自 `21e3b27` 起)由 @zjy0516 以 maintainer edits 推送;原修订版本由 PR 作者撰写,曾限制 gather 为 256 MiB 分块。
- 涉及文件:`vllm/v1/worker/utils.py`、`vllm/model_executor/warmup/kernel_warmup.py`、`tests/v1/worker/{test_attn_utils,test_utils}.py`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
