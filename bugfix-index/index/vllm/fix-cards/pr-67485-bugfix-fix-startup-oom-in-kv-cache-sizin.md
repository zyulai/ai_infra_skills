# PR #67485: [Bugfix] Fix startup OOM in KV cache sizing warmup by releasing kernel warmup's KV blocks

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#67485](https://github.com/vllm-project/vllm/pull/67485) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-11-13 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
extensible KV cache 的 warmup 提交只增不减,sizing warmup 被遗留块挤到启动 OOM;sizing 前 shrink 回最小提交修复。

## 问题现象
#67071 把 `gpu_memory_utilization: 0.90` 从 Kimi-K3-pruned75 DSpark TP4 eval config 移除后,服务器启动即死于 `run_kv_cache_sizing_warmup` 的 CUDA OOM(#67307 中的一项)。复现(4×B200,178.39 GiB usable,vLLM `0.34.0rc1.dev398+g153dc0464`):基线 `153dc04` 3/3 次 CUDA OOM(失败时 1.58–1.67 GiB 空闲,Tried to allocate 2.04 GiB,栈顶 `trtllm_mxfp4_moe.py` apply);`vllm/vllm-openai:nightly-153dc046…` 镜像同样 3/3 复现。修复后 6/6 次(其中 3 次由独立会话复跑)`Application startup complete.` 并正常返回一条 chat completion(`finish_reason: "length"`,64 tokens),日志无 CUDA OOM/Traceback;修复前后日志均显示 `Available KV cache memory: 47.88 GiB`(同一默认配置)。默认配置 rank 0 最终 KV cache 629 blocks(25.92 GiB);用 0.90 时 402 blocks(16.59 GiB)。单元测试无修复时 `assert 1025 == 1` 失败(sizing warmup 带着 1024 个 kernel warmup 请求的块加 null block 起步),修复后整个文件 35 passed。

## 根因
正文披露(rank 0 临时 debug 日志数字,非本 PR 内容):(1) KV cache init 时 `warmup_committable_blocks` 由空闲内存 51.76 GiB 减 profiled activation peak 6.51 GiB 与 margin 4.88 GiB 得出,warmup 可提交 967 个 42.2 MiB 的块;(2) `warmup_kernels` 跑最多 `max_num_seqs` 个小请求,Kimi-K3 有 3 个 KDA(Mamba-like)组、每请求各持 1+8 个 speculative 块(块 3200 token),33 个请求提交 958 块(39.47 GiB),几乎占满预算;(3) 提交是 grow-only,唯一的释放在 CUDA graph capture 之前,而 capture 排在 sizing warmup 之后;JIT warmup 又把 torch reserved 涨约 6 GiB;全量 sizing 步骤(16384 tokens、`prompt_logps=20`)以 6.34 GiB 空闲起步且只寻址 83 个已提交块,FlashInfer TRT-LLM MXFP4 MoE kernel 请求 2.04 GiB workspace 失败;(4) 0.90 之所以有效:KV cache 容量 30.04 GiB 封顶 `committable_blocks()`,warmup 少提交约 12 GiB,步骤就放得下。

## 修复方案
`vllm/v1/worker/gpu_worker.py` `_compile_or_warm_up_model`:在 `run_kv_cache_sizing_warmup` 之前调用 `kv_cache.commit(kv_cache.min_committed_blocks, shrink=True)`(复用 #66523 在 CUDA graph capture 前释放 warmup 块的同一调用),`kv_cache = ...` 查找与 assert 随之上移;释放的块只含已完成 warmup 请求数据,每个 sizing 步骤仅提交它寻址的块(`ensure_kv_cache_blocks`)。修复后 debug 运行中全量步骤以 42.34 GiB 空闲起步,测得 transient 6.27 GiB < profiled 6.51 GiB。新增回归测试 `test_sizing_warmup_gets_the_memory_kernel_warmup_committed`。未取的备选:让 `ensure_kv_cache_blocks` 可 shrink(会在 `warmup_kernels` 内重映射约 39 GiB 两次)、调低 `warmup_committable_blocks`、把 sizing warmup 提到 `warmup_kernels` 之前(JIT 编译会进入测量窗口)、恢复 0.90(关闭文档默认的 stopgap)。

## 影响与回归风险
- 影响窗口:自 #67071 从该 eval config 移除 `gpu_memory_utilization: 0.90` 起;仅影响 enable_extensible_kv_cache 路径。
- 回归面:`gpu_worker.py` `_compile_or_warm_up_model` 中 sizing warmup 前的一次 shrink(`run_kv_cache_sizing_warmup` 仅此一个调用方;CPU 与 XPU worker 无 extensible KV cache);最终 KV cache 大小的测量方式不变(sizing 读 torch allocated peak,最终大小用 free + committed,均不依赖 warmup 持有的块数);遗留限制:`warmup_committable_blocks` 仍在 KV cache init 时固定(先于 JIT warmup 涨约 6 GiB),`warmup_kernels` 结束时仍剩约 6.3 GiB 空闲——另一机制,本 PR 未处理;验证仅覆盖 4×B200 的启动与单条 chat,CI 的 GSM8K eval 未跑。

## 关联
- [#67307](https://github.com/vllm-project/vllm/issues/67307)(修复其中一项)
- [#67071](https://github.com/vllm-project/vllm/pull/67071)(移除 0.90 的 PR)
- [#66523](https://github.com/vllm-project/vllm/pull/66523)(复用其释放调用)
- [#67296](https://github.com/vllm-project/vllm/pull/67296)(allocator fragmentation headroom,本 PR 未改)
- [#67244](https://github.com/vllm-project/vllm/pull/67244)(ROCm config stopgap)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
