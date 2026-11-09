# PR #30206: fix(server): capture legal multi-request prefill CUDA graph batches

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @nvpohanh |
| 合并 | 2026-08-03 |
| 分类 | CUDA Graph / Compiler |
| Label | performance, run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/30206 |

## 一句话结论

修复 prefill CUDA graph 捕获时合成请求超出 `--context-length` 导致的 illegal memory access(#21112),改为把 bucket 拆成多个合法合成请求,保留多请求 prefill 的图覆盖。

## 动机

当 `--context-length` 小于 prefill CUDA graph 捕获 bucket 时,旧捕获路径生成一个 `seq_len=num_tokens` 的合成请求,超出单请求上下文上限,在 graph warm-up 期间触发 #21112 的 illegal-memory-access。直接把聚合 token 上限 clamp 到 `context_length` 虽能避免崩溃,却把"总 token 数超过单请求上下文但本身合法的多请求 prefill"也一并禁用了图 replay,损失覆盖。

## 优化点

修复捕获形状而非对聚合 token 设置做 per-request 上限。具体:在 `ModelRunner.init_prefill_cuda_graph()`(capture_prefill_graph)一次性解析合法 bucket 并回写 `cuda_graph_config.prefill.bs`,使所有后端消费同一配置;所有 prefill 后端的捕获 bucket 保持"聚合 token"语义,上限 cap 到 `context_length * max_capture_requests`(tc_piecewise/breakable 取 request-pool 大小,full 取解析后的固定 request-slot 数);捕获时把 `num_tokens` 的 bucket 划分成最少数量的合成请求,使每个请求 `seq_len <= context_length`(`range(0, num_tokens, context_length)` 切分);full 后端保持固定 request 轴契约,在上下文受限的合成请求之后用零长度 sentinel 补满 slot;过滤后无合法 bucket 则记录后端/上下文长度/request-pool 容量并回退 eager prefill;移除此前 ServerArgs 阶段 `context_length * max_running_requests` 的计算(该阶段 `max_running_requests` 不一定已解析);DeepSeek MHA prefill 在 BCG 下于 CUDA 与 HIP 都注册并选择 `attn_mha` companion,使捕获的 attention op 拿到 MHA head metadata 而非同 layer id 的 `attn_mqa` metadata。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py: capture_prefill_graph 中按 context_length/request 容量过滤合法 bucket 并回写 config
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py: capture_prepare 把 bucket 拆成最少合成请求(full 后缀零长度 sentinel)
python/sglang/srt/server_args.py: 移除 ServerArgs 期 context_length clamp(max_running_requests 未解析)
python/sglang/srt/model_executor/cuda_graph_config.py: 注释更新(bs 对所有 prefill 后端统一为聚合 token)
```

## 性能声明

PR 未注明吞吐/延迟加速数字;给出捕获与验证数据:4xGB200 DSR1(DeepSeek-R1-0528 ModelOpt NVFP4,TP4,FlashInfer,`context_length=2048`,`chunked_prefill_size=4096`,backend=breakable、bs=[4096],唯一 bucket 构造为 2 个合法合成请求):4 rank 完成 4096 token 的 "Capture target prefill CUDA graph" 各约 20.9 秒,捕获显存约 1.41 GiB/GPU;服务器就绪后双 prompt `/generate` 正常返回(prompt 长度各 1301,聚合 2602 > 单请求 2048 上限),两次运行时 prefill 均记录 `cuda graph: True`。H100 80GB(Qwen2.5-1.5B-Instruct,tc_piecewise):所有 bucket 至 8192 聚合 token(4x context_length)捕获完成。

## 验证与回归风险

验证:`git diff --check`、5 个改动文件 `py_compile`、Black 格式检查(lmsysorg/sglang:dev-cu13)。风险面:bucket 列表回写 config 属启动期行为变化,捕获失败时静默降级为 eager(有 warning 日志);`capture_seq_lens` 切分依赖 `bs <= max_bs` 断言;full 后端 `full_prefill_max_req` 现在必须在 runner 构造前解析(assert);DeepSeek MHA prefill companion 注册同时影响 CUDA/HIP 两条路径。

## 关联

同区域:PR #27988(full prefill CUDA graph 引入,其 bs=1 捕获契约被本 PR 扩展为多请求)、PR #29458(Breakable CUDA Graph 转 default)。修复 #21112 / #22516 的捕获形状根因。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
