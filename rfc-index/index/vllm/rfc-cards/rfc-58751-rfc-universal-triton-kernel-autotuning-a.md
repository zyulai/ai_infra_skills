# RFC #58751: [RFC]: Universal Triton kernel autotuning at warmup

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58751](https://github.com/vllm-project/vllm/issues/58751) |
| 作者 | giuseppegrossi |
| 创建日期 | 2026-09-25 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议 opt-in 的 `--kernel-config.enable_triton_autotune`:warmup 时为所服务模型实际使用、但 JSON 配置表中缺失的 Triton kernel 自动调优并持久化。

## 提案动机
vLLM 运行的大量 Triton kernel 的 launch 参数(`BLOCK_*`、`num_warps`、`num_stages)依赖 shape、device 与 dtype,目前参数有三种选择方式:1) `@triton.autotune`(约 50 处调用点,Triton 在 `profile_run` 中首次见到每个 key 时惰性计时候选列表);2) 离线 JSON 表(fused MoE、W8A8 block FP8、LoRA、Mamba `selective_state_update`,按 `(shape, device, dtype)` 查配置);3) 纯 `@triton.jit`(约 500 处,无法自动调优)。缺口在第 2 类:vLLM 只捆绑一组精选的 shapes/devices,当服务的模型+加速器组合不在表里时,kernel 静默回退到硬编码启发式默认,可能显著变慢。如今唯一的 workaround 是手动跑 `benchmarks/kernels/benchmark_*.py` 并把结果放进 `VLLM_TUNED_CONFIG_FOLDER`,而且用户要为各种架构(含新架构)维护大量配置文件,更想要开箱即用、为所服务模型用到的所有 kernel 自动选好配置的方案。FlashInfer 已用 `--kernel-config.enable_flashinfer_autotune` 解决同类问题(对实际服务的模型在 kernel warmup 时调优),Triton 没有对应机制。

## 提出的方案
- 新增 opt-in flag `--kernel-config.enable_triton_autotune`(默认 `False`),仿照 `enable_flashinfer_autotune`。启用后在 kernel warmup(模型加载与 KV cache 分配之后、**CUDA graph capture 之前**)调优所服务模型实际会使用、但尚无配置的每个 Triton config-table kernel,结果持久化到本地缓存供后续启动复用。范围有意限定为第 2 类(读 JSON 配置表的 kernel)。
- "family" 是一张 config 表/查找函数而非单个 kernel;每张表实现一个小接口 `TunableConfigTable`:`pending_items(worker)` 返回所服务模型会查但无配置的条目(`TuningItem`:table、shape、bucket);`tune(item)` 在 scratch buffer 上对单项基准测试(无效返回 None);`commit(results)` 持久化结果并清空表的查找缓存,在所有 rank 上运行。
- 分布式正确性要求:每个 rank 都必须到达每个 collective、最终持有相同配置(使捕获的 CUDA graphs 在 TP/PP rank 间匹配)。流程:各 rank 计算自身 `pending_items`(shape 为 per-rank,如 TP sharding 后,但不含 device index 等 rank 特有内容)→ `all_gather_object` 取并集 → 确定性排序后按 `items[rank::world_size]` 分片,每项恰好被调优一次 → 各 rank 调优自己的分片,失败被捕获以保证不跳过任何 collective → `all_gather_object` 收集结果,**每个** rank 都调 `commit`,保证所有 rank 写同样的文件、清同样的缓存。
- 执行时机:从 `kernel_warmup()` 中 `process_local_only` 早退之后启动,因此总在每个 rank 的正常启动路径上、`capture_model` 之前运行。
- 持久化:结果存于 `VLLM_TRITON_AUTOTUNE_CACHE_DIR`(默认 `$VLLM_CACHE_ROOT/triton_autotune/triton=<version>/<table>/`),复用每张表现有的磁盘 JSON schema(按 shape/device/dtype 键控),同一文件也可经 `VLLM_TUNED_CONFIG_FOLDER` 使用;config loader 增加一条查找路径(user folder → autotune cache → bundled default),二次启动命中已调优文件即跳过;`VLLM_TRITON_AUTOTUNE_FORCE=1` 可在已有配置时强制重调;默认尊重精选捆绑配置,调优只补缺口。

## 优化点与收益
正文未给出量化数字。定性收益:表外的模型+加速器组合不再静默使用可能显著变慢的启发式默认配置;免去手动运行 benchmark 脚本、维护大量 per-架构配置文件的负担;为所服务模型实际使用的所有相关 kernel 开箱即用地选出合适配置。

## 现状与落地
🟢 open(2026-09-25 创建),labels 仅 RFC,Feedback Period 一栏为 No response,处于等待评审阶段;正文说明本 RFC 是作者初始 PR #48980 向更通用方案的 re-scope,并注明草稿经 AI 辅助撰写、由作者审校修订并担责设计。

## 关联
- 作者初始 PR: https://github.com/vllm-project/vllm/pull/48980
- 同类机制的先例:`--kernel-config.enable_flashinfer_autotune`(正文提及,未附链接)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
