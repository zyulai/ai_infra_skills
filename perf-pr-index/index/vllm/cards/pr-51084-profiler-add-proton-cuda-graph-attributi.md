# PR #51084: [Profiler] Add Proton CUDA graph attribution for MRV2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Luosuu |
| 合并 | 2026-09-14 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | documentation, performance, ready, nvidia, mrv2, verified |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(1 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/51084 |

## 一句话结论

为 V2 model runner 的 Proton profiler 增加 CUDA graph replay 归因,使回放 kernel 保留 capture 上下文,Proton 不再要求 `--enforce-eager`。

## 动机

CUDA graph 回放时,profile 中的 kernel 失去 capture 时的上下文,Proton 此前直接不支持 graphs——开 Proton 必须禁用 CUDA graph,否则回放 kernel 会被静默漏掉。这使生产配置(FULL/PIECEWISE graphs)无法用 Proton 做低开销聚合分析,只能退回 eager。本 PR 是 #48789 的扩展。

## 优化点

机制:一个 Proton session 同时观察 capture 与 replay。worker 在真正的 decoder/encoder graph capture 前进入 `WorkerProfiler.capture_cuda_graphs()` 钩子(默认 no-op),Proton wrapper 在此创建/激活常驻 session;capture 阶段的活动通过 phase API 丢弃(`advance_phase` 后 `clear`),因此 profile 只含 replay;之后每个 profiling interval 在 `/stop_profile` 时 flush、导出并清理自己的 phase,同时保留 graph 元数据(`<captured_at>`、Triton launch metadata),实现 interval 间隔离与 rank 唯一输出文件。约束与校验:attribution 需要 `proton_data='tree'`、Triton >= 3.7(phase API)、MRV2;启用 graphs 时必须开 attribution(否则报错),PC sampling 与 Chrome trace 仍要求 graphs 关闭;`periodic_flushing` 与 attribution 互斥(两者都管理 phase,phase 导出/清理必须单一 owner),校验前移到配置层,把原先的 SIGABRT 变成 ValueError。encoder-only graph 用独立 predicate 接入同一钩子,无需 encoder 实现改动。

## 关键改动

```text
vllm/profiler/wrapper.py: ProtonProfilerWrapper 新增 capture_cuda_graphs()/phase 管理/shutdown 清理
vllm/v1/worker/gpu_worker.py: _get_cudagraph_capture_context() 包住 capture_model(),profile() 复用 graph session
vllm/config/profiler.py + vllm/config/vllm.py: proton_graph_attribution 配置与校验(从 __post_init__ 移到 runner 选择后)
vllm/v1/worker/gpu/model_runner.py: needs_cudagraph_capture() 判断是否还有 graph 待捕获
```

## 性能声明

无性能数字(可观测性能力 PR,不改变执行路径)。验证环境:2x NVIDIA H20,Python 3.12.13,PyTorch 2.13.0+cu130,Triton 3.7.1。功能验证:GPU replay 测试覆盖 shadow/python context × Hatchet/MessagePack × decoder/encoder-only,精确 kernel count 断言(每个 interval 恰含 1 个 replay kernel、不含 capture-only 活动)、正 GPU 时间、数值正确性;真实 HTTP serving(facebook/opt-125m,9 种场景 × 2 intervals)135 个 completion 与确定性 eager baseline 全部一致,54 次 start/36 次 stop 返回 200,20 份 profile 全部可读且含 capture 上下文,关机后无残留 session 文件,无 missing-graph warning。

## 验证与回归风险

`tests/v1/worker/test_gpu_profiler.py` 大幅扩展(配置拒绝组合、phase 生命周期、重复 start、导出失败恢复、shutdown finalize);`test_gpu_model_runner_v2_cudagraph_profiling.py` 为 GPU 端到端;合计 134 passed。风险:校验逻辑从 `VllmConfig.__post_init__` 移至 `_validate_profiler_config`(runner 选择之后),报错时机变化;Proton+graphs 现在强制要求 attribution,普通 Proton 用户配置 graphs 会收到新报错(需改配或 `--enforce-eager`);periodic flushing 与 attribution 互斥被硬性拒绝;仅支持 NVIDIA CUDA 与 MRV1 不支持;multimodal 完整服务未测,encoder-only 用 minimal runner state 验证。

## 关联

同区域:#48789(本 PR 扩展的前作,Proton 基础接入);profiler/CUDA graph 相关卡(pr-35963 ViT full CUDA graph、pr-21146 GC freeze 等)。PR 正文未列出其他关联编号。
