# PR #48789: [Profiler] Add minimal Triton Proton profiling backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Luosuu |
| 合并 | 2026-08-11 |
| 分类 | Attention / Kernel(major;pack 注明 diffstat 缺失,不改变模型输出) |
| Label | documentation, performance, frontend, intel-gpu, ready, v1, cpu, verified |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/48789 |

## 一句话结论

为 worker profiler 新增 Triton Proton 低开销 GPU 剖析后端(NVIDIA/CUPTI),不影响模型输出。

## 动机

此前 vLLM 只有 PyTorch profiler 和 CUDA profiler 两种 worker 剖析手段。Proton 是 Triton 自带的 GPU profiler,能以低开销收集聚合 tree 或 Chrome trace,但 vLLM 尚未接入。本 PR 刻意做成最小可评审集成:仅支持 NVIDIA(CUPTI)与 graph-free 执行,把 CUDA graph 归因、分布式回滚等拆到后续 PR。

## 优化点

新增 `ProtonProfilerWrapper`,复用既有 profiler 生命周期,不重写 start/stop/shutdown。关键机制:triton.profiler 惰性导入,其他 profiler 用户无需 Proton-capable Triton;每次 profiling 请求独立 finalize,输出名由 prefix、rank、pid、实例 uuid、run 序号限定,避免分布式 worker / 重启 / 多次 run 互相覆盖;按安装的 Triton 版本 gate 新能力(`hatchet_msgpack`、`periodic_flushing` 需 Triton 3.7+);仅在选中 proton 时校验 Proton 专属选项,非 CUDA 平台或开启 CUDA graph 时启动即报错;剖析窗口外跳过 annotation 构建,`annotate_profile` 返回 nullcontext。

## 关键改动

```text
vllm/profiler/wrapper.py: 新增 ProtonProfilerWrapper(start/finalize/annotate 委托 triton.profiler)
vllm/config/profiler.py: ProfilerKind 增加 "proton",新增 proton_* 配置项与组合校验
vllm/config/vllm.py: proton 强制 CUDA 且 cudagraph_mode=NONE
vllm/v1/worker/gpu_worker.py: proton 分支;stop 后重建 wrapper 使新 profile_prefix 生效
docs/contributing/profiling.md: Proton 用法文档
```

## 性能声明

PR 正文未给出性能对比数字(本 PR 为剖析基础设施)。功能性结果:focused profiler tests 62 passed;NVIDIA H20 上以 `facebook/opt-125m` 冒烟,`/start_profile`、生成、`/stop_profile` 均返回 HTTP 200,Hatchet 输出可在 Triton viewer 打开并显示 CUDA kernel 计时,Chrome trace 为合法 JSON 且含 1,416 个 complete events。

## 验证与回归风险

新增/扩展 `tests/v1/worker/test_gpu_profiler.py`:配置组合校验、输出名唯一性、Triton 版本 gate、worker 重建 wrapper 等。回归风险:使用 Proton 必须加 `--enforce-eager` 或 `cudagraph_mode=NONE`(CUDA graph 归因在 #51084);ROCm 不支持(需 Ray executor 的设备可见性传播,延后到专门 follow-up);Triton < 3.7 无法使用部分输出格式,会在建 session 前报错;不改变模型输出与 serving 语义。

## 关联

拆分的后续 PR:#51084(CUDA graph capture/replay 归因)、#51085(分布式 partial-start rollback、HTTP 错误映射、通用生命周期恢复);两者仅依赖本 PR 且互相独立。卡内未列出同分类其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
