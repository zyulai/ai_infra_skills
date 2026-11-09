# PR #32017: [Model Loading] Overlap checkpoint staging with CUDA graph capture during startup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yh0903 |
| 合并 | 2026-08-13 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | run-ci, apple-silicon, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/32017 |

## 一句话结论

启动时把 checkpoint 预取(stage 进 page cache)与 CUDA graph capture 重叠,Qwen3-32B 启动缩短 8.6-11.7%(相对 serial+prefetch)。

## 动机

默认模型加载路径会先把真实权重完整加载并后处理,然后才开始 CUDA graph capture。对大型 checkpoint 而言,checkpoint I/O 与 CUDA graph capture 都是启动时间的重要组成部分,但两者此前完全串行——GPU 在 capture 时空闲等 I/O 之后的阶段,host 在 I/O 时空闲等 capture。这不是推理路径优化,而是引擎冷启动(扩容、故障恢复、频繁重建)的时延优化。

## 优化点

新增 opt-in 启动路径 `--startup-weight-load-mode {serial,overlap}`(默认 serial,不改变既有行为)。机制分四段:1) 把 `DefaultModelLoader` 的启动生命周期拆成 model/storage 初始化、checkpoint 解析、capture-safe 初始化(`initialize_capture_safe_weights` 填 sentinel)、capture 后原地提交四步;2) 在 capture 开始前显式启动 checkpoint 预取——把 safetensors 分页 stage 进 OS page cache(复用既有 coordinated prefetcher,加可取消生命周期句柄),与内存池初始化、attention backend 初始化、CUDA graph capture 并发;3) capture 完成后原地加载并后处理真实权重(`commit_model_weights`),不改变被捕获 graph 引用的 tensor 存储;4) 提交后做双重校验——`ModelStorageManifest` 断言每个 parameter/buffer 的对象身份、data_ptr、shape、stride、dtype、device、storage offset 均未变,并断言所有浮点参数都已从 capture sentinel 改写,任何不一致直接 raise。TP barrier 从 load_model 移到 finalize(commit 失败的 rank 不得上报就绪)。支持矩阵收得很紧(CUDA、DefaultModelLoader、auto/safetensors+mmap、dense Llama/Qwen2/Qwen3、FP16/BF16、TP1/TP2、单 checkpoint 源),不支持配置显式报错。为三段式 rollout 的 PR 1(PR2 泛化加固、PR3 auto 默认),prefetch 失败自动回退正常加载。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py(新):
  StartupWeightLoadManager 状态机(prepare/start_prefetch/finalize)、
  TensorStorageManifest 存储不变量校验、支持矩阵门控
python/sglang/srt/model_executor/model_runner.py + managers/{scheduler,tp_worker}.py:
  start/finalize_startup_weight_load 生命周期接线,barrier 移至 finalize
python/sglang/srt/model_loader/loader.py:
  DefaultModelLoader 拆出 initialize_model_for_startup/resolve_model_weights/
  prepare_model_for_capture/commit_model_weights;ResolvedSource 复用
```

```python
changed_names = manifest.changed_names(self._model)   # data_ptr/shape/stride/... 不变
if changed_names:
    raise RuntimeError("Startup weight commit changed graph-visible tensor storage")
unchanged = manifest.unchanged_parameter_names(CAPTURE_SAFE_WEIGHT_SENTINEL)
```

## 性能声明

配置:Qwen3-32B BF16,H100,Triton decode CUDA graph 到 bs 256,prefill graph 关闭,每配置 3 个计时样本(Latin-square 排序),每次运行前 `POSIX_FADV_DONTNEED` + `mincore` 验证 checkpoint 页全部冷,预取 4 线程/rank。主结果(B: serial+prefetch → C: overlap,即本 PR 启动顺序的净贡献):local NVMe TP1 38.94±0.56 s → 35.60±0.83 s(-8.6%,1.09x);TP2 51.12±1.16 s → 45.16±0.29 s(-11.7%,1.13x);TP1 用 8 聚合预取线程时 43.25 s → 36.71 s(-15.1%,1.18x)。端到端语境(A: 默认 serial → C):TP1 84.83 → 35.60 s(-58.0%,2.38x)、TP2 78.60 → 45.16 s(-42.5%,1.74x),但作者注明 A→C 的大部分收益来自既有的 coordinated prefetch,B→C 才是本 PR 的归因结果;NFS 重跑因共享存储竞争被排除,不作 NFS 声明;样本量小,不声称紧统计置信度。

## 验证与回归风险

新增注册 1-GPU e2e 测试对比 serial 与 overlap 的 token ID、文本与 logprob(1e-5 容差);CPU 单测覆盖兼容性选择、显式拒绝、生命周期状态转换、graph-visible 存储不变量、checkpoint 源复用、prefetch 取消/等待、serial/overlap 调度路由、server args 与 MLX 路由(MLX 显式 raise 不支持)。Qwen3-32B 三种启动模式输出一致;H100 上手动验证 safetensors engine 与 one_batch 路径。风险面:重叠正确性完全依赖"commit 不改 graph 可见存储"这一不变量与 manifest 校验;capture-sentinel 检查要求所有浮点参数都被 checkpoint 改写,`__init__` 中计算出的浮点参数会 fail-loud(同时是扩展架构白名单的准入门);失败路径沿用既有 startup 合同(不恢复、不清理),commit 失败表现为 peer 端 barrier timeout;scope 门控之外的配置(EP/PP/DP/量化/spec 等)全部显式拒绝。

## 关联

作者注明为三段式 rollout 的 PR 1(PR2 泛化与加固、PR3 auto 默认 rollout,编号未定);同区域(启动/CUDA graph):本批次 #22128(PCG 与 spec decoding 共存)。pack 未列出其他直接关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
