# PR #25205: [AOT compilation] support torch.compile inductor artifacts in VllmCompiledFunction

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @dolpm |
| 合并 | 2026-01-20 |
| 分类 | CUDA Graph / Compiler（tier: major） |
| Label | performance, rocm, ready, ci/build, v1, nvidia |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 因超长截断（5 个文件未展开），可见部分改动 `vllm/compilation/{backends,caching,compiler_interface,decorators}.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/25205 |

## 一句话结论

让 torch.compile/inductor 的 AOT 编译产物可进 VllmSerializableFunction 序列化（去重存储、批量加载、免重切分重编译），cache 命中的 warm 启动较 baseline 快 1.49x/1.38x。

## 动机

vLLM 的 AOT 编译缓存此前只覆盖 VllmCompiledFunction 自己的产物，dynamo+inductor 路径的 piecewise 子图编译结果不在 `VllmSerializableFunction` 序列化范围内，导致 cache 命中时仍要重新切图、重新编译各 shape。目标="所有预编译产物（如 dynamo+inductor）都归入同一屋檐（VllmSerializableFunction）"，改善冷启动并保持输出语义。

## 优化点

四步机制：(1) compile decorator 推迟 AOT save——等所有 shape range 编译完成后再触发（最坏在 warmup 阶段遇 cache miss、user-defined compile_sizes 走完 piecewise interpreter 之后），经 `set_on_compilation_complete` 回调 `save_aot_compiled_function`，且从磁盘加载过则跳过；(2) `cuda_piecewise_backend` 新增 `to_bytes()`，返回每个 shape 到其编译字节的映射；(3) 序列化时遍历所有已编译子图，装入 `StandaloneCompiledArtifacts`——SHA256 内容寻址两级间接（"{submod}_{shape}"→hash→bytes）对相同 transformer 层编译出的重复产物去重，反序列化时用线程池 `load_all()` 批量加载再按 submodule/shape 对重新挂回；(4) 从 cache 加载时经 `reconstruct_serializable_fn_from_mega_artifact` 用缓存产物直接重建 split_gm：`PiecewiseBackend` 以 `compiled_runnables`（graph=None）构造，经共享的 `wrap_with_cudagraph_if_needed` 包 cudagraph，`make_copy_and_call` 处理符号 shape 输入到静态 buffer 的拷贝。配套：`compiler_interface` 支持 `standalone_compile(aot=True)`（PyTorch>=2.10，否则报 CRITICAL 并回退）、hash 因子加入 `get_inductor_factors()`、functorch `bundled_autograd_cache` 仅在非 mega-AOT 时关闭、pickle allowlist 扩展。整体由 `VLLM_USE_MEGA_AOT_ARTIFACT`（与 `VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS`）环境变量门控。

## 关键改动

```text
vllm/compilation/caching.py: StandaloneCompiledArtifacts（SHA256 去重存储/批量加载）+ reconstruct_serializable_fn_from_mega_artifact
vllm/compilation/backends.py: VllmBackend.collect_standalone_compile_artifacts/to_bytes 采集；wrap_with_cudagraph_if_needed 与 make_copy_and_call 抽取共享
vllm/compilation/compiler_interface.py: standalone_compile(aot=True) 返回 AOTCompiledArtifact 供序列化
vllm/compilation/decorators.py: AOT save 推迟到编译完成后回调；cache-hit 标记跳过重复保存
```

```python
def insert(self, submod_name: str, shape: str, entry: bytes):
    hex_digest = hasher.hexdigest()          # SHA256 内容寻址
    self.submodule_bytes[f"{submod_name}_{shape}"] = hex_digest
    if hex_digest not in self.submodule_bytes_store:
        self.submodule_bytes_store[hex_digest] = entry
```

## 性能声明

PR 正文给出的 BENCHMARK SUMMARY（未注明硬件/精度/workload 条件，按日志应为编译/启动耗时基准）：

- Qwen/Qwen3-32B：Inductor Cold=1m21.44s、Warm=35.00s（warm 相对 cold 2.33x speedup）；Baseline Cold=1m10.50s、Warm=52.24s（1.35x）；Inductor vs Baseline：Cold 0.87x（慢 10.94s）、Warm 1.49x（快 17.25s）。
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B：Inductor Cold=1m8.44s、Warm=36.10s（1.90x）；Baseline Cold=1m1.66s、Warm=49.95s（1.23x）；Inductor vs Baseline：Cold 0.90x（慢 6.78s）、Warm 1.38x（快 13.85s）。

即：cache 命中的 warm 启动收益 1.38–1.49x，代价是 cold（首次编译+保存）慢约 7–11s。

## 验证与回归风险

测试（据 PR 附注）：覆盖 AOT save/load、tuple 输出一致性、shape-env 保留、AOT load 上的 partition-wrapper、GPT2 cache hit、backend/cache 单元测试。风险面：功能默认关闭（需 `VLLM_USE_MEGA_AOT_ARTIFACT=1` 且要求 `VLLM_USE_STANDALONE_COMPILE=1`），PyTorch <2.10 时 aot 参数不可用并回退非 AOT 模式；序列化路径对 split_gm 做 deepcopy（内存开销）；`graph_returns_tuple`/`sym_shape_indices` 等元数据不匹配会导致重建失败（有 assert 校验 cache 完整性）。

## 关联

同区域（vLLM 编译/AOT 体系）：索引内另有 #24123（WideEP piecewise CUDA graph）、#34299（torch.compile AR+RMS fusion 默认开启）。PR 正文未列关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
