# PR #27139: [FEAT] Support fast engine recovery through weight cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @QiuMike |
| 合并 | 2026-07-25 |
| 分类 | Quantization(classify 归类;实际为引擎快速恢复/weight cache 特性) |
| Label | documentation, npu, run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用,另有 7 个文件因超长被剔除) |
| 链接 | https://github.com/sgl-project/sglang/pull/27139 |

## 一句话结论

Weight Cache Daemon 把引擎重启的权重加载从 ~306-327s 降到 <1s,Qwen3-235B FP8 端到端启动 ~390s → ~80s。

## 动机

引擎崩溃、滚动更新或重启时必须重新从磁盘加载权重、初始化分布式、重捕 CUDA graph、预热 JIT kernel,大模型(Qwen3-235B FP8,4×GPU TP=4)完整启动约 390s(~6.5 分钟),其中磁盘权重加载占 ~79%(306-327s)。重启窗口内所有 in-flight 请求失败或排队,直接造成 P99 尾延迟尖刺与可用性/SLA 问题。框架目标:冷重启 <10s、warm standby 切换 <1s;本 PR 是第一阶段(消灭最大瓶颈),后续 CUDA graph 序列化、kernel 预热持久化等阶段仍在 Planned。

## 优化点

核心机制是"持久化 GPU 进程 + CUDA IPC 零拷贝":每个 GPU 一个 daemon 进程,从磁盘完成完整加载流水线(disk → TP shard → 量化),缓存 `process_weights_after_loading()` 之后的完整 post-quantization 状态(含 weight_scale、workspace、repacked weights 等),并通过 Unix socket 向引擎导出每个参数的 `cudaIpcMemHandle`。引擎侧在 meta device 上初始化模型(不占 GPU/CPU 内存),再把各参数的 data 指针直接映射到 IPC 显存,零拷贝。提供三种模式:`off`(默认磁盘加载)、`daemon`(engine 代启 daemon,与 engine 同生命周期,本身不加速重启)、`client`(连接独立常驻 daemon,重启 <1s)。CacheConfig 指纹校验 model_path/model_arch/tp_size+tp_rank/dp_size/quant_method+quant_config_hash/dtype,任一不匹配即回退磁盘加载保证正确性。IPC 路径本身也做了优化:预构建 param/buffer 字典做 O(1) 查找,替代逐条遍历 `model.named_parameters()`,参数映射 4.7s → 0.95s,IPC 总加载 5.05s → 1.30s。

## 关键改动

```text
python/sglang/srt/weight_cache/daemon.py: 新增 WeightCacheDaemon,加载权重并经 Unix socket 服务 IPC handle
python/sglang/srt/weight_cache/ipc_loader.py: 新增 IpcModelLoader(BaseModelLoader 子类),meta device init + IPC 零拷贝映射
python/sglang/srt/weight_cache/protocol.py: CacheConfig 指纹、socket 协议与序列化
python/sglang/srt/configs/load_config.py: LoadFormat.IPC_CACHE 内部分发格式与 weight_cache_mode/socket 字段
python/sglang/srt/entrypoints/engine.py: daemon 模式下 _launch_weight_cache_daemons 与优雅回收
python/sglang/srt/server_args.py: --weight-cache-mode/-socket/-timeout 参数与校验(禁与投机解码同用)
```

## 性能声明

PR 正文数据(Qwen3-235B FP8、4×GPU TP=4,除非另注):
- 磁盘 vs IPC 零拷贝权重加载:7B FP16(~14GB)~30s → <1s;70B FP16(~140GB)~180-300s → <1s;235B FP8(~235GB)~306-327s → <1s(约 3000×)。
- E2E 启动(client 模式):总时长 ~390s(6.5min)→ ~80s(1.3min),节省 ~310s;其中 Load weight ~306-327s → ~0.63s。其余阶段(DeepGEMM JIT 预热 ~22.2s、CUDA graph 捕获 ~33.9s)基本不变。
- 环境参考:H20 HBM,GPU 内拷贝带宽 ~500-900 GB/s,IPC handle 映射 ~10k handles/ms。
- 目标(框架整体):冷重启 <10s、warm standby <1s。

## 验证与回归风险

PR Breakdown 列出单元测试(CacheConfig、protocol、dispatch、跨进程 IPC)与真实模型加载的 E2E 测试,状态 Done;diff 中未包含测试文件内容。风险与限制(均由代码显式守护):weight cache 激活时拒绝 `update_weights_from_disk/distributed/tensor/ipc` 与 `release/resume_memory_occupation`(避免破坏 daemon 母本);IPC 零拷贝模式下禁用 memory saver 的 CPU 权重备份;不支持 `dp_size>1` 的 daemon 模式、不支持与 `--speculative-algorithm` 同用(daemon 不导出 draft 权重);仅支持 CUDA-alike 平台(CUDA/ROCm);daemon 崩溃不影响 client 推理(CUDA 引用计数保活显存),配置不匹配自动回退磁盘加载。

## 关联

同批无同区域 PR。本 PR 为 Fast Recovery Framework 的 Phase 1(Weight Cache Daemon);Phase 2-6(CUDA graph 序列化、kernel 预热持久化、server init 优化、多节点/DP/EP 支持、daemon 容错)在 PR 正文中为 Planned,无对应已合并编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
