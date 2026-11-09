# PR #50157: [Kernel] Add support for Flashinfer Mamba SSU algorithm selection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amitz-nv |
| 合并 | 2026-08-04 |
| 分类 | Attention / Kernel |
| Label | performance, ready, nvidia |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/50157 |

## 一句话结论

新增 `--mamba-ssu-algorithm` 参数选择 FlashInfer Mamba selective state update 算法,强制 "horizontal" 在 Nemotron 3 Nano NVFP4 ISL/OSL=1k/8k 场景吞吐 +6.1%。

## 动机

FlashInfer Mamba 后端的 selective state update(SSU)内核自带多种算法(auto/simple/vertical/horizontal),vLLM 此前未暴露选择入口,一律走 FlashInfer 的 "auto"(在部分 workload 上会选择 "vertical")。在 Nemotron 3 Nano NVFP4 的长输出场景(ISL/OSL=1k/8k)上,"horizontal" 明显更优,但没有开关就无法利用。

## 优化点

这是配置暴露型 PR,不改动内核本身:新增 `MambaSSUAlgorithm = Literal["auto","simple","vertical","horizontal"]` 类型与 `MambaConfig.ssu_algorithm` 字段(None 默认走 FlashInfer "auto");`EngineArgs` 增加 `--mamba-ssu-algorithm` CLI 参数并透传到 mamba_config;`MambaConfig.validate_ssu_algorithm` 校验取值合法性并强制仅在 `--mamba-backend flashinfer` 下可用。分发层 `FlashInferSSUBackend` 加 `_algorithm` property(None 映射为 "auto"),调用 `flashinfer.mamba.selective_state_update` 时透传 `algorithm=` 参数,启动时打一条 info 日志。Triton 后端不受影响。

## 关键改动

```text
vllm/config/mamba.py: MambaSSUAlgorithm 类型、ssu_algorithm 字段与校验(仅 FlashInfer 后端可用)
vllm/engine/arg_utils.py: --mamba-ssu-algorithm CLI 参数与配置透传
vllm/model_executor/layers/mamba/ops/ssu_dispatch.py: FlashInferSSUBackend 调用内核时透传 algorithm
tests/kernels/mamba/test_ssu_dispatch.py: mock 内核断言 algorithm 逐值透传(None→auto)
```

## 性能声明

单卡 H100,Nemotron 3 Nano NVFP4(nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4),AIPerf 流式、concurrency 512、ISL 1024/OSL 8192(固定)、temperature 0、ignore_eos,3 次取均值:FlashInfer 默认(auto)mean output TPS/GPU 13,075.998,horizontal 为 13,869.302,提升 +6.067%(三次分别 +6.535%/+5.440%/+6.226%)。服务端附带 `--mamba-cache-mode align --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding` 等配置与 `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random`。gsm8k(TP=2,5-shot):默认 flexible-extract 0.4390 / strict-match 0.8423,horizontal 为 0.4261 / 0.8438(flexible-extract 略降,在 ±stderr 范围边缘)。

## 验证与回归风险

- 测试:`test_ssu_dispatch.py` 新增 `test_flashinfer_forwards_ssu_algorithm`,用 Mock 内核断言 None/auto/simple/vertical/horizontal 五种输入都被正确透传;端到端验证为上述吞吐与精度测试。
- 风险:非 FlashInfer 后端传 `ssu_algorithm` 会在 `__post_init__` 直接 ValueError;强制算法必须被 FlashInfer 在当前 GPU/state dtype/decoding mode 下支持,否则由 FlashInfer 侧报错;默认(None→auto)路径行为与之前完全一致,回归面小。

## 关联

- 同区域:Mamba/SSM 系列优化(`--mamba-backend flashinfer`、mamba cache stochastic rounding 等 MambaConfig 字段);此 PR 是 FlashInfer Mamba 集成之上的调优入口。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
