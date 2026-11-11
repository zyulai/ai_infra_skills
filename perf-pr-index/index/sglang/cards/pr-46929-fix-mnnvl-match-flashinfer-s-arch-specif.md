# PR #46929: [Fix] mnnvl: Match FlashInfer's arch-specific CuTe DSL AR profiles

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @leejnau |
| 合并 | 2026-11-08 |
| 分类 | Attention / Kernel（tier: major） |
| Label | dependencies, run-ci |
| Diff | +49/-28, 2 files（pack diffstat 缺失，按 pack diff 文本计数） |
| 链接 | https://github.com/sgl-project/sglang/pull/46929 |

## 一句话结论

修复 MNNVL CuTe DSL AR profile 选择：传入架构信息用 find() 匹配，GB300 tp=4 恢复 FlashInfer 实测路由，AR 快 6–12%。

## 动机

`_config_for_shape` 决定 FlashInfer MNNVL CuTe DSL 的 `DEFAULT_CONFIG` 能否直接服务当前 shape，否则 SGLang 需提供 re-target 的单 profile 配置。它此前调用 `StaticProfile.matches()` 时不带 compute capability：FlashInfer 发布的 tp=4/hidden=8192/top_k=10 profile 分别绑定 GB300（sm_103）与 Rubin（sm_107），绑定某架构的 profile 永远匹配不了 None 架构。于是在这些 GPU 上 SGLang 找不到匹配、丢弃 FlashInfer 实测的 tp=4 路由，改用 re-target 的 GB300 tp=8 预设——在 tp=4 场景是错误的 fallback。

## 优化点

给 `_config_for_shape` 增加 `compute_capability` 参数（调用点传 `torch.cuda.get_device_capability(device)`），查找改为 `MNNVLCuteDSLConfig.find()`——与 FlashInfer workspace 的 `resolve()` 相同的查找逻辑——只在 FlashInfer 未为当前架构发布该 shape 的 profile 时才 re-target；re-target 日志带上 sm 版本，并修正了"DEFAULT_CONFIG 只有 GB300 H=8192/K=10"的过时注释。依赖 FlashInfer 0.7.2.post1+（`find()` 与 `compute_capability` 参数）；main 自 #48468 起为 0.7.3，其 tp=4/H8192/K10 profile 仍只绑定 (10,3) 与 (10,7)。

## 关键改动

```text
python/sglang/srt/layers/flashinfer_mnnvl_cutedsl.py: _config_for_shape 增加 compute_capability，matches() 改 find()
test/registered/unit/layer_boundary/test_cutedsl_ar_fusion.py: 新增"本架构发布的 profile 不被 re-target"测试
```

```python
profile = default_config.find(
    tp_size=tp_size, hidden_size=hidden_size, top_k=top_k,
    dtype=torch.bfloat16, compute_capability=compute_capability,
)
if profile is not None:
    return default_config
```

## 性能声明

4×GB300（dlcluster，sm_103），`lmsysorg/sglang:dev` + flashinfer-python 0.7.3 + 本分支；tp=4、H=8192、top_k=10、bf16、含 RMSNorm。方法沿用 FlashInfer `benchmarks/comm/bench_mnnvl_cutedsl_h5120.py`：每 CUDA graph 20 次调用、30 次 replay 取中位、各 rank 取 max，重复两次一致（~0.1 µs）。all-reduce 在 M≥512 快 6–12%（M=1024：48.61→43.42 µs；M=2048：87.97→79.71 µs；M=4096：164.76→151.33 µs），M=16 快 12%（5.84→5.20 µs）；finalize 在 8≤M≤48 更快或持平，其余 ±2–3%。最大回退约 0.19 µs（AR M=32）与 ~2.5%（finalize M=1/2048），来自 FlashInfer 自身的 GB300 tp=4 路由表而非本选择逻辑。

## 验证与回归风险

`test_cutedsl_ar_fusion.py` 33 通过，含新增参数化测试：sm_103 与 sm_107 保留 DEFAULT_CONFIG、sm_100 被 re-target。设备上验证 `_config_for_shape(DEFAULT_CONFIG, tp=4, H=8192, K=10, cc=(10,3))` 返回 DEFAULT_CONFIG、无 re-target 日志、`find()` 命中 (10,3) profile。无 Rubin 硬件，sm_107 路径仅单测覆盖。回归面很小：只影响 MNNVL CuTe DSL all-reduce fusion 的配置选择；需要 FlashInfer ≥0.7.2.post1。

## 关联

#48468（FlashInfer 升级到 0.7.3，本 PR 的依赖前提）。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
