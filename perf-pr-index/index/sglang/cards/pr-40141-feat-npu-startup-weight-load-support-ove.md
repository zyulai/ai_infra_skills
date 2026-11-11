# PR #40141: [Feat] [NPU] startup-weight-load: Support overlap mode on NPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wl55721 |
| 合并 | 2026-11-11 |
| 分类 | Overlap / Scheduler(tier: major) |
| Label | run-ci |
| Diff | +14/-3,2 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/40141 |

## 一句话结论

把 startup weight load 的 overlap 模式从 CUDA 放宽到 NPU,缩短 NPU 上的模型启动时间。

## 动机

`--startup-weight-load-mode=overlap` 将 checkpoint 预取与 decode CUDA-graph capture 重叠以缩短模型启动,但选项校验硬编码 CUDA-only:解析时判断 `is_cuda_platform`、`device == "cuda"`,NPU 直接被拒。NPU 上 decode graph 仍会捕获,而 prefill graph(默认 `TC_PIECEWISE`)是禁用的——`tc_piecewise` prefill 会在 `torch.compile` 期间烘焙权重,commit-after-capture 不安全,因此 NPU 满足启用 overlap 的前提条件。

## 优化点

`StartupWeightLoadOptions` 增加 `is_npu` 字段,由 `current_platform.is_npu()` 填充;overlap 的不支持原因判定从 CUDA-only 放宽为 CUDA-or-NPU(`not (is_cuda_platform or is_npu) or device not in ("cuda","npu")`),拒绝消息改为 "CUDA or NPU only";把硬编码的 `torch.cuda.synchronize()` 换成设备无关的 `current_platform.synchronize()`。CUDA 路径行为不变。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py: is_npu 选项、CUDA-or-NPU 校验规则、设备无关同步
test/registered/unit/model_executor/model_runner_components/test_startup_weight_load.py: 单测补 is_npu=False,更新拒绝消息断言
```

```python
(
    not (options.is_cuda_platform or options.is_npu)
    or options.device not in ("cuda", "npu"),
    "CUDA or NPU only",
),
```

## 性能声明

PR 未给出具体加速数字。e2e 测试度量启动 wall-clock(进程启动 → 首个 `/health_generate` 200),对 Qwen3-32B / tp=2 的 serial 与 overlap 各跑并打印 speedup,同时 guard "overlap 不比 serial 慢 2× 以上"(防 CI 计时噪声)。PR 说明该改动只影响启动延迟,per-token 吞吐不受影响。

## 验证与回归风险

NPU e2e `test_npu_startup_weight_load_mode.py`(配套测试分支 `test/npu-startup-weight-load-mode`):Qwen3-32B、tp=2、prefill graph 关闭,serial/overlap 两模式断言生成文本含 "Paris"——证明 capture 期烘焙的 sentinel 权重在 commit 后被真实权重替换(否则输出乱码)。单测覆盖新 `is_npu` 选项与更新后的拒绝消息。风险:NPU 上依赖 prefill graph 关闭这一前提;若 NPU capture-commit 语义变化,sentinel 权重可能泄漏到输出;CUDA 路径未改动。

## 关联

无(素材未提及其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
