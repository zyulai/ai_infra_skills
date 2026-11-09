# PR #49381: [ModelOpt] Redesign the LinearMethod classes using the generic QuantKey-driven method

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @juhi10071998 |
| 合并 | 2026-09-01 |
| 分类 | Quantization（tier: major） |
| Label | ready, needs-rebase, quantization, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「2 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/49381 |

## 一句话结论

6 个 ModelOpt LinearMethod 合为 1 个 QuantKey 驱动类，6 格式 logits 字节一致，属重构非提速。

## 动机

正文指出 ModelOpt linear quantization 当时是六个近乎重复的 `LinearMethod` 类，每种格式一个（FP8 per-tensor、FP8 per-channel/per-token、FP8 block-weight-only、NVFP4 W4A4、NVFP4 W4A16、MXFP8）。这些类约 80% 结构相同，却各自内联实现 `create_weights` / `process_weights_after_loading` / `apply`，且各格式的 quirks 是 load-bearing 且"fail silently"——一个错误假设会直接产出垃圾输出而不报错。NVFP4 W4A4 与 W4A16 两个完整类只差激活是否量化，三个 FP8 类主要差在 scale 形状，kernel 选择分散，新增格式要在每个 dispatch 点接线。设计出发点是分清哪些部分可由格式的数值描述推导、哪些真正需要 per-format 代码；结论是几乎一切都可由 `(weight, activation)` 的 `QuantKey` 对推导。

## 优化点

用 `(weight, activation)` 两个 `QuantKey` 描述一个 ModelOpt linear 格式：`resolve(algo, config, prefix)` 把 checkpoint 的 algo 字符串映射为 `QuantSpec(weight, activation)` 加只读的 `CkptCtx`，一个通用 `ModelOptLinearMethod` 跑固定生命周期——`create_weights`（wkey/akey 各自建权重、`select_linear_kernel(spec)` 选 fp4/mxfp8/fp8 内核族、`expose_input_quant_key`）→ `process_weights`（key scheme 处理、maybe_fuse_global_scales、FormatScheme hooks、kernel post-load）→ `apply`（FormatScheme 包裹、kernel.apply_weights）。`QuantKeyScheme` 按内容键控、按角色参数化，七个 scheme 覆盖六种格式且可共享（NVFP4 W4A4 与 W4A16 共用 `KNvfp4Static` 权重 scheme）。新增格式变成数据：一条 `resolve()` 行加可复用 scheme，而非新类。两个扩展缝为 `FormatScheme`（可选 per-format hooks）与 `LINEAR_METHOD_BUILDERS` 注册表（无法用 key pair 表达的格式注册自己的 `LinearMethodBase`）。前端各 config 只暴露 `linear_algo()`，`LinearBase` 统一调用 `build_linear_method`。正文明确 MoE 与 mixed-precision 保持不变，`routed_experts.py` 依赖的 `ModelOpt*` 类名耦合被保留，mixed MoE fork byte-identical。

## 关键改动

```text
vllm/model_executor/layers/quantization/modelopt.py: 定义 ModelOptLinearMethod、QuantKeyScheme/SCHEME_FOR、resolve()、FormatScheme、LINEAR_METHOD_BUILDERS 与 build_linear_method
vllm/model_executor/layers/linear.py: LinearBase 改为经 build_linear_method 构造量化方法
vllm/model_executor/layers/attention/mla_attention.py: 适配新 linear 方法（diff 中改动）
vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py: MXFP8 内核侧适配
tests/quantization/test_modelopt.py: 单测由断言各格式类改为断言 ModelOptLinearMethod + 解析出的 QuantSpec
```

```python
    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")
    assert isinstance(method, ModelOptLinearMethod)
    assert method.spec.weight is kNvfp4Static
    assert method.spec.activation is kNvfp4Dynamic
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据（pack 的 perf claims 检测为 none），本 PR 定位为行为保持的重构。正文给出的是正确性与等价格证据（Qwen3-8B；Parity 门控 `VLLM_BATCH_INVARIANT=1` + `--enforce-eager`，逐层 weight/kernel/expose byte-hash 与 per-token prefill logit-diff，GSM8K 为 1319 Q · 5-shot · T=0 · 512 tok）：

| Format | weight/kernel/expose hash | logit Δ | GSM8K | Verdict |
| --- | --- | ---: | ---: | --- |
| NVFP4 W4A4 | byte-identical | 0.0 | 0.89765 | identical |
| W4A16 NVFP4 | byte-identical | 0.0 | 0.87794 | identical |
| FP8 per-tensor | byte-identical | 0.0 | 0.88476 | identical |
| FP8 PcPt | byte-identical | 0.0 | 0.89689 | identical |
| MXFP8 | byte-identical | 0.0 | 0.89538 | identical |
| FP8 PbWo | weights differ | 0.0 (eager) | 0.89689 | bug fix |

六种格式 logits 全 byte-identical，GSM8K 跨 build 差异 ≤2 Q（torch.compile autotune 噪声）、invalid-rate 0。附带修复的定量描述：FP8 PbWo 旧类因 misnamed guard 跳过 block kernel post-load，编译态 serving 下产出 garbage（acc 0.000、invalid-rate 1.0），通用方法执行 post-load 后正确（0.897）。硬件 PR 未注明测试条件。

## 验证与回归风险

Unit：`pytest tests/quantization/test_modelopt.py -k "not checkpoint_setup"` → 33 passed；`ruff check`、`ruff format --check`、`typos` 均 clean；正文说明已 rebase 到含 #53132（Kimi K3 nvfp4）的 main。Per-format evidence 在 Qwen3-8B 上做 byte-level parity 与 GSM8K。可能影响的路径：六种 ModelOpt linear 格式的权重创建/加载后处理/apply 全流程、kernel 选择、`WEIGHT_LOADER_V2_SUPPORTED` 注册（MXFP8 补齐）、带 on-disk `input_scale` 的 W4A16 checkpoint 兼容（`_DropInputScale`）、以及宽度非 128 倍数的 partial-block FP8_PB_WO（`_Fp8PbWoPartialBlock`，正文称在宽度 2624 验证 bit-identical）。正文声明 MoE 与 mixed-precision 不变，但 FP8 PbWo 编译态 bug 修复会改变该格式行为；pack 末尾注明 2 file(s) dropped from diff。

## 关联

正文"Not duplicating existing work"段明确引用：#48422（FP8_PB_WO post-load bug，本 PR 顺带修复，若其先落地则此处 PbWo 变 no-op 并 rebase）、#53132（Kimi K3 nvfp4，merged，其 partial-block FP8_PB_WO 支持被本 PR 吸收）、#50617（serve `modelopt_mixed` with FP8_PB layers，将据此 PR 关闭，其 `has_blocked_weights()` 修复已并入）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
