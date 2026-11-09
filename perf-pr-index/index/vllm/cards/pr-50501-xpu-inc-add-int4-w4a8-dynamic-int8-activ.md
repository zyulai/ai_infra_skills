# PR #50501: [XPU][INC] Add int4 w4a8 (dynamic int8 activation) backend for INC linear layers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tthakkal |
| 合并 | 2026-08-21 |
| 分类 | Quantization \| tier: major |
| Label | intel-gpu, quantization, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/50501 |

## 一句话结论

为 XPU INC int4 线性层加入 w4a8 后端，FLUX.1-dev / B70 上比 ARK 默认快 1.39×。

## 动机

正文指出 INC int4 layers on XPU 在 ARK 可导入时总是 dispatch 到 ARK，oneDNN `int4_gemm_w4a16` 仅作 fallback，`int4_gemm_w4a8` 从未接入。int8 activations 避免把权重反量化到 activation dtype，对 compute-bound 形状有利（FLUX 1024×1024 是 M=4608）。保持 opt-in 因最佳 kernel 依设备而定：B70 上 ARK 对下述两个 workload 都不是最快，两个 oneDNN 路径都胜过它，故做成可选择。

## 优化点

新增 `INCXPUW4A8LinearMethod`（int4 权重 + 动态 per-token int8 activations，复用 w4a16 权重布局不变，低于 512 tokens 回退 w4a16）与 `VLLM_XPU_INC_WNA16_BACKEND` 选择器：`auto | ark | w4a16 | w4a8`；纯增量（一个新测试文件，不改现有测试），`auto` 保持当前行为不变。另校验 partitioned in/out dims 为 8 的倍数：`XPUW4A8IntLinearKernel.can_implement()` 要求此点，但 INC 自建 linear method、不走 `MPLinearKernel` 选择，且形状在 `create_weights()` 之后才存在，检查不可达；未对齐会被静默截断（int4 packing 把两维除以 8，`out=890` 打包 111 个 int32 只覆盖 888 列、丢失 2 列），TP 切分也可能让对齐的模型维度产生未对齐分区。

## 关键改动

```text
vllm/model_executor/layers/quantization/inc/schemes/inc_w4a8_linear.py: 新增 INCXPUW4A8LinearMethod（w4a8 kernel 绑定、512-token batch gate、分区对齐校验）
vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py: 接入后端选择器
vllm/envs.py: 新增 VLLM_XPU_INC_WNA16_BACKEND 环境变量
tests/quantization/test_inc_xpu_w4a8.py: 新增 32 个测试（29 mocked 处处可跑，3 real-model 需 XPU 有 int4_gemm_w4a8）
tests/quantization/test_auto_round.py: 回归测试
```

注释片段（摘自 inc_w4a8_linear.py 的 diff）：

```diff
+# per-token symmetric int8, so the GEMM stays on the int8 datapath instead of
+# upconverting weights to the activation dtype. It is opt-in through
+# VLLM_XPU_INC_WNA16_BACKEND because which kernel is fastest depends on the
+# device: ARK can use XMX int8 on some XPUs, and the default ("auto")
+# preference order is deliberately left unchanged.
```

## 性能声明

Per-layer GEMM：1.32× at M=4608、1.13× at M=512、0.84× at M=128，故设 512-token fallback。E2E `FLUX.1-dev`, 1024×1024, 28 steps, B70, TP=2：`black-forest-labs/FLUX.1-dev` bf16 dense 19.46 s；`vllm-project-org/FLUX.1-dev-AutoRound-w4a16` 的 ARK（current default）23.63 s、oneDNN w4a16 21.87 s、oneDNN w4a8 **16.99 s**（1.39× over ARK default，唯一优于 bf16 dense 的量化配置；16.99 s 复现 4 次；cosine 0.99996 vs w4a16 with bf16 activations）。文本生成 smoke-test：Intel Arc Pro B70、`OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc`、32 prompts × ~710 prompt tokens + 1024 generated each、`enforce_eager`、greedy、warmup 后 3 次中位数：`unset (auto)` / ARK 为 12.675 s（12.598–12.903）、2,585 gen tok/s；`w4a8` 为 **8.354 s**（8.272–8.647）、**3,922 gen tok/s**，即 1.52× over ARK default；共 22,742 prompt tokens、32,768 generated tokens，两后端输出一致，kernel 绑定确认（auto→96 ARK、w4a8→96 w4a8、w4a16→96 w4a16）。

## 验证与回归风险

新 `tests/quantization/test_inc_xpu_w4a8.py` 共 32 tests，覆盖 backend selection、weight processing、apply_weights、partition alignment、end to end；29 个 mocked 处处可跑，3 个 real-model 在无 XPU `int4_gemm_w4a8` 时 self-skip；另有 `test_auto_round.py` 回归与 ruff；mutation-tested（把 env var 翻到 `w4a16` 会让 `test_e2e_selects_w4a8_for_every_linear` 报 "no linear layer used the w4a8 method"，非空绿）。回归风险：`auto` 保持当前行为、纯增量；最佳 kernel 依设备故 opt-in；低于 512 tokens 回退 w4a16（M=128 仅 0.84×）；文档由 `docs/configuration/env_vars.md` 从 `envs.py` 自动生成。其余路径 PR 未注明。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
