# PR #64687: [Models][XPU] Port eager Qwen3.5 models and fuse MoE decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hzjane |
| 合并 | 2026-11-19 |
| 分类 | MoE（tier: major） |
| Label | documentation, intel-gpu, ci/build, qwen, quantization, verified |
| Diff | pack 标注 +None/-None, None files；pack 末尾说明「12 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/64687 |

## 一句话结论

把 Qwen3.5 与 MTP 迁移到 XPU 的 eager、breakable-graph 设计（#51012），并在 XPU 模型内直接调用 vllm-xpu-kernels #670 的 routed/shared expert provider 实现 MoE 小批量 decode 融合：TP2 graph-on MoE B1 流式 TPOT −21.0%～−20.1%、输出吞吐 +16.3%～+25.4%。

## 动机

XPU 上的 Qwen3.5 此前走 torch.compile（mode 3）路径，小批量（1–8 token）MoE decode 的融合依赖编译器 pass（如 #61163 的 compiler pass 方案）。#51012 已在 NVIDIA/ROCm 引入 eager + breakable graph 设计。本 PR 把该设计扩展到 XPU：不用模型级 torch.compile 或编译器 fusion pass，而是由 eager XPU 模型直接调用 companion kernel PR vllm-xpu-kernels#670（已合并；本地 B1 smoke 用 wheel 0.1.17.1）的 provider。融合限定于符合条件的 1–8 token FP16/BF16 调用、per-tensor FP8-e4m3 权重，routed/shared expert 输出共享一次 TP 归约；其他配置保留现有 MoE 路径。当前分支上 MoE 融合为 opt-in（`--additional-config '{"decode_backend":"latency_specialized"}'`），默认用 stock MoE；无 provider 的 wheel 默认保持 stock MoE 且无法启用显式特化。

## 优化点

- **平台模型迁移**：`vllm/models/qwen3_5/xpu/`（model.py、mtp.py、specialized_decode/ 等）使 XPU 加入 NVIDIA/ROCm 的 eager 变体行列；`test_qwen3_5_models_use_platform_specific_implementation` 增加 xpu 分支；`default_breakable_cudagraph_architectures` 在 XPU 上并入 `_QWEN3_5_ARCHITECTURES`。
- **MoE decode 融合**：`Qwen3_5SparseMoeBlock` 经 `initialize_specialized_decode` 挂上 decode runtime；`_xpu_moe_shared_fused_plan` 绑定 native provider（`_xpu_C::moe_shared_fused_decode_interface`）；plan 绑定 shared_w2 等权重指针，graph 捕获后 reload 就地恢复参数存储即可复用（`data_ptr`/workspace 不变）。
- **provider 复用**：`XPUExpertsFp8` 新增 `initialize_weights` 与 `_weights_match`——权重绑定未变时复用已 relayout 的 `XpuFusedMoe`；`process_weights_after_loading` 在 reload 恢复捕获参数存储前先准备布局；mxfp4 权重按 `float4_e2m1fn_x2` 视图重绑。
- **约束**：shared 投影必须用 `XPUW8A16FP8LinearKernel`；显式 W8A8 选择经 fallback 保留其激活量化；大 prefill 步（actual_rows > 8，reason="token_rows"）保留 stock MoE；Auto MTP/weight updates 与特化解耦（`block._decode is None` 时目标/draft graph 与权重更新会话仍独立）。

## 关键改动

```python
# vllm/model_executor/layers/fused_moe/experts/xpu_moe.py
+    def initialize_weights(self, w1, w2, topk, activation, expert_map) -> None:
+        """Reuse the provider while its source weight bindings remain unchanged."""
+        if self._weights_match(w1, w2):
+            return
...
+        self.fused_moe_impl = XpuFusedMoe(w13=w1, w13_scales=..., w2=w2, ...)
+        self._weight_bindings = weight_bindings
+    def process_weights_after_loading(self, layer):
+        """Prepare layout before reload restores the captured parameter storage."""
+        self._layer = cast("RoutedExperts", layer)
+        self.initialize_weights(self._layer.w13_weight, self._layer.w2_weight, ...)
vllm/config/vllm.py: default_breakable_cudagraph_architectures() 增加 XPU 分支
tests/kernels/moe/test_moe.py: + test_xpu_qwen3_5_shared_expert_decode（B1 FP16/FP8 layer smoke:
  native fused dispatch、与 stock MoE 一致 rtol=1e-2、provider 复用、profiler 计数 == 1）
tests/models/qwen3_5/xpu_specialized_decode/test_integration.py（新, 564 行）: TP2 reload/graph、
  A8 shared expert fallback、default MTP weight update graph、auto 与 latency_specialized 的状态迁移
  （同一请求跨两条 MoE 路径且 KV/SSM state 指针不变）
tests/models/test_qwen3_5.py: 平台实现选择测试加 xpu
.buildkite/intel_jobs/kernels_intel.yaml: 新增 "(B70) Qwen3.5 Shared Expert Decode (B1)" CI 步骤
docs/models/hardware_supported_models/xpu.md: 说明 Qwen3.5 on XPU 使用 eager、MoE decode fusion opt-in
```

## 性能声明

（环境：两块 Intel Xe2 32GB GPU，TP2，PyTorch 2.14.0+xpu，FP16 + online `fp8_per_tensor`；性能 checkpoint 为 Qwen3.6-35B-A3B（MoE，用 Qwen3.5 实现）与 Qwen/Qwen3.5-9B（BF16 无量化 TP1 card 0，TP2 greedy 校验于 card 0/1）。Baseline main `fa3b9d1377` compiled（mode 3），本 PR `dd6f1ccb15` eager（mode 0），同 kernel build `0becdcc76c`（#670 Qwen native 实现）。）

- **Claims**：TP2、async、graph-on MoE B1：streamed TPOT −21.0%～−20.1%、输出吞吐 +16.3%～+25.4%（1K/8K/32K 输入、512 输出），对照 compiled XPU baseline。
- **小批量 sweep（中位 streamed TPOT ms，B1/B2/B4/B8；graph-on）**：1024 in：9.26/10.22/11.86/13.34 → 7.35/8.34/10.44/12.08（TPOT −20.6%/−18.4%/−12.0%/−9.4%；吞吐 +25.4%/+21.9%/+13.2%/+10.0%）；8192 in：−21.0%/−14.6%/−11.2%/+0.1%（吞吐 +23.7%/+15.0%/+10.8%/+0.8%）；32768 in：−20.1%/−11.5%/−9.6%/−3.9%（吞吐 +16.3%/+9.2%/+7.5%/+3.6%）。**graph-off**：1024 in TPOT −31.7%/−31.4%/−31.1%/−30.0%（吞吐 +46.2%/+45.2%/+44.4%/+41.8%）；8192 in −30.3%～−27.5%；32768 in −31.4%～−18.8%。
- **vllm bench serve（#51012 式随机负载）**：35B-A3B decode cc1 ITL 8.90 → 6.97 ms（−21.7%，吞吐 +26.7%）；prefill 4k TTFT 297.71 → 280.99 ms（−5.6%）；SPEED-Bench 8K（32 行固定子集，subset measurement）cc1 ITL 9.17 → 7.22（−21.3%，+23.8%）、cc2 −15.4%、cc4 −11.9%、cc8 −5.1%、cc32 −1.2%。9B BF16 TP1 大多 ±1% 内；decode cc128 ITL +5.2% 为本测量中 >1% 的回退（正文标注 regressions above 1%: qwen35_9b nv_decode_cc128）。
- **隔离 forward/MTP**：35B-A3B decode B1 8.846 → 6.944 ms（−21.5%）、B32 +0.1%；prefill B1 −4.9%/B8 −4.4%；MTP decode B1 −6.0%。9B 全部 ±1.2% 内。
- **精度**：GSM8K 5-shot 16K 预算：96.74% → 96.59%（−0.15 pp）；GPQA-Diamond 43.43% → 41.92%（−1.52 pp）；AIME25 avg1 56.67% → 53.33%（−3.33 pp）。正文明确 "Accuracy falls more than one percentage point below the reference in gpqa_diamond, aime25_avg1; no accuracy-parity claim is made"（截断数 GPQA 52/55、AIME 13/14）。Greedy smoke：35B-A3B TP2 3/4 token-identical、MTP 4/4、9B 2/4～3/4；token-exact equivalence 未确立。

## 验证与回归风险

当前 Intel CI 选择为一块 B70 上的一个 B1 FP16/FP8 layer smoke test（`test_xpu_qwen3_5_shared_expert_decode`，kernel wheel 0.1.17.1 本地 1 passed）：检查 native fused dispatch、与 stock MoE 一致（rtol=1e-2）、provider 复用且不加载全模型权重。三个扩展 reload/graph layer 回归本地通过（在缩减的 CI 套件之外）。历史验证（早期 revision）：native fused/fallback/TP2 model-block 18 passed、既有 model/platform/quantization 6 passed、真权重 TP2 reload/graph 回归 1 passed、stock-wheel fallback 1 passed（保留为历史证据）。Changed-file hooks 通过（mypy 以显式 package root 单独运行）；本 revision 的 remote CI 待确认；MTP 任务经 engine metrics 验证非零 drafted/accepted tokens。回归风险：融合仅限 1–8 token、FP16/BF16、per-tensor FP8-e4m3，需 #670-capable kernel wheel（否则无法启用且默认 stock MoE）；shared 投影必须 `XPUW8A16FP8LinearKernel`；GPQA-Diamond/AIME25 精度下降超过 1 pp 且不主张精度持平；9B decode cc128 +5.2% 回退；remote CI 未确认。

## 关联

- #51012（eager/breakable-graph 设计源头）、#61163（compiler pass 方案，本 PR 未采用）、vllm-xpu-kernels#670（routed/shared expert provider，已合并）。
- 精度基准来源：RFC #59665。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
