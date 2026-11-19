# PR #63086: [ROCm][Kimi-K3] Enable encoder graph capture for Kimi-K3 Vision Tower

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @samutamm |
| 合并 | 2026-11-16 |
| 分类 | CUDA Graph / Compiler（tier: major） |
| Label | rocm, ready, torch.compile, ci/build, multi-modality, nvidia, kimi, k3 |
| Diff | pack 标注 +None/-None, None files；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/63086 |

## 一句话结论

为 Kimi-K3 视觉塔（MoonViT3dPretrainedModel + KimiK25MultiModalProjector）在 AMD ROCm (MI355X) 上实现 `SupportsEncoderCudaGraph`，encoder 单次调用 7.1 → 4.5 ms，serving geomean 吞吐 +3.12%、mean TTFT −3.92%。

## 动机

Kimi-K3 的 vision tower 由大量小 kernel 组成，eager 执行时 launch overhead 与小 kernel 间 idle gap 占比高。PR 早期版本也尝试对 vision tower 加 `torch.compile`（compile_mm_encoder），但 profiling 显示 Inductor 只能融合 norms、residual adds 与 GELU（全塔 301 → 275 kernels，约占 1% GPU 时间），因为 GEMM、AITER FMHA 与 `packed_qk_rope`（约 94% 时间）是不透明 kernel；该部分在 review 后被丢弃——收益来自 graph replay 而非 fusion。#44394 与 #56820 已为其他模型（PaddleOCR-VL、Ling-3.0-flash-VL）在同一 `SupportsEncoderCudaGraph` 接口上加了 encoder CUDA graph 支持，本 PR 只为 Kimi-K3 实现该接口。

## 优化点

- 新增 `vllm/models/kimi_k3/amd/encoder_cudagraph.py` 的 `KimiK3EncoderCudaGraphMixin`：实现 item specs、item 选择、capture inputs、replay buffers、graph/eager forward。
- token 预算：自动推断的 budgets 上限 512 tokens（`_MAX_ENCODER_CUDAGRAPH_TOKEN_BUDGET = 512`，最小 64）；超过预算的输入走 `EncoderCudaGraphManager` 既有的 eager 路径（replay 只在 ≤512 output tokens 时划算，padding 更大的 graph 并不更快）。
- `get_encoder_cudagraph_config` 校验：与 vision encoder 动态 FP8 scaling 不兼容（amax history 是无法捕获的 host 侧状态，需静态 scale 文件）；FLASHINFER / TORCH_SDPA ViT attention backend 不可捕获（FlashInfer 在 host 读 max_seqlen、TORCH_SDPA 对 cu_seqlens 调 `.tolist()`），需 FLASH_ATTN 或 ROCm 的 ROCM_AITER_FA。
- buffer keys：pixel_values、pos_embeds、rope_freqs_cis、cu_seqlens、max_seqlen、sequence_lengths、merge_gather_idx；modalities 仅 image。
- `KimiK3ForConditionalGeneration` 挂上 mixin 并保存 `self.model_config`。
- 特性 opt-in（`cudagraph_mm_encoder`，默认关闭），默认行为不变；仅 ROCm，NVIDIA 模型不变；`amd/model.py` 与 `nvidia/model.py` 之间 encoder-graph 方法去重留作后续 PR。

## 关键改动

```python
# vllm/models/kimi_k3/amd/encoder_cudagraph.py（新文件, 271 行）
class KimiK3EncoderCudaGraphMixin(SupportsEncoderCudaGraph):
    supports_encoder_cudagraph: ClassVar[Literal[True]] = True
    # Replay removes launch overhead, which only pays off up to ~512 output
    # tokens; padded larger graphs are not faster, so larger batches run eagerly.
    _MAX_ENCODER_CUDAGRAPH_TOKEN_BUDGET: ClassVar[int] = 512
...
    def get_encoder_cudagraph_budget_range(self, vllm_config):
        min_budget = 64   # 224x224 图像 2x2 merge 后 64 tokens
        max_budget = min(vllm_config.scheduler_config.max_num_batched_tokens,
                         self.model_config.max_model_len,
                         self._MAX_ENCODER_CUDAGRAPH_TOKEN_BUDGET)
...
    def encoder_cudagraph_forward(self, values, path="default"):
        metadata = {k: v for k, v in values.items() if k != "pixel_values"}
        hidden_states = self.vision_tower(values["pixel_values"], None, encoder_metadata=metadata)
        output = self.mm_projector(hidden_states.to(dtype))
        return output.reshape(-1, output.shape[-1])
# vllm/models/kimi_k3/amd/model.py
 class KimiK3ForConditionalGeneration(nn.Module, SupportsMultiModal,
+    KimiK3EncoderCudaGraphMixin,
     SupportsPP, SupportsQuant, SupportsEagle3, ...):
tests/models/multimodal/generation/test_vit_cudagraph.py: 新增 kimi_k3 配置与
  test_kimi_k3_vit_cudagraph_replay_and_threshold（fills_smallest_budget_64 / padded_budget_196 /
  fills_largest_budget_512 / above_largest_budget_544 / two_images_share_replay / batch_over_budget_all_eager）
```

## 性能声明

- **Encoder 级 profile**（MI355X，完整 vision tower，512x512 图像）：每次 encoder 调用 wall time eager 7.1 ms，cudagraph replay 4.5 ms；kernel 列表相同，收益来自消除 launch overhead 与小 kernel 间 idle gap，小图收益最大。
- **Serving**（MI355X，TP=8，9 点 sweep，每配置 3 次重复，A/B 交错 A/D/A/D/A/D，每次 fresh server boot；A = eager vision tower，D = 仅 `cudagraph_mm_encoder=true`，budgets 自动推断为 [64, 128, 256, 512]；每点 96 prompts）：geomean Total throughput 3341 ± 72 → 3445 ± 16 tok/s（+3.12% ± 1.85）；geomean Mean TTFT 490.6 ± 9.4 → 471.3 ± 1.4 ms（−3.92% ± 1.60）；geomean Median TTFT −0.87% ± 5.39；mean TPOT −2.83%、median TPOT −2.11%（预期持平，decode 路径未动）。仅一点（256x256 @ conc=16，mean TTFT −13.35%）单独超出噪声带，其余各点单独均在噪声内，geomean 是真实效应的证据。Above-cap 对照点（768x1024，D 下仍 eager）：throughput +2.71%、mean TTFT +3.30%——flat/noisy，确认 cutoff 之上的流量未回退。
- **噪声底线**（3 次重复 CV% 的 RMS/最大值）：mean TTFT ≤4.0%/点可靠；median TTFT 每点最高约 25% 仅方向性；throughput ≤7.7%/点。正文另注明 "below ~4% for mean TTFT, ~8% for throughput/TPOT, ~25% for median TTFT isn't distinguishable from noise"，以及 "up to 25%" 的 per-point median TTFT 噪声。
- **精度**：MMMU（N=900，单次运行）A (eager) 0.6734，D (cudagraph only) 0.6678，−0.8%（−0.56 points），正文判定 "0.6% difference is not significant, no regression"。

## 验证与回归风险

`pytest -v -s tests/models/multimodal/generation/test_vit_cudagraph.py -k kimi_k3`：单 GPU dummy weights 下 7 passed、1 skipped（video modality 不支持），约 292.99 s；覆盖 graph-vs-eager 输出一致性（容差为输出尺度 2 个 bf16 ulp，`bound = 2 * expected.abs().max() * 2**-7`）、token 上限、graph_hits/misses 计数、两图共享 replay、超预算全 eager。Serving 验证用 MI355X docker（nightly-76c6f3f2a95f6049657db1e513133b824a5c8175 + 本 PR），`VLLM_USE_BREAKABLE_CUDAGRAPH=1`、AITER/quick-reduce 为标准 Kimi-K3 配方（非本 PR 特有）。回归风险：仅 ROCm；与动态 FP8 encoder scaling、FLASHINFER/TORCH_SDPA backend 不兼容（会显式报错）；默认 budgets 上限 512 tokens，更大输入仍 eager；serving 增益只有 geomean 级别可分辨，单点不可靠；amd/nvidia model.py 去重留作后续，存在双份实现短期并存。

## 关联

- #44394、#56820：其他模型（PaddleOCR-VL、Ling-3.0-flash-VL）的 encoder CUDA graph 支持（同一接口）。
- #53011：Kimi K2.5 vision helper 的 eager torch.compile 修复（bugfix，非 encoder cudagraphs）。
- 讨论：https://github.com/vllm-project/vllm/pull/63086#discussion_r4440377966 （512-token 预算上限）。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
