# PR #36205: [mla] Support fused FP8/NVFP4 output quantization in MLA attention (#35792)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @carlyou |
| 合并 | 2026-04-03 |
| 分类 | Quantization（tier: major） |
| Label | documentation, performance, ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「5 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/36205 |

## 一句话结论

把 FP8/NVFP4 输出量化折进 MLA attention custom op，省掉每层一个 quant kernel（Phase 1，未主张加速）。

## 动机

issue #35792 的 Phase 1 (of 2)。此前 MLA attention 的输出量化需要一个独立的 quant kernel（每层一次），`mla_attention.py` 的 `forward_impl` 在 `output_scale` / `output_block_scale` 非空时直接 `raise NotImplementedError("fused output quantization is not yet supported for MLA")`。本 PR 把量化折进 `unified_mla_attention_with_output` custom op，消除每层独立的 quant kernel。

## 优化点

在 `mla_attention.py` 的 `forward_impl` 实现 fused output quant：当 `output_scale` 或 `output_block_scale` 非空（`use_quant`）时，因 fusion pass 已把 output 分配为量化 dtype（FP8 或 uint8 for FP4），改用 temp-buffer swap——先把 `output` 换成同形状 `q.dtype` 的临时 `torch.empty` 供计算，末尾再量化写回真实 `quant_output`；新增 `self._quant_fp8_op = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR, compile_native=True)`。`backend.py` 为 MLA 基类加 `fused_output_quant_supported()` stubs。新增 `vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`（pattern matcher + `MLAAttnFusionPass`）做 MLA attn→quant 融合，并接入 `pass_manager.py` / `config/compilation.py` 的 `fuse_attn_quant` 开关。正文注明该 swap 是 "temporary until kernels support fp8 output"。

## 关键改动

```text
vllm/model_executor/layers/attention/mla_attention.py: forward_impl 用 temp-buffer swap 实现 fused output quant，新增 _quant_fp8_op。
vllm/compilation/passes/fusion/mla_attn_quant_fusion.py（新增）: pattern matcher + MLAAttnFusionPass，做 MLA attn→quant 融合。
vllm/compilation/passes/pass_manager.py、vllm/config/compilation.py: 接入 fuse_attn_quant 开关。
vllm/v1/attention/backend.py: MLA 基类新增 fused_output_quant_supported() stubs。
tests/compile/passes/test_mla_attn_quant_fusion.py、tests/compile/fusions_e2e/{models.py,test_tp1_quant.py}: FP8/NVFP4 融合测试与 e2e 模型注册。
```

```text
        use_quant = output_scale is not None or output_block_scale is not None
        if use_quant:
            # The fusion pass has allocated output with quantized dtype
            # (FP8 or uint8 for FP4). We can't write into it directly,
            # so we swap in a temp buffer for computation, then quantize
            # into the real output at the end.
            # NOTE(carlyou): this is temporary until kernels support fp8 output
            quant_output = output
            output = torch.empty(
                output.shape[0],
                self.num_heads * self.v_head_dim,
                dtype=q.dtype,
                device=output.device,
             )
```

## 性能声明

正文明确 "Check E2E accuracy and perf without and with fusion on this PR so we're not regressing either (not expecting speedup yet but not expecting a regression)"。gsm8k（Config `mla_quant_fusion/h100_fp8_eval`，Model `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8`，GPU NVIDIA H100 80GB HBM3 / 81559 MiB，CUDA 13.0，PyTorch 2.10.0+cu130，Python 3.12.3，x86_64，1319 questions；格式 Accuracy / Latency(s) / Q/s / Tok/s）：`fuse_off_eager` 0.783 / 78.95 / 16.71 / 2124.2；`fuse_on_eager` 0.782 / 72.48 / 18.20 / 2317.8；`fuse_off_compiled` 0.773 / 18.56 / 71.08 / 9035.3；`fuse_on_compiled` 0.779 / 44.92 / 29.37 / 3772.3。benchmark（GPU NVIDIA H100 NVL / 95830 MiB，TP=1，max-model-len 4096，`vllm bench serve --num-prompts 1000 --request-rate inf --random-input-len 128 --random-output-len 128 --num-warmups 50 --ignore-eos --max-concurrency 16`），六列依次为 `fuse_off_eager` / `fuse_on_eager` / `fuse_off_compiled` / `fuse_on_compiled` / `fuse_off_compiled_igp` / `fuse_on_compiled_igp`：Output token tput (tok/s) 470.69 / 466.21 / 2117.15 / 2069.74 / 2042.51 / 2036.29；Request throughput (req/s) 3.68 / 3.64 / 16.54 / 16.17 / 15.96 / 15.91；Mean TTFT (ms) 111.73 / 110.19 / 47.69 / 88.88 / 54.58 / 52.11；Mean TPOT (ms) 33.11 / 33.45 / 7.19 / 7.04 / 7.41 / 7.45。可见 eager 下 gsm8k 略好、compiled 路径 Q/s 与 TTFT 反而回退（71.08→29.37、47.69→88.88），与"不主张加速"一致。

## 验证与回归风险

验证：Unit tests（CI，B200）新增 `test_mla_attn_quant_fusion.py`（MLA + FP8、MLA + NVFP4 融合）；E2E FP8（1xH100）serve `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-FP8` 带 `fuse_attn_quant=true`；E2E NVFP4（1xB200）serve `carlyou/DeepSeek-V2-Lite-NVFP4` 带 `fuse_attn_quant=true`；E2E accuracy 与 perf 的 fusion on/off 对比。风险：正文明确这是 issue #35792 的 Phase 1 (of 2)，temp-buffer swap 是 "temporary until kernels support fp8 output"；compiled 路径下吞吐与 TTFT 有回退；`tests/compile/fusions_e2e/models.py` 注明 MLA attn + per-group FP8 quant 仍不支持（DeepSeek-V3 `attn_quant_fusion=0`）。

## 关联

issue #35792（Phase 1 of 2，标题亦引用 #35792）；Phase 2 后续 PR 正文未给编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
