# PR #47419: [ROCm] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @larryli2-amd |
| 合并 | 2026-07-10 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 6 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/47419 |

## 一句话结论

把 DeepSeek-V4-Pro DSpark 投机解码 port 到 ROCm gfx950,8×MI350X 下每 GPU 吞吐 32.6→51.9 tok/s、TPOT 1.66×。

## 动机

DSpark speculative decoding for **DeepSeek-V4-Pro (1.6T)** 此前在 vLLM 中是 NVIDIA-only:`vllm/models/deepseek_v4/__init__.py` 在 ROCm 上把 `DSparkDeepseekV4ForCausalLM = None`,导致 `--speculative-config '{"method":"dspark",...}'` 在 AMD GPU 上模型加载即失败。基础 DeepSeek-V4 模型、ROCm sparse-MLA / sparse-SWA backends 与 MHC `aiter`/`triton` kernel 在 ROCm 上均已存在,故本 PR 是 wiring port 而非新 kernel 工作。

## 优化点

把 draft model port 到 AMD subpackage 并补齐两个缺失依赖:目标模型的 EAGLE3 aux-hidden-state 接口,以及 non-causal sliding-window-attention 的 ragged-buffer sizing。`amd/dspark.py` 是 `nvidia/dspark.py` 的 ROCm 移植:改用 AMD `DeepseekV4DecoderLayer`、把 MHC head 经 `HCHeadOp`/`MHCPostOp` CustomOp dispatcher(aiter/tilelang/triton)路由而非 NVIDIA-only tilelang kernel、按 `use_fused_mhc` 门控尾部 `mhc_post`,并去掉 AMD `model.py` 中不存在的 mega-MoE 权重路径。`amd/model.py` 加 `EagleModelMixin` / `SupportsEagle3` 并在 DSpark 目标层 `dspark_target_layer_ids = [58, 59, 60]` 收集 mean-pooled aux hidden states。`amd/rocm.py` 把 sparse-SWA ragged buffer 按 noncausal 索引宽度 `cdiv(window_size + num_speculative_tokens, 128) * 128` 分配:此前只按 causal 宽度 `max_tokens * window_size` 分配,DSpark draft 的非因果窗口会撑爆 buffer 报 `RuntimeError: The size of tensor a (896) must match ... (1792)`;修法对既有 causal 路径是 no-op(`noncausal_index_width == 0`)。`config/speculative.py` 对 `method="dspark"` 拒绝 `num_speculative_tokens < dspark_block_size`(半自回归 block drafter,更短会产生 garbled output)。

## 关键改动

```text
vllm/models/deepseek_v4/__init__.py: 解开 ROCm 分支的 stub,从 .amd.dspark 导入 DSparkDeepseekV4ForCausalLM 而非设为 None
vllm/models/deepseek_v4/amd/dspark.py: 新增,nvidia/dspark.py 的 ROCm 移植(MHC CustomOp 路由、use_fused_mhc 门控、去 mega-MoE 权重路径)
vllm/models/deepseek_v4/amd/model.py: 加 EagleModelMixin / SupportsEagle3 与 DSpark 目标层 [58,59,60] 的 aux hidden states 收集
vllm/models/deepseek_v4/amd/rocm.py: 按 noncausal_index_width 调整 sparse-SWA ragged buffer 尺寸与 max-entries-per-row
vllm/config/speculative.py, tests/models/test_registry.py: dspark_block_size 校验;registry 测试改为仅在非 CUDA 且非 ROCm 时 skip
```

关键代码(dspark 的 block-size 下限校验):

```python
                    dspark_block_size = getattr(
                        self.draft_model_config.hf_config,
                        "dspark_block_size",
                        None,
                    )
                    if (
                        dspark_block_size is not None
                        and self.num_speculative_tokens < dspark_block_size
                    ):
```

## 性能声明

Results(8× AMD MI350X、gfx950):镜像 `vllm/vllm-openai-rocm:nightly-09663abde0f50944a8d5ea30120666024b503faa`(vLLM `0.23.1rc1.dev714+g09663abde`),Recipe InferenceX `dsv4-fp4-mi355x-vllm`(TP=8、`--moe-backend aiter`、FP8 KV cache、full CUDA graph `mode:3 FULL_AND_PIECEWISE`),Workload `ISL/OSL = 1024/1024`、`--ignore-eos`(固定 1024 output tokens)、greedy,模型 `deepseek-ai/DeepSeek-V4-Pro-DSpark`(混合精度:MXFP4 MoE experts + FP8 block-scaled dense/attention + FP8 `fp8_ds_mla` KV cache)。Throughput @ concurrency = 8(Output tok/s **per GPU**):STP baseline(no draft)32.6、TPOT 29.7 ms、1.00×;DSpark `num_speculative_tokens=6` 51.9、TPOT 17.9 ms、1.59×,Mean acceptance length 2.94、Avg draft acceptance 32.4%、Per-position acceptance `0.819 / 0.403 / 0.247 / 0.193 / 0.158 / 0.122`。正文另注 "TPOT speedup vs baseline: DSpark n=6 = 1.66×"。Quality:固定 10-prompt set(raw `/v1/completions`、temperature 0)下 STP 与 DSpark n=6 输出一致且正确;从本 PR 提交代码直接重跑 DSpark n=6 复现约 52 tok/s/GPU(在 run-to-run 方差内)。Caveats:验证在 MI350X,InferenceX 参考 recipe 针对 MI355X(同 gfx950、更高 clocks/TDP),绝对吞吐可能不同,但相对 speedup 在同硬件同镜像下可直接比较。

## 验证与回归风险

`tests/models/test_registry.py` 的 `DSparkDraftModel` 从"仅 CUDA"改为"CUDA + ROCm 支持、XPU 仍 stub"(diff 可见);正文给出 Reproduction 全套命令(serve / InferenceX `benchmark_serving.py` / `curl /metrics` 计算 mean acceptance length 与 per-position acceptance),并注明 DSpark 运行需 `--use-chat-template --dsv4` 否则随机 token 上 acceptance 退化。风险面:正文 "Note: known limitation" 明确 greedy DSpark 偶发 prompt/run-dependent garbled output(不特定于某个 `num_speculative_tokens`),根因留作 follow-up;`config/speculative.py` 的 dspark 校验影响所有 `method="dspark"` 配置;non-causal ragged-buffer 改动触及 ROCm sparse-SWA 共享路径(正文称对 causal 路径 no-op)。

## 关联

无(PR 正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
