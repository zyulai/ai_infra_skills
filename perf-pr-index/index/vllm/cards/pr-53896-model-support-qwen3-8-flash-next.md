# PR #53896: [Model] Support Qwen3.8-Flash-Next

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @peakcrosser7 |
| 合并 | 2026-08-31 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | new-model, speculative-decoding, ready, ci/build, qwen, kv-connector, nvidia, quantization, mrv2, kimi, k3, scheduler, kv-cache-manager |
| Diff | diffstat 缺失(pack 未记录);diff 极大,pack 仅保留 12 个测试/内核文件,108 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/53896 |

## 一句话结论

为 vLLM 接入 Qwen3.8-Flash-Next(内部代号 qwen4_exp)hybrid 架构,含 MTP 投机解码与 PLE n-gram embedding CPU offload 支持。

## 动机

Qwen 发布了新模型 Qwen/Qwen3.8-Flash-Next,vLLM 需要模型接入才能服务它。该模型是 hybrid 线性注意力架构(层类型含 linear_attention/full_attention),带 hyper-connection(HC)机制、MTP 投机解码头,以及可选的 PLE(Prefetch Layer Embedding)n-gram embedding CPU offload;其 TP 切分后的中间维度(如 640/4=160)与 checkpoint 的 128 FP8 block 不对齐,也需要量化路径上的专门处理。

## 优化点

机制上分四块:(1) 新增 `vllm/models/qwen4_exp/` 模型实现(common/NVIDIA/AMD 分层、`Qwen4ExpConfig`、hyper-connection 算子 `hc_combine`/`hc_gate_mix`/`grouped_gemma_rmsnorm`、PLE layer 与 `Qwen4ExpModelState` 输入准备),配套 `short_conv_attn.py` 注意力后端与 QSA pre-indexer;(2) fused GDN decode MTP CUDA kernel 增加 `output_gate_activation` 模板参数,支持 silu 与 sigmoid 两种输出门控;(3) 投机解码:`SpeculativeConfig` 草稿配置覆盖(Qwen4ExpMTP、`index_share_for_mtp_iteration` 语义),服务命令为 `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`;(4) TP 不对齐的 blockwise FP8 MoE:将 checkpoint 128 块的 scale 网格无损细化为 32 块(每个 32 块均落在某个全局 128 块内),使 Triton fused MoE 可消费精确的 per-shard scale。PLE offload 依赖 PR #53899(`VLLM_PLE_CPU_OFFLOAD=1`)。

## 关键改动

```text
vllm/models/qwen4_exp/: 模型实现、config、HC ops、PLE layer、nvidia/AMD
  分层(pack diff 省略,由测试与 CI 依赖推断其存在)
csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu: 增加 SigmoidGate
  模板参数与 output_gate_activation 入参
.buildkite/test_areas/models_basic.yaml: 新增 H200/MI300/CPU qwen4_exp 测试步
tests/models/qwen4_exp/: test_config、test_hc_ops、test_ple、test_qsa_* 等
```

关键机制摘录(细粒度 scale 细化,tests/kernels/moe/test_block_fp8.py 语义):

```python
# per-rank scale grid is refined from 128 to 32 (lossless: each 32-block
# lies inside one global 128-block)
ckpt_block = [128, 128]
refined_block = [32, 32]
factor = ckpt_block[0] // refined_block[0]  # 4
```

## 性能声明

PR 正文未给出量化性能数字。验证矩阵由 Inferact 提供(详见 vLLM recipe):无 offload 时 BF16、FP8、NVFP4 权重,平台 GB300、GB200、H200、MI355X(仅 BF16),并行 TP2/TP4/TEP4;开启 N-gram embedding offload 时 BF16 与 FP8(NVFP4 暂不支持),平台 GB200,并行 TP2/TP4/DEP4。

## 验证与回归风险

新增测试:`tests/models/qwen4_exp/`(config 校验、MTP draft 覆盖、PP+PLE 拒绝路径、ngram 上下文/dummy 输入稳定性、HC 算子数值、PLE 分片加载)、`tests/kernels/mamba/test_gdn_fused_mtp.py` 与 `test_fused_gdn_post_conv.py` 增加 sigmoid/silu 双激活参数化、`tests/kernels/moe/test_block_fp8.py` 增加 TP 不对齐 refined-block-scale 测试(4 个 TP rank)、`test_trtllm_bf16_moe.py` 改用 `convert_to_unquantized_kernel_format` 并验证 N pad 到 128 倍数。另新增 B200/H200 FP8 accuracy eval 配置。风险面:diff 规模极大(108 文件省略)且作者注明 review 期间可能不稳定,期间应使用 PR #53899 构建;涉及共享内核改动(fused GDN MTP、MoE format 转换)会波及既有 kimi/k3、qwen3_next 等模型路径。

## 关联

- 依赖/配套:PR #53899(PLE offload 支持,期间作为稳定构建入口)
- 同区域:qwen3_next 的 shared expert 对齐处理(`_should_replicate_misaligned_shared_expert` 复用)、GDN fused MTP 内核族

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
