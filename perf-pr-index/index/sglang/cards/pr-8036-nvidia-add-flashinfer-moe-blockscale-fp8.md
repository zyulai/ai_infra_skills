# PR #8036: [NVIDIA] Add Flashinfer MoE blockscale fp8 backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kaixih |
| 合并 | 2025-07-27 |
| 分类 | MoE |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 8 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/8036 |

## 一句话结论

在 Blackwell EP MoE 上接入 FlashInfer TRTLLM block-scale FP8 后端,低延迟场景 e2e 最高提升 3 倍。

## 动机

为低延迟(low latency)场景启用 FlashInfer 的 MoE blockscale FP8 后端。PR 正文极简,未展开具体瓶颈分析,仅说明 e2e 性能最高提升 3 倍(数据放在 PR 评论 #issuecomment-3104285508,pack 未收录该评论)。

## 优化点

新增 `FlashInferEPMoE`(EPMoE 子类):forward 不再走 SGLang 侧 topk + 分组 GEMM,而是直接调用 `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`,把 routing 融合进 kernel——传入 `routing_logits`(fp32)、`routing_bias`(即 correction_bias)、`routing_method_type=2`(DeepSeek 风格 routing)、`n_group`/`topk_group`/`routed_scaling_factor` 等,因此启用时模型侧 `self.topk` 置为 None,`experts(...)` 改收 `router_logits`。输入激活用 `sglang_per_token_group_quant_fp8` 做 128x128 block 量化,且 hidden_states scale 需转置后传入。tile 大小由 `_get_tile_tokens_dim` 按每专家 token 均匀假设估算、pad 到 2 的幂并 cap 在 8-64(kernel 支持范围)。权重加载侧:FlashInfer 假定 w13 权重为 w31 排布,weight loader 在启用时对调 w1/w3 shard(weight scale 同理)。后端选择优先级:deepep > flashinfer cutlass > flashinfer trtllm > ep_moe > fused_moe。CLI 上原 `--enable-flashinfer-moe` 更名为 `--enable-flashinfer-cutlass-moe`,新增 `--enable-flashinfer-trtllm-moe`(强制要求 `--enable-ep-moe`)。

## 关键改动

```text
python/sglang/srt/layers/moe/ep_moe/layer.py: 新增 FlashInferEPMoE 与 trtllm_fp8_block_scale_moe 接入
python/sglang/srt/models/deepseek_v2.py: topk=None 分支,experts 直接收 router_logits
python/sglang/srt/server_args.py: 新增 --enable-flashinfer-trtllm-moe,旧 flag 更名
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: enable_flashinfer_moe 改名 cutlass
python/sglang/srt/models/qwen2_moe.py / qwen3_moe.py: 同步 flag 更名
python/sglang/srt/layers/quantization/modelopt_quant.py: 属性更名 enable_flashinfer_cutlass_moe
```

```python
a_q, a_sf = sglang_per_token_group_quant_fp8(hidden_states, self.block_shape[1])
a_sf_t = a_sf.t().contiguous()  # NOTE: scales of hidden states have to be transposed!
return fi_fused_moe.trtllm_fp8_block_scale_moe(
    routing_logits=router_logits.to(torch.float32),
    routing_bias=self.correction_bias.to(hidden_states.dtype),
    hidden_states=a_q, hidden_states_scale=a_sf_t, ...)
```

## 性能声明

- "The e2e perf shows up to 3x improvement",数据位于 PR 评论(issuecomment-3104285508);pack 未包含评论内容,具体模型/硬件/workload/精度条件 PR 正文未注明
- 适用定位:低延迟场景;后端仅 Blackwell + BlockScale FP8 + EP MoE(要求 `--enable-ep-moe`)

## 验证与回归风险

PR 正文未注明新增测试。风险与约束(代码 assert):仅支持 silu activation、renormalize 必须开启、不支持 fused shared experts;w1/w3 shard 对调使权重加载路径分叉,错配会静默加载错位;`deepseek_v2.py` 的 normal/dual_stream forward 因 topk=None 改变调用形态;CLI flag 更名对现有启动脚本是不兼容变更(旧 `--enable-flashinfer-moe` 失效)。

## 关联

并列后端:FlashInfer CUTLASS MoE(modelopt_fp4,原 enable_flashinfer_moe 路径)、DeepEP、EPMoE;同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
