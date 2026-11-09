# PR #17613: [Perf] refactor piecewise cuda graph support of Qwen3-Next

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zminglei |
| 合并 | 2026-02-14 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,改 5 个文件含 1 个测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/17613 |

## 一句话结论

Qwen3-Next piecewise cuda graph 从"整层 fake op"改为只跳过 attention 本体,gsm8k 吞吐 2592 → 2964 tok/s(+14.3%)。

## 动机

此前 Qwen3Next 把 `Qwen3GatedDeltaNet.forward` 内所有算子藏进一个自定义 fake op,即使开启 piecewise cuda graph(PCG),GatedDeltaNet 整个 forward 仍以 eager 模式运行,输入投影、split/reshape、norm、输出投影都无法被捕获。本 PR 重构 `RadixLinearAttention`,对齐 `RadixAttention` 的模式:PCG 捕获时只跳过 attention 计算本身,让更多算子进入图内,提升 prefill 性能。

## 优化点

机制上是把 split 点从"整层"下移到"attention 核心"。`RadixLinearAttention.forward` 在 extend + PCG 上下文中改为调用新注册的自定义 op `unified_linear_attention_with_output`(以 `mutates_args=["output"]` 注册 custom op,并 `register_split_op` 切图):该 op 内部仍然走 `attn_backend.forward`,但只包住线性注意力计算,`Qwen3GatedDeltaNet.forward` 里的 `_forward_input_proj`、split/reshape、gated norm、`_forward_out_proj` 全部变为可被 torch.compile/PCG 捕获的普通算子;原先包整层的 `gdn_with_output` 路径删除。配套两处:`layernorm_gated.py` 的 `calc_rows_per_block` 在 PCG 启用时返回常量 `MAX_ROWS_PER_BLOCK=4`,避免 torch.compile 在动态 batch 维上生成 guard;`model_runner.init_piecewise_cuda_graphs` 的 attention 层注册适配 `layer.linear_attn.attn` 的新属性名(Qwen3GatedDeltaNet 中 `linear_attn` 更名为 `attn`)。

## 关键改动

```text
python/sglang/srt/layers/radix_linear_attention.py: 新增 unified_linear_attention_with_output split op,只跳过 attention 核心
python/sglang/srt/models/qwen3_next.py: 删除整层 fake-op 包装,linear_attn 更名 attn,forward 直接受控于 PCG
python/sglang/srt/layers/attention/fla/layernorm_gated.py: PCG 启用时 rows_per_block 用常量,避免动态 shape guard
python/sglang/srt/model_executor/model_runner.py: attention 层注册适配新属性名
test/registered/models/test_qwen3_next_models_pcg.py: 解除 PCG 测试禁用(原因 #17039 间歇失败)
```

```python
@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_linear_attention_with_output(
    mixed_qkv: torch.Tensor, a, b, output, layer_id: int,
) -> None:
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layer = context.attention_layers[layer_id]
    ret = forward_batch.attn_backend.forward(
        layer=attention_layer, forward_batch=forward_batch,
        mixed_qkv=mixed_qkv, a=a, b=b,
    )
    output.view(ret.shape).copy_(ret)
```

## 性能声明

测试条件:Qwen3-Next-80B-A3B-Instruct,`--tp 4 --enable-piecewise-cuda-graph`(启动命令同 PR),gsm8k `bench_sglang.py` 200 条:

- Before:Accuracy 0.950,Latency 12.379 s,Output throughput 2592.343 token/s
- After:Accuracy 0.960,Latency 11.041 s,Output throughput 2963.644 token/s
- 吞吐提升 14.3%(PR 正文原话 "The throughput is improved by 14.3%")

PR 未注明 GPU 型号与并发/workload 细节(仅给出上述启动参数与基准命令)。

## 验证与回归风险

重新启用此前因间歇失败(#17039,5-10% 概率)被禁用的 `test_qwen3_next_models_pcg.py` CUDA CI 测试(stage-c-test-4-gpu-h100,est_time 400s),即以该测试作为回归防线;PR 正文 Accuracy Tests 一节为空(仅以 gsm8k Accuracy 0.950→0.960 作参考)。回归面:Qwen3-Next 的 PCG 路径(仅 extend + forward context 存在时走新 op,decode/普通路径不变);`unified_linear_attention_with_output` 从全局 context 取 layer,依赖 `init_piecewise_cuda_graphs` 注册顺序;layernorm_gated 的常量 rows_per_block 也影响非 PCG 下该 kernel 启动配置的读法(仅 PCG 分支改变)。

## 关联

同区域:Qwen3-Next MTP/FLA 相关优化(如 #11133,同模型线性注意力路径);pack 正文未列出其他 PR 编号,仅提及 issue #17039(PCG 间歇失败调查)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
