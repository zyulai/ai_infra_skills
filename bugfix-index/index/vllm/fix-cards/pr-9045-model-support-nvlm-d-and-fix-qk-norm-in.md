# PR #9045: [Model] Support NVLM-D and fix QK Norm in InternViT

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9045](https://github.com/vllm-project/vllm/pull/9045) |
| 作者 | DarkLight1337 |
| 合并日期 | 2024-10-07 |
| labels | ready |
| 热度 | 👍 5 · 💬 27 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
基于 InternVL 新增 NVLM-D 模型支持,并修复 InternViT 并行 attention 与 QK normalization 不兼容的问题(TP 下先 all-gather 再归一化)。

## 问题现象
在测试 NVLM-D 模型时,@ywang96 发现 InternViT 现有的并行 attention 实现无法与 QK normalization 一起工作(即 TP>1 且模型开启 `qk_normalization` 时 attention 结果错误;具体报错/数值表现正文未给出,由 @Isotr0py 修复)。该问题影响 InternVL 系列及使用同款 InternViT 的模型;NVLM-D 的接入也因此依赖此修复(正文同时为两个 issue 的修复:FIX #9040 NVLM-D 支持、FIX #9041 QK Norm)。

## 根因
正文披露:InternViT 的并行 attention(InternParallelAttention)与 QK normalization 不兼容,但未展开机制细节。从 diff 描述改动所暗示的机制:原实现把 TP 分片后的 q/k 在本地 `flatten(-2,-1)` 后各自做 RMSNorm,分片内的归一化统计量不等价于对完整 head 维的归一化,结果随 TP 切分方式改变;同时 head 数不能被 TP 整除时并行路径根本不可用,只能回落 SDPA(由 diff 推断)。

## 修复方案
- `vllm/model_executor/models/intern_vit.py`:新增 `_apply_qk_norm`——TP>1 时先 `tensor_model_parallel_all_gather` q/k,用 `forward_native` 对完整张量做 q_norm/k_norm,再按 tp_rank 切回本地分片;norm 在 view 成 head 维之前执行:

```python
def _apply_qk_norm(self, q, k):
    if self.tp_size > 1:
        q = tensor_model_parallel_all_gather(q.contiguous())
        k = tensor_model_parallel_all_gather(k.contiguous())
    q = self.q_norm.forward_native(q)
    k = self.k_norm.forward_native(k)
    if self.tp_size > 1:
        q = splitter(q)[self.tp_rank]
        k = splitter(k)[self.tp_rank]
    return q, k
```

- 引入 `num_dummy_heads`「哑头」机制:qkv/proj 按 `(num_dummy_heads + num_heads)` 建,使常见 GPU 数下 `(num_heads + num_dummy_heads) % tp_size == 0` 可走 TP;q_norm/k_norm 用新参数 `RMSNorm(dummy_dim, var_hidden_size=embed_dim)`。
- `vllm/model_executor/layers/layernorm.py`:RMSNorm 新增 `var_hidden_size`(variance_size_override),方差只在真实 embed_dim 维上计算、权重覆盖 padding 后的 dummy_dim;有 override 时 CUDA/XPU 回落 native 实现。
- 另:InternVL 输入处理重构为 `InternVLInputPipeline` 类,`chat_utils.py` 将 `NVLM_D` 模型类型的占位符映射为 `<image>`,examples 新增 NVLM-D(单图/多图)示例;diff 中 `InternVisionModel` 删除了 `resize_pos_embeddings`。
- 注:pack 中 diff 在 40000 字符处截断(截断处位于 internvl.py 重构中段),其后改动未见于 pack,如实注明。

## 影响与回归风险
- 回归面:InternViT 并行 attention 路径(TP 分片、QK norm 顺序与 TP 通信量——QK norm 开启时每次 forward 多两次 all-gather)、RMSNorm 的 CUDA/XPU native 回退分支、InternVL 输入管线注册方式;新增 dummy heads 改变 qkv/proj 形状与权重加载映射;SDPA 回退条件从 `num_heads % tp_size` 变为 `(num_heads + num_dummy_heads) % tp_size`。

## 关联
FIX #9040、FIX #9041

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
