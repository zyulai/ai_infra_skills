# PR #32950: [MLA] Fuse cat and qaunt for fp8 kv-cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-01-24 |
| 分类 | Quantization |
| Label | documentation, performance, ready, deepseek |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),实际改动集中在 mla_attention.py |
| 链接 | https://github.com/vllm-project/vllm/pull/32950 |

## 一句话结论

把 MLA decode fp8 路径里 "拼接 q_pe/ql_nope 再量化" 的三段操作(cat + reshape + scaled_fp8_quant)收进一个 custom op,消除 eager 逐算子开销。

## 动机

MLA decode 走 fp8 attention 时,需要把 `decode_ql_nope` 与 `decode_q_pe` 在最后一维拼成完整 q、flatten 后做 per-tensor FP8 量化再 view 回原形状。旧实现是 empty+两次 copy_(注释称 torch.cat 开销大)+ `ops.scaled_fp8_quant` + view,eager 下每个小算子都有 launch/调度开销,decode 小 kernel 序列里占比可观。

## 优化点

新增 `_DecodeConcatQuantFP8` custom op(继承 `QuantFP8`,注册名 `mla_decode_concat_quant_fp8`):forward 统一为 `cat -> reshape -> quant_fn -> view`,并把 `forward_native`/`forward_cuda`/`forward_hip` 都指向同一工厂函数。关键在两点:一是 `static=True, group_shape=GroupShape.PER_TENSOR` 的静态 per-tensor FP8 量化路径;二是 `compile_native=True`——当 custom op 未启用 CUDA 实现时,forward_native 会被 torch.compile 整体编译,把 cat/reshape/quant/view 融合成一个 region,替代原先在 eager 里逐个执行的多个小 op。调用点从 11 行手写拼接/量化代码换成一次 op 调用。该 op 本质是把量化前的 q 拼接(cat)与 FP8 量化输入准备合并,服务于 fp8 KV-cache 的 q 量化。

## 关键改动

```text
vllm/model_executor/layers/attention/mla_attention.py: 新增 _DecodeConcatQuantFP8 custom op 并替换 decode fp8 路径的手写 cat+quant
```

```python
def forward(self, decode_ql_nope, decode_q_pe, scale, scale_ub=None):
    decode_q0 = torch.cat((decode_ql_nope, decode_q_pe), dim=-1)
    decode_q_flat = decode_q0.reshape(decode_q0.shape[0], -1)
    decode_q, _ = quant_fn(self, decode_q_flat, scale, scale_ub)
    return decode_q.view(decode_q0.shape)
```

## 性能声明

PR 正文仅附两张 benchmark 截图(Main vs PR 的对比数字在图片里,素材文本未包含具体数值),未写明测试模型/硬件/workload。PR 未注明可引用的量化测试条件。

## 验证与回归风险

- diff 中未见新增/修改测试;行为等价性由 cat+flatten+量化+view 的数学等价保证,但依赖既有 MLA fp8 测试回归。
- 风险:`_DecodeConcatQuantFP8` 用 static/per-tensor 量化,与原先 `ops.scaled_fp8_quant(decode_q0.view(...), layer._q_scale)` 相同 scale 语义;`expects` 抽象改变 decode_q 的产生路径,若 `fp8_attention` 且 DCP(dcp_world_size>1)分支交互处有形状假设,需要留意;非 fp8 路径不受影响。

## 关联

- 同区域:#20411(SM100 cuDNN MLA prefill)、#34732(FA4 MLA prefill)、#20034(flashinfer ragged prefill)、#21078(flashinfer trtllm-gen decode)、#12528(MLA decode 优化)。
- 依赖组件:`QuantFP8` input_quant_fp8 量化算子体系。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
