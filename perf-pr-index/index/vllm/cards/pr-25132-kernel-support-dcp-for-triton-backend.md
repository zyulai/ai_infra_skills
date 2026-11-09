# PR #25132: [Kernel] Support DCP for Triton backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @frank-wei |
| 合并 | 2025-09-25 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, gpt-oss |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/25132 |

## 一句话结论

Triton MLA decode kernel 返回 LSE,使 Triton attention 后端可以接入 DCP(Decode Context Parallel)。

## 动机

DCP 需要跨 rank 合并部分注意力结果,依赖 decode kernel 输出 LSE(log-sum-exp);Triton decode kernel 此前只返回 attention 输出、丢弃 LSE,导致 TRITON_MLA 后端无法参与 DCP(#23734 已为其他后端铺路,本 PR 是其 follow-up)。另发现 deepseek_v2 有一个会改写 residual 的隐患。

## 优化点

_fwd_kernel_stage2 在归并各 KV split 时计算并存储 lse_val = e_max + log(e_sum),lse 张量贯穿 decode_attention_fwd 全链路(含 normal/grouped 两条路径与 stage2 的 stride 参数);TritonMLAImpl 声明 can_return_lse_for_decode = True,_forward_decode 返回 (o, lse) 并用 q.shape[1] 替代 self.num_heads 以兼容 head 维变化。另修 deepseek_v2.py:residual 为 None 时改为 residual = hidden_states.clone(),避免后续 input_layernorm 分支潜在地修改共享的 residual 变量。

## 关键改动

```text
vllm/attention/ops/triton_decode_attention.py: stage2 kernel 输出 LSE,签名与调用链透传 lse
vllm/v1/attention/backends/mla/triton_mla.py: can_return_lse_for_decode=True;_forward_decode 返回 (o, lse)
vllm/model_executor/models/deepseek_v2.py: residual = hidden_states.clone() 防共享修改
tests/kernels/attention/test_triton_decode_attention.py: 适配新签名
```

```python
lse_val = e_max + tl.log(e_sum)
tl.store(lse + cur_batch * stride_lse_bs + cur_head, lse_val)
```

## 性能声明

PR 未给出性能数字。验证为 tests/distributed/test_context_parallel.py 2 passed(212.42s,4 卡 CUDA_VISIBLE_DEVICES=4,5,6,7,VLLM_ATTENTION_BACKEND=TRITON_MLA)。收益为 DCP 对 Triton 后端可用这一能力,而非单点加速。

## 验证与回归风险

tests/distributed/test_context_parallel.py + triton decode attention 单测签名适配。风险:decode_attention_fwd 签名变化影响所有直接调用该 op 的外部代码;stage2 多一次 lse store(开销极小);deepseek_v2 的 clone 引入一次 hidden_states 拷贝,是非 DCP 路径上唯一可能的微小开销点。

## 关联

前序:#23734(DCP 支持的铺垫工作,本 PR 自述 follow-up)。同分类 Attention 卡:#12528(MLA decode 优化)、#42555(MLA prefill 参数清理)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
