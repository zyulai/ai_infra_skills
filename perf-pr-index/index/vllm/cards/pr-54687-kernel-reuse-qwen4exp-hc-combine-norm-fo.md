# PR #54687: [Kernel] Reuse Qwen4Exp HC combine-norm for MTP input

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gcanlin |
| 合并 | 2026-09-04 |
| 分类 | Speculative Decoding |
| Label | performance, ready, qwen |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/54687 |

## 一句话结论

Qwen4Exp MTP 输入改复用 HC combine-norm kernel(unit injection 特化),该步快至 4.84x。

## 动机

MTP 路径此前单独维护一个 add-and-norm kernel(`_forward_mtp_layer` 及其重复 kernel),而 Qwen4Exp 已有 HC combine kernel 体系。MTP 的 `fc_embedding` 要进入 normal decoder,本质就是一次无 gating 的 residual add 加 grouped RMSNorm——用"unit injection"(injection=None)即可让现有 kernel 表达,免去维护两份实现。与 #54560 不重复:那个针对一般 Qwen4Exp `M=1` SM90 decode 路径,本 PR 针对 MTP 入口的 residual add + grouped RMSNorm(任意 token 数)。

## 优化点

将 `injection_logits=None` 定义为 unit injection,Triton kernel 直接对可选 `inj_ptr` 特化:MTP 路径完全省掉 injection load 与 sigmoid。`fc_embedding` 作为 pending HC block output 传入 normal decoder,由现有 combine-norm kernel 融合 BF16 broadcast add 与 grouped Gemma RMSNorm;两个 HC combine 变体统一到同一 optional-injection 契约,删除 `_forward_mtp_layer` 与重复 MTP kernel。实现上:residual/block load 先于 FP32 转换发射;gated 与 unit 两路共享同一二维 tiled layout 和 RMSNorm reduction;launch grid 从 `(N, hc_count)` 改为展平的 token-major 一维 `(N * hc_count,)`,各 token 的 stream 相邻,无需 injection 专属 grid 分支;唯一 unit 特有调度是把 norm weight load 提到 reduction 之前,gated 路径仍保持 reduction 后 load 以限制 live range。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/ops/hc.py: hc_combine/hc_combine_norm 支Optional injection,grid 展平、inj_ptr 特化
vllm/models/qwen4_exp/nvidia/mtp.py: MTP 不再自行 add,改传 prev_block_output=inputs_embeds
vllm/models/qwen4_exp/nvidia/model.py: Qwen4ExpDecoderLayer 按 prev_block_output 是否存在走 combine_and_mix
vllm/models/qwen4_exp/nvidia/hyperconnection.py: combine/combine_and_mix 签名改为 injection 可空
tests/models/qwen4_exp/test_hc_ops.py: 新增 unit-injection 两个测试
```

```python
if inj_ptr is not None:
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
```

## 性能声明

Kernel microbench:NVIDIA B200、CUDA 13.0、PyTorch 2.13.0+cu130、BF16、H=2560、HC=4,CUPTI medians、CUDA graphs、cold L2,双方均 `torch.compile(fullgraph=True)` 并先校验输出。Without injection(vs main 的 broadcast add + grouped RMSNorm 编译序列):1 token 1.533x、64 token 2.302x、2048 4.508x、8192 **4.843x**、32768 4.817x。With injection(vs main 既有 injected 路径):小 token 数在噪声内持平(≈1.0x),8192/32768 token 约 +20%(1.203x/1.208x)。E2E benchmark:PR 注明"pending final paired rerun on TP4 B200",未给出最终数字。GSM8K 全量 1319 题、5-shot、greedy:flexible 0.9666±0.0049,strict 0.9659±0.0050。

## 验证与回归风险

`pytest tests/models/qwen4_exp/test_hc_ops.py`:8 passed,覆盖 gated 与 unit-injection combine 路径,token 数含 1/17/2048;新增两个 unit-injection 测试断言与参考实现 `torch.equal`。风险:共享 kernel 的 grid 结构(二维→一维)与签名契约变化影响 qwen4_exp 全部 HC 路径(含 injected combine 与下游 mixer),非 MTP 路径需回归;microbench 显示 injected 路径大 token 数才受益,小 token 数持平;e2e 数字 PR 注明 pending。

## 关联

#54560(同模型相邻优化,PR 正文明确区分不重复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
