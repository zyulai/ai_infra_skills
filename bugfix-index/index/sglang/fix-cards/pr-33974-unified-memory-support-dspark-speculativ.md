# PR #33974: [unified memory] Support DSPARK speculative decoding + fix two NaN root causes (page hand-out zeroing, CuTe int32 slot-stride wrap)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33974](https://github.com/sgl-project/sglang/pull/33974) |
| 作者 | ch-wan |
| 合并日期 | 2026-08-10 |
| labels | blackwell, run-ci, jit-kernel, bypass-fastfail |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
--enable-unified-memory 原本硬禁投机解码;接入 DSPARK chain,并修复页未清零与 CuTe int32 slot-stride 回绕两个 NaN 根因。

## 问题现象
`--enable-unified-memory` 下 `server_args.py` 直接 assert 不允许任何投机算法。在 Kimi-K3(TP8, 2x4 GB300)与 Kimi-Linear-48B 端到端验证 DSPARK 时发现两个真实 NaN/损坏根因:POISONED 池 + rand draft 时 GSM8K 0.005 / Invalid 0.985、accept 假 1.00/8.00(argmax(NaN)=0 token 链);只修清零、clean 池仍 0.120。int32 回绕在 clean 启动即静默损坏(GSM8K 0.120、fake accept-rate 突发)或 illegal access,故此前表现为 unified-only、K3-only、load-only 的 heisenbug。

## 根因
正文披露两点:
1. trtllm-gen MLA kernel 读整页、对 seq_len 之外的行做算术掩码(NaN-unsafe);unified 池仅在启动时清零 raw buffer,页回收后历史字节重现,请求最后部分页尾部的 NaN 泄入 attention 输出。投机解码每个 verify 窗口都落在序列尾部新页,放大暴露。
2. DSPARK KDA verify kernel(nv_cutedsl 快路径)以静态 CuTe layout 编译,state-pool slot stride 是编译期常量;envelope-strided 视图 slot stride ~14M(fp32 ssm)/~28M(bf16 conv)元素,单个常量可放进 int32,`slot * stride` 折叠成 32 位算术,slot 超过 ~153(conv)/~306(ssm)即 mod 2^32 回绕。静态池 slot stride ~197K 元素,永不触界。

## 修复方案
五个独立 commit:①放开 server_args 门(仅 DSPARK chain、topk∈{None,1}、限定已审计 attention 后端),trtllm_mla 的 capture-stable DENSE write-loc 扩展到 target_verify,mamba scatter 的 dst 契约放宽为「任意 layer/slot 步长 + 尾部 entry 连续」,VIRTUAL mamba_track_indices 在单点翻译为物理索引,draft pool 按统一分配器虚拟 id 空间(size_full,页对齐)定容;②新增 `SGLANG_DEBUG_POISON_POOL` 永久回归开关(启动用 bf16-NaN 填池);③分配器 hand-out 时以 `zero_pages` 清零页 envelope;④`kda_decode_mtp_kernel` 将 slot 提升为 int64;⑤核实写集顺序后保持 lazy compaction ON。

```python
+    # int64: `slot * stride` overflows int32 on envelope-strided pools.
+    slot = cutlass.Int64(slot)
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:server_args 门控、trtllm_mla target-verify 元数据、unified 分配路径(每次 hand-out 多一次页 envelope memset,正文 A/B 实测零开销)、KDA verify slot 运算、mamba scatter 契约、draft pool 尺寸与布局(关闭 page-major)。EAGLE/tree/DFLASH/NGRAM 仍被阻止;fa3/flashinfer 的 verify 未验证;compaction MOVE 与 in-flight verify 写的竞态未动态演练(正文 Limitations)。修复后 POISONED 池 GSM8K 0.985 与 clean 一致,real draft 0.990 / accept 0.53 / 4.74。

## 关联
- https://huggingface.co/RadixArk/Kimi-K3-DSpark(draft 模型)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
