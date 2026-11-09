# PR #39947: [Feat] [NPU] DSpark: Support dsv4 dspark on ascend 950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @randgun |
| 合并 | 2026-10-30 |
| 分类 | Speculative Decoding |
| Label | deepseek, npu, run-ci, jit-kernel, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/39947 |

## 一句话结论

把 DeepSeek-V4 DSpark 投机解码扩展到 NPU Ascend 950(arch35),补齐其专用 sparse attention 算子、权重加载与数值兼容。

## 动机

已有 DSpark 实现只支持 CUDA 与旧 NPU(arch22);arch35(Ascend 950)需要不同的 sparse attention 算子、权重加载策略与数值兼容 workaround。另有此前一次改动把断言条件从 `if fp8:` 改成 `if quantize_wo_a:` 引入回归:DSpark 的 `wo_a` 不量化,在 arch35 上触发错误断言,需一并修复。

## 优化点

为 arch35 新增 DSpark 专用的 v2 sparse attention 算子对(经新 `TorchOpLoader` 扩展加载,在 TP worker 与 DSpark draft worker 初始化时装入):A5 上使用专用量化 sparse-attention kernel `kv_quant_sparse_attn_sharedkv`,metadata 走 host 版本(从 CPU mirror 读序列长度,缺 CPU mirror 时显式报错);`_sparse_attn_ops(is_dspark=...)` 统一按 draft worker/架构选择算子对。修 `use_npu_arch35_mxfp8_wo_a()`:新增 `wo_a_keeps_quant_config` 参数,DSpark 显式关闭 wo_a 量化时返回 False,模型侧缓存该值不再逐调用点重算。新增 `_dspark_stage_config()`:给 draft stage 应用 draft 专属 expert 数、对纯文本 stage 关闭 vision 层;arch35 回退到标准词表/融合行为而非 arch31 专有路径(`uses_own_vocab_modules = _is_npu and not is_npu_arch35()`)。权重加载把 NPU `weight_packed`→`weight` 改名移到 stacked 参数映射之前,消除重复分支;`_fused_commit_kv_proj_supported` 在 NPU 上直接返回 False 走 per-linear torch 路径;`_kernel_metadata_from_parts` 删除 block_tables/max_seqlen_q/is_nextn 等不再需要的参数。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py: DSpark 专用 sparse attention 算子与 host metadata
python/sglang/srt/hardware_backend/npu/utils.py: use_npu_arch35_mxfp8_wo_a 增加 wo_a_keeps_quant_config
python/sglang/srt/models/deepseek_v4.py / deepseek_v4_dspark.py / deepseek_v4_nextn.py: wo_a 判定缓存与 arch35 权重加载
python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py: NPU 禁用 fused commit KV proj
```

```python
def _sparse_attn_ops(is_dspark=False):
    if is_npu_arch35():
        if is_dspark:
            from sgl_kernel_npu.operators.attention import (
                kv_quant_sparse_attn_sharedkv,
                kv_quant_sparse_attn_sharedkv_metadata_host,
            )
            return (kv_quant_sparse_attn_sharedkv_metadata_host,
                    kv_quant_sparse_attn_sharedkv)
        return (torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata,
                torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv)
```

## 性能声明

功能启用型 PR,无前后对比。DeepSeek-V4-Flash,GPQA-Diamond 198 题:GPQA-Diamond 准确率 89.9%;Avg Latency 149.111 s,Avg TTFT 1943.3 ms,Avg TPOT 14.3 ms,Avg Throughput 67 tok/s,平均输入 344 / 输出 9991 tokens。测试基线模型为 DeepSeek-V4-Flash-0731;硬件为 Ascend 950(arch35),其余测试条件 PR 未注明。

## 验证与回归风险

更新 `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py`:metadata 调用签名删参、DSpark host metadata 用显式 mocked `metadata_op` 断言收到 CPU 序列长度;精度验证为 GPQA-Diamond e2e。风险面:DSpark host metadata 强制要求 CPU query/KV 长度 mirror(否则 RuntimeError);NPU 上禁用 fused commit KV proj 走 per-linear 路径可能影响 draft 侧开销;wo_a 判定与权重加载分支改动同时影响 target/draft/nextn 模型的参数分配。pack 显示 CI Base、Extra、AMD ROCm 10 三项均失败。

## 关联

同区域:同批 #32207(NPU PP prefill 节点支持 EAGLE)、#43838(DeepSeek-V4.1 小批量 decode 融合扩到 TP2)。DSpark 前序实现与回归引入 commit 素材未注明编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
