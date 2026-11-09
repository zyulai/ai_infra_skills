# PR #45589: [Bugfix] Fix MoE model load OOM in FlashInfer_TRTLLM  backend with sleep mode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45589](https://github.com/vllm-project/vllm/pull/45589) |
| 作者 | andakai |
| 合并日期 | 2026-06-17 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | resource |

## 一句话结论
FlashInfer TRTLLM BF16 MoE 权重转换在 sleep mode 下反复产生 20-50 MiB 中间张量导致加载 OOM;改为预分配输出、index_select 直写,顺带提速约 4 倍。

## 问题现象
#43951:加载 `Qwen/Qwen3-235B-A22B-Instruct-2507` 且开启 `--enable-sleep-mode`、后端为 FlashInfer TRTLLM MoE 时,权重加载阶段(早于 CUDA graph profiling)OOM,失败分配可小至 20 MiB;traceback 指向 `convert_moe_weights_to_flashinfer_trtllm_block_layout`。作者在 H20-141GB 上强制让 FlashInfer TRTLLM 后端通过 SM100 capability gate 复现(TP=4、dummy load、bfloat16);仅 FlashInfer TRTLLM 后端出错,Triton、flashinfer_cutlass 后端可成功。

## 根因
正文完整披露:该模型 94 层 x 128 专家,原转换对每个专家执行 `clone -> index reorder -> contiguous -> convert_to_block_layout -> append -> torch.stack`,反复创建中等大小临时张量(clone、重排张量、连续拷贝、block-layout 输出、stack 输出),尤其 20-50 MiB 区间——w13 的 int64 permute indices 约 48 MiB,w2 的约 24 MiB。#41268 为防加载期碎片化设了 `max_split_size_mb=20` 的分配器策略,实测该模型 20 → OOM、25/32/64/128 → 成功,二者冲突;且 sleep mode 下这些分配走 vLLM 的 CuMem weights pool,pool 在加载中途不完全回收临时 backing segment。FlashInfer CUTLASS 与 Triton 无此问题,因其后处理不产生同样的重复临时分配。作者论证了不应绕过 `max_split_size_mb=20`(绕过后 warmup 阶段仍 OOM)也不应全局调大阈值(依赖模型形状与 TP,只是 case-specific workaround)。

## 修复方案
改 `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` 的 `convert_moe_weights_to_flashinfer_trtllm_block_layout`:1) 预分配最终 w13/w2 block-layout 输出张量;2) 每个 expert 以 uint8 view 处理;3) 内联 block 布局变换(reshape 成块 + permute);4) 用 `torch.index_select(..., out=...)` 把置换后的 expert 直接写入预分配输出;5) 把 `swap_w13_to_w31` 折叠进 W13 permute indices(避免完整物化交换后的 W13):

```python
    def _copy_permuted_expert_to_block_layout(
        out: torch.Tensor,
        expert_uint8: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        expert_blocks = expert_uint8.view(
            expert_uint8.shape[0], out.shape[0], block_k
        ).permute(1, 0, 2)
        torch.index_select(
            expert_blocks,
            1,
            source_indices.to(expert_uint8.device),
            out=out,
        )
```

同时删除 `unquantized.py` 中转换前单独的 `swap_w13_to_w31` 调用(已并入 permute indices)。验证:dummy load 内存 109.61 GiB、加载 3.0s;无 sleep mode 下新转换 2.39s vs 旧 10.2-10.7s(约 4x);新旧转换对 Qwen-like TP4、DeepSeek-like TP4/TP2 形状做随机 BF16 张量 bitwise uint16 对比,`ALL_BITWISE_EQUAL`。

## 影响与回归风险
- 影响窗口:开启 sleep mode + FlashInfer TRTLLM MoE 后端加载大 BF16 MoE 模型时(#41268 与该转换共存即可能触发);正文未给出具体版本号。
- 回归面:FlashInfer TRTLLM BF16 MoE 权重转换路径(所有使用该转换的模型/TP 配置);权重布局语义应不变(位级对比通过),但作者注明验证是在 H20 强制过 capability gate 完成的,SM100/B 系列上的最终运行时 kernel 正确性仍需在 B 系列硬件上测试。

## 关联
- #43951(所修 issue)
- #41268(max_split_size_mb=20 分配器策略来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
