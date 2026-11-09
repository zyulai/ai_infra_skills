# PR #40543: fix(cuda-graph): remove unnecessary CP batch alignment

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AMD-yanfeiwang |
| 合并 | 2026-10-03 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/40543 |

## 一句话结论

去掉 decode CUDA graph 捕获时不必要的 attn_cp_size batch 对齐,组合栈下 DeepSeek-V4 高并发 TTFT 降 21-27%。

## 动机

Decode CUDA graph 捕获此前把 batch size 对齐到 `attn_cp_size`,但 prefill CP(context parallel)对 `DECODE` 和 `TARGET_VERIFY` 模式本就不生效;且 DCP 与 `--enable-cp-decode-attn-tp` 是按 KV/head 维度切分而非按 token 行切分,不需要 CP 宽度的 batch 对齐。这个多余的对齐因子让 decode graph 的捕获/回放 batch 档位变稀,高并发下浪费调度与显存档位。

## 优化点

`get_cuda_graph_batch_size_alignment()` 中,把 `alignment *= attn_cp_size` 的 CP 因子在 HIP 上移除(`not is_hip()` 才保留原判断),同时保留既有的 TBO 与 gathered-buffer attention-TP 对齐因子。代码以 TODO 注明"NVIDIA 上未验证,待 CUDA 侧验证后删除该 gate"——即当前 CUDA 平台行为不变,仅 ROCm/HIP 侧去掉该因子。diff 为单文件单点修改。

## 关键改动

```text
python/sglang/srt/utils/common.py: get_cuda_graph_batch_size_alignment 移除 HIP 上的 attn_cp_size 对齐因子
```

```python
def get_cuda_graph_batch_size_alignment() -> int:
    ...
    # TODO: unverified on NVIDIA; drop the gate once validated on CUDA.
    if not is_hip() and alignment % get_parallel().attn_cp_size != 0:
        alignment *= get_parallel().attn_cp_size
    return alignment
```

## 性能声明

性能结果依赖组合栈 #34200(HIP CP-v2 支持)+ #39950(AITER B-preshuffled 切片修正)+ #40543(本 PR),非本 PR 单独贡献:

- 条件:单节点 DeepSeek-V4-Pro-0813,8x MI355X,TP8,DSpark gamma 6,static verify,每点 3600 s;`PCP8 + DTP` 即 `--enable-prefill-cp --attn-cp-size 8 --enable-cp-decode-attn-tp`;C4/C16 用 GPU KV,C32/C48/C64 用 HiCache。
- 对比 TP8 → PCP8+DTP:C32 TTFT avg 2.176 → 1.711 s(-21.4%)、p90 -26.0%,TPOT avg -8.4%、p90 -15.1%,per-GPU 总吞吐 +4.4%;C48 TTFT avg -24.6%、p90 -27.5%,TPOT avg -13.9%、p90 -16.7%,吞吐 +8.6%;C16 吞吐基本持平(-0.4%);C64 无 TP8 参照,达到 28,413 total tok/s/GPU。
- 汇总:在 C32/C48,平均/P90 TTFT 降低 21-27%,平均/P90 TPOT 降低 8-17%,per-GPU 吞吐提升 4-9%。五个 PCP8+DTP 点零请求错误。

## 验证与回归风险

diff 本身无新增测试;依赖组合栈的端到端 AgentX 测量(零 profiled 请求错误)与 CI。风险:改动显式保留 CUDA 侧旧行为(仅 HIP 生效),NVIDIA 上是否同样可以去掉 CP 对齐尚未验证(TODO);CUDA graph batch 对齐变化会影响捕获显存与可回放 batch 档位,若未来 CUDA 侧 CP decode 语义变化需重新评估该 gate。

## 关联

性能依赖组合栈:#34200(HIP CP-v2)、#39950(AITER B-preshuffled slicing);三者的合并效果才体现为上述端到端数字。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
