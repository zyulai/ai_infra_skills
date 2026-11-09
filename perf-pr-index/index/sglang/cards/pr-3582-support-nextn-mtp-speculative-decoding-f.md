# PR #3582: Support NextN (MTP) speculative decoding for DeepSeek-V3/R1

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2025-02-14 |
| 分类 | Speculative Decoding(tier: major) |
| Label | high priority |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/3582 |

## 一句话结论

为 DeepSeek-V3/R1 接入官方 NextN(MTP)层做投机解码,bs=1 输出吞吐从 38.19 提升到 77.73 token/s。

## 动机

DeepSeek-V3/R1 官方 checkpoint 自带一个 MTP 模块(NextN layer),此前 sglang 的投机解码只支持 EAGLE,需要单独的 draft 模型,没有利用这个现成的单层 draft。作者基于 EAGLE 2 on Triton backend(#3466)实现,发现 MTP 模块同样能像 EAGLE 一样自回归预测,accept rate 很高:draft 2 tokens 平均接受长度约 1.9;2 steps + topk 4 draft 4 tokens 可达 2.5~3。预测更多 token 会增加开销、在大 batch 下影响吞吐,延迟与吞吐的最优权衡需进一步实验。

## 优化点

新增 draft 模型 `DeepseekV3ForCausalLMNextN`:只取 checkpoint 的 1 个 NextN 层,结构为 enorm/hnorm 归一化目标 hidden_states 与 draft hidden_states 后由 `eh_proj` 拼接投影,进入一个 `DeepseekV2DecoderLayer(is_nextn=True)`,shared_head 复用目标 LM head 语义(支持 dp-attention 时用 ReplicatedLinear)。投机框架层面把 `SpeculativeAlgorithm` 增加 `NEXTN` 并复用 EAGLE worker,与 EAGLE 的区别是 draft 自带 embed_tokens,不做 embed/head 共享;`server_args` 接受 `--speculative-algorithm NEXTN`,与 EAGLE 一样自动禁用 radix cache、chunked prefill 与 overlap schedule。配套提供 `scripts/export_deepseek_nextn.py` 把 NextN 层权重导出为独立 draft checkpoint。MLA forward 中 normal computation 分支排除 target-verify 与 draft-extend 模式,避免投机路径误入非权重吸收分支。

## 关键改动

```text
python/sglang/srt/models/deepseek_nextn.py: DeepseekV3ForCausalLMNextN draft 模型与权重加载
python/sglang/srt/speculative/spec_info.py: SpeculativeAlgorithm 增加 NEXTN(基于 EAGLE)
python/sglang/srt/server_args.py: --speculative-algorithm 接受 NEXTN,附同样的调度降级
python/sglang/srt/models/deepseek_v2.py: decoder 层 is_nextn 分支 + forward_normal 条件修正
scripts/export_deepseek_nextn.py: NextN 层权重导出脚本
```

draft 前向的核心拼接(摘自 `DeepseekModelNextN.forward`):

```python
hidden_states = self.eh_proj(
    torch.cat(
        (
            self.enorm(hidden_states),
            self.hnorm(forward_batch.spec_info.hidden_states),
        ),
        dim=-1,
    )
)
```

## 性能声明

`bench_one_batch_server`,batch size 1,input 256 / output 256,DeepSeek-V3 `--tp 8`(硬件 PR 未注明):

- baseline(main 分支):latency 6.70 s,output throughput 38.19 token/s,(input+output) 76.39 token/s。
- NEXTN 投机解码(steps 2 / topk 4 / draft 4):latency 3.77 s,output 67.93 token/s,(input+output) 135.87 token/s。
- NEXTN + Torch.compile(`--enable-torch-compile --torch-compile-max-bs 1 --disable-radix`):latency 3.29 s,output **77.73 token/s**,(input+output) 155.45 token/s(相对 baseline 输出吞吐约 2.04x;正文概括为 "1.76x speed up with CUDA Graph and Torch.compile compatibility"、bs=1 输出吞吐 77 token/s)。

接受长度:MTP 模块 draft 2 tokens 约 1.9;steps 2 / topk 4 / draft 4 为 2.5~3。

## 验证与回归风险

PR 正文未列出专门的单测或 CI 结果(素材中无测试章节,pack 亦无 CI 状态)。风险:启用 NEXTN 会与 EAGLE 一样强制禁用 radix cache、chunked prefill 和 overlap schedule;fp8 checkpoint 下因 `bmm_fp8` 只支持 per-tensor scale,`kv_b_proj` 权重会被 requant,代码注释自认可能影响 fp8 精度(HIP 走 e4m3fnuz 归一化);`forward_normal` 的模式排除改动触及 DeepSeek-V2/V3 的 extend/verify 路径;实现断言只支持 `num_nextn_predict_layers == 1`。

## 关联

前序:基于 EAGLE 2 on Triton backend 的 #3466。同区域:Speculative Decoding 分类的后续 MTP/EAGLE 优化(如 DeepSeek 系列后续 spec 支持均沿用 NEXTN/EAGLE 框架)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
