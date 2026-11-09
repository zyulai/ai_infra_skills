# PR #28587: [Hybrid][torch.compile] Refactor mamba2 forward to avoid obscuring linear projections under custom op

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tomeras91 |
| 合并 | 2025-11-19 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | performance, ready, torch.compile |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/28587 |

## 一句话结论

把 mamba2 的 in_proj/out_proj 与 gated RMSNorm 移出 custom op,让 Inductor 能融合这些算子,hybrid 模型吞吐 +10-15%、TTFT -22%。

## 动机

`mamba_mixer2` custom op 之前封装了 Mamba2 block 的整个 forward:输入投影、conv+SSSM token mixing、gated RMSNorm、输出投影。custom op 存在的理由是 token mixing 部分有输入依赖控制流(如 cu_seqlens),无法直接被 piecewise CUDA graph 捕获;但投影与 RMSNorm 并无输入依赖控制流,被包进 custom op 后对 torch.compile 不可见,大量简单融合机会被错过。这一点对量化模型尤其重要:RMSNorm 本可与投影前的量化算子融合,但即使同时启用 `mixer2_gated_rms_norm` 与 `quant` 两个 custom op,也没有 pass 把该模式替换成融合 kernel。

## 优化点

仿照 Attention 的实现方式(QKV/输出投影在 Attention custom op 之外)重构 Mamba2 forward:`in_proj`(含 mup_vector 缩放)与 `out_proj`、gated RMSNorm(`self.norm(ssm_output, gate)`)移到 custom op 外的普通 eager/compile 代码中;`mamba_mixer2` custom op 收窄为只含 token mixing(conv + SSM),签名改为接收 `projected_states` 输出 `ssm_output`。split 卷积权重的 `conv_weights` buffer 与 `split_hidden_states_B_C_fn` 移到 `__init__` 预计算,profile-run 路径复用;输出 preallocation 改为直接 split 调用方传入的 output buffer。6 个 hybrid 模型(bamba、falcon_h1、granitemoehybrid、mamba2、nemotron_h、zamba2)的调用点同步改为 `output = self.mamba(hidden_states)`。nsys 显示 SSM 更新后、out_proj 前的 RMS+quant 算子被融合,kernel 数从 15 降到 2。

## 关键改动

```text
vllm/model_executor/layers/mamba/mamba_mixer2.py: forward 拆分为
  in_proj → mamba_mixer2 custom op(conv+SSM)→ gated RMSNorm → out_proj
vllm/model_executor/models/{bamba,falcon_h1,granitemoehybrid,mamba2,
  nemotron_h,zamba2}.py: 调用点签名同步(去掉 output 出参)
```

关键机制摘录(mamba_mixer2.py,custom op 只剩 token mixing):

```python
torch.ops.vllm.mamba_mixer2(
    projected_states,
    ssm_output,
    self.prefix,
)
# 4. gated MLP
gate = projected_states[..., : self.tped_intermediate_size]
hidden_states = self.norm(ssm_output, gate)
# 5. Final linear projection
output, _ = self.out_proj(hidden_states)
```

## 性能声明

nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8,单卡 H100,ISL/OSL/BS = 1K/1K/256(`vllm bench serve`,1280 prompts,max-concurrency 256,async-scheduling、关闭 prefix caching):吞吐 +10%-15%,TTFT 约降 22%。对照表:Request throughput 5.66→6.36 req/s(+12.37%),Output token throughput 5791.47→6510.57 tok/s(+12.42%),Mean TTFT 2021.85→1576.73 ms(-22.02%),Mean TPOT 42.21→37.77 ms(-10.52%),Mean E2EL 45204.97→40210.64 ms(-11.05%)。nsys(decode forward):RMS+quant 融合后 kernel 数 15→2,耗时约 91.8 μs → 约 14.5 μs。

## 验证与回归风险

质量验证:ibm-granite/granite-4.0-h-tiny 与 nvidia/NVIDIA-Nemotron-Nano-9B-v2 的 GSM8K(lm_eval,5-shot),main 与 PR 结果在误差棒内一致(如 Nemotron flexible-extract 0.7983 vs 0.7961)。风险面:所有 hybrid/Mamba 架构模型(bamba、falcon_h1、granitemoehybrid、mamba2、nemotron_h、zamba2)的 forward 签名变化;custom op 边界内移后,piecewise CUDA graph 捕获范围与 `ssm_output` 中间 tensor 的处理(注释说明 split 留在 custom op 内以避免被 torch.compile 当作 intermediate tensor)是需要关注的正确性点。

## 关联

- 同区域:hybrid/mamba 系列的 compile 与 CUDA graph 支持(同分类 CUDA Graph / Compiler,如 piecewise cudagraph 相关 PR); Attention 的 custom op 边界设计是其参照对象

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
