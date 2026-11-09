# PR #40190: [XPU] Enable fused QK-norm + RoPE for Qwen3-MoE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mehersai07 |
| 合并 | 2026-10-04 |
| 分类 | MoE(major) |
| Label | intel, xpu, run-ci, jit-kernel |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 2 个文件(qwen3_moe.py 模型接线、新增 XPU 测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/40190 |

## 一句话结论

在 XPU(Intel Arc Pro B60)上为 Qwen3-MoE 启用已有的 `fused_inplace_qknorm_rope` AOT kernel,q/k-norm + RoPE 阶段从每层 3 次 launch 降到 1 次,E2E 吞吐 +4.7-7.0%。

## 动机

Qwen3-MoE 本来就有 fused QK-norm + RoPE 路径,但 `Qwen3MoeAttention.__init__` 的 gate 写死 `_is_cuda`,XPU 上只能退回每层 2 次 RMSNorm launch + 1 次独立 RoPE launch,共 3 次;而 `sgl-kernel-xpu` 早已以 AOT 方式提供 `fused_inplace_qknorm_rope` kernel(使用 rotary embedding 的 cos/sin cache),只是没接上。

## 优化点

不新增 kernel,只做模型侧接线:kernel 内的 device dispatch 由 wrapper 负责,模型保持单一代码路径。`apply_qk_norm_rope` 在 CUDA 分支之外新增 XPU 分支:把 packed qkv 按 q/k/v 尺寸 split 后以 strided view 传入 `fused_inplace_qknorm_rope`,kernel 直接读 `rotary_emb.cos_sin_cache` 并接受模型原生的 int64 positions,免去逐层 cast。`use_fused_qk_norm_rope` gate 从 `_is_cuda` 放宽为 `_is_xpu or (_is_cuda and can_use_fused_qk_norm_rope(...))`,保持由既有 `--enable-fused-qk-norm-rope` flag opt-in,与 CUDA 行为对齐。XPU 分支跳过了 CUDA 路径的 head_dim/neox/yarn 兼容性检查(由测试覆盖对应组合)。

## 关键改动

```text
python/sglang/srt/models/qwen3_moe.py: gate 加入 _is_xpu;apply_qk_norm_rope 新增 XPU fused_inplace_qknorm_rope 分支
test/registered/xpu/test_fused_inplace_qknorm_rope_xpu.py: 新增,注册 nightly-xpu-1-gpu 套件
```

关键调用片段:

```python
elif _is_xpu:
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    fused_inplace_qknorm_rope(
        q=q.view(-1, self.num_heads, self.head_dim),
        k=k.view(-1, self.num_kv_heads, self.head_dim),
        q_weight=self.q_norm.weight,
        k_weight=self.k_norm.weight,
        cos_sin_cache=self.rotary_emb.cos_sin_cache,
        positions=positions.view(-1),
        ...
    )
```

## 性能声明

PR 注明条件:Qwen3-30B-A3B TP=4,4x Intel Arc Pro B60。

- q/k-norm + RoPE 阶段(rank 0,40 个 profiled decode steps,launches/device ms):q/k RMSNorm 3,858/79.6ms + RoPE 1,920/23.07ms → 融合 kernel 1,920/40.80ms;stage 合计 5,778/102.67ms → 1,938/41.32ms,launches -66%、device ms -60%。rank 0 总 device 时间 5,672→5,495ms(-3.1%)。
- E2E `bench_serving`(input 1024 / output 256,每臂各跑一次,baseline 抖动约 1%):conc=1 输出吞吐 21.4→22.9 tok/s(+7.0%),TPOT 46.5→43.5ms;conc=8 158.4→168.4 tok/s(+6.3%);conc=32 557.6→583.9 tok/s(+4.7%)。合计 +4.7-7.0% 吞吐,TPOT 降 4.6-6.6%。

## 验证与回归风险

新增 XPU 测试对比融合调用与模型 unfused 路径(`apply_qk_norm` + `RotaryEmbedding`),参数化覆盖 default/YaRN rope_scaling(theta=1e6)、head_dim=128 且 rotary_dim 128/64、neox 与非 neox、num_tokens 1/257;容差 ATOL=8e-2/RTOL=1e-2(融合 kernel 跳过了 norm 与 RoPE 之间的 bf16 rounding,与 diffusion rope 测试同容差)。E2E:GSM8K 1319 例 greedy,TP=4,baseline 94.14 vs fused 94.6,两臂均经 `/get_server_info` 确认 flag 生效。风险:XPU 分支绕过 CUDA 的 `can_use_fused_qk_norm_rope` 检查,head_dim <128 时 `RotaryEmbedding.forward_xpu` 有 bf16-cache fallback 路径未覆盖;CI Base/Extra/AMD 快照均失败(疑与 XPU 平台无关的 CI 环境)。

## 关联

CUDA 侧同功能前序:Qwen3-MoE fused QK-norm+RoPE(CUDA 路径,`fused_qk_norm_rope`);后续 PR #43345 将该 flag 在 XPU 上默认打开。同分类其他 XPU PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
