# PR #45202: [Perf] [AMD] Kimi-K3: prepare the attention interface for mono fusion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @billishyahao |
| 合并 | 2026-11-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | amd, dependencies, run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 2 个文件(pyproject_other.toml、kimi_k3.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/45202 |

## 一句话结论

为 AMD monokernel 后端(K1/K2 融合)准备 Kimi-K3 attention 接口,配合 sgl-amd-mono 使 decode ITL P50 降低 25.2%、decode steps/s 提升 32.8%。

## 动机

AMD monokernel 后端(sgl-amd-mono,经 `--monokernel-backend amd` 选择,通过 `sglang.srt.plugins` 入口加载)将 Kimi-K3 的 decode/DSpark verify 每层压缩为两个 persistent launch:K1(KDA 前置 launch)覆盖 69 个 KDA 层的 attention 侧计算,K2(layer launch)覆盖全部 92 个 MoE 层的 o_proj、attention-TP all-reduce 与 MoE。插件内部能替换内核,但从外部无法接入模型 forward,因此需要 SGLang 侧打开接口:这是插件唯一无法自行完成的改动。该工作遵循 sgl mono kernel RFC(#43949)。

## 优化点

1. **KDA in-projection gate 放宽**:`_may_fuse_kda_inproj()` 在 monokernel backend 被选中时也允许 qkvbfg 融合(`do_fuse_qkvbfg or get_exec().kernel.enable_amd_monokernel`)。不选后端时谓词与原来完全等价,刻意保持 identity;权重安全检查(`_is_unquantized_mergeable` 要求三权重同 dtype 且为 bf16/fp16,per-channel FP8/MXFP4 拒绝)不变。
2. **新增 `forward_pre_o_proj()`**:在 `KimiK3DeltaAttention`(plain extract,`forward()` 改为调用新方法并保留 o_proj 尾部)与 `KimiK3MLAAttention`(经实例级 gated o_proj wrapper 拦截,返回 output gate 之后、o_proj 之前的 TP-local attention 输出)上实现,满足 `o_proj(forward_pre_o_proj(x)) == forward(x)`,让 K2 接管 o_proj 与其携带的 attention-TP reduction。
3. **MLA 拆分的四道护栏**:`use_output_gate` 必须开启;方法不可重入;一次 forward 内二次到达 o_proj 报错而非覆盖;量化 o_proj(MXFP4/block-scale fp8 输入)直接 raise。
4. **依赖 pin 更新**:pyproject_other.toml 将 sgl-amd-mono 从 b10f3b7 更新到 e52803f。

## 关键改动

```text
python/sglang/srt/models/kimi_k3.py: _may_fuse_kda_inproj() 放宽 monokernel
  条件;KimiK3DeltaAttention/KimiK3MLAAttention 新增 forward_pre_o_proj();
  MLA 侧 wrapper 增加 _return_pre_o_proj/_pre_o_proj_out 拦截与护栏
python/pyproject_other.toml: sgl-amd-mono 依赖 pin 升级(配合 mono patch
  sgl-amd-mono#13)
```

关键片段(diff 摘录):

```python
if self._return_pre_o_proj:
    if isinstance(x, tuple):
        raise RuntimeError(
            f"forward_pre_o_proj: layer {self.layer_id} has a "
            "quantized o_proj input, so there is no gated bf16 "
            "output to hand back"
        )
    self._pre_o_proj_out = x
    return x, None
return _orig_o_proj_forward(x, *args, **kwargs)
```

## 性能声明

PR 正文,未独立核实。硬件 8 × MI355X,TP8/DCP1,FP8 KV cache,DSpark block size 7,`/data-models/Kimi-K3`:

- **精度**:GSM8K 200 题、`--parallel 2`:无 monokernel 0.990,`--monokernel-backend amd` 0.985。
- **短 prompt decode steps/s**:41.63 → 55.26(**+32.8%**),区间不重叠;收益限于 8-row verify steps,更宽的 step 与全部 prefill 走标准路径。
- **定长长上下文 ITL**:64k/128k/256k ISL 下 P50 ITL 6.17→4.62ms(−25.2%)、6.67→5.29ms(−20.6%)、7.70→6.19ms(−19.6%);P90 同步移动,TTFT 不变;按 speedup 计为 +33.7%/+25.9%/+24.3%。
- **AgentX agentic-coding replay**(每臂 1 小时):TPOT mean 6.28→4.94ms(−21.3%),interactivity P50 160.4→203.7 tok/s(+27.0%),output 97.5→112.0 tok/s(+14.9%),完成请求 205→222。
- 作者强调:每步 admitted 两个请求时(16 行,超出 spec 支持宽度)收益被稀释(单请求 step 快 9.8%,双请求 step 仅 2.2%);要拿满收益需保持 verify step 在支持宽度内。

## 验证与回归风险

- 精度与速度均为作者手测(8×MI355X),CI 无对应 runner;Checklist 中 unit tests / documentation / benchmark 项未勾选。CI 状态(pack 记录):Base 与 AMD ROCm 10 进行中,Extra 失败。
- MLA `forward_pre_o_proj` 路径在树内无调用者(只有 monokernel 插件调用),出错表现为数值错误而非异常;四道护栏(尤其量化 o_proj raise)是主要防线。
- 融合 GEMM 与 split kernel 是不同内核,两路径仅到 bf16 rounding 一致、非 bitwise;gate 放宽会影响所有 ROCm Kimi-K3 用户的 decode 输出(一位 bf16 rounding)。
- 不指定 mono 参数时标准路径 `forward()` 输出不变。

## 关联

sgl-amd-mono#13(配套 mono patch)、sgl-project/sglang#43949(sgl mono kernel RFC)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
