# PR #41794: [Perf] [AMD] kimi-k3: Run the fused KDA decode with a BF16 state and Quark weights

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuychang |
| 合并 | 2026-11-11 |
| 分类 | Quantization(tier: major) |
| Label | documentation, quant, amd, run-ci, jit-kernel |
| Diff | +610/-79,9 files(pack diffstat 缺失,按 pack 内 diff 文本计数;pack 注明另丢弃 3 个文件、末文件 diff 中途截断) |
| 链接 | https://github.com/sgl-project/sglang/pull/41794 |

## 一句话结论

修复 Kimi-K3 融合 KDA decode 对 Quark checkpoint + BF16 state 不点亮的三个 blocker,MI355X TPOT 降 2.2-2.9%。

## 动机

ROCm 上 Kimi-K3 已有 fused KDA decode kernel(`f_b` 投影 + 卷积 + recurrent update + gated RMSNorm 单个 gfx950 launch),但 `aiter` backend 下对部署的 Quark checkpoint(`amd/Kimi-K3-Quark-MXFP4-AttnFP8`)从不生效,decode 每步跑 unfused 链并多付 launch 开销。三个 blocker:`SGLANG_K3_FLYDSL_SOURCE=aiter` 导入的模块没有任何 AITER build 提供(kernel 实际发布在 ROCm/aiter#6026 的 `aiter.ops.flydsl.linear_attention_kernels`);adapter 只接受 FP32 recurrent state,而推荐的 `--mamba-ssm-dtype bfloat16` pool 是 BF16;checkpoint 的 `f_b` 是 per-channel FP8,kernel 要 dense BF16。另有两个 env var 用裸 `os.environ` 读取,不在 `Envs` 体系内。

## 优化点

从 AITER 新模块绑定 `flydsl_kda_decode_with_f_b`,AITER 缺 #6026 或缺 `state_dtypes` 标签时一次性告警并 fail-closed 回退;arm 规则要求 kernel 声明的 `state_dtypes` 覆盖 pool 的 state dtype,`auto` 来源在 FP32-only vendored kernel 不收 BF16 时继续尝试 AITER 的 kernel。Quark per-channel FP8 `f_b` 只为 fused kernel 一次性 dequant 出 BF16 副本,FP8 层继续服务 unfused 路径;`covered()` 镜像 AITER 的 cache-slot/stride 规则,不支持布局回退而非 raise;graph capture 前只 warmup pool 的 state dtype,避免 capture 期 JIT 编译。两个 env var 改名进 `Envs`(旧名 alias + DeprecationWarning)。ReplaySSM ring 开启时 decode 不再 defer `f_b`(ring 会把 raw `f_a` 直接送 recurrence)。仅 ROCm,CUDA 不变。

## 关键改动

```text
python/sglang/kernels/ops/attention/kda_fused_decode_aiter_hip.py: _Kernel/_bind(state_dtypes 匹配)、covered/run/warmup 按 state_dtype 收敛
python/sglang/srt/models/kimi_k3.py: _fused_decode_f_b_weight(Quark FP8→BF16)与 _prepare_fused_decode 门控、ReplaySSM 分支
python/sglang/srt/models/kimi_k3_rocm_quant.py: k3_fused_decode_f_b_weight(QuarkW8A8Fp8 per-channel 检测与 dequant)
python/sglang/srt/environ.py: SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND / SGLANG_USE_HIP_K3_FLYDSL_SOURCE(EnvStrWithAlias)
```

```python
for candidate in ("sglang", "aiter") if source == "auto" else (source,):
    try:
        kernel = _load_kernel(candidate)
    except ImportError as error:
        errors.append(f"{candidate}: {error}")
        continue
    if state_dtype in kernel.state_dtypes:
        return kernel
    loaded.append(kernel)
```

## 性能声明

8×MI355X TP8,random 8192 in/1024 out,radix cache off,FP8 KV,AITER attention,`--mamba-ssm-dtype bfloat16`;base `main`@dd9ea06c00,PR 侧加两个 env;AITER be461934f 双侧。TPOT:c2 18.22→17.70 ms(-2.9%)、c4 -2.9%、c8 -2.3%、c16 -2.2%;c32 -0.8%、c64 -0.9%(约为噪声两倍,不主张)。TTFT 变化在 ±0.8% 内。GSM8K 1319 题:main 97.27% vs PR 97.12%(0.15 点=2 题,标准误约 ±0.45 点,视为噪声)。

## 验证与回归风险

`test_kimi_k3_kda_decode.py` 扩充:AITER kernel 绑定、BF16 state 对照 reference(relative RMSE <1e-3,store 为 round-to-nearest-even)、cache pool 布局覆盖、warmup dtype、旧名 alias;在带 #6026 的 AITER 上 25 passed,pinned image 15 passed/10 skipped。`test_kda_fused_decode_gfx950.py` 新增 BF16 pool 不 arm 用例;`test_kimi_k3_rocm_quant.py` CPU 11 passed;新增 CPU 测试验证 ReplaySSM ring 下 decode 不 defer `f_b`。风险:依赖 AITER 版本(需 ROCm/aiter#6026 与 `state_dtypes` 标签);`f_b` 已被 ShardedStateLoader/RemoteModelLoader post-process 时不 arm;性能数字来自较早修订(后续改动只碰加载检查/告警/warmup/ReplaySSM 门控)。

## 关联

素材未提及其他 sglang PR 编号(依赖上游 ROCm/aiter#6026)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
