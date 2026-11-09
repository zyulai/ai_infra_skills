# PR #36607: [AMD] Enable GLM-5.3-Flash on gfx942 and gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @andyluo7 |
| 合并 | 2026-08-28 |
| 分类 | MoE (tier: major) |
| Label | amd, jit-kernel |
| Diff | 素材未提供 diffstat;diff 含 6 个文件,另有 22 个文件因篇幅被省略 |
| 链接 | https://github.com/sgl-project/sglang/pull/36607 |

## 一句话结论

让 GLM-5.3-Flash 在 ROCm gfx942/gfx950 正确高效运行:MI355X c32 总吞吐达 B200 的 101%,mHC 集成带来 4.3-5.4x。

## 动机

#36507 加入上游 GLM-5.3-Flash 模型实现,但该实现尚不能在 AMD ROCm gfx942(MI300X)与 gfx950(MI355X)上正确高效运行。本 stacked PR 补齐 ROCm 使能,且不向共享引擎路径添加模型名检查;同时为 gfx950 打通 AITER mHC 等 ROCm 专属性能路径。

## 优化点

ROCm 使能侧:新增 ROCm KPool/AITER paged-MQA 路径及所需 preshuffled cache layout(KPool 缓存读写 kernel 通过 `PRESHUFFLE_TILE` 参数支持 AITER 预重排布局);非 gfx95 的 ROCm 走可移植的 unfused Torch DSA top-k,gfx950 保留 fused SGL-kernel 路径(JIT fused 路径支持 128–512 的 group-topk,group_topk=2048 的 AOT 专用路径保留 HIP unfused fallback);DSA index transform 支持 KPool tail 宽度(新增 tiled decode kernel,保留 2,048 列单程序快路径);处理 GLM 的零宽 RoPE tail(TileLang kernel 对 `D_tail > 0` 条件化);在 ROCm 上屏蔽不可用的 DeepGEMM/TileLang mHC 路径;支持 mixed Quark MXFP4 与显式 block-quantized FP8 checkpoint 加载。gfx950 性能侧:AITER gfx95 路径激活时启用 GLM shared-expert fusion;HIP 上按 `SGLANG_USE_AITER=1` 将 mHC pre/post 派发到 AITER(导入/执行失败自动回退并禁用快路径);通过既有 AITER boundary 实现把 attention mHC post 与下一层 FFN mHC pre 融合(`hc_attn_to_mlp`)。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py: ROCm KPool indexer,AITER paged-MQA logits 与 HIP fp8_mqa_logits 派发
python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py: KPool 缓存 preshuffle 偏移计算与 HIP unfused top-k fallback
python/sglang/srt/layers/attention/dsa/transform_index.py: 新增 tiled decode transform kernel 支持任意 topk 宽度
python/sglang/kernels/ops/layernorm/mhc.py: AITER mHC pre/post 派发(HIP+gfx95+SGLANG_USE_AITER,失败自动回退)
python/sglang/srt/layers/communicator_mhc.py: hc_attn_to_mlp 融合 attention mHC post 与 FFN mHC pre
python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py: 零宽 RoPE tail 条件化(D_tail > 0)
```

```python
def _use_aiter_mhc() -> bool:
    return (
        not _AITER_MHC_RUNTIME_DISABLED
        and is_hip()
        and is_gfx95_supported()
        and get_bool_env_var("SGLANG_USE_AITER")
    )
```

## 性能声明

MI355X 性能验证:8x MI355X(gfx950),PR head `9d208769`,模型 `zai-org/GLM-5.3-Flash`,TP8/EP1、FP8 E4M3 KV、AITER MoE、TileLang DSA,full decode graphs(bs 1/32)、prefill graphs 关闭,random dataset(seed 42,1,024 入 / 1,024 出,ignore EOS,radix cache 关闭,无限请求率):

- concurrency 1:总吞吐 **287.85 tok/s**,median TPOT 6.8675 ms,interactivity 145.61 tok/s/user。
- concurrency 32:总吞吐 **5,731.36 tok/s**,TPOT 10.6696 ms,interactivity 93.72 tok/s/user。

同 workload B200 对照(TP8/EP8、DeepGEMM MoE、TRT-LLM DSA):总吞吐 c1 达 B200 的 90.18%、c32 达 **101.07%**;interactivity c1 89.11%、c32 98.35%,均超过 80% B200 的目标。mHC 集成前后(同配置):c1 53.08 → 287.85 tok/s(**5.42x**),c32 1,326.39 → 5,731.36 tok/s(**4.32x**)。kernel 级:1K→2K decode 区间 fused k-pool 操作约 8.8-9.6 µs,旧 ROCm 路径 159-164 µs,即 **16.9-18.2x** 加速。精度(更早 engine head `9e692c92` 验证):MI300X/gfx942 GSM8K 全量 1,284/1,319 = 97.35%;MI355X/gfx950 1,288/1,319 = 97.65%,均 0 错误/空生成;当前性能 head 未重跑全量 GSM8K。

## 验证与回归风险

AITER mHC 与 fused-boundary 测试 MI355X 上 18 passed + 5 subtests;GLM/shared-expert/k-pool focused selection 53 passed + 4 subtests + 1 deselected;全部 pre-commit hooks 与 GitHub lint 通过;CI 等 maintainer 打 label(记录时均为 pending)。风险:mHC/AITER 路径由 HIP + gfx95 + `SGLANG_USE_AITER` 门控,CUDA 派发与行为不变;fused k-pool select-all 快路径与 `torch.topk` 的选中集排序可能不同,DSA attention 对选中集置换不变,但非 bitwise 等价;依赖 ROCm/aiter#5060 的 BF16 调优表,合并后需推进 `AITER_COMMIT_DEFAULT`。

## 关联

前序:#36507(GLM-5.3-Flash 模型实现)。配套:#36732(cookbook/性能配方)、ROCm/aiter#5060(BF16 调优表)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
