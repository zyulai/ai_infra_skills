# PR #46346: [GDN] Improve kkt kernel of CuteDSL prefill backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-06-30 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 4+ 个文件(kernel_kkt_inv_uw.py、kernel_h.py、gdn_chunk_cutedsl/__init__.py、cute_utils),另有 2 个文件未含于 pack |
| 链接 | https://github.com/vllm-project/vllm/pull/46346 |

## 一句话结论

优化 GDN CuteDSL prefill 后端 KKT kernel 的 inverse 流水线:beta/gate 前移到专用 prep warps、Ab/Abg 融合进 inverse 步骤,TP1 提升约 10%。

## 动机

对 #43273 引入的 KKT kernel 做插桩分析后发现:tcgen05 MMA 快到 `W MMA + U MMA + next KKT MMA` 的耗时小于 INV warps 加载并准备 beta/gate 的耗时——"Prepare gate/beta" 与 "Store Ab/Abg to tmem" 反而成为瓶颈。此外,`inverse(I+A)` 之后计算 `Ab = inv(I+A)*beta`、`Abg = inv(I+A)*beta*gate` 的 Ab/Abg 准备占 inverse 逻辑本身时间的 20-25%,存在一次多余的 smem roundtrip,并且对上三角零 tile 做了无谓的计算与存储。在 Blackwell 上乘法本身也可能成为瓶颈。

## 优化点

两方面核心改动:(1) beta/gate 准备移到新增的专用 prep warps(warp 配置从 2+4+4 改为 4+4+4),prep warps 做 warp-local prefix scan 并与 INV warps 重叠、领先一个 stage,INV warps 需要时 beta/gate 已在 smem 中,无需等待;(2) Ab/Abg 准备与 inverse 步骤融合——对角 16x16 tile 的 Newton-Schulz inverse 与非对角 tile 的 block forward-substitution 在结果仍在寄存器时直接算出 Ab/Abg 写 tmem,消除独立阶段。上三角零 tile 不再计算/存储:原 4x4 个 16x16 tile,现在只需 1+2+3+4 个。附带清理:elect_sync 移入 tcgen05 helper、去掉 sA/sAi 的 swizzled layout、TmaInfo 对象直接传递、用 cutlass.range(vectorize=True) 应用 f32x2 数学、用内建 PTX(neg.bf16x2/sub.bf16x2)替代 adhoc BF16 位操作。作者指出 tcgen05 仍大部分空闲,后续可做 FlashAttention 式 pingpong 调度。

## 关键改动

```text
vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_kkt_inv_uw.py: KKT kernel 重构——prep warps 前移 beta/gate、Ab/Abg 融合进 inverse、warp 布局 4+4+4
vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_h.py: TMA 参数改为 TmaInfo 传递、elect_one 下沉、f32x2 向量化
vllm/cute_utils/_tcgen05.py / __init__.py: tcgen05 helper 内置 elect_one;新增 bf16x2 neg/sub 内建 PTX 封装
```

关键代码摘录(新 inverse 流水线结构,节选,<=15 行):

```text
# current
1. compute inverse for diagonal 16x16 tiles (Newton-Schulz)
2. compute inverse for off-diagonal tiles (block forward-substitution)
3. compute Ab/Abg, and store to tmem
# new
1.  diagonal inverse (Newton-Schulz)
1b. while result still in registers, compute Ab/Abg -> tmem
2.  off-diagonal inverse (block forward-substitution)
2b. while result still in registers, compute Ab/Abg -> tmem
```

## 性能声明

微基准(GB200,FlashInfer 为 baseline,prefill kernel 计时):
- Qwen3.6-27B TP1:1x8192 0.422 → 0.352 ms(1.20x);8@8192 mixed 0.410 → 0.336 ms(1.22x);1x16384 0.714 → 0.662 ms(1.08x)。
- Qwen3.5-397B-A17B TP1:1x8192 0.426 → 0.425 ms(1.00x,旧版 0.92x);8@8192 mixed 0.463 → 0.425 ms(1.09x)。
- Qwen3.5-397B-A17B TP4:1x8192 1.75x(不变);8x1024 0.175 → 0.123 ms(1.43x);多数 TP4 case 基本不变。
- 总结:TP1 约 10% 改进(仅 KKT kernel,另有 H 与 O kernel);TP4 性能基本不变。8x2048 大 batch case 仍慢于 FlashInfer(0.77x/0.69x)。

E2E:Qwen3.6-27B,8k-1 prefill only,GB200,concurrency 64:TPGS 25938 → 25964 tok/s,TTFT 20.00s → 19.98s,作者认为属噪声水平。AIME25 exact match 0.9167。

## 验证与回归风险

- E2E 用 Qwen3.6-27B 的 AIME25 与 prefill benchmark 验证正确性与性能;pack 未列 pytest 单测文件(diff 后部被截断,可能有测试改动)。
- 影响路径:GDN(gated delta rule)CuteDSL chunked prefill 后端的 kkt/inv/uw 与 h kernel,主要惠及 Qwen3.5/3.6 等 GDN 模型单卡(TP1)prefill;TP4 路径不变。smem 布局改动(去 swizzle、tmem 断言从 `(BT//2)*num_stages` 收紧为 `BT//2`)是与硬件分配约束耦合的改动,新 warp 布局对 grid/launch(min_blocks_per_mp=1)有依赖。

## 关联

- 前序:PR #43273(引入 CuteDSL prefill backend,本 PR 为其 follow-up)。
- 同区域:索引库已有 pr-31167、pr-31722、pr-40711、pr-43534、pr-59735 等 GDN kernel 优化卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
