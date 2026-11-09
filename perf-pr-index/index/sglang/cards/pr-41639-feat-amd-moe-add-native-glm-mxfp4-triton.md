# PR #41639: [Feat] [AMD] moe: Add native GLM MXFP4 Triton Gluon backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiejingzhangamd |
| 合并 | 2026-11-02 |
| 分类 | MoE(major) |
| Label | quant, amd, deepseek, run-ci, jit-kernel, run-ci-extra, run-ci-nv-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/41639 |

## 一句话结论

为 GLM-5.2/5.3 Quark MXFP4 在 MI355X 上新增整层 Triton Gluon MoE 后端,rank-local MoE 层最高 6.78x,GSM8K 0.9437。

## 动机

GLM-5.2/5.3 在 MI355X 上 decode 时 MoE 层占用每步的显著部分。现有 runner 边界拿到的是已物化的 top-k ID 与权重,模型专用 kernel 无法跨 router 投影、top-k、路由排序、专家调度、激活量化、routed/shared expert 做整体优化。需要一个显式、严格、fail-closed 的整层后端,把"选路"直接变成"专家调度",消除通用的 `topk_ids → moe_sorting → expert GEMM` 交接。

## 优化点

新增 `gluon` 作为 `MoeRunnerBackend` 选项(`--moe-runner-backend gluon --speculative-moe-runner-backend gluon` 显式启用,`auto` 永不选择,无 AITER 回退)。核心机制:router 投影与输入量化作为独立 CTA 在同一个 launch grid 里协同调度(省一次 host launch);固定 256 专家 sigmoid top-8 的选择核直接以 atomic ticket 把路由写入专家 bucket 并产出 job descriptor,供上/下投影消费;按 active-token M 分三个 kernel 家族——tiny M(split-K CDNA4 MFMA router + 选择/专家 CTA 共享 launch,expert CTA 重算所需 route rank)、medium M(选择直接发 expert ticket 与排序路由)、large M(chunk 直方图 + prefix/scatter + bucketed grouped projection,32K family 拆分 buffer 保证 gfx950 每个地址在 4-GiB 窗口内);专家 GEMM 用 CDNA4 `mfma_scaled` 直接吃 e2m1 MXFP4 与 scale;TP/EP-aware shared expert 处理并复用 SGLang 既有 post-expert collective。绑定阶段严格校验 model_type、256 专家、sigmoid top-8、quant ABI、gfx950、TP/EP 形状等,不支持组合直接报错。

## 关键改动

```text
python/sglang/kernels/ops/moe/gluon_moe_gfx950/: 新增目录;按 (TP, M) 形状家族特化的 fused MoE gluon kernel(如 fused_moe_tp4_m128_4192.py)
新增 GluonMoeBackend 整层接口:模型绑定、eager 权重准备、按 TP/EP/target/NextN/M 调度、输出 ABI 校验
test/registered/unit/models/test_gluon_moe_backend.py 等: 72 个单测,枚举每个 M 断言审计范围外严格拒绝
```

```python
@gluon.jit
def _select_routes(L, Bias, Ids, Counts, ...):
    prob = 1.0 / (1.0 + gl.exp(-gl.load(L + m * 256 + e).to(gl.float32)))
    score = prob + gl.load(Bias + e).to(gl.float32)
    ...
    for j in gl.static_range(8):        # 固定 256 专家 sigmoid top-8
        maximum = gl.max(score, 0)
        ...
    owned = (selected_id >= expert_start) & (selected_id < expert_start + LOCAL_EXPERTS)
    ticket = gl.atomic_add(Counts + ..., 1, (e < 8) & owned, sem='relaxed')
    record = (selected_id * TICKET_STRIDE + ticket).to(gl.uint64)
    record |= weight.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
```

## 性能声明

MI355X 单卡(TP8/EP1 rank),AITER 基线含 SGLang 小 M 排序 patch,20 warmups/200 次 HIP-event 中位数,交替 arm 顺序:

- 完整 rank-local MoE 层:M=1 574.97 → 99.42 µs(5.78x)、M=8 6.78x、M=16 5.77x、M=32 4.43x、M=128 3.32x;后续 M=1 融合再降 18.71%(100.80 → 81.94 µs),MTP M=6 降 29.30%。
- Large-M 完整层:M=1024 2.52x、M=8192 2.46x、M=16384 2.63x、M=32768 2.60x(延迟降约 60%)。
- rocprofiler-sdk(HEAD `317b7156d`,每 arm 100 次):M=1 GPU active 60.82 → 13.47 µs,launches 15 → 4;M=32 119.22 → 64.30 µs,15 → 5 launches。
- AgentX 端到端(TP8/EP1,GLM-5.2-MXFP4,MTP 3/4/1,32K chunk):C1 3/4/1 P90 ITV +22.98%,C2 3/4/1 +29.63%,C1 5/6/1 +14.62%,C2 5/6/1 -6.28%(vs 09/24 baseline)。

## 验证与回归风险

正确性:新旧 Gluon 调度 M=1..8 bitwise 一致;router/top-k 隔离 top-8 ID 全匹配、权重差 ≤5.96e-8;EP2/4/8 vs rank-masked AITER 相对 L2 0.0038-0.0046;large-M TP8/EP1 相对 L2 1.81-2.33%(MXFP4 累加顺序差异)。端到端:GLM-5.2-MXFP4 + EAGLE/NextN、FP8 KV、Triton DSA、CUDA graph 下 AITER vs Gluon 8/8 输出文本与 token ID 完全一致(TP4/EP4、TP4/EP1 融合与非融合);GSM8K 全量 1314 题得 0.9437(1,240/1,314,5-shot,temperature 0),高于既有 0.92 阈值;AgentX canary TP8/EP2/EP4/EP8 零跳过调度、零错误、零回退。风险:仅审计了 GLM `glm_moe_dsa` Quark W4A4 MXFP4 target + BF16 NextN 的 6 种 TP/EP 拓扑;其他模型/量化/拓扑被严格拒绝而非回退;M=16802 初次暴露覆盖边界后扩展到 32K;独立 top-k 与 AITER 约持平,收益来自消交接而非 kernel 本身。

## 关联

对照基线为现有 AITER MoE runner(以及 Triton/FlashInfer/MegaMoE/DeepEP 等,本 PR 不改变其 `auto` 行为)。pack 未列出其他前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
