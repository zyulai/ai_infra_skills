# PR #18856: [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-02-16 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,仅改 rotary_embedding.py 一个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/18856 |

## 一句话结论

为 Ernie4.5-VL 的 3D MRoPE 写融合 Triton kernel,rotary embedding 耗时 670µs 降到 123µs(5.4x),实测负载 E2E 提升约 17%。

## 动机

Ernie4.5-VL 的 MRoPE 在推理耗时中占大头:实现由多个小 PyTorch op(index_select/chunk/stack/reshape/cat)组成,产生大量 GPU bubble。Ernie 采用特殊的频率重排 [h, w, h, w, ..., t, t, t](源自 THW 位置),与现成 fused rotary kernel 假设的 [t, h, w, t, h, w, ...] 布局不同,因此无法直接复用已有融合 kernel,只能保留多 op 的 eager 路径。

## 优化点

新增 Triton kernel `_triton_ernie45_rope_qk_fused`:每 token 一个 program,一次性对 Q 和 K 原地(in-place)施加 3D MRoPE。关键机制是不在 Python 侧构造 (cos, sin) 张量,而是在 kernel 内按 rotary pair 索引直接选择轴向位置——`ridx < section_hw` 时空间段按偶/奇交替取 hpos/wpos,其余取 tpos——再从 cos_sin_cache 直接 gather,消除了中间张量分配、减少 kernel launch 次数并把整个操作收敛到单次 Triton launch。支持 neox 与 GPT-J 两种 pair 布局;`forward_cuda` 在 positions 为 2D([3, num_tokens])时走新 kernel,1D(text-only)时回退到已有 `apply_rope_with_cos_sin_cache_inplace`,否则 `forward_native`(其 `_apply_rotary_emb` 额外包了 `torch.compile(dynamic=True)`)。入口 `forward` 从固定 `forward_native` 改为路由到 `forward_cuda`。kernel 侧断言 `section_h == section_w`(Ernie 布局假设)且 mrope_section 之和等于 rotary_dim//2。

## 关键改动

```text
python/sglang/srt/layers/rotary_embedding.py: 新增 _triton_ernie45_rope_qk_fused kernel 与 triton_ernie45_rope_fused_inplace 入口;Ernie4_5_VLRotaryEmbedding.forward_cuda 接入
```

```python
# ridx < section_hw: even->hpos, odd->wpos ; else -> tpos
use_hw = ridx < section_hw
use_h = (ridx & 1) == 0
pos = tl.where(use_hw, tl.where(use_h, hpos, wpos), tpos)
cos = tl.load(cos_sin_cache_ptr + pos * rd + ridx, mask=rmask, other=0.0)
sin = tl.load(
    cos_sin_cache_ptr + pos * rd + (ridx + half_rd), mask=rmask, other=0.0,
)
```

## 性能声明

- 融合 kernel 将 Ernie4.5 VL rotary embedding 耗时从 670µs 降到 123µs,kernel 维度 5.4x speedup,所测负载约 17% E2E 提升(PR 原话:"5.4x speedup. (≈17% end-to-end improvement for the measured workload)")
- 单请求图编码实测(`bench_n_image.sh`,prompt 973 tokens 含图):real 1.317s → 1.085s
- 模型 `baidu/ERNIE-4.5-VL-28B-A3B-PT`;pack 未注明 GPU 型号与所测 workload 的具体参数

精度(lmms_eval,mmtasks mmmu_val 900 题,batch_size 16):Main 基线 mmmu_acc 0.2644,PR 0.2622(两轮均在噪声范围;PR 未做显著性分析)。

## 验证与回归风险

PR 正文 Accuracy Tests 仅给出上述 mmmu_val 对比;Checklist(单测/文档等)未勾选,diff 中无新增测试。回归面:仅 Ernie4_5_VLRotaryEmbedding 的 forward 路径——2D positions 走 Triton kernel(要求 q/k 最后一维 stride=1、positions stride(1)==1、cos_sin_cache 连续,不满足时 assert 失败);1D text-only 路径行为不变;`forward` 从 native 改为 cuda 优先,非 CUDA 平台的 `forward_cuda` 末尾仍回退 `forward_native`。

## 关联

同区域:vLLM 系 MRoPE 融合 kernel(`apply_rope_with_cos_sin_cache_inplace`,本 PR 在 text-only 路径复用);pack 正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
