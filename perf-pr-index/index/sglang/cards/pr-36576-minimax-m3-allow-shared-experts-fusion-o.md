# PR #36576: MiniMax-M3: allow shared-experts fusion on ROCm gfx942 and newer

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-17 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | amd, run-ci |
| Diff | diffstat 缺失(pack 未记录);diff 可用,涉及 3 个文件(topk.py、minimax_m3.py、minimax_m3_vl.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/36576 |

## 一句话结论

解除 MiniMax-M3 shared-experts fusion 的 CUDA-only 平台限制,让 ROCm(代码实际 gate 在 gfx950)也能启用该融合。

## 动机

`shared_experts_fusion_disable_reason` 此前在任何非 CUDA 设备上直接拒绝,导致 ROCm 即使硬件支持也永远拿不到 shared-experts fusion。作者认为这一平台限制在 gfx942+ 上没有存在理由。另一个关键动机:实际服务的 MiniMax-M3-MXFP8 checkpoint 是 VL variant(`model_type: minimax_m3_vl`),`minimax_m3_vl.py` 里有一份完全相同的 gate——只 patch `minimax_m3.py` 一个文件会在用户真正跑的模型上静默失效,所以两个文件必须同时改。

## 优化点

这是纯粹的 enablement gate 改动,不新增内核。第一处:两个模型文件里把 `if not _is_cuda` 的拒绝条件放宽为 `if not (_is_cuda or _is_hip)`,CUDA 侧保留 SM80 门槛,ROCm 侧新增 `is_gfx95_supported()` 检查(拒绝信息写明 "validated on gfx950 only";注意标题写 gfx942+,代码 gate 实际是 gfx950)。第二处:`topk.py` 的 `select_experts` 中,`num_fused_shared_experts_for_gate` 的计算新增 `_use_aiter` 分支——因为 aiter 后端在 `_post_process_topk_ids` 里自行追加 shared expert,gate 侧不能再计入,否则与 DeepEP remap 的 expert id 偏移逻辑(384 -> 392)冲突。

## 关键改动

```text
python/sglang/srt/models/minimax_m3.py: 放宽 shared_experts_fusion_disable_reason 到 CUDA/ROCm,ROCm gate 在 gfx95
python/sglang/srt/models/minimax_m3_vl.py: VL variant 上同步同样的 gate 改动(实际服务的 checkpoint 走这里)
python/sglang/srt/layers/moe/topk.py: select_experts gate 对 _use_aiter 不计入 fused shared experts
```

关键代码(topk.py gate):

```python
# aiter appends the shared expert in _post_process_topk_ids, so the gate must not.
num_fused_shared_experts_for_gate = (
    0
    if (has_per_rank_fused_shared_slots(num_fused_shared_experts) or _use_aiter)
    else num_fused_shared_experts
)
```

## 性能声明

PR 自己没有任何 benchmark 或精度数字,作者明确不声称性能。正文提到的 +46.6%..+55.4% output throughput(8x MI350X)属于更早的一个组合改动(本 gate + alternate-stream overlap + `prefill_attention_backend=aiter` serving default),不属于本 PR;且分解显示 input throughput 收益全部来自 serving default。更关键的是:所有测量运行中服务端均报告 `shared_experts_fusion=False`——本 PR 启用的融合从未真正生效,因此这个 gate 本身的吞吐效应是未测量的。

## 验证与回归风险

diff 未新增测试。PR 页面 CI 状态显示 Latest PR Test (Base)、(Extra)、(AMD ROCm 10) 三个 run 均失败(:x:)。回归风险:`topk.py` 的 gate 改动影响所有走 `select_experts` 的 MoE 模型的 aiter/DeepEP 路径;ROCm 侧 fusion 在非 gfx950 卡上仍会被拒绝;CUDA 路径行为不变(SM<80 依旧拒绝)。作者建议由真正能打开 fusion 的配置在合并后再测性能。

## 关联

正文提到同一 bundle 的另一半(serving defaults 与 cookbook)是单独的 change,不含在本 PR 内,编号未注明。同区域无其他明确关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
