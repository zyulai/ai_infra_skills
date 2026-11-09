# PR #38440: [Feat] [NPU] DSpark: DeepSeek-V4 support DSpark compact mode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @randgun |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, quant, deepseek, hicache, npu, run-ci, diffusion, model-gateway, jit-kernel, unified-radix-cache, memory-pool, bypass-fail-fast, rust-processor |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留部分源码文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/38440 |

## 一句话结论

在 Ascend NPU 上启用 DSV4 的 DSpark compact 投机解码:按 SPS 逐请求压缩 verify 窗口并把 epilogue 折进图,TPOT 降 20.6%、吞吐 +4.0%。

## 动机

DSpark 在 Ascend DSV4 后端此前只能回退到 uniform verify window 或 eager 侧后处理,存在两处可避免开销:(1) SPS 策略已为每个请求选出更短的 verify 长度,target 模型却仍验证完整 draft 窗口;(2) accept/finalize 与 hidden/KV commit 不在 target verify 图内,产生额外 host 侧 launch。

## 优化点

三部分:(1) NPU compact verify——由 DSpark planner 的逐请求 verify 长度构建 packed/ragged verify 窗口,新增 NPU 安全的 compact row mapping、verify-ID gathering、cache-location 处理与 compact-to-strided scatter,DSV4 的 SWA、sparse-attention、C4/C128 compressor、position、state、page-table 元数据全部从 live compact layout 构建;后端不支持 ragged verify graph 时保留 eager fallback。(2) compact target-verify graph——DSV4 Ascend 后端声明 `supports_ragged_verify_graph`,按 compact token tier 选择图并保留固定 graph-owned buffer,replay 前原地刷新 q indptr、seq lens、cache/page table、compressor location/state 与 sparse indices;新增 NPU-local bucketed-layout cache 供 Ascend 与 DSV4 元数据消费者复用同一派生布局。(3) verify epilogue graph——DSV4 NPU 专属 target-verify tail hook,把 greedy accept/finalize 与符合条件的 hidden/KV commit 折进 compact verify 图;sampling、grammar 约束、logits 调整等不合格场景留在 eager epilogue;改动限定 NPU/DSV4,CUDA 行为不变。另:ScheduleVerifyLensTopk 在 NPU arch 上走 bit-exact torch 参考实现(triton-ascend 无法 lower 该 triton kernel),NPU 侧调小 BLOCK 尺寸。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py: ragged verify layout/bucket cache + 图元数据按 token tier 组织
python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py: NPU 走 torch 参考 + BLOCK 调整
python/sglang/srt/speculative/dspark_components/...: compact 行映射/scatter/epilogue tail hook
```

## 性能声明

PR 正文表格(Compact vs Static,Ascend NPU,未注明具体模型/拓扑/负载条件,仅 "DSpark on Ascend NPU"):

- Output throughput:2027.13 vs 1949.98 tok/s(+4.0%)
- Mean TPOT:20.87 vs 26.27 ms(-20.6%)
- Mean TTFT:36563.48 vs 38385.26 ms(-4.7%)

精度:GPQA-Diamond 0.91(DSpark 开启,NPU)。

## 验证与回归风险

新增 scheduler 覆盖:后端不支持 ragged verify graph 时 eager token 数精确;GPQA-Diamond 0.91 为唯一精度数据。正文 checklist 中格式化、文档、speed benchmark 三项未勾选;CI Base/AMD 显示进行中(hourglass)、Extra 为红。风险面:改动限定 NPU/DSV4、CUDA 保持不变,但 ragged verify graph 的 metadata 刷新路径复杂(q_pa/q indptr/compressor state/page table 多处原地更新),`_npu_ragged_verify_bucket_cache` 以 layout 对象标识做缓存,存在 Python id 复用隐患(代码注释已处理);epilogue 折图仅覆盖 greedy 路径。

## 关联

同区域:DSpark 投机解码系列;PR #23882(DeepSeek V4 主支持)、PR #37771(HiSparse spec decoding);同分类 Speculative Decoding 其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
