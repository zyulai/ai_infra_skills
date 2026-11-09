# PR #27928: [AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @At1a8 |
| 合并 | 2026-06-16 |
| 分类 | Attention / Kernel (tier: major) |
| Label | deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,3 个文件,含 1 个新 CI 测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/27928 |

## 一句话结论

为 DeepSeek-V4 的 unified_kv(Triton)attention 路径补上 AMD prefill context parallel 支持,1024k 输入吞吐最高 2.97x,TTFT 提升 1.33-3.82x。

## 动机

DeepSeek-V4 在 AMD(HIP)上走 `DeepseekV4HipRadixBackend` 的 unified_kv 路径,但 prefill context-parallel(interleave/round-robin 切分)在该路径缺失:模型侧 `_forward_prepare` 中 `if unified:` 分支先于 `elif use_cp:` 短路,跨 rank 的 KV all-gather 被跳过;后端的 per-query 元数据也没有按 CP rank 本地化。长上下文 prefill 全部压在 8 卡上串行做,阻塞正在进行的 decode 请求。目标是提升 AMD 长上下文 prefill 延迟与吞吐,同时保持模型质量;所有改动用 `is_hip` 限制在 AMD 路径,对 NVIDIA(B200/H200)零影响。

## 优化点

三处机制:(1) 模型侧 unified prefill 分支在 `use_cp and _is_hip` 时调用 `cp_all_gather_rerange_output`,把各 rank 的 bf16 KV 在 CP 组内 all-gather 成完整 chunk KV(2-source prefill 与 ring 写入都需要全量);(2) 后端 `_forward_unified_kv` 在 CP 激活(cp_size>1、round-robin 策略、kv 长度为 cp_size*T)时,把 state_slot/chunk_start/cu_q/final_pos 等 per-query 字段按 `[cp_rank::cp_size]` 切成本地 T 条,positions 同样按 round-robin 切(取前 T 个连续位置会得到错误的输出 token),而 KV 值保持全局使每个本地 query 仍能关注完整 all-gathered KV;(3) SWA ring 写入改为每个 rank 都写完整窗口(用全量 state_slot/positions/final_pos),否则 decode 与后续 chunk 读本地 ring 会出错。另对齐 NV 路径补 `c4_sparse_raw_indices = torch.empty_like(c4_sparse_page_indices)`。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: unified 分支补 CP KV all-gather(HIP-only guard)。
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: per-query 元数据 round-robin 本地切片、全窗口 SWA ring 写入、raw indices。
test/registered/amd/test_deepseek_v4_pro_fp4_cp.py: 新增,MI35x 8 卡 nightly,DeepSeek-V4-Pro FP4 + interleave CP,GSM8K > 0.92。
```

CP 元数据本地切片核心:

```python
_sl = slice(get_attention_cp_rank(), None, _cp_size)
state_slot = state_slot[_sl].contiguous()
...
positions = forward_batch.positions.to(torch.int64)[_sl].contiguous()
# SWA ring 必须每个 rank 都写 FULL 窗口
positions_full = forward_batch.positions.to(torch.int64)[: state_slot_full.shape[0]]
```

## 性能声明

AMD 平台(测试注册为 MI35x 8 卡),DeepSeek-V4-Pro,tp8、dp1、cp8(TP8 CP8)对比 TP8 baseline,输出 1k,`--enable-prefill-cp --cp-strategy interleave`:吞吐在 1024k 输入(conc 2)达 2.97x(18561.71 vs 6240.67 token/s),512k 2.01x,256k 1.40x,64k conc64 1.53x;短输入低负载时 baseline 更好(8k conc2 为 0.78x)。TTFT 提升 1.33x-3.82x(1024k 处 3.82x:59904.83 vs 228545.73 ms);受通信开销影响,ITL 在多数条件下回退(约 0.74-0.81x,PR 表述为 "less than 0.26x regression")。PR 强调 CP 的收益来自更快 prefill 减少 decode 阻塞,而非 decode 本身变快(解码略慢)。精度:GSM8K 1319 题准确率 0.947,Invalid 0.000。

## 验证与回归风险

新增 `test_deepseek_v4_pro_fp4_cp.py` 注册到 nightly-amd-8-gpu-mi35x-deepseek-v4-pro 套件,断言 GSM8K > 0.92。pack 中 CI Base 与 Extra 均标红(失败)。回归风险:(1) round-robin 与 sequential 位置切片的正确性陷阱(注释明确 sequential [:T] 会产出错误 token),其他 CP 策略或后端若复用该逻辑需小心;(2) SWA ring 必须全 rank 写全窗口,漏写会导致精度错误;(3) `_cp_active` 判定依赖 `kv.shape[0] == cp_size * T` 与 `state_slot.shape[0] != T` 的形状启发式;(4) 大量环境变量组合(unified_kv_triton、AITER indexer 等)仅在该配置下验证。

## 关联

同区域 DeepSeek-V4 AMD 优化:PR #24584(AMD DeepSeek V4 attention kernel)、PR #24930(AMD DeepSeek V4 Triton);CP 相关前序为 NSA CP 路径(use_cp 已有实现)。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
