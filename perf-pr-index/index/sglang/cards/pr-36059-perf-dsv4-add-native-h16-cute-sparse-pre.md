# PR #36059: [Perf] dsv4: Add native H16 CuTe sparse prefill backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xuqianmamba |
| 合并 | 2026-11-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未统计行数;正文提及 refactor 后约 1,253 additions / 12 deletions、11 files);diff 素材省略 9 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/36059 |

## 一句话结论

为 DSV4 sparse prefill 新增 SM90 原生 H16 CuTe DSL kernel,免掉 48 个 padded query head,8x H20 serving 吞吐 +9.9%-17%。

## 动机

DeepSeek-V4-Pro 有 128 个全局 query head,TP=8 下每 rank 仅 16 个逻辑 query head;现有 SM90 FlashMLA sparse-prefill 路径按 H64 padding 的 query 运算,需要计算并物化 48 个 padded head。本 PR 新增显式、实验性的原生 H16 CuTe DSL 后端(`--dsv4-prefill-backend cutedsl_h16`),返回紧凑的 `[TQ, 16, 512]` 输出;默认 `auto` 行为不变。

## 优化点

新增 SM90 BF16 CuTe DSL sparse-prefill kernel,采用 `m64n16` WGMMA 分解与 online softmax,sink 在 epilogue 并入分母;cp.async 双缓冲流水(QK 分组提交、PV rolling 与下一 tile 预取重叠),输出 epilogue 复用 KV staging 共享内存。启动契约校验:CUDA SM90、BF16 query 与 sparse workspace、恰好 16 个 TP-local 逻辑 head、Q/K/V 宽 512、合并 TOPK 对齐 128 且不超过 16384(默认 1M context 的 8320 满足)。用配置级定长的 C128 TOPK bucket(对齐 padding)避免 chunk 长度变化引入新的 JIT specialization;缓存 padded combined-index 表示供同层复用。仅普通 eager EXTEND 走 H16;decode、speculative、TBO、CP、bounded-tail、非 eager EXTEND 与 CUDA graph 路径保持原后端;HIP/NPU 上显式选择 `cutedsl_h16` 直接报错。C128 索引 padding 复用 `get_c128_combined` 定宽缓存。refactor 后 runtime 再减 183 行,15 个设备侧函数与已验证版本 AST 一致;review 中修复了 Int64 偏移越界(TQ/KV 行跨 2^31 边界的非法内存访问)。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/cute_sparse_mla_h16.py: SM90 CuTe H16 sparse-MLA kernel(约 746 行)+ 编译缓存
python/sglang/srt/layers/attention/deepseek_v4_backend.py: cutedsl_h16 选择、契约校验、C128 定宽 bucket 与 prefill 分支
python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py: SparsePrefillChunkCache.get_c128_combined 定宽索引缓存
python/sglang/srt/layers/attention/attention_registry.py: 注册 cute_sparse_mla_h16_fwd kernel(SM90 限定)
python/sglang/srt/arg_groups/fields/exec_.py: --dsv4-prefill-backend 新增 cutedsl_h16 选项
```

## 性能声明

端到端 serving(2026-10-29,DeepSeek-V4-Pro,8x H20-3e TP=8,context 49152,`flashinfer_mxfp4` MoE,chunked prefill 4096,GSM8K 派生定长 prompt,3 次取中位数,吞吐含输入+输出 token;两臂仅 prefill backend 不同):

| Input/Output | 并发 | BF16 H64 tok/s | Native H16 tok/s | 吞吐增益 | Mean TTFT 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 / 1 | 8 | 5286.37 | 5969.72 | 12.93% | 11.45% |
| 16384 / 1 | 8 | 5028.81 | 5839.60 | 16.12% | 13.87% |
| 32768 / 1 | 4 | 4875.40 | 5702.65 | 16.97% | 14.49% |
| 4096 / 128 | 16 | 4224.03 | 4641.04 | 9.87% | 11.64% |

算子微基准(TQ=4096, SKV=49152,3 次取中位):CuTe BF16 H16 相对 FlashMLA BF16 H64 约 2.9x(如 TOPK=128:690.032 -> 236.368 us)、相对 FlashMLA Q8 H64 约 2x;正文强调两者输出契约不同,不可当作 e2e 加速读。

准确性:GSM8K 5-shot 1314 题(temperature 0),flashmla_sparse 95.3577% vs cutedsl_h16 95.0533%(held-out 95.0344% vs 95.3400%),H16 少对 4 题(配对:基线独对 9 题,H16 独对 5 题),invalid 均为 0。历史版记录(2026-08-23,旧环境):latency -2.3%、输出吞吐 +3.0%(0.951 -> 0.952)。

## 验证与回归风险

多轮 H20 验证:最终 32 passed(numerical/boundary 含 TOPK 8320/16384、backend/selector、dispatch/cache、snapshot/metadata、CLI、docs coverage),Int64 边界用例在旧代码上可复现 IMA;C4/C128 parity 改用生产 `make_core_attn_metadata` 与真实 gather 路径,并做 fault-injection 验证测试能拒绝旧 mock。风险:仅 eager EXTEND 生效,graph/spec/TBO/CP 路径不走 H16;TOPK > 16384 显式报错;DSV4.1 不支持;算子输出契约(H16 vs H64+max/LSE)不同,跨后端对比需谨慎;TOPK=128 最小档在 offset fix 后测得 +1.702% 算子延迟(后续更新中消失)。

## 关联

同区域:FlashMLA sparse prefill(FlashMLA BF16/Q8 H64 为对照与默认路径)、`flashmla_sparse_q8`(Q8KV8 sparse prefill,同 flag 的既有选项)。其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
