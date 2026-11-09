# PR #45339: [Perf] [AMD] dsv41: Skip unneeded layer 20 computation under decoder-bounded-replay

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-10-20 |
| 分类 | Speculative Decoding | tier: major |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 3 个文件:`deepseek_v4.py`、`deepseek_v4_platform_stage.py`、`deepseek_v4_backend_hip_radix.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/45339 |

## 一句话结论

AMD 路径在 decoder-swa-bounded-replay 下跳过 DeepSeek-V4.1 layer 20 的全 token 计算,200K prompt 的 TTFT 提速 7.1%。

## 动机

开启 `--enable-decoder-swa-bounded-replay` 时,DeepSeek-V4.1 的 decoder 层在 prefill 中只处理每个请求的最后 128 个 prompt token。但 layer 20(第一个 decoder layer)仍必须为每个 token 写 KV cache——只是不必对全部 token 跑 indexer、attention 与 MoE。此前 AMD 代码路径对每个 token 都跑完整的 layer 20。NVIDIA 侧已在 #42901 启用该跳过;AMD 被排除的原因是其 fused layer path 把整个 layer 20 在一次调用中跑完,KV write 无法从层的其余部分拆分出来。

## 优化点

全部改动限于 HIP/AMD 路径(NV 路径不变),核心是把 layer 20 拆成"全 token KV 写入 + 仅 tail 128 token 的层计算"。`deepseek_v4.py`:AMD 上 `tail_writes_source_kv=True`,使 tail 从 layer 20(而非 21)开始;layer 20 的 KV-write 输入改由新增的 `fused_boundary_attn_input` 提供——gfx950 的 fused boundary 把每层的 mHC post 延迟到下一层开头执行,layer 20 开始时上一个 FFN 的 hc_post 还未运行,需要先补算。`deepseek_v4_platform_stage.py`:新增 `fused_boundary_attn_input()`,复用 fused boundary kernel 的前段为每个 token 计算 pending mHC post 并构建 layer 20 的 attention 输入。`deepseek_v4_backend_hip_radix.py`:复用 `DeepseekV4AttnBackend.write_tail_source_kv` 为每个 token 写 layer 20 的 KV cache;layer 20 的 indexer 只在最后 128 token 上运行,结果直接写入 tail buffer;`enter_late_layer_tail` 相应大幅简化——不再切分 candidate mask、不再把 source 层的 sparse page indices/topk lengths/raw indices 从 full-token buffer 拷贝到 tail buffer(source 层现在在 tail 内部发布自己的 top-k 与 candidates),删除相应拷贝代码。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: AMD 上 tail_writes_source_kv=True,tail 起点提前到 layer 20
python/sglang/srt/models/deepseek_v4_platform_stage.py: 新增 fused_boundary_attn_input(补算 pending mHC post 并构建全 token attention 输入)
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: 复用 write_tail_source_kv;indexer 仅跑 tail 128 token;删除 full→tail buffer 拷贝
```

```python
if _is_hip and source.hc_boundary_fused:
    # The previous FFN hc_post has not run yet, so run it here too.
    source_x = fused_boundary_attn_input(source, state)
else:
    source_x = mhc.combine(source.attn_hc, state)
attn_backend.write_tail_source_kv(layer=source.self_attn, x=source_x, ...)
```

## 性能声明

条件:SA MI355 TP2 设置,DSpark block size 5,模拟 accept length 3.51,模型 DeepSeek-V4.1-Flash(tp 2、`--fp8-gemm-backend aiter`、`--enable-decoder-swa-bounded-replay`、`--swa-prefix-tails 2048`、AITER 环境变量组)。

- 单个 16K-token prefill chunk 中 layer 20 耗时:prompt 首 chunk 13.4 ms → **3.0 ms**;200K-token prompt 的最后一个满 chunk 34.4 ms → **3.0 ms**。
- TTFT(新 prompt):590 tokens 83.8→82.1 ms(**2.1% faster**);6,318 tokens 152.9→146.8 ms(**4.0%**);14,795 tokens 259.0→248.0 ms(**4.3%**);64K tokens 1163.0→1103.9 ms(**5.1%**);200K tokens 3811.1→3540.3 ms(**7.1%**)。缓存前缀后追加新 token(2K/128 new tokens after 16K/64K/200K cached)均无变化。
- GSM8K(1319 题,两轮):无 spec 0.905/0.900 → 0.906/0.900;DSpark bs5 0.902/0.898 → 0.917/0.901;radix cache off 0.903/0.906 → 0.904/0.898。

## 验证与回归风险

测试:GSM8K 1319 题三组配置前后对比(上表);运行环境含 `SGLANG_USE_AITER=1` 等一组 AMD 特性环境变量。风险:仅 AMD/HIP + `--enable-decoder-swa-bounded-replay` 生效;tail 起点提前意味着 `enter_late_layer_tail` 的前提变化(write_tail_source_kv 必须先于 tail 进入,新增 assert);fused boundary 下 layer 20 输入依赖 `fused_boundary_attn_input` 补算 pending mHC post,与 `hc_boundary_fused` 标志耦合;unified_kv_triton 后端不支持 bounded replay(既有 NotImplementedError);checklist 各项未勾选;CI Base/Extra 显示失败(外部状态,正文未说明)。

## 关联

#42901(NV 前序:NVIDIA 侧同一 layer-20 skip 的启用)。无正文列出的其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
