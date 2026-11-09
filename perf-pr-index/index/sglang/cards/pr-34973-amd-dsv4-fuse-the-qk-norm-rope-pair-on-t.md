# PR #34973: [AMD] DSv4: fuse the qk-norm-rope pair on the MTP target-verify path

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @karverma-amd |
| 合并 | 2026-08-21 |
| 分类 | Speculative Decoding(major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/34973 |

## 一句话结论

DeepSeek-V4 MTP target-verify 路径复用 decode 的 fused qk-norm-rope kernel(关闭 cache store),输出吞吐 +1.67%~+2.60%。

## 动机

DeepSeek-V4 unified-KV 路径上,fused qk-norm-rope kernel 只接到了 decode;MTP target-verify 仍以每层两次独立 launch 执行同样的 norm+RoPE(`sglang::fused_q_norm_rope` 处理 q、`sglang::fused_norm_rope` 处理 kv),61 个 target layer 共 122 次 launch/verify step。kernel 本身是 token-indexed 的,verify batch(每请求多个 draft token)本来就是它能处理的形状,只是 store 语义不同。

## 优化点

`MQALayer._forward_prepare` 中,新 env 门控 `SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY`(默认开)让 `is_target_verify()` 也进入融合 kernel 调用,但**store 半边刻意关闭**:`swa_cache`/`swa_loc` 传 None(不调 `get_unified_swa_loc`),kernel 只做 q-norm+RoPE 与 kv-norm+RoPE、不写 ring cache。原因:decode 是单源消费者,attention 前必须写 ring;而 target-verify 是双源——attention 从 ring 读历史、当前 chunk 直接作为 k/v 传给 attention,若提前把当前 chunk 写入 ring,draft token 之间失去因果掩码(GSM8K 会从 0.950 崩到 0.027,却伪装成约 7% 吞吐收益,因为跳过的恰是正确性所需的工作)。kv 先 `.contiguous()`(它是 qkv_a 的 strided slice,ring store 需要连续缓冲;unfused 路径在 `_compute_kv_bf16` 内付同样的拷贝,属 parity)。返回的 kv 非空使调用方 `save_kv_cache = kv is not None` 让后端照旧在 attention 后做因果索引的 store。decode、prefill 与非 unified 路径不动。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: _forward_prepare 允许 verify 走融合 kernel(store 关闭,返回 kv)
python/sglang/srt/environ.py: 新增 SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY(默认 True)
```

```python
fuse_verify = (
    envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.get()
    and forward_batch.forward_mode.is_target_verify()
)
do_fused_qk_norm_rope = (unified and (is_decode or fuse_verify)) or (...)
if unified and fuse_verify:
    kv = kv.contiguous()
    swa_cache, swa_loc = None, None   # 跳过 kernel 自身的 store
```

## 性能声明

DeepSeek-V4-Pro,MI355X(gfx950),TP8,8k 输入/1k 输出,steady state(off/on/on/off 顺序各 2 次取均值;噪声 0.25% 1σ;用 `SGLANG_SIMULATE_ACC_LEN=2.49` 钉住接受长度):

| concurrency | 输出 tok/s(base → patched) | uplift |
| ---: | ---: | ---: |
| 4 | 332.43 → 341.07 | +2.60% |
| 8 | 543.73 → 555.57 | +2.18% |
| 16 | 839.00 → 857.32 | +2.18% |
| 32 | 1167.14 → 1186.66 | +1.67% |

kernel 级(TP8,8k/1k,conc 16,graphs-off trace):CSA 30 层 launches/fwd 1920→1800、GPU µs/fwd(不含 collective)15779→15488(-1.8%);HCA 31 层 1526.8→1410.5、12385→12014(-3.0%)。

## 验证与回归风险

GSM8K(1319 题,5-shot,greedy,MI355X TP8):baseline(gate 0)0.950,patched 0.949 / 复跑 0.945,invalid 均 0.000;且明确不使用 `SGLANG_SIMULATE_ACC_LEN`,让 accept/reject 真实路径被测到。新增单元测试素材未在 diff 中展示(测试覆盖情况 PR 未注明)。回归风险:正确性依赖"store 关闭 + 后端因果索引 store"这套约定,若 unified_kv 后端的 per-token causal index stream 行为变化,verify 输出会静默失真;gate 可设 0 回退 unfused 链。

## 关联

同区域:SGLANG_OPT_USE_FUSED_QK_NORM_ROPE(decode 侧融合开关,本 PR 的前序机制)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
