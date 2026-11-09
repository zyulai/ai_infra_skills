# PR #37577: [Feature] Add Aaronson-Gumbel text watermarking

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(44 个文件未收入,仅含 proto/api_types 与 textseal_selector.py 新文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/37577 |

## 一句话结论

新增 opt-in 的 Aaronson-Gumbel 文本水印:有限 top-k 时输出吞吐几乎无损(+0.13%/+0.02%),无限 top-k 场景每流吞吐 -2.8%~-7.2%,支持 normal 与 NGRAM/EAGLE/EAGLE3/NEXTN 投机解码。

## 动机

部分部署需要机器可读的"文本由 AI 生成"信号,用于溯源流程与 EU AI Act Article 50 等透明度义务。目标是在采样生成中内嵌水印而不增加 token、不改变默认 serving 行为,并可从 token ID 端到端检测(Gamma-test)。

## 优化点

在 post-penalty、grammar-masked、top-k/top-p/min-p 分布之上做 keyed Aaronson-Gumbel 采样:对每个候选 token 用 Murmur32 把 (key, context_hash, token_id) 混成伪随机数并转成 log-uniform 值,再除以概率得到 Gumbel 分数,选分数最大者作为强制 token,通过把 logits 清为 -inf 再写 0 实现 force,返回的 logprob 仍反映 pre-watermark 目标分布(与投机解码语义对齐)。实现上是纯 Triton kernel(textseal_selector.py,改编自 facebookresearch/textseal):有限 top-k 走 `_watermark_force_topk_kernel`(含 boundary-tie 处理的 `_canonicalize_topk_kernel`),无限 top-k 走 partial-argmax 分片 + finalize kernel;fused normal/speculative state kernel 把 context hash 计算、重复上下文防环(已用 context hash 去重)与 token history 追加合入投机路径;host 侧对完全禁用水印的 batch 不启动任何 watermark kernel。支持 per-request 的 enabled/key/context_window、双 key 按 per-context 确定性硬币切换,以及对不支持模式(DFLASH/DSPARK、PP 投机、beam search、PD 分离、非 CUDA 等)的 fail-closed 拒绝。

## 关键改动

```text
python/sglang/kernels/ops/sampling/textseal_selector.py: 新增 Triton 水印 selector/force/argmax/context-hash/记录 kernel 全家桶(1400+ 行)
python/sglang/api/v1/api_types.py + proto/sglang/api/v1/sampling.proto: SamplingParams 新增 watermark(enabled/key/context_window)请求字段,Rust wire schema 同步
python/sglang/srt/sampling/watermarking/: 服务器配置 --enable-watermark/--watermark-config、检测器与 README(pack diff 未含,见 PR)
```

```python
scores = tl.where(is_candidate,
    _log_uniform_from_hash(hashed) / safe_probabilities, -float("inf"))
max_score = tl.max(scores, axis=0)
token_id = tl.min(tl.where(scores == max_score, token_ids, 0x7FFFFFFF), axis=0)
```

## 性能声明

PR 正文给出 ON/OFF 相对对比(随机 1024/512 请求,c8,temperature 0.8,top-p 0.95,top-k 20,10 warmup/100 measured/3 次重复,除注明外):

- 有限 top-k 无投机:Qwen3-8B H200 c1/c8 输出吞吐 **+0.13% / +0.02%**(run-to-run 噪声内);GB300 无限 top-k c8:**224.88 → 212.84** tok/s per stream(**-5.35%**);Limitations 注明无限 top-k 实测 **-2.8% ~ -7.2%** per-stream(需全词表 sort)。
- 投机:Qwen3.5-9B GB300 TP2 NEXTN 3/1/4 TPOT **2.246 → 2.451 ms(+9.17%)**;投机 accept length **2.759 → 2.783(+0.85%)**;dual key vs single key GB300 输出吞吐 -0.21% c1 / -1.50% c8(重复方差内)。
- 开销护栏:能力启用但 EAGLE3 流量完全禁用时 TPOT 2.465 → 2.469 ms(**+0.14%**,vs merge-base)。
- 精度/检测:Qwen3-8B GSM8K-100(sgl-eval)在 B200 normal、H200 NGRAM(--disable-overlap-schedule)、EAGLE3 下 95–97% accuracy、100% stop;检测 ON p=1.09e-20~4.18e-27,OFF 对照 p=0.325–0.995;DP2/TP2 gate 两 rank 均检测到。

## 验证与回归风险

测试面很广:CUDA selector/state parity 套件、OpenAI endpoint e2e、Rust `wire.rs` schema 测试、`test_parity.py`/`test_model_overrides.py` registry、pre-commit/taxonomy 检查、公开模型 GSM8K/检测/greedy/投机/TP/DP service gates;实测覆盖 normal + NGRAM/EAGLE/EAGLE3/NEXTN 与 TP2/TP4、DP2/TP2。风险:`SamplingParams`/proto 新增字段触及 API 兼容面(unknown keys 仍 DENY);fail-closed 列表外的组合(如未来投机模式)需显式加入;水印在 min(4096, per-request token-pool) 个 context 后停止、temperature 0 无信号、paraphrase/翻译可抹除信号;Murmur32 非密码学,key 持有者可检测可伪造。CI 状态:Base 运行中(hourglass),Extra 通过,AMD ROCm 10 失败(:x:)。

## 关联

与投机解码采样路径(NEXTN/EAGLE/EAGLE3)深度集成;实现参考 facebookresearch/textseal(Apache-2.0);pack 未列出同分类关联 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
