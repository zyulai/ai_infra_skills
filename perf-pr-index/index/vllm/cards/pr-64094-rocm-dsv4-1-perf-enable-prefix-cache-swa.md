# PR #64094: [ROCm][DSv4.1][Perf]  Enable prefix-cache SWA replay on gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @karverma-amd |
| 合并 | 2026-10-31 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | rocm, ready, multi-modality, deepseek, verified, glm, DSv4.1 |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `attention.py` 的门控修改、amd/model.py 注释更新与新测试,核心逻辑约 20 行 |
| 链接 | https://github.com/vllm-project/vllm/pull/64094 |

## 一句话结论

在 gfx950/MI355X 上恢复 DeepSeek-V4.1 prefix-cache SWA bounded replay,最佳配置 e2e 吞吐 c128 +13.5%、c256 +12%。

## 动机

DeepSeek-V4.1-Flash 40 层中仅第 2/8/14/20 层写 KV,滑窗 128 token:第 20 层之后只需最近 128 token。decoder replay(#59039,由 `cache_config.swa_bounded_replay` 控制)已在 ROCm 生效,但 prefix-cache 路径在 ROCm 被一刀切 force-off——当时假设 FlashInfer/FlashMLA prefill kernel 内有窗口 clamp,而该路径实际不依赖。gfx950 的 AITER sparse MLA 消费的是 `_compute_swa_indices_and_lens_kernel` 已 clamp 到 `replay_start` 的 slot 列表,完全可以安全走 replay。

## 优化点

把 `vllm/models/deepseek_v41/attention.py` 中针对 ROCm 的 `swa_bounded_replay` force-off 收窄为按 CDNA 版本门控:`get_cdna_version() != 4` 时才关闭并告警,CDNA4(gfx950,MI350/MI355)保留 bounded replay,滑窗 cache 从"参与 prefix cache 存储"改为"命中时从 replay_start 重放窗口"。默认开启,`--no-swa-bounded-replay` 仍可关闭;gfx942 及 CDNA3 等其他 ROCm 架构维持 force-off。prefix-cache 语义变化通过 KV cache spec 的 `bounded_replay`/`prefix_cacheable`/`prefix_replay_tokens` 表达。启用方式:去掉 recipe 里的 `VLLM_NO_SWA_BOUNDED_REPLAY=1`,保持 V2 model runner 与可打断 CUDA graph(4096-token prefill chunk 超过 768-token replay trim 阈值,走 trimmed graph)。

## 关键改动

```text
vllm/models/deepseek_v41/attention.py: ROCm force-off 改为 CDNA!=4 才关
vllm/models/deepseek_v41/amd/model.py: 注释同步门控语义
tests/v1/attention/test_deepseek_v4_swa_visible.py: 参数化验证 CDNA4/
  CDNA3 × flag 的 bounded_replay 与 prefix_cacheable spec
```

关键片段:

```python
if swa_bounded_replay and current_platform.is_rocm():
    from vllm.platforms.rocm import get_cdna_version
    if get_cdna_version() != 4:
        logger.warning_once("Prefix-cache SWA replay is only enabled "
                            "on ROCm CDNA4; ...")
        swa_bounded_replay = False
```

## 性能声明

MI355X 最佳配置 e2e:并发 128 吞吐 +13.5%,并发 256 +12%(未注明具体模型 tok/s 数值)。单请求 A/B(nightly 00b6c4c,chunk 2048,DEP8,warm):8k TTFT 492 → 316 ms;32k TTFT 1.84 → 1.23 s;128k TTFT(1k 输出)7.42 → 4.83 s;128k prefill 17,667 → 27,136 tok/s;128k decode 52 → 49-62 tok/s(decode 不变,一步本就只差窗口内 1 token)。精度:GSM8K 1319 题 5-shot 0.904 → 0.902;RULER niah_single_1(131,072 tokens)1.0 → 1.0。prefix-cache 命中 sweep 显示 replay 命中 token 数与 stock 一致或更多(如 4,096 prefix 时 32,640 → 32,768)。

## 验证与回归风险

新增参数化测试覆盖 is_rocm × CDNA 版本 × flag 四种组合,断言 spec 的 bounded_replay/prefix_cacheable/prefix_replay_tokens;CI 无 MI355X checkpoint,Buildkite 不覆盖该路径(作者注明)。风险:flag 默认开启,任何未传 `--no-swa-bounded-replay` 的 gfx950 DeepSeek-V4.1 server 行为都会改变;正确性依赖两条消费路径的 clamp——opt-in AITER sparse MLA 与默认 Triton gather(ComputePrefillMetadataKernel 与 combine_topk_swa_indices);测量基于 AITER Triton sparse MLA 路径,其他路径未验证。

## 关联

前序:#57906(引入 prefix-cache SWA replay)、#59039(decoder replay,已在 ROCm 生效)。同分类 KV Cache/ROCm:#60084(TRITON_ATTN 32 位寻址)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
