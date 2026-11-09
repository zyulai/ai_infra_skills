# PR #59039: [ROCm][Perf][DSv4.1] Enable decoder SWA bounded replay

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @frida-andersson |
| 合并 | 2026-10-13 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | rocm, ready, ci/build, deepseek, DSv4.1 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/59039 |

## 一句话结论

在 ROCm 上为 DeepSeek-V4.1 开启 decoder 侧 SWA bounded replay:16384-token prefill 的 Median TTFT 从 1176 ms 降到 630 ms(−46%),吞吐 +27.8%。

## 动机

#58132 已为 CUDA 模型加入 decoder 侧 SWA bounded replay,但 ROCm 上被禁用(此前 warning 说明 sparse SWA metadata builder 依赖的窗口 clamp 只存在于 FlashInfer/FlashMLA prefill kernel 中,padded slot 会 fault)。DeepSeek-V4.1 的 AMD 模型使用 `DeepseekV41ModelState` 与 #59532 的 replay graphs,且 AMD attention 类不覆写 `forward_kv`,replay 调用共享的 `DeepseekV4Attention.forward_kv`(经 `fused_deepseek_v4_kv_rope_insert` 写 uint8 SWA cache),具备开启条件。

## 优化点

机制:eager prefill 时,最后一个 KV source 层(生产配置为 layer 20)对每个 token 写 KV,然后从该层起(`DecoderReplayLayers`)每请求只在其最后 128 个 token(sliding_window)上重跑该层剩余部分与 layers 21–39,最终 `mhc_post` 在 replay batch 内运行;返回 `(hidden_states, pre_mix, *aux)`,cut 处的 aux 状态在裁剪后的行上捕获。`write_kv` 只做该层 forward 中写 KV 的部分(mhc_pre_delayed + attn_norm + forward_kv),跳过其余计算;metadata prefixes 收集 replay 层读的 SWA/compressed/indexer cache。NVIDIA 侧 `model_state.py` 配合:存在 decoder replay layers 且 prefix-cache window 为 0 时仍构建 `replay_start` 并 warm gather kernel(pad kernel 留在 prefix-cache 路径)。Gating:`cache_config.swa_bounded_replay` 开启 + v2 model runner;`--no-swa-bounded-replay` 关闭;最后 KV source 需 `compress_ratio <= 1` 且其后无 Engram 层;ROCm 不读 `swa_cache_layer.bounded_replay`(那是 prefix-cache 路径);replay graphs 跟随 `cudagraph_decoder_replay`(默认开)与 `decoder_replay_trim_threshold`(默认 768)。

## 关键改动

```text
vllm/models/deepseek_v41/amd/model.py: write_kv 抽取、DecoderReplayLayers 构建(cut=max(kv_source_layer_ids))、_run_layers/_run_replay_layers 拆分
vllm/models/deepseek_v41/nvidia/model_state.py: window=0 时仍构建 replay_start 并 warm gather kernel,pad kernel 留在 prefix-cache 路径
vllm/models/deepseek_v41/amd/vl_model.py: VL 模型转发 decoder_replay_layers
vllm/models/deepseek_v41/attention.py: ROCm 禁用文案改为仅限 prefix-cache SWA replay
tests/models/test_deepseek_v41_amd_replay_forward.py: 300-token 前向的裁剪/双 mhc_post/cut-aux/SWA insert 校验
```

replay 层运行核心:

```python
def _run_replay_layers(self, hidden_states, ...):
    # cut-aux 在裁剪后的行上捕获;最终 mhc_post 在 replay batch 内运行
    return (hidden_states, pre_mix, *cut_aux, *aux)
```

## 性能声明

测试条件:4× MI325X(gfx942),TP4,DeepSeek-V4.1-Flash,spec decode off、prefix cache off,HIP INT4 QuickReduce,`FULL_AND_PIECEWISE`,扩展为 ROCm nightly `0.31.1rc1.dev1+gbb87d227d`;基准在 rebase 到 main `c75cb63cd2bb` 之前测得、未复跑。4 prompts、16384 in、128 out、并发 1、seed 7,双臂各 4 请求:

- Median TTFT 1176.37 → 629.97 ms(−46.4%);Mean TTFT 1176.09 → 629.65 ms(−46.5%)。
- Median/Mean TPOT 基本不变(10.59 → 10.58 ms)。
- Output throughput 50.75 → 64.87 tok/s(+27.8%)。

## 验证与回归风险

测试:`test_deepseek_v41_amd_replay_forward.py`(300-token 前向:裁剪到窗口、两次 `mhc_post`、cut-aux 捕获、cut 前缀集合、SWA insert 与 `fused_deepseek_v4_kv_rope_insert` 字节一致、poison 块未被写)与 `test_deepseek_v41_replay_batch.py` 新增 window=0 warm gather/跳过 pad 用例,GPU 4 上 11 passed。精度:RULER `niah_single_2` 500 样本 @16384 在前一 head 上 off/on 均 1.0(本 drop 未复跑,65536 未测);GSM8K 5-shot 1319 题、并发 8:strict-match 0.9287→0.9257、flexible-extract 0.9280→0.9257。风险面:replay 与 sequence parallel、prefill context parallel、microbatching 不兼容;drafter 的 sliding window 必须不大于目标窗口;PP stage 必须持有 cut 及其后所有层;Engram 层位于 cut 之后则关闭。

## 关联

依赖(已在本 base,不在本 diff):#59894(write_kv/first_swa_prefix 替代 row buffers)、#56638(ROCm prefill SWA 索引在 replay start 处封顶)、#59532(replay graphs);前序 #58132(CUDA decoder SWA bounded replay)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
