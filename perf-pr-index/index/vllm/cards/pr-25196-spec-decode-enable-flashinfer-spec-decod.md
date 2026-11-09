# PR #25196: [Spec Decode] Enable FlashInfer Spec Decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2025-09-24 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/25196 |

## 一句话结论

让 spec decode 走 FlashInfer 的 trtllm-gen decode kernel，ShareGPT 200 请求 TPS 530（1.14x），attention kernel 约 2.5x。

## 动机

此前 FlashInfer 后端未用于 speculative decoding。本 PR 启用它：条件允许时用 `trtllm-gen` decode-optimized kernel 做 spec decode，回退方案是 prefill kernel（可处理任意 query length 但性能较差），目标是把 decode 阶段的 attention 换到更快的专用 kernel。

## 优化点

FlashInfer backend 引入 trtllm-gen workspace buffer 与 `max_q_len_prefill`，把 `reorder_batch_threshold` 改为实例字段并按 `can_use_trtllm_attention(num_qo_heads, num_kv_heads)` 初始化，使 spec decode 的 padded drafter batch 能以 decode 路径规划；`vllm/utils/flashinfer.py` 提供 `can_use_trtllm_attention` 等判定，`eagle.py` 放宽 `allowed_attn_types` 以接纳新 metadata，并同步多个 attention backend（MLA、mamba、linear、short_conv、xformers、gdn）的 metadata 接口与 `split_decodes_and_prefills`。正文还说明修复了非均匀 batch（如 `[2, 1]`）下 `max_q_len > max(query_lens)` 造成的 TRTLLM-gen 非法访存：切分 batch 后手动计算 `max_q_len_prefill`。正文给出的示例启动命令带 `--no-enable-prefix-caching`，并把 `disable_padded_drafter_batch` 作为 non-uniform batch 的可选回退；回退走 prefill kernel，可处理任意 query length 但性能较差。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: trtllm-gen workspace、max_q_len_prefill、spec-as-decode 的 reorder threshold
vllm/utils/flashinfer.py: can_use_trtllm_attention 等后端能力判定
vllm/v1/spec_decode/eagle.py: allowed_attn_types 放宽以接纳新 metadata
vllm/v1/attention/backends/utils.py: split_decodes_and_prefills 与 metadata 切分
tests/v1/attention/test_attention_splitting.py: 切分逻辑测试
```

```python
+trtllm_gen_workspace_buffer = None
+
+
+def _get_trtllm_gen_workspace_buffer():
+    global trtllm_gen_workspace_buffer
+    if trtllm_gen_workspace_buffer is None:
+        trtllm_gen_workspace_buffer = torch.zeros(
+            FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device='cuda')
+    return trtllm_gen_workspace_buffer
...
+        supports_spec_as_decode = \
+            can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads)
+        self._init_reorder_batch_threshold(1, supports_spec_as_decode)
```

## 性能声明

Benchmarking with **200 requests from ShareGPT** 的 TPS：**Padding enabled + FlashInfer 530 TPS (1.14x)**、Padding disabled + FlashInfer 465 TPS (1.0x)、Padding enabled + FlashAttention 492 TPS (1.06x)、Padding disabled + FlashAttention 466 TPS (1.0x)。`nsys` traces 显示使用 decode-optimized trtllm-gen kernels 时主模型 attention kernels **约 2.5x faster**。测试条件：示例启动命令为 EAGLE3（1xB200、`meta-llama/Llama-3.1-8B-Instruct` + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`、`num_speculative_tokens=4`、`--max-model-len 2048`、`--no-enable-prefix-caching`）；PR 未注明精度。

## 验证与回归风险

Correctness：GSM8k（limit 500、5-shot、batch_size 128）在 with/out spec 与 with/out FlashInfer 四种组合下均成功：FlashInfer+Padded 0.796/0.774、FlashInfer+Non-uniform 0.792/0.774、Baseline FlashInfer 0.796/0.778、Baseline FlashAttn 0.794/0.772（flexible-extract / strict-match exact_match）。风险面：正文记录 TRTLLM-gen kernels 在 Blackwell 上非均匀 batch 曾出现非法访存导致崩溃（后在切分后计算 `max_q_len_prefill` 修复）；`disable_padded_drafter_batch` 的 non-uniform 回退走 prefill kernel，性能较差；CUDA graph padding 的 planning 逻辑曾假设 `num_decode_tokens` 与 `num_decodes` 相同。

## 关联

PR 正文明确"This PR depends on #25183"（batch reordering threshold 变量重构）。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
