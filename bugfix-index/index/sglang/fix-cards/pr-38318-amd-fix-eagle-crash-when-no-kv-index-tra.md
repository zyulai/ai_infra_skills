# PR #38318: [AMD] Fix EAGLE crash when no kv_index_translator is bound on the DSA fp8 read door

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38318](https://github.com/sgl-project/sglang/pull/38318) |
| 作者 | jiaryang |
| 合并日期 | 2026-09-07 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
EAGLE 草稿模型走 DSA 后端时对未绑定 kv_index_translator 的后端强做翻译,首个请求即崩;加 None 判断跳过。

## 问题现象
gfx950(MI355X)上以 `--kv-cache-dtype fp8_e4m3` 启动带 EAGLE 的 DeepSeek 类模型时,服务器正常起来,但第一个请求到达时所有 4 个 TP rank 全部崩溃。报错栈:
```
File "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py", line 659, in _get_mla_kv_buffer_from_fp8_for_dsa
    kv_indices = get_attn_backend().kv_index_translator.translate_dcp_read_ids(
AttributeError: 'NoneType' object has no attribute 'translate_dcp_read_ids'
```
仅投机解码(speculative decoding)触发:target-only 运行没有 draft model,不会走到这条路径。正文给出的复现命令为 `python3 -m sglang.launch_server --model amd/GLM-5.2-MXFP4 --tp 4 --trust-remote-code --kv-cache-dtype fp8_e4m3 --disable-radix-cache --chunked-prefill-size 16384 --dsa-prefill-backend triton --dsa-decode-backend triton --speculative-algorithm EAGLE --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-eagle-topk 1`。

## 根因
正文披露:`_get_mla_kv_buffer_from_fp8_for_dsa` 这个「读门」无条件地通过 attention backend 的 `kv_index_translator` 翻译读取的 ids,但只有 FlashAttention 和 FlashInfer 后端绑定了 translator;`BaseAttentionBackend` 文档化其 `None` 默认值含义是「no translate」,而 EAGLE 草稿模型所运行的 DSA 后端正是保留该默认值——于是对 None 调 `translate_dcp_read_ids` 抛 AttributeError。

## 修复方案
按 `BaseAttentionBackend` 的契约在读门处加保护:没有绑定 translator 时跳过翻译(此时 ids 从未被扩宽到 VIRTUAL,无需折叠);绑定了 translator 的后端行为不变。改动仅一个文件 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`:
```diff
-            kv_indices = get_attn_backend().kv_index_translator.translate_dcp_read_ids(
-                kv_indices
-            )
+            translator = get_attn_backend().kv_index_translator
+            if translator is not None:
+                kv_indices = translator.translate_dcp_read_ids(kv_indices)
```
正文确认这是「guard only, no functional change on any path that previously worked」。另附 GSM8K 精度验证(GLM-5.2-MXFP4, MI355X, TP4, 三次运行):EAGLE 0.941/0.932/0.927,target-only 0.929/0.932/0.932,处于同一水平。

## 影响与回归风险
- 影响窗口:正文未说明(指出当前 main 即崩溃,本 PR 后正常)。
- 回归面:仅 `_get_mla_kv_buffer_from_fp8_for_dsa` 一处读路径;FlashAttention/FlashInfer 等绑定了 translator 的后端行为字节级不变;变化只发生在「translator 为 None」的此前必崩路径上。开销为一次 None 检查。

## 关联
- 正文引用代码位置:https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py#L659

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
