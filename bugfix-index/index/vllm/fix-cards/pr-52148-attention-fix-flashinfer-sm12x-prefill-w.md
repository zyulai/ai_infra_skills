# PR #52148: [Attention] Fix FlashInfer SM12x prefill with sinks

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52148](https://github.com/vllm-project/vllm/pull/52148) |
| 作者 | askliar |
| 合并日期 | 2026-08-13 |
| labels | bug, nvidia, ready |
| 热度 | 👍 5 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
SM12x 上启用 attention sinks 时,FlashInfer 通用 FA2 prefill 路径收下 `sinks` 参数却不生效,prefill 输出错误;改用 sink-aware 的 `BatchAttentionWithAttentionSinkWrapper`。

## 问题现象
SM12x(device capability family 120)上开启 attention sinks 的模型(如 GPT-OSS):#49718 让 decode 走 XQA,但 prefill 走通用 FA2 路径,产生错误的 prefill 输出。作者验证:`test_flashinfer_native_prefill_with_sinks` 通过、SM12x GPT-OSS 模型 eval 修复并通过、Nemotron 与 Qwen 模型均正常。影响面:SM12x + FlashInfer + sinks 注意力模型的 prefill 正确性。

## 根因
正文披露:通用 FA2 prefill 路径(`BatchPrefillWithPagedKVCacheWrapper`)接受 `sinks` 参数但并不应用它,所以 #49718 的方案(decode 用 XQA)在 prefill 侧输出错误结果。作者说明这不是自动 revert #51987 的重复:#51987 移除了 SM12x XQA 支持,而本 PR 保留 XQA、只修 sinks 模型在 SM12x 上的 prefill 路径。

## 修复方案
改 `vllm/v1/attention/backends/flashinfer.py` 的 `_get_prefill_wrapper` 与 `forward`:当 `has_sinks` 且平台是 device capability family 120(SM12x)时,non-causal 与普通 prefill wrapper 均改用 FlashInfer 的 sink-aware `BatchAttentionWithAttentionSinkWrapper`(按激活 dtype、KV dtype、head_dim、window_left 等特化);`forward` 中按 wrapper 类型分支调用,sink wrapper 直接传 `self.sinks` 与合并后的 softmax scale:

```diff
+                if self.has_sinks and current_platform.is_device_capability_family(120):
+                    self._noncausal_prefill_wrapper = (
+                        BatchAttentionWithAttentionSinkWrapper(
+                            self._get_workspace_buffer(),
+                            get_kv_cache_layout(),
+                            backend="auto",
+                            q_data_type=self.q_data_type_prefill,
+                            kv_data_type=self.kv_cache_dtype,
+                            head_dim_qk=self.head_dim,
+                            head_dim_vo=self.head_dim,
+                            window_left=self.window_left,
+                        )
+                    )
```

sink 路径对 NVFP4 KV cache 断言不允许(`assert not self.is_kvcache_nvfp4`)。DCP、NVFP4、SM90/SM100、无 sinks 路径均不变。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围(涉及 #49718 合入后的 SM12x + sinks 场景)。
- 回归面:仅 SM12x 且 `has_sinks` 的 FlashInfer prefill 分支切换到新 wrapper,并改用其 `run` 签名(sinks、scale 直接传参);无 sinks、NVFP4、DCP、SM90/SM100 平台走原路径不受影响。sink wrapper 对 NVFP4 KV cache 直接断言失败,属显式约束。

## 关联
- #49718(decode 走 XQA、prefill 路径出错的相关 PR)
- #51987(自动 revert,作者说明与本 PR 不重复)
- 正文未列出其他 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
