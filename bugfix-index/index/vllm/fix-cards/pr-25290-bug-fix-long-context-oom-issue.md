# PR #25290: [Bug] Fix Long Context OOM Issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25290](https://github.com/vllm-project/vllm/pull/25290) |
| 作者 | yewentao256 |
| 合并日期 | 2025-09-23 |
| labels | ready, v1 |
| 热度 | 👍 3 · 💬 2 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
MLA chunked prefill 工作区按 128k token 预留导致 128k 长输入启动即 OOM;把预留上限降到 64k token(与注释一致),单条 128k 请求可正常完成。

## 问题现象
超长上下文单请求 OOM。用户(@smarterclayton)用 DeepSeek-V3.1 测试 128k 输入、1 个输出 token(显式设 `--max-model-len`,DeepSeek V3.1 本身即 128k),单条请求即报:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.12 GiB. GPU 0 has a total capacity of 178.35 GiB of which 3.91 GiB is free. ...
```

配置为 DP=16、2 节点 B200,每 GPU 约 53GB KV cache。修复后同场景基准测试成功:129,999 input tokens / 1 generated token,Mean TTFT 167585.20 ms。

## 根因
PR 正文披露:主要原因是 MLA chunk padding 分配了过多内存。代码中 MLA chunked prefill workspace 的预留上限写成 `128 * 1024` 个 token,而同处注释明确建议长上下文模型不要过度挤占 KV cache 空间、应限制在 64k token(按 576 MLA head dim、fp16 估算 workspace 约 144MB,up-projected context 约 3GB)。正文也说明:若上下文更长、显存有限,OOM 是合理的,应增加 tp 或调低 `--gpu-memory-utilization`。

## 修复方案
改动在 `vllm/v1/attention/backends/mla/common.py`,把 chunked prefill workspace 的 token 上限从 `128 * 1024` 改为 `64 * 1024`,同时消除注释与代码的不一致:

```diff
             # which would result in up-projected context being
             #   2*(192*128)*(64*1024) = 3gb
             # (assuming 192 QK head dim, 128 heads, and fp16)
-            128 * 1024)
+            64 * 1024)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:影响所有走 `vllm/v1/attention/backends/mla/common.py` 该工作区分配的 MLA 后端路径;工作区减半后,若单 chunk 实际需要的 up-projected context 超过 64k token 预算,相关行为可能变化(正文未讨论该点)。正文 Test 仅给出上述单请求基准结果,无更广的回归验证数据。

## 关联
正文未列出(问题上下文引自用户 @smarterclayton 的报告)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
