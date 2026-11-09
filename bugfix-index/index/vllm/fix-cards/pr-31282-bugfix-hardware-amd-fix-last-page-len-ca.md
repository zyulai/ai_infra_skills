# PR #31282: [Bugfix][Hardware][AMD] Fix last_page_len calculation in AITER MLA decode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31282](https://github.com/vllm-project/vllm/pull/31282) |
| 作者 | c0de128 |
| 合并日期 | 2026-01-02 |
| labels | rocm, ready, v1 |
| 热度 | 👍 0 · 💬 30 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论

AITER MLA 解码把整串长度当 last_page_len,致注意力算错/越界,改为恒为 1 的预置缓冲。

## 问题现象

ROCm AITER MLA 解码路径下,`paged_kv_last_page_len` 被算错。按正文:非标准长度的序列注意力分数计算不正确,MLA decode 内核可能出现越界内存访问,素数长度的序列行为不可预测。作者用 DeepSeek-V2-Lite(MLA 架构)配合素数 token 数(127、131)做验证;测试标准为 MLA decode 不崩溃不挂起、输出连贯、素数长度可用。AMD CI 当时因基础设施 flake(Exit 22)未通过,与代码改动无关。

## 根因

按正文披露:AITER MLA 内核的 block size 是 1(每页恰好 1 个 token),但解码元数据构建时把 `paged_kv_last_page_len` 错误地设成了完整序列长度:

```python
paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device)
```

对 127 个 token 的序列,这会把 `last_page_len` 设为 127——告诉内核最后一页有 127 个 token,而实际只有 1 个。

## 修复方案

采用 @ganyi1996ppo 的建议,在 `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` 中:`__init__` 预先把持久缓冲 `self.paged_kv_last_page_len` 初始化为全 1(`torch.ones(max_num_reqs, ...)`),`_build_decode` 里直接切 `[:num_reqs]` 使用,eager 与 cudagraph 两种模式复用同一缓冲,并删除 cudagraph 路径里原来的 `copy_`/`fill_(1)` 逻辑。正文还给出与 FlashInfer 后端的对照:其用 `seq_lens_np % page_size` 计算,对 `page_size=1` 恒得 0 再变 1,与本修复直接置 1 等价且更高效。

关键 diff 摘录:

```diff
-        paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device)
+        # kernel block size is always 1, so each page has exactly 1 token.
+        # last_page_len is always 1 - just slice the pre-initialized buffer.
+        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅 ROCm AITER MLA 解码路径。last_page_len 从动态计算变为恒 1,与内核 block size=1 的语义一致;cudagraph 路径少了两次缓冲写入操作。若未来该内核支持其他 block size,预置全 1 的假设需重新审视。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
