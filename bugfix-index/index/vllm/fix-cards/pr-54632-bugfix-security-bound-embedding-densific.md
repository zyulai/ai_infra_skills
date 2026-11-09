# PR #54632: [Bugfix][Security] Bound embedding densification before to_dense()

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54632](https://github.com/vllm-project/vllm/pull/54632) |
| 作者 | LeizhenZhang |
| 合并日期 | 2026-09-01 |
| labels | bug, documentation, ready, multi-modality |
| 热度 | 👍 6 · 💬 8 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
客户端稀疏张量可声明超大形状令 to_dense() 巨额分配(DoS);新增 safe_to_dense 在分配前按字节上限拒绝。

## 问题现象
客户端提供的 embedding 载荷反序列化后立即 densify:`weights_only=True` 阻止任意代码执行、invariant 检查拒绝越界索引,但都不限制声明的 shape 本身——一个仅含 1 个非零元素、声明 `(2**20, 2**20)` 的稀疏 COO 张量完全合法,序列化约 1.5 KB,`to_dense()` 会申请 4 TiB(该尺寸下分配器直接拒绝;危害在"装得下"的形状:分配由声明形状而非请求限定,攻击者可选形状,数个接近主机内存的并发请求即可造成瘫痪)。实测:(20000,20000) 稀疏张量 1 个存储元素,载荷 1560 字节,RSS 337MB→1866MB(+1.49 GiB);(30000,30000) 形状 2612 字符 base64 花费 3.4 GiB。入口:`/v1/chat/completions` 的 image_embeds / audio_embeds / video_embeds(需 `--enable-mm-embeds`)与 `/v1/completions` 的 prompt_embeds(需 `--enable-prompt-embeds`),均为 opt-in;dict 形式 image_embeds 可一次携带多个载荷。

## 根因
正文:检查顺序缺陷——内存分配发生在 `to_dense()`,而所有形状校验(`ImageEmbeddingItems`/`AudioEmbeddingItems` 的 ndim、`safe_load_prompt_embeds` 的 hidden_size/dim())都排在 `to_dense()` 之后运行;invariant 检查只管索引是否越界,不管声明形状,因此合法张量可携带任意大的声明形状造成内存放大。

## 修复方案
`vllm/utils/sparse_utils.py` 新增 `safe_to_dense(tensor, *, parameter)`:先校验反序列化产物确为 `torch.Tensor`(三个 media IO 类此前缺此检查,非张量载荷会以 AttributeError 变成 500),再在 `to_dense()` 分配之前检查 `numel() * element_size()` 是否超过 `VLLM_MAX_EMBED_DECODE_BYTES`(默认 2 GiB,设 0 禁用),超限抛 `VLLMValidationError` 使客户端得到 4xx。全部 7 处 `to_dense()` 调用点(audio/image/video 三个 `*EmbeddingMediaIO` 与 `safe_load_prompt_embeds`)改走该函数;`docs/usage/security.md` 增补该环境变量条目;新增 `tests/multimodal/test_embedding_decode_limit.py`(16 passed)。

```diff
+    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
+    if max_bytes > 0:
+        dense_bytes = tensor.numel() * tensor.element_size()
+        if dense_bytes > max_bytes:
+            raise VLLMValidationError(
+                f"`{parameter}` declares shape {tuple(tensor.shape)}, which "
+                f"would allocate {dense_bytes} bytes "
+                f"({dense_bytes / MiB_bytes:.0f} MiB) once densified, "
+                f"exceeding the {max_bytes} byte limit. Set "
+                f"VLLM_MAX_EMBED_DECODE_BYTES to increase this limit.",
+                parameter=parameter,
+                value=dense_bytes,
+            )
+
+    return tensor.to_dense()
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:所有客户端 embedding 解码路径受 2 GiB 上限约束(正文称 2 GiB 恰为 128K token float32 hidden_size=4096 的 prompt_embeds 开销,再低会误伤真实请求);非张量载荷从 500 变 4xx;面向不可信客户端的部署可调低 `VLLM_MAX_EMBED_DECODE_BYTES`,设 0 关闭检查。另注:正文报告 2 个既有测试失败(test_empty_prompt、test_out_of_vocab_token_ids)系本地环境 `FileNotFoundError: ... 'vllm'`,与本改动无关。

## 关联
正文未列出(作者检索 open/closed PR 与 issue 未发现重叠)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
