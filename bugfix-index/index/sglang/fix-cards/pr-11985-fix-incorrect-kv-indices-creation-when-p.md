# PR #11985: Fix incorrect KV indices creation when page_size=32 in TRTLLM MLA backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11985](https://github.com/sgl-project/sglang/pull/11985) |
| 作者 | cicirori |
| 合并日期 | 2025-10-23 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
TRTLLM MLA 后端 page_size=32 时 KV indices 循环次数按 token 而非 page 计算,索引不完整;改为按 page_size 动态求 NUM_PAGE_PER_BLOCK。

## 问题现象
TRTLLM MLA 后端在 `page_size=32` 时,`create_flashmla_kv_indices_triton` 生成的 KV indices 不完整(KV indices 被 incompletely processed),导致 KV cache 行为不正确。`page_size=64` 时无此问题。正文 Testing 一节记录:page size 32 现在能创建正确的 KV indices;page size 64 行为不变;测试验证所有需要的 block 都被处理。

## 根因
正文明确披露:`create_flashmla_kv_indices_triton` 的迭代次数按 token 而不是按 page 计算,循环计数使用了固定的 `BLOCK_SIZE`(4096 个 token):

```python
num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)  # Wrong: uses tokens
```

这只在 `page_size=64` 时碰巧正确(64 pages × 64 tokens/page = 4096 tokens);`page_size=32` 时 64 pages × 32 tokens/page = 2048 tokens ≠ 4096,循环次数不足,KV indices 处理不完整。

## 修复方案
1. `python/sglang/srt/layers/attention/utils.py`:删除硬编码的 `TRITON_PAD_NUM_PAGE_PER_BLOCK = 64`,新增 `_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096` 与 `get_num_page_per_block_flashmla(page_size)`(返回 `4096 // page_size`,保持每轮迭代处理 4096 个 token);kernel 内改为动态计算 `NUM_PAGE_PER_BLOCK = FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE`,不再接受外部传入参数。
2. `python/sglang/srt/layers/attention/trtllm_mla_backend.py`:`_calc_padded_blocks` 的约束 LCM 改用动态的 `get_num_page_per_block_flashmla(self.page_size)`;三处调用点(`_create_block_kv_indices`、cuda graph capture/replay)移除硬编码的 `NUM_PAGE_PER_BLOCK=TRITON_PAD_NUM_PAGE_PER_BLOCK` 参数。
3. `python/sglang/test/attention/test_trtllm_mla_backend.py`:补充测试 setup(全局 server args)与校验。

```diff
-    BLOCK_SIZE: tl.constexpr = 4096
+    NUM_PAGE_PER_BLOCK: tl.constexpr = (
+        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
+    )
```

diff 中还顺带把 `flashinfer_mla_backend.py` 里 `SGLANG_ENABLE_TORCH_COMPILE` 的读取从裸 `os.environ[...]` 改为 `envs.SGLANG_ENABLE_TORCH_COMPILE.get()`(避免未设置该环境变量时 KeyError)。

## 影响与回归风险
- 回归面:影响所有使用 `create_flashmla_kv_indices_triton` 的路径(TRTLLM MLA 后端及其继承的 flashinfer MLA 后端)的 KV indices 生成与 `_calc_padded_blocks` padding 计算;正文声明无性能回退(每轮迭代仍处理 4096 token)、无 breaking change;page_size=64 行为不变,page_size=32 由错转对。kernel 签名移除了 `NUM_PAGE_PER_BLOCK` 参数,外部若有直接调用需同步。

## 关联
正文未列出
