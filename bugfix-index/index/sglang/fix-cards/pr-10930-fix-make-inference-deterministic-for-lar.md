# PR #10930: fix: make inference deterministic for large TP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10930](https://github.com/sgl-project/sglang/pull/10930) |
| 作者 | JustinTong0323 |
| 合并日期 | 2025-09-27 |
| labels | run-ci |
| 热度 | 👍 5 · 💬 4 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
TP>1 时确定性推理不可用;设 NCCL_ALGO=allreduce:tree 并禁用 custom all-reduce,使大 TP 确定性推理可用。

## 问题现象
此前启用 `--enable-deterministic-inference` 仅支持 TP=1,TP>1 无法使用确定性推理(修复前代码在 tp_size>1 时直接 raise,见 diff;正文仅表述为使 TP>1 的确定性推理可用)。修复后用 `sglang.test.test_deterministic` 验证:Dense/MoE 模型 TP=2,4,8 在 fa3、flashinfer、triton 后端通过。复现残留问题的命令(正文照抄):

```
python3 -m sglang.launch_server \
    --model Qwen/Qwen3-30B-A3B \
    --port 30000 \
    --enable-deterministic-inference \
    --attention-backend fa3 \
    --tp-size 2
```

再跑 `python3 -m sglang.test.test_deterministic --test-mode prefix`,可见 `Prompt 0 with prefix length 1: total samples: xxx, Unique samples: 2`。

## 根因
正文未披露根因,仅说明修改方式。(由 diff 推断)修复前 `_handle_deterministic_inference` 在 tp_size>1 时直接抛 ValueError,未为多卡 allreduce 做任何确定性配置;新代码通过指定 NCCL tree allreduce 并禁用 custom all-reduce 获得确定性,暗示默认 allreduce 路径在 TP>1 下不保证结果一致。

## 修复方案
按正文 Modifications:Set `NCCL_ALGO="allreduce:tree"` and disable custom all-reduce when TP > 1。diff 上 `python/sglang/srt/server_args.py` 的 `_handle_deterministic_inference`:

```diff
             # Check TP size
             if self.tp_size > 1:
-                raise ValueError(
-                    "Currently only TP size 1 is supported for deterministic inference."
+                os.environ["NCCL_ALGO"] = "allreduce:tree"
+                self.disable_custom_all_reduce = True
+                logger.warning(
+                    "NCCL_ALGO is set to 'allreduce:tree' and custom all reduce is disabled for deterministic inference when TP size > 1."
                 )
```

同时移除了原先对 MoE 模型的警告;`python/sglang/test/test_deterministic.py` 把 long_prompt.txt 的读取路径改为基于 `__file__` 的绝对路径,消除对工作目录的依赖。

## 影响与回归风险
- 影响窗口:正文未给出版本/提交范围。
- 回归面:按 diff,凡 `--enable-deterministic-inference` 且 TP>1 的启动配置不再报错,而是被强制 `NCCL_ALGO=allreduce:tree` 并 `disable_custom_all_reduce=True`(性能影响正文未给数据);其余启动路径不变。已知残留:Qwen3-30B-A3B TP=2 + fa3 的 prefix 场景仍产生 2 种输出(Mixtral-8x7B-v0.1 可通过,正文 Accuracy Tests)。

## 关联
- [#10785](https://github.com/sgl-project/sglang/issues/10785)(正文 Close #10785)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
