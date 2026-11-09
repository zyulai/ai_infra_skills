# PR #29826: [Perf] Avoid pageable HtoD transfer in MinTokensLogitsProcessor

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jthomson04 |
| 合并 | 2025-12-02 |
| 分类 | Sampler / Detokenization |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供);diff 可用,仅涉及 `vllm/v1/sample/logits_processor/builtin.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/29826 |

## 一句话结论

消除 MinTokensLogitsProcessor 引发的 pageable HtoD 同步传输,修复 min_tokens 请求在低并发下 TPOT 最多升高 30% 的问题。

## 动机

`MinTokensLogitsProcessor.apply` 里 `logits[self.logits_slice] = -float("inf")` 会触发一次 pageable host-to-device 拷贝,且该拷贝相对设备是同步的。在 `--async-scheduling` 下,这种同步行为会打断调度器与模型执行的重叠(scheduler 不再能被 overlap),导致带 min_tokens 的请求在低并发时 TPOT 显著升高(正文引用 up to 30%)。作者附了 nsys trace 对比图(无 min_tokens vs 有 min_tokens)佐证。

## 优化点

两处赋值改为设备端操作,根因是"用 host 标量做 tensor 索引赋值会隐式发起 H2D 拷贝":

1. `MinTokensLogitsProcessor`:在 `__init__` 预建常驻设备端的 `neg_inf_tensor`(float32、-inf),`apply` 中把 Python 标量赋值改为 `logits.index_put_(self.logits_slice, self.neg_inf_tensor)`,拷贝源变为 GPU 内存,不再有 pageable HtoD 传输。
2. `MinPLogitsProcessor`:顺带把布尔索引赋值 `logits[invalid_token_mask] = -inf` 改为 `logits.masked_fill_(invalid_token_mask, -inf)`,避免同类隐式拷贝路径。

采样语义不变,只是写 -inf 的执行方式变为纯设备端 kernel。

## 关键改动

```text
vllm/v1/sample/logits_processor/builtin.py: MinTokens/MinP logits processor 改用设备端写 -inf
```

```python
+        self.neg_inf_tensor = torch.tensor(
+            -float("inf"), dtype=torch.float32, device=self.device
+        )
...
     def apply(self, logits: torch.Tensor) -> torch.Tensor:
         if self.min_toks:
-            logits[self.logits_slice] = -float("inf")
+            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
```

## 性能声明

PR 正文只给出问题侧数字:带 min_tokens 的请求在低并发下 TPOT 增加"up to 30%"(使用 `--async-scheduling` 的场景),并附 nsys trace 截图展示同步 HtoD 打断 overlap。修复后的端到端复测数字未提供(Test Result 一节为空),即修复后具体 TPOT 恢复幅度 PR 未注明。

## 验证与回归风险

Test Plan 写明:沿用 `MinTokensLogitsProcessor` 已有测试,认为足够,不新增测试。影响路径:min_p、min_tokens 两个 logits processor 的所有用户(带 min_tokens/min_p 的请求);`is_argmax_invariant` 语义未变(greedy 下 min-tokens 仍会改变 argmax 结果)。`index_put_`/`masked_fill_` 与索引赋值在数值上等价,主要回归面是这两个算子在不同后端/硬件上的行为一致性,PR 未做额外硬件验证。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
