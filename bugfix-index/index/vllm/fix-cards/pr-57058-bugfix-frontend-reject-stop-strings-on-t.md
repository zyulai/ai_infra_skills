# PR #57058: [Bugfix][Frontend] Reject stop strings on --tokens-only servers instead of silently ignoring them

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57058](https://github.com/vllm-project/vllm/pull/57058) |
| 作者 | shimib |
| 合并日期 | 2026-09-17 |
| labels | bug, frontend, ready |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

--tokens-only 服务器在请求校验后才强制 `detokenize=False`,client 的 stop 字符串被静默忽略;在 `serve_tokens` 校验处返回 400 拒绝该组合。

## 问题现象

--tokens-only 服务器上,客户端传的 stop 字符串被静默忽略:生成一直跑到 EOS 或 `max_tokens`,没有错误、警告或日志。因 `stop_token_ids` 仍然有效(无需反解码),问题容易被漏看。本 PR 部分解决 #57052(--tokens-only 一侧;derender 侧 stop 字符串截断是另一个设计讨论,仍在该 issue 跟踪)。

## 根因

正文披露:--tokens-only 服务器在请求已经通过校验之后设置 `sampling_params.detokenize = False`,绕过了 `SamplingParams` 中「无 detokenize 时禁止 stop 字符串」的守卫;下游无 tokenizer 的 `IncrementalDetokenizer` 不会调用 `check_stop_strings`,stop 字符串于是从不生效。#46310 守卫的是 CLI 层 `--skip-tokenizer-init` 路径,拦不住这里「请求本身校验通过、之后才被翻转」的情况。

## 修复方案

`vllm/entrypoints/scale_out/token_in_token_out/serving.py` 的 `serve_tokens`:当 `self.force_no_detokenize` 且 `sampling_params.stop` 非空时返回 400,错误信息提示在 coordinator 侧检查 stop 字符串或改用 `stop_token_ids`,与客户端自行设 `detokenize=False` 时得到的错误一致。新增测试 `test_stop_strings_rejected_when_tokens_only`(stop → 400;stop=null 同服务器成功)。

```diff
+        if self.force_no_detokenize and sampling_params.stop:
+            # SamplingParams rejects stop with detokenize=False at request
+            # validation, but this server forces detokenize=False afterwards,
+            # so the combination must be rejected here or stop strings are
+            # silently never applied.
+            return self.create_error_response(
+                "stop strings are not supported on a --tokens-only server "
+                "because detokenization is disabled. Check stop strings on "
+                "the coordinator, or use stop_token_ids."
+            )
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 scale-out token_in_token_out 服务器的请求校验;被接受的请求输出 byte-identical(正文声明);默认(非 --tokens-only)服务器的既有 guard 不受影响(回归测试通过);Rust 前端不支持 --tokens-only,新测试相应 skipif。

## 关联

#57052、#46310

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
