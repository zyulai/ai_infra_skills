# PR #11982: bugfix: Fix signature mismatch in benchmark's `get_tokenizer` function

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11982](https://github.com/vllm-project/vllm/pull/11982) |
| 作者 | e1ijah1 |
| 合并日期 | 2025-01-13 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

benchmarks 目录的 `get_tokenizer` 签名与 vLLM 内部版本不一致,缺少 tokenizer_mode/kwargs 参数导致基准脚本调用报错;对齐签名并支持 slow/mistral 模式修复。

## 问题现象

正文仅写「FIX #11977」,未描述具体现象。可确定问题位于 `benchmarks/backend_request_func.py` 的 `get_tokenizer`:基准测试脚本按 vLLM 内部 `get_tokenizer` 的调用约定使用它时签名不匹配(标题所述 signature mismatch),导致相关 benchmark 运行报错。

## 根因

正文未披露根因(仅引用 #11977)。(由 diff 推断)benchmark 里的 `get_tokenizer` 旧签名为 `(pretrained_model_name_or_path: str, trust_remote_code: bool)`,与 vLLM 主代码 `vllm/transformers_utils/tokenizer.py` 中的同名函数不一致:不接收 `tokenizer_mode` 与任意 `**kwargs`,`trust_remote_code` 还是必填位置参数。当 benchmark 脚本(如 benchmark_serving 等)按主代码签名传入 `tokenizer_mode` 等参数时产生 TypeError;且旧实现也不支持 slow/mistral 分词器模式。

## 修复方案

修改 `benchmarks/backend_request_func.py` 的 `get_tokenizer`,对齐 vLLM 内部实现:新签名 `(pretrained_model_name_or_path, tokenizer_mode="auto", trust_remote_code=False, **kwargs)`;`tokenizer_mode == "slow"` 时强制 `use_fast=False`(若 kwargs 里传了 `use_fast=True` 则报 ValueError);`tokenizer_mode == "mistral"` 时返回 `MistralTokenizer.from_pretrained(...)`(vllm 未安装则给出安装提示);否则走 `AutoTokenizer.from_pretrained(..., trust_remote_code=trust_remote_code, **kwargs)`。关键 diff:

```diff
-    pretrained_model_name_or_path: str, trust_remote_code: bool
+    pretrained_model_name_or_path: str,
+    tokenizer_mode: str = "auto",
+    trust_remote_code: bool = False,
+    **kwargs,
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 `benchmarks/` 下的工具函数,不影响 vLLM 服务本体。`trust_remote_code` 从必填变为默认 False,若 benchmark 中有调用点依赖旧的位置传参且未显式传值,行为会从「必传」变为「默认不信任远程代码」;新增 kwargs 透传到 AutoTokenizer,极端情况下可能改变分词器构造参数。

## 关联

- FIX #11977

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
