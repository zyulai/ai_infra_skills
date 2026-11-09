# PR #53939: [Bugfix][Rust Frontend] Fix LogprobsTensors wire schema mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53939](https://github.com/vllm-project/vllm/pull/53939) |
| 作者 | jungjiyu |
| 合并日期 | 2026-08-26 |
| labels | bug, rust, verified |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

#52242 给 Python `LogprobsTensors` 加第五个位置字段,NamedTuple 序列化成 5 元而 Rust `WireLogprobs` 只收 4 元解码失败;补 `serde(default)` 字段修复。

## 问题现象

Rust 侧(engine-core-client)解码 Python logprobs 线格式载荷失败,报 `Decode { target_type: "...EngineCoreOutputs", message: "array had incorrect length, expected 4; ..." }`。该失败出现在 CI 中(H200 E2E 未在本地运行,作者用 Rust 回归测试复现同样的 wire decode 失败)。作者核对其他 Rust 元组类型:`EngineCoreOutputs` 8/8、`EngineCoreOutput` 17/17、`UtilityOutput` 3/3 均匹配,`WireLogprobs` 是唯一 4/5 不匹配的。

## 根因

正文披露:#52242 把 `cu_num_generated_tokens_tensor` 加为 Python `LogprobsTensors` 的第五个位置字段;NamedTuple 序列化包含所有位置字段(即使值为 None),序列化结果变成 5 个元素,而 Rust `WireLogprobs` 在 `Deserialize_tuple` 下仍期望 4 个元素,解码即失败。

## 修复方案

`rust/src/engine-core-client/src/protocol/logprobs/wire.rs`:`WireLogprobs` 增加尾部 `#[serde(default)] pub cu_num_generated_tokens_tensor: Option<OpaqueValue>`,兼容既有 4 字段载荷;`logprobs.rs` 的 `resolve()` 中拒绝非 None 的第五值(错误信息不含载荷内容,因其可能内嵌完整 tensor blob),并在文档中写明位置式线格式兼容要求(新增 Python 字段须同变更内在 Rust 尾部加匹配的 default 字段)。新增 2 个回归测试(第 5 位 Nil 可解码、非 Nil 被拒)。与 #53886(回退 #52242)不同,本 PR 保留 #52242 并做其推荐的更窄 Rust 侧修复。

```diff
+    #[serde(default)]
+    pub cu_num_generated_tokens_tensor: Option<OpaqueValue>,
 }
```

## 影响与回归风险

- 影响窗口:#52242 合并之后(引入第五字段)。
- 回归面:仅 Rust 侧 wire 解码(logprobs.rs、wire.rs 及测试);正文声明不改变模型计算与 logprob 数值;Dispatcher decode-error 处理与 CI 依赖覆盖超出本 PR 范围,由 #53932 跟踪。

## 关联

#53932、#52242、#53886

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
