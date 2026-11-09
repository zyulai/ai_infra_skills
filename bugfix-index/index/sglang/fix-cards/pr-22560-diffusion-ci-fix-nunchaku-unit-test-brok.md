# PR #22560: [Diffusion][CI] Fix nunchaku unit test broken by #22365

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22560](https://github.com/sgl-project/sglang/pull/22560) |
| 作者 | alisonshao |
| 合并日期 | 2026-04-11 |
| labels | high priority, quant, diffusion |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#22365 使 nunchaku 单测把假 /tmp 路径当 HF 仓库下载;测试中 mock 掉 maybe_download_model。

## 问题现象
multimodal-gen 单测 `test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook` 在 CI 失败(正文附失败示例:[Run 24257571982 / job 70866905056](https://github.com/sgl-project/sglang/actions/runs/24257571982/job/70866905056))。原因:测试使用不存在的假本地路径 `/tmp/svdq-int4_r32.safetensors`,而 #22365 新增的 `_resolve_quant_config_from_transformer_override` 会对 transformer weights path 调用 `maybe_download_model`,导致其尝试把该假路径当作 HF repo 下载,HF Hub 校验失败。Test plan 为 `multimodal-gen-unit-test` CI job 通过。

## 根因
正文披露:#22365 添加的 `_resolve_quant_config_from_transformer_override` 会调用 `maybe_download_model`,而该测试用例使用不存在的 `/tmp` 假路径,无法通过 HF Hub 校验,属 #22365 引入的测试回归。

## 修复方案
`python/sglang/multimodal_gen/test/unit/test_transformer_quant.py`:为该测试新增一层 `@patch`,把 `sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model` mock 为直接返回入参路径(`side_effect=lambda path, **kw: path`),并在测试签名中新增 `_mock_download` 参数。

```diff
+    @patch(
+        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
+        side_effect=lambda path, **kw: path,
+    )
     def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
         self,
+        _mock_download,
         mock_metadata,
         _mock_quant_metadata,
         _mock_nvfp4,
```

## 影响与回归风险
- 回归面:仅改单测的 mock,不触碰生产代码;该测试不再触发真实下载/HF Hub 校验,quant load spec 与 nunchaku hook 的解析逻辑仍被覆盖;代价是 `maybe_download_model` 的真实下载行为在该用例中不再被覆盖。

## 关联
- [#22365](https://github.com/sgl-project/sglang/pull/22365)(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
