# PR #55485: [MyPy][4/N] Fix mypy errors in medium-hard tests/ dirs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55485](https://github.com/vllm-project/vllm/pull/55485) |
| 作者 | theSatvik |
| 合并日期 | 2026-10-20 |
| labels | ready, ray, tool-calling, qwen, deepseek, gpt-oss, nvidia, mistral, dflash, kimi, k3, glm, minimax, inkling, cohere, DSv4.1, pooling |
| 热度 | 👍 0 · 💬 32 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
修复 tests/ 九个中高难度目录全部 511 个 mypy 错误并纳入强制检查,顺带清理类型检查暴露的死代码;声明无行为变化。

## 问题现象
这是 #49569 系列的第 4 个 PR,覆盖 "medium-hard" 测试目录:这些目录在更严格的 `follow_imports=silent` mypy 设置下共有 511 个类型错误——`tests/v1/kv_offload` 232、`tests/v1/worker` 94、`tests/parser` 64、`tests/model_executor`(含 model_loader、kernels)49、`tests/renderers` 28、`tests/entrypoints/speech_to_text` 20、`tests/v1/metrics` 13、`tests/v1/streaming_input` 11(`tests/models/test_initialization.py` 为 0)。属类型检查/工程质量问题,不影响运行时功能。

## 根因
非单一根因,为批量类型错误;但类型检查暴露出若干真实缺陷,正文列明:1) `RoutingMethodType.FUSED_TOPK` 不存在,读取 `DummyRouter.routing_method_type` 会抛 AttributeError(FusedMoE 确会读该属性),改为 `TopK`;2) `tests/v1/worker/test_utils.py` 存在上游重构遗留的死 setup 写入(`worker.refines_row_mirrors`、`worker._layer_mirror_callbacks` 在 vllm/ 中不存在),已移除;3) `test_gpu_autoregressive_speculator.py` 的死 observer 属性,已移除;4) `CPUExpertsFp8`→`CPUExpertsInt4`(#59455 已在 main 独立修复,不在本 diff);5) `test_obj_tier.py` 的 `lookup_and_wait` 注解 `-> list[bool]` 与实际返回 `list[LookupResult]` 不符;6) `test_factory.py` 缺少兄弟测试都有的 `assert isinstance(spec, CPUOffloadingSpec)`;7) `DUMMY_TOOLS` 原始 dict 不经 pydantic 校验的问题被发现但未在本 PR 修改。

## 修复方案
1. 修复上述 9 个目录全部 mypy 错误(511→0),手法按优先级:runtime-true narrowing(`assert x is not None`/`assert isinstance`)、按文件修正 stub、仅必要时 `MagicMock(spec=...)`、带理由的 scoped `# type: ignore[code]`;全程不新增 `cast()`。
2. 将这些目录从宽松组迁入 `tools/pre_commit/mypy.py` 的 `SILENT_GROUPS`,使修复被 pre-commit 强制执行(新增 tests/parser、tests/renderers、tests/v1/kv_offload、tests/v1/metrics、tests/v1/worker 等,合并 tests/model_executor 为单条目)。
3. 顺带修正 vllm/ 注解:`RayPrometheusMetric.labels() -> Self`、`TokenizeParams.apply_post_tokenization` 加 @overload、`AudioAssetName` 增加 `"azacinto_foscolo"`、`StreamingInput.prompt: PromptType | EngineInput`、`EMABackpressureDetector.default_config` 收窄为 `-> dict`、`TritonKernelDispatcher` 协议补 `get_warmup_keys`/`compile`/`__name__`。
注:pack 的 diff 在 40000 字符处按文件粒度截断,tests/ 下其余文件的改动未在素材中展示。

## 影响与回归风险
- 影响窗口:正文未说明(针对当前 main `10f6ab01a` 上 511 个错误)。
- 回归面:正文明确「Behaviour changes: None」,被质疑的三处(DUMMY_TOOLS、mock_request、block_ids)已回退;vllm/ 侧仅注解、overload、Literal/Union 类型与返回类型注解变化,均为类型层;测试侧为 mock 写法(monkeypatch.setattr)与断言收窄。CPU 验证中 89 个失败与 main 完全一致(需 CUDA/编译算子),无本分支新增失败。

## 关联
- 系列总 issue:https://github.com/vllm-project/vllm/issues/49569 (认领 comment: issues/49569#issuecomment-5552556980)
- https://github.com/vllm-project/vllm/pull/51043
- https://github.com/vllm-project/vllm/pull/59455

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
