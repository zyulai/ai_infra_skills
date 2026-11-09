# PR #10045: [Model][OpenVINO] Fix regressions from #8346

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10045](https://github.com/vllm-project/vllm/pull/10045) |
| 作者 | petersalas |
| 合并日期 | 2024-11-06 |
| labels | ready, ci/build |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#8346 使 OpenVINO 路径缺 multimodal placeholder 索引映射(#10025)、molmo dummy 数据返回元组(#10042),分别补齐修复。

## 问题现象
OpenVINO 后端在 #8346 之后出现两处回归(通过 issue 报告):#10025 与 #10042。正文未附具体报错日志;从修复内容看,前者是 OpenVINO 注意力元数据缺少多模态 placeholder 索引映射,后者是 molmo 的 dummy 数据构造返回了错误的类型。

## 根因
正文/标题披露两处回归均由 #8346 引入:
- #10025:`OpenVINOAttentionMetadata` 没有 `multi_modal_placeholder_index_maps` 字段,#8346 的改动要求把多模态 embedding 与 placeholder 的索引映射传给 `model_executable`,OpenVINO 路径缺失即失败(正文引述自 `AttentionMetadata` 的注释:该字段其实不属于 attention 元数据、更适合放在 `ModelInputs`,但作者不想进一步分叉 OpenVINO/非 OpenVINO 路径)。
- #10042:`dummy_data_for_molmo` 返回 `(dummy_seqdata, {"image": dummy_imgdata})` 元组,而接口约定应返回 `DummyData`(由 diff 推断 #8346 统一了该接口)。

## 修复方案
- `vllm/attention/backends/openvino.py`:`OpenVINOAttentionMetadata` 增加 `multi_modal_placeholder_index_maps: Optional[Dict[str, MultiModalPlaceholderMap.IndexMap]]` 字段,并附注释说明这是让该数据对 `model_executable` 可用的临时方案。
- `vllm/model_executor/models/molmo.py`:`dummy_data_for_molmo` 改为 `return DummyData(dummy_seqdata, {"image": dummy_imgdata})`。
- `.buildkite/run-openvino-test.sh`:docker 内脚本路径从 `/workspace/vllm/examples/offline_inference.py` 修正为 `/workspace/examples/offline_inference.py`。

## 影响与回归风险
- 影响窗口:#8346 合并之后。
- 回归面:OpenVINO 后端(尤其 multimodal/molmo 推理与 profiling 的 dummy 数据路径);`OpenVINOAttentionMetadata` 新增字段为带默认 Optional 语义的追加,需确认 OpenVINO 构造处已提供该映射;molmo dummy 数据改动仅影响 profiling/benchmark 路径。

## 关联
- Fixes [#10025](https://github.com/vllm-project/vllm/issues/10025)
- Fixes [#10042](https://github.com/vllm-project/vllm/issues/10042)
- 引入回归的 #8346(正文标题指明,未给链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
