# PR #56090: [Frontend][EPD] Use JSON arrays for multimodal metadata

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gty111 |
| 合并 | 2026-09-10 |
| 分类 | Overlap / Scheduler |
| Label | documentation, frontend, ready, kv-connector |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/56090 |

## 一句话结论

EPD 占位元数据从 torch.save+base64 改传 JSON 数组,免去序列化开销:MUIRBench 对比吞吐 60.48 → 62.05 req/s。

## 动机

EPD 场景下 encoder 向 decoder 传递的小型占位元数据(如 `image_grid_thw`)此前要经 `torch.save`/`torch.load` 加 base64 编解码,对几个数字的数组而言纯属序列化开销,且下游必须依赖 PyTorch 才能解码;vllm-project/router 与 Rust frontend(#55842)未来要支持 EPD,需要语言中立的元数据格式。本 PR 是从 #55842 中抽出的 Python-only 前置。

## 优化点

前端接收侧:`image_embeds`/`audio_embeds`/`video_embeds` 的 dict 值类型扩展为 `MultiModalEmbedsPayload = str | dict[str, str | list[int | float]]`,允许数值 JSON 数组;`_parse_metadata_array` 只接受模型声明的 metadata 字段(`placeholder_metadata_fields`),经 `is_list_of_numbers`(排除 bool、NaN/Inf、嵌套 list)校验后,含 float 用 float64、否则 int64 建 tensor;legacy base64 字符串路径与 proxy 的 raw-media 路径完全保留。代理侧 `disagg_epd_proxy.py` 直接删除 `_b64_tensor`(torch.save+base64),metadata 以扁平 JSON 数组原样下发;`_load_embeds_dict` 统一处理 str(异步 fetch base64 tensor)与数组(直通)混合的 dict。

## 关键改动

```text
vllm/entrypoints/chat_utils.py: MultiModalEmbedsPayload 类型、_parse_metadata_array、_load_embeds_dict
vllm/utils/collection_utils.py: is_list_of_numbers 校验
examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py: 删除 _b64_tensor,JSON 数组直发
tests/entrypoints/unit_tests/test_chat_utils.py + tests/v1/ec_connector/unit/test_epd_proxy_round_robin.py: 单测
```

```python
def _parse_metadata_array(key: str, value: list, metadata_fields: set[str]):
    if key not in metadata_fields:
        raise VLLMValidationError(f"JSON arrays are only supported for metadata: {key}")
    if not is_list_of_numbers(value):
        raise VLLMValidationError(f"Metadata {key} must be a finite numeric array.")
    dtype = torch.float64 if any(isinstance(v, float) for v in value) else torch.int64
    try:
        return torch.tensor(value, dtype=dtype)
    except (ValueError, TypeError, OverflowError, RuntimeError) as error:
        raise VLLMValidationError(f"Invalid metadata array: {key}") from error
```

## 性能声明

- MUIRBench,Qwen3.5-35B-A3B + Mooncake RDMA,4E+4PD 部署,V2,并发 96,CUDA graphs 至 16384 tokens(无 eager);200 题 warmup 后对前 1200/2600 题重复五次:Base64 → JSON:accuracy 59.02% → 59.10%,mean throughput 60.48 → 62.05 req/s(+2.59%);两臂均 0/6000 失败。
- PR 注明:两臂为不同 node pairs 上的顺序运行,吞吐差异不能视为该改动的独立因果估计。

## 验证与回归风险

`test_chat_utils.py`(image_embeds / json_metadata / json_arrays 相关 17 个)与 `test_epd_proxy_round_robin.py`(11 个)通过,pre-commit hooks 通过。新增:数值精度保留测试(int/float 精度与 dtype 选择)、非法输入校验测试(bool、NaN/Inf、嵌套数组、超界值均抛 VLLMValidationError)、JSON 数组与 legacy base64 混用测试、proxy rewrite 输出断言。风险:JSON 数组仅限模型声明的 metadata 字段,其他字段会拒绝;image/audio/video 三类 embeds 解析路径都受影响,依赖各模型 processor 正确声明 `placeholder_metadata_fields`;E2E 收益数字非受控 A/B。

## 关联

前序:#55842(Rust frontend,本 PR 为其 Python 前置)、#52409(相关)。索引同分类(frontend/E2E 开销)另有 pr-7162、pr-7957。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
