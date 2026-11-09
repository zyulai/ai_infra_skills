# PR #26355: API Perf: Replace pydantic per-element validation with C loop validation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Jialin |
| 合并 | 2026-05-27 |
| 分类 | Other |
| Label | npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 4 个文件,其中 3 个新文件(validator、bench、test) |
| 链接 | https://github.com/sgl-project/sglang/pull/26355 |

## 一句话结论

把 FastAPI 请求体里 `input_ids` 的 pydantic 逐元素校验换成基于 `array('q')` 的 C 循环校验,100K token 校验 50ms → 2ms(25x)。

## 动机

Breakdown 分析显示,FastAPI 触发的 pydantic per-element validation 在长 prompt 上很慢:`GenerateReqInput.input_ids` 类型为 `Optional[Union[List[List[int]], List[int]]]`,pydantic 默认校验器会逐元素做类型检查,100K 输入 token 时校验耗时约 50ms,全部落在 HTTP 请求处理的关键路径上。这是纯 API 层开销,与模型推理无关,但对长上下文 / agentic 高流量场景直接推高 TTFT。

## 优化点

新增 `field_validators.py` 提供轻量校验器并经 `pydantic.PlainValidator` 接线,替换 `input_ids` 字段的默认校验:`validate_optional_list_i64_1d_2d` 按首元素类型分派——int 走 1d 路径、list 走 2d 逐行校验;核心是 `array("q", v)` 一次调用让 CPython 的 C 循环完成全部元素的 int64 范围/类型检查(TypeError/OverflowError 转为 ValueError),不再逐元素 Python 循环;首元素快路径 + 空列表/None 直通。语义收窄:pydantic 原校验接受任意 int 子类型并可隐式转换,新校验要求元素为 int 且落在 int64 范围(float、str、超界值直接拒绝,2D 时报行号)。同时提供 `benchmark/io/bench_input_ids_validator.py` 微基准与注册进 CPU CI 的单元测试。

## 关键改动

```text
python/sglang/srt/utils/field_validators.py: 新文件,validate_list_i64_1d / validate_optional_list_i64_1d_2d
python/sglang/srt/managers/io_struct.py: GenerateReqInput.input_ids 换成 Annotated[..., PlainValidator(...)]
benchmark/io/bench_input_ids_validator.py: 新文件,pydantic vs C-loop 校验微基准
test/registered/unit/utils/test_field_validators.py: 新文件,边界值/拒绝路径单元测试(base-a-test-cpu 套件)
```

关键机制(C 循环校验):

```python
if not isinstance(v[0], int):
    raise ValueError(f"elements must be int; got {type(v[0]).__name__}")
try:
    array("q", v)
except (TypeError, OverflowError) as e:
    raise ValueError(f"contains non-int64 element: {e}") from None
return v
```

## 性能声明

PR 正文微基准(`bench_input_ids_validator.py`,30 次取均值,input_ids 为 1D int 列表):

| n_tokens | pydantic 默认 (ms) | C-loop validator (ms) |
| --- | --- | --- |
| 1,000 | 0.52 | 0.02 |
| 10,000 | 4.97 | 0.19 |
| 100,000 | 50.77 | 1.91 |
| 1,000,000 | 503.65 | 19.85 |

自述 "25x faster (e.g. 100K input tokens, 50ms -> 2ms)"。PR 未提供端到端(带服务)的 TTFT 对比数据。

## 验证与回归风险

`test_field_validators.py` 覆盖:int64 边界(±2^63)、None/非 list/浮点/字符串/溢出拒绝、2D 空行、错误信息带行号;CI PR Test Base 通过,Extra 失败(x)。风险点:校验语义比 pydantic 默认更严格——此前能通过的 float(整数值)或大整数现在报错,属有意收窄但可能影响极端客户端;2D 路径仍对每行做一次 `array('q')` 构造,行数极多的 batch 请求收益小于 1D;只覆盖 `input_ids` 字段,`input_embeds` 等其他大字段未替换。

## 关联

同区域(API/IO 层优化):#17575(model gateway hashring)、#26355 与 #8577(禁 GC)同属请求处理路径;pack 中未列出其他 validator 相关 PR,写"无"级别的直接关联。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
