# PR #7724: [Multimodal][Perf] Use `pybase64` instead of `base64`

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2025-07-08 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | performance |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 内容可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/7724 |

## 一句话结论

用 SIMD 加速的 pybase64 替换标准库 base64,多模态服务 Mean TTFT −49%、P99 TTFT −77%。

## 动机

多模态请求的图像/音频/视频以 base64 字符串传输,服务端编解码此前走纯 Python 标准库 `base64`,是明显的 CPU 开销点。pybase64 提供 SIMD 加速的 C extension,其 benchmark 显示相比标准库解码最高快 28×、编码最高快 9×,值得整体替换。

## 优化点

把代码中所有 `base64` 的 import 与调用点替换为 `pybase64`,保留标准与 URL-safe 接口行为;在适用的解码处加 `validate=True`(作者称此举是为"maximal performance")。覆盖面包括:`srt/utils.py` 的 `decode_video_base64`、`load_audio`、`load_image`、进程间 `serialize`/`deserialize`;`srt/multimodal/mm_utils.py` 的 `load_image_from_base64`;`sglang/utils.py` 的 `encode_image_base64`、`encode_video_base64`;`srt/entrypoints/http_server_engine.py`;以及 `bench_serving.py` 的 MMMU 采样路径。同时在 `python/pyproject.toml` 把 `pybase64` 加入 `runtime_common` 依赖组。

## 关键改动

```text
python/sglang/srt/utils.py: 视频/音频/图像 base64 解码与 serialize/deserialize 换 pybase64(validate=True)
python/sglang/srt/multimodal/mm_utils.py: load_image_from_base64 换 pybase64
python/sglang/utils.py: encode_image_base64 / encode_video_base64 编码换 pybase64
python/pyproject.toml: runtime_common 新增 pybase64 依赖
python/sglang/bench_serving.py: MMMU 数据集采样图像编码换 pybase64
```

## 性能声明

测试条件:PR 注明在 H100 80GB HBM3 + Sapphire Rapids CPU 上,`bench_serving.py --model Qwen/Qwen2.5-VL-7B-Instruct --backend sglang --dataset-name mmmu --request-rate 10 --num-prompts 100`,98 个成功请求。结果:

- Mean TTFT 99.84→50.87 ms(−49.06%);Median TTFT 51.62→47.37 ms(−8.23%);P99 TTFT 563.09→127.30 ms(−77.39%)。
- Max ITL 645.31→181.86 ms(−71.83%);Mean/Median ITL 约 −1.8/−1.9%。
- Mean E2E 15275.51→14942.63 ms(−2.18%),Median E2E −2.44%。
- 吞吐:Request throughput 4.26→4.30 req/s(+0.94%),Input/Output/Total token throughput 各 +0.84%;Concurrency −1.35%。
- pybase64 本身的 28× 解码/9× 编码加速引自其上游 benchmark,非本 PR 端到端数字。

## 验证与回归风险

PR 附 `python -m pytest srt/test_vision_openai_server_common.py -vv` 结果:7 项测试全部 PASSED(覆盖 audio、multi-images、mixed batch、regex、single image、video chat completion)。回归风险:`validate=True` 会对输入做严格 base64 校验,非法字符会抛错,与标准库默认宽松行为有差异;新增 pybase64 运行时依赖;编码/解码结果需逐字节一致(标准 alphabet 下无差异)。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
