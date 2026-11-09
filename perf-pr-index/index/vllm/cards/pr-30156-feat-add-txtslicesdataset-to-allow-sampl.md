# PR #30156: feat: add TxtSlicesDataset to allow sampling slices from txt file for benchmarking

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jdebache |
| 合并 | 2026-04-14 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, performance, frontend, ready |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 2 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/30156 |

## 一句话结论

新增 TxtSlicesDataset 基准数据集类型:从用户提供的 txt 文件按 token 切片采样 prompt,兼顾随机数据集的长度可控性与真实语料的分布真实性,补齐投机解码/EP 场景的 benchmark 输入。

## 动机

直接从 tokenizer 随机生成 token 做基准测试,产生的是无语义的伪文本;这类输入对 speculative decoding 或 expert parallelism 场景不理想——投机解码的接受率、MoE 的专家路由都依赖真实 token 分布,随机 token 会给出失真的性能画像。而随机数据集的优势是完全可控的输入/输出序列长度与可复现性。本 PR 引入 `TxtSlicesDataset` 作为两者之间的折中:从真实文本文件随机切片,同时保持长度参数化。

## 优化点

机制上分三部分:(1) 新增 `create_txt_slices_dataset.py` 脚本:把本地或 URL 的纯文本 tokenize 后,按 `--input-len`/`--output-len`(可配 `--range-ratio`)随机选起点切片(切片越过文本末尾时循环取模),decode 成 prompt 并连同 `output_tokens` 写成 JSONL,供 `--dataset-name custom` 的 serving benchmark 直接使用;种子化保证可复现。(2) `datasets.py` 重构:把 `get_sampling_params` 从 `RandomDataset` 抽到共享 `vllm/benchmarks/datasets/utils.py`,新增 `RangeRatio` 类型——单一 float 同时作用于 ISL/OSL,JSON dict(`{"input": 0.3, "output": 0.5}`)可分别设置,CLI `--random-range-ratio` 相应改为 str 解析;`SampleRequest` 的 `expected_output_len` 放宽为可 None、`prompt` 支持 list[dict];oversample 从 `deepcopy` 改为 `dataclasses.replace`(只改 request_id);统一各 dataset `sample()` 签名与类型标注。(3) 新增 `tests/benchmarks/test_sampling_params.py` 单元测试与新的数据集类型注册。

## 关键改动

```text
vllm/benchmarks/datasets/create_txt_slices_dataset.py: txt -> JSONL 切片采样脚本(tokenize 后随机起点循环切片)
vllm/benchmarks/datasets/utils.py: 共享 get_sampling_params 与 RangeRatio(ISL/OSL 独立 ratio)
vllm/benchmarks/datasets/datasets.py: 各 dataset 采样逻辑复用共享工具、typing 修复
vllm/benchmarks/datasets/__init__.py: datasets 包化并集中导出
tests/benchmarks/test_sampling_params.py: 采样参数单元测试
```

关键片段(切片与循环取模):

```python
start_pos = rng_py.randint(0, num_available_tokens - 1)
prompt_token_ids = [
    token_ids[(start_pos + j) % num_available_tokens]
    for j in range(req_input_len)
]
prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
```

## 性能声明

PR 正文未提供 benchmark 数字(此 PR 是基准工具链增强,本身不产生性能数据;perf claims 检测为 none)。其价值在于让 speculative decoding / EP 的基准输入更接近真实分布。

## 验证与回归风险

新增 `tests/benchmarks/test_sampling_params.py`,用 FakeTokenizer 覆盖 `get_sampling_params` 的区间计算、ratio 校验与 dict 形式 ratio。回归风险:`--random-range-ratio` CLI 从 float 改为 str(接受 float 或 JSON dict),下游解析方式变化;`datasets.py` 大面积签名/类型调整波及所有 dataset 子类的 `sample()`;`expected_output_len` 可为 None 可能影响依赖该字段的统计路径。TxtSlicesDataset 类本体的注册文件在 diff 截断部分,素材中未见其实现细节。

## 关联

与 vLLM benchmark 自带的 RandomDataset/SonnetDataset/CustomDataset 同属基准数据集体系(本 PR 重构其共享采样逻辑)。索引内同分类(Speculative Decoding)卡片:#13365(n-gram 匹配优化)、#14930(rejection sampling 优化)、#25784(suffix decoding);PR #34109 的 benchmark 命令即使用 sonnet.txt 数据集。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
