# PR #34732: [Attention] Use FA4 for MLA prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-03-12 |
| 分类 | Attention / Kernel |
| Label | documentation, performance, ready, ci/build, v1, nvidia |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 9 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/34732 |

## 一句话结论

让 MLA prefill(`forward_mha`)可用 FA4,并把 Blackwell 默认 MLA prefill 从 TRT-LLM 切到 FA4——对 chunked prefill 的 extend 操作尤其有利。

## 动机

Blackwell 上 MLA prefill 此前默认走 TRT-LLM ragged deepseek prefill。FA4(FlashAttention 4)在 Blackwell 上对 extend 类操作(vLLM chunked prefill 的主要形态)性能优于 TRT-LLM,但依赖 flash-attention 上游的两个 PR(#123/#126)提供 diff-headdim(qk=192, v=128)支持,此前 FA4 无法用于 MLA 的 qk/v 头维不等场景。

## 优化点

vLLM 侧把 `AttentionConfig` 默认值翻转:`disable_flashinfer_prefill` 默认 True、`use_trtllm_ragged_deepseek_prefill` 默认 False,使 MLA 的 `vllm_flash_attn_version` 解析到 FA4;`mla_attention.py` 中 `_pad_v` 逻辑加入 `vllm_flash_attn_version == 4`(FA4 与 Hopper FA3 一样原生支持 qk/v 不同 head dim,免 V padding);`fa_utils.py` 放开 FA4 在 SM100 上 head_size>128 的 TMEM 限制例外:`head_size == 192`(MLA qk=192)被上游 commits 1a15733e/1b36ab19 支持。CMake 中 vllm-flash-attn FetchContent tag 升级到含 FA4 的 1488682bb。同时大幅扩展 `benchmarks/attention_benchmarks`:新增 `--prefill-backends`(fa2/fa3/fa4/flashinfer/cudnn/trtllm)与 `_PREFILL_BACKEND_CONFIG` 映射(设置 flash_attn_version 与三个 attention_config 开关),prefill 路径改走 `forward_mha`,并在批跑前清掉 `use_flashinfer_prefill` 等 `@functools.cache` 的缓存、断言实际选中的 prefill 后端与请求一致;`mla_prefill.yaml` 重写为 6 后端 x 纯 prefill/chunked extend(短/中/长上下文 1k-64k)的对比矩阵,原 sparse 配置拆到 `mla_sparse_prefill.yaml`。

## 关键改动

```text
vllm/config/attention.py: 默认关闭 flashinfer prefill 与 trtllm ragged prefill(FA4 成为默认)
vllm/model_executor/layers/attention/mla_attention.py: FA4 加入免 V padding 分支
vllm/v1/attention/backends/fa_utils.py: FA4 SM100 head_size==192 例外(MLA diff-headdim)
cmake/external_projects/vllm_flash_attn.cmake: vllm-flash-attn 升级到含 FA4/diff-headdim 的 tag
benchmarks/attention_benchmarks/{benchmark.py,mla_runner.py}: 新增 prefill 后端对比模式与配置校验
```

## 性能声明

PR 正文未给出可引用的具体数字:性能对比在一张截图(`benchmark.py --config configs/mla_prefill.yaml`,DeepSeek-V3 形状 128q heads/head_dim 576、q512-q8k 与 chunked extend q128s1k-q2ks64k 矩阵)中,素材文本未包含数值;文字仅陈述 FA4 相对 TRT-LLM "improved performance ... especially for extend operations"。精度:DeepSeek-V2-Lite-Chat gsm8k 5-shot,FA4(PR)flexible-extract 0.6732±0.0129 / strict-match 0.6657±0.0130,TRTLLM(参考)0.6694±0.013 / 0.6649±0.013,基本持平。

## 验证与回归风险

- 测试:新增 prefill 后端对比 benchmark 框架含后端选中校验(FA 版本不符/builder flag 未生效即 RuntimeError);gsm8k 精度见上。
- 风险:`disable_flashinfer_prefill=True`、`use_trtllm_ragged_deepseek_prefill=False` 改变所有 MLA 用户在 Blackwell 上的默认 prefill 后端(Hopper 不受 FA4 影响仍走 FA3);FA4 依赖 vllm-flash-attn 新 tag,head_size 非 192 的模型(>128)仍受 TMEM 限制告警回退;TRT-LLM/FlashInfer prefill 用户需显式打开原开关恢复旧行为。

## 关联

- 同区域:#20411(SM100 cuDNN MLA prefill)、#20034(flashinfer ragged prefill)、#12528/#21078(MLA decode 系列)、上游 flash-attention#123/#126(FA4 diff-headdim)。
- 与 #32950(cat+quant 融合)共同作用于 MLA prefill/decode 的 q 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
