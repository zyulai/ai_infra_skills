# PR #31382: [Qwen3.5][MTP] Support FlashInfer CuTe DSL for online NVFP4 draft MoE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-07-31 |
| 分类 | Speculative Decoding (tier: major) |
| Label | quant, blackwell |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/31382 |

## 一句话结论

Qwen3.5 MTP 草稿 MoE 的 `nvfp4_online` 量化改走 FlashInfer CuTe DSL 路径,吞吐 +4.5%~+5.3%。

## 动机

Qwen3.5 ModelOpt FP4 checkpoint 中内嵌的 MTP draft expert 权重保持 BF16;`nvfp4_online` 会在加载时把它们转成 NVFP4,但此前只支持 FlashInfer TRT-LLM MoE runner。需要让已有的 FlashInfer CuTe DSL MoE 路径也能服务这个在线量化的 draft 模型。

## 优化点

不引入新 kernel/参数/checkpoint 格式,是路径放行与配置解析改动:(1) `_verify_quantization` 中当 draft model 且显式指定 `nvfp4_online`、checkpoint quant_method 为 `modelopt_fp4` 时,保留该选择不被 checkpoint 检测覆盖;(2) `nvfp4_online` 的支持条件从仅 TRT-LLM 扩到 `flashinfer_cutedsl` + FlashInfer A2A 或 DeepEP `low_latency`,复用现有 CuTe DSL 权重布局与已注册 DeepEP runner;(3) FlashInfer autotune 阶段按 target/draft 分别解析 speculative MoE backend(此前直接读 `moe_runner_backend`),并继续跳过会超出 DeepEP low-latency 每 rank token 上限的 CuTe DSL v1 dummy dispatch;(4) online loader 拒绝非浮点源权重,防止对已打包 FP4 权重再次量化;(5) `nvfp4_online` 的 per-tensor input scale 预填 1.0。

## 关键改动

```text
python/sglang/srt/configs/model_config.py: draft + 显式 nvfp4_online 优先于 checkpoint quant_method 检测
python/sglang/srt/layers/quantization/modelopt_quant.py: supports_nvfp4_online_moe 条件扩展到 cutedsl + A2A/DeepEP low_latency
python/sglang/srt/layers/quantization/nvfp4_online.py: 同步支持条件;拒绝非浮点源权重
python/sglang/srt/model_executor/runner/flashinfer_autotune.py: 按 speculative_draft/target 解析 backend 再决定是否 autotune
python/sglang/srt/layers/moe/ep_moe/layer.py: cutedsl runner 接受 nvfp4_online quant config
```

## 性能声明

4x GB300 聚合 serving,TP/DP/EP=4,CUDA Graph 开启,随机 1k 输入/1k 输出,并发 64,每次 64 完成请求;M0/M1 为三次运行中位数,M2 单次:

- M0 DeepEP + DeepGEMM,BF16 draft:5,377.303 tok/s,TPOT 8.9780 ms(基线)
- M1 FlashInfer A2A + CuTe DSL `nvfp4_online`:5,621.619 tok/s(+4.543%),TPOT 8.5038 ms(-5.282%)
- M2 DeepEP low-latency + CuTe DSL `nvfp4_online`:5,662.671 tok/s(+5.307%),TPOT 8.5514 ms(-4.752%)

PR 注明这是端到端 MTP 改动(A2A、runner、draft 精度同时变化),delta 不是 CuTe DSL vs DeepGEMM 的隔离 kernel 结果。

## 验证与回归风险

GSM8K(Qwen3.5-397B-A17B-NVFP4,4x GB300,TP/DP/EP=4,EAGLE 3 steps 4 draft tokens):BF16 draft 0.990 vs `nvfp4_online` 0.990(100 例),mean accept length 3.4472→3.4206(-0.77%),accept rate 0.8157→0.8068(-1.09%);更早的完整 GSM8K 资格评测 0.9771 vs 0.9794(1,311/1,311)。DeepEP low-latency 变体的 exact-HEAD 资格在 PR 合并时仍 pending。风险面:quantization 校验逻辑对 draft 的覆盖、autotune 对 speculative backend 的解析、online loader 对非浮点权重的拒绝路径;CuTe DSL v1 + DeepEP dummy dispatch 限制保留。

## 关联

同区域:Qwen3.5 MTP、`nvfp4_online` 在线量化与 FlashInfer CuTe DSL MoE runner;pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
