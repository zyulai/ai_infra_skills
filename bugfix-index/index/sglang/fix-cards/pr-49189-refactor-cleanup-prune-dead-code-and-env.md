# PR #49189: [Refactor] cleanup: Prune dead code and env vars, fix silent failures found by the code-cleanup skill

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#49189](https://github.com/sgl-project/sglang/pull/49189) |
| 作者 | BBuf |
| 合并日期 | 2026-11-13 |
| labels | documentation, quant, amd, lora, Multi-modal, deepseek, hicache, npu, run-ci, diffusion, mthreads, apple-silicon, jit-kernel, unified-radix-cache, memory-pool |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
第三轮 code-cleanup(261 文件 +986/-5,217)顺带修复 LoRA TypeError、视频 AttributeError、env 布尔解析失效等十余处真实 bug 并删除死代码。

## 问题现象
正文" Bugs fixed"表逐条:LoRA `--lora-backend torch_native` 对每个单投影 `ReplicatedLinear`(fc*_latent_proj、DSA indexer 投影、无 q_lora 的 `fused_qkv_a_proj_with_mqa`)抛 `TypeError`(调用漏传 `output_offset_cpu`);Ernie4.5-VL/NVILA 每个视频请求 `AttributeError`(processor 仍对 `load_video` 返回的 `VideoDecoderWrapper` 调 decord API:get_avg_fps/get_batch/asnumpy);Gemma4 PP+PLE 拒绝 `--cuda-graph-backend-{decode,prefill} disabled` 拼写,报错还让用户传废弃 flag;`configure_logging` 的运行时改日志级别命令会顺带关掉 `--log-requests` 并把请求内容 dump 到 `/tmp/sglang_request_dump`;Ascend sparse KV offload 首次分配即 `AttributeError`(读 `req.req_pool_idx`,该字段在 #37094 已移到 `req.kv`);HiCache hf3fs 的 page_first_kv_split/page_head/page_unified 布局 `UnboundLocalError`;`SGLANG_MOE_PADDING=true` 崩掉 Triton MoE 模块 import(`bool(int(...))`);`SGLANG_SKIP_P2P_CHECK`、`SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION`、`SGLANG_BLOCK_NONZERO_RANK_CHILDREN` 按 `== "1"`/`== "0"` 解析,`true` 被忽略,rope fusion 设 true 反而关闭;`SGLANG_MM_RS_THREADS` Rust 端只读旧名 `SGL_MM_RS_THREADS`,import 警告引导用户设的新名无效;FlyDSL diffusion norms 忽略 `eps` 恒用 1e-6;LTX-2 两段式默认 `"H200" in device_name` 误匹配 96 GB 的 GH200;Janus-Pro vision tower 被遮蔽的 `Normalize` 算成 L_p 归一化、`sam` 分支 `vision_tower` 未绑定;DSpark SPS profiler 接受 decode cuda graph 被禁的服务器量出无用表;docs 含 main 拒绝的命令(`--max-context-length`、`--kv-cache-dtype fp8`、`--cuda-graph-bs` 别名等)。

## 根因
正文按条披露(见上),每条引用 code-cleanup skill(#49163)的模式并通过其"not this pattern"案例;这是继 #44690、#48366、#48399 之后的第三轮清理、首次用该 skill,覆盖 code 与 env vars 一起。

## 修复方案
覆盖 srt runtime、layers/models、kernels、diffusion、全部 env vars、CLI flags 与 docs:上表逐条修复(补传 `output_offset_cpu` 并用真实 backend 测试、改用 wrapper API、改查 `cuda_graph_fully_disabled()`、只发送用户传入的 keys、改读 `req.kv` 并删除读已移除字段的死 `init_req`、hf3fs 未知 layout 改抛带支持列表的 `ValueError`、env 改经 `envs`/`EnvBool` 声明并只读一次、Rust 先读新名再回退旧名、`eps` 作为编译期常量传入 kernel、LTX-2 仅按显存判断、改用 `torchvision.transforms.Normalize` 且 `sam` 分支抛 `NotImplementedError`、docs 命令逐个对照 parser 校正)。另:删除死代码(jump-forward 遗留 -381、Inkling banded/one-shot ssconv-norm kernel -920、`gpu_tensor_hash`、Hunyuan3D FlashVDM -477、Qwen-Image 2.1 Q/K RMSNorm 路径等)、AST 级重复副本合一、21 个 attention backend 的死分支 `accept_output_buffer`、8 个 `_DEPRECATED_ENVS` 墓碑;`--triton-attention-reduce-in-fp32`/`--show-time-cost`/`--disable-outlines-disk-cache` 改为解析+弃用警告;`--attention-backend compressed` 由翻译成 dsv4 改为直接报错。验证:GB300 2,092 个测试文件两树无 branch-only 失败;`SGLANG_MOE_PADDING=true` 在 base import 失败、本分支可 import;grammar backends(xgrammar/outlines/llguidance)输出 byte-identical;MI350X 各项 pass。注:pack 的 diff 在 40000 字符处截断,以上按正文归纳,截断部分未逐行核对。

## 影响与回归风险
- 影响窗口:正文未给出。
- 回归面:261 文件的宽面改动:被删的 jump-forward 代码(#47228 仍在修其 bug,但自 2025-03 起无调用方)、8 个废弃 env 墓碑从警告改为忽略、3 个 no-op flag 改为弃用警告、`--attention-backend compressed` 从静默翻译改为硬失败、outlines 磁盘缓存不再打开、env 布尔语义变化(`true`/`false` 现在被尊重);`fused_experts_mxfp8` 本 PR 未改(#45442 负责);留给 owner 的未改问题(`--modelopt-quant` 被忽略、`/update_weights_from_disk` 忽略部分参数等)列在正文;未跑:sm90、Ascend、多节点、e2e 视频、speculative/constrained server tests(gated HF repo)。

## 关联
- [#49163](https://github.com/sgl-project/sglang/pull/49163)(code-cleanup skill)
- [#44690](https://github.com/sgl-project/sglang/pull/44690)(前轮 cleanup)
- [#48366](https://github.com/sgl-project/sglang/pull/48366)(前轮 cleanup)
- [#48399](https://github.com/sgl-project/sglang/pull/48399)(前轮 cleanup)
- [#47228](https://github.com/sgl-project/sglang/pull/47228)(修被删 jump-forward 代码 bug 的 open PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
