# PR #24256: [core/loader] Add presharded load format

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ch-wan |
| 合并 | 2026-07-25 |
| 分类 | Model-specific Fast Path | tier: major |
| Label | dependencies, npu, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 展示 5 个文件(`load_config.py`、`loader.py`、`server_args.py`、两个 `pyproject.toml`),另有 4 个文件因超长从 diff 中略去 |
| 链接 | https://github.com/sgl-project/sglang/pull/24256 |

## 一句话结论

新增 `--load-format presharded`:把 post-process 后的推理就绪权重落盘缓存,DSV3 FP8 TP8 权重加载 53.9 s→16.0 s(3.4x)。

## 动机

大型 MoE 模型(DeepSeek-V2 系列 `--quantization fp8`、DeepSeek-V3、DeepSeek-V4-Pro mxfp4)的初始权重加载加量化处理非常昂贵,而且每次以相同 parallelism + quantization 配置启动引擎都会原样重跑:source shard 逐个迭代、model-specific `post_load_weights` 转换、quant-method `process_weights_after_loading` 的 reshape。对于反复起停、评测、同配置多实例的服务,这部分是纯粹的重复劳动。本 PR 引入 `presharded` load format,把 `process_weights_after_loading` 之后的推理就绪状态缓存到磁盘,让后续启动跳过全部三个阶段。

## 优化点

首次运行(dump):各 rank 正常加载 → `post_load_weights` → `process_weights_after_loading` 得到推理就绪 layout,然后 rank 间协调 dump 一套 per-rank 去重的 safetensors 到 `<model_path>/presharded/<config subfolder>/`。后续运行(reload):初始化模型后,对未初始化参数只跑一次 `process_weights_after_loading`(该变换只依赖 shape 不依赖值,纯粹为 materialize post-process 形状),随后多线程读取所需文件并 `copy_` 进参数——不再迭代 source ckpt,不再对真实数据重跑 `post_load_weights`。dump 协议:每 rank 用 xxh3-128 对 `(shape, dtype, 原始字节)` 多线程哈希(默认 8 线程,xxhash 释放 GIL,CUDA 张量用 pinned streaming D2H 避免整块 host 分配),写 JSON manifest;rank 0 汇总去重、按 rank-list 分组、bin-pack 成 ≤20 GiB 文件,计算 per-rank aggregate checksum(sum mod 2^64),发布 `checksum.json`;最后 barrier 后 rank 0 写 `READY` sentinel,并发读者不会看到半写 dump。reload 默认 `verify_on_load=false` 纯多线程 `copy_`(可开启重哈希校验);关键效率点还包括:**无 mid-load `empty_cache`**——参数用 in-place `copy_` 填充,caching allocator 不逐文件增长(此前 per-file `empty_cache` 主导了上百 shard 的大 MoE reload);多线程 I/O 滑窗把 host RAM 限制在约 `num_threads` 个 in-flight 文件。`post_load_weights` 安装的非 Parameter Tensor 属性(`w_kc`、`w_vc`、`compressor.freqs_cis` 等)作为 extras 持久化并以 `setattr` 重装;GemmaRMSNorm 的 `gemma_weight` 与 GDN Parameter 别名在 copy 后 rebind。post-process 时机还避免了大 quant 模型的加载期内存翻倍(DSV4-Pro mxfp4:pre-process 约 283 GB vs post-process 约 223 GB/rank)。子目录名由 TP/DP/EP/PP、quantization、dtype、EPLB、expert location 等 knob 的结构签名生成(如 `TP-8-sig-...`)。

## 关键改动

```text
python/sglang/srt/model_loader/loader.py: 新增 PreshardedModelLoader(dump 计划/哈希/校验/reload 全流程,约 800 行)
python/sglang/srt/configs/load_config.py: LoadFormat.PRESHARDED
python/sglang/srt/server_args.py: load-format 选择与文档;extra-config 旋钮说明
python/pyproject.toml / pyproject_cpu.toml: 新增 xxhash 依赖
```

```text
--model-loader-extra-config 旋钮:presharded_path(缓存根,默认 model_path 下)、
draft_presharded_path(draft 模型)、max_file_bytes(默认 20 GiB)、
hash_num_threads(默认 8)、verify_on_load(默认 false)、
enable_multithread_load(默认 true)/num_threads(默认 8)
```

## 性能声明

方法:仅 weight-load 计时(无 warmup generate,`disable_cuda_graph=True`,`max_total_tokens=10000`),指标为 scheduler `Load weight end. elapsed=` 的跨 rank 最大值;硬件 8xH200;reload 默认 `verify_on_load=false`;dump 为一次性。

- DeepSeek-V3-0324 FP8,TP=8(约 642 GB source,632 GB presharded):auto 53.9 s;**presharded reload 16.0 s(3.4x faster)**;首次 dump 约 501 s(一次性)。reload rank 间 15.0-16.0 s;早前 serial + per-file `empty_cache` 路径同一份 dump 约 160 s。
- 主矩阵 12/12 OK(reload vs auto):llama32-tp1 1.82x、qwen35-4b-tp2 2.32x、qwen15moe-tp2 3.57x、dsv2-tp2 2.17x 等,全部命中缓存。
- 附加配置 13/18 full OK:1.56x-4.55x(qwen15moe-tp4 4.55x 最高);2 个 ERR 为无关的 server-args 断言,非 load-format bug。
- DeepSeek-V4-Pro mxfp4(GB300,cold-cache e2e):baseline auto 246 s;presharded reload 约 83-93 s(约 2.6-3.0x);首次 dump 405 s(一次性)。

## 验证与回归风险

准确性:DeepSeek-V4-Pro mxfp4 TP=4 EP=4 DP-attn、4xGB300 上全参数哈希 diff(`verify_on_load=true`,state_dict 全量 + post_load_weights 安装的非 Parameter Tensor 属性):**9288/9288 参数跨 4 rank bit-identical**;DeepSeek-Coder-V2-Lite-Instruct 多 TP/quant 配置与 Qwen3.5(含 GDN/GemmaRMSNorm rebind)reload 后生成文本一致;篡改 `rank_checksums["0"]` 一个字节在 `verify_on_load=true` 下被检出。checklist 声明有 plan 构造、哈希(CPU/CUDA digest 一致)、多线程文件迭代、READY 生命周期的单测。风险:dump 目录必须跨 rank/node 共享(否则计划文件不可见而报错);缓存 key 绑定 parallelism+quant 结构签名,配置一变即 miss 并重新 dump(首次 dump 成本高于正常加载);新增 xxhash 依赖;`verify_on_load` 默认关闭,静默损坏需用户显式开启校验;structural signature 构建失败时返回 None 并告警,降级为仍可 dump/reload。

## 关联

无正文列出的前序/后续 PR;与 HF/default 加载路径共享多线程 I/O 旋钮(`enable_multithread_load`/`num_threads`)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
