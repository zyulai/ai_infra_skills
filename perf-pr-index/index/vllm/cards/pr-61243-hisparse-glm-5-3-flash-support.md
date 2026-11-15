# PR #61243: [HiSparse] GLM-5.3-Flash support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-11-12 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | performance, rocm, ready, ci/build, cpu, kv-connector, nvidia, mrv2, kv-cache-manager, glm, hisparse |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/61243 |

## 一句话结论

让 HiSparse 支持 GLM-5.3-Flash(首个 hybrid 模型:45 层中 11 层 sparse MLA、其余 KDA linear attention),并重排 GPU pool 布局,使可并发的长请求容量提升 3.4–4.8x;128k/16k c256 下 TTFT 634 s → 384 s。Fixes #59176。

## 动机

HiSparse 在 GLM-5.3-Flash 上无法工作:启动失败、CUDA graph capture 挂起、page 落到 host 后 buffer 溢出、Hopper 上 `FLASH_ATTN_MLA_SPARSE` 因 indexer top-k 输出有间隙而崩溃;此外所有 KV cache group 共享一个按最大 group(Flash 的约 1.1 MB KDA state)定尺寸的 GPU pool,HiSparse 自己的 group 大部分块空间被浪费。正文给出四个修复方向与显存利用优化(见下)。

## 优化点

- 让 Flash 启动并正确服务:从 text config 读 top-k(Flash 顶层 config 是 multimodal);修 CUDA graph capture 期间的挂起;GPU model runner 一行改动——piecewise CUDA graph step 也给 HiSparse connector 提供 attention metadata(full-graph step 本来就给),否则 host 上有 page 时 Flash 溢出一个 buffer;修 Hopper 上 `FLASH_ATTN_MLA_SPARSE` 因 top-k 输出间隙导致的崩溃(`kpool_compress` 的 tail 紧跟所选 pools,有效项构成前缀);attention backend 现在声明是否支持 HiSparse(`supports_hisparse`,默认 False 并进 `validate_configuration`)——Hopper 上的 FlashInfer(prefix-cache hit 结果错误)与 SM120(未验证)选择退出。
- GPU 显存高效利用(HiSparse groups 填满共享 pool 的大块):hot-cache layers 打包进每个块(22 groups → 4);每个 indexer block 容纳多个 kernel page——page 除不尽 block 时 planner 在"pad 每 page"与"增大 block"之间取给请求留 room 更多的方案;KDA block 缩到 256 tokens 以免抬高 scheduler block;`worker/utils.py` 一行小改,使 kernel 能寻址 padded pages。

## 关键改动

> pack 的 diff 被截断(13 个文件未展示,含 `vllm/v1/core/` 的 pool/planning 实现与 `worker/utils.py` 等),以下仅覆盖已展示部分。

```text
vllm/v1/attention/backend.py: 新增 supports_hisparse()(默认 False)与 validate_configuration 的 use_hisparse 检查
vllm/v1/attention/backends/mla/flashattn_mla_sparse.py: supports_hisparse = True
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py: base 支持HiSparse;SM120 子类显式 False
vllm/models/glm5next/{nvidia,amd}/ops/kpool_compress.py: history_len = min(tail_start, topk),tail 紧跟所选 pools 构成前缀(修 Hopper top-k 间隙崩溃)
vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py: _in_step 标志,layer mirror 只发生在 start_step 与 finish_forward 之间(graph capture forward 不 mirror)
vllm/config/vllm.py: index_topk 改从 hf_text_config 读取
vllm/model_executor/layers/attention/sparse_mla_attention.py: 移除 hot buffer 必须 contiguous 的 assert(hot block 只需整行间隔)
tests/: selector/registry/kv_cache_planning/kv_cache_utils/attn_utils/gpu_model_runner 等相应测试
```

```python
pool_len = seq_len // POOL_SIZE
tail_start = pool_len * POOL_SIZE
tail_count = seq_len - tail_start  # in [0, POOL_SIZE)
# The tail follows the selected pools: valid entries must form a prefix.
history_len = tl.minimum(tail_start, topk)

# History region: expand selected pool g = cols // POOL_SIZE.
is_history = cols < history_len
g = cols // POOL_SIZE
o = cols % POOL_SIZE
pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
hist_val = (pid * POOL_SIZE + o).to(tl.int32)
hist_out = tl.where(pid >= 0, hist_val, -1)
```

## 性能声明

- 容量(启动时 vLLM 报告的最大并发请求数):4xH200 bf16、139k 请求长度:28.45 → 136.90(4.8x);4xB200 fp8、64k:117.5 → 440.7(3.8x);4xB200 fp8、128k:75.0 → 251.6(3.4x)。即正文的 3.4–4.8x more concurrent long requests。
- 性能(4xB200,fp8 KV,200 GiB host pool,与 B200 容量行同 server;`vllm bench serve --dataset-name random --random-range-ratio 0 --ignore-eos --seed 42`;无 HiSparse → HiSparse):64k/2k c1:tok/s 174.0 → 162.5、TPOT 5.01 → 5.43 ms;64k/2k c32:985.3 → 971.0 tok/s、TPOT 25.36 → 25.84 ms;64k/16k c256:tok/s 4309 → 3835、TTFT 295 s → 185 s、TPOT 29.1 → 54.9 ms;128k/16k c256:tok/s 2707 → 2756、TTFT 634 s → 384 s、TPOT 29.1 → 68.2 ms。作者解读:同并发下 HiSparse 代价很小;并发 256 时它一次跑完全部请求而不是把大多数排队,故 TTFT 大降、单 token 延迟上升。
- 精度(GSM8K 5-shot,全 1319 题):4xB200 无 HiSparse(main)0.911;HiSparse 0.916;HiSparse fp8 KV 0.919;4xH200 HiSparse 0.918;4xH100 HiSparse FlashMLA backend 0.913;100-shot 从 host(~17k tokens,仅 4 GB GPU KV cache)0.935 / 0.945 / 0.915。长上下文 run 从 host 向 GPU 流式读取超过 1 TB,确实压到了 host-memory 路径。

## 验证与回归风险

- Server 命令(4 GPU,bf16 KV):`vllm serve zai-org/GLM-5.3-Flash -tp 4 --kv-cache-dtype bfloat16 --max-model-len 8192 --attention-config '{"hisparse_config":{}}' --kv-transfer-config ...(HiSparseConnector,host_pool_gib 32)`。
- 精度:GSM8K 5-shot 全 1319 题(`tests/evals/gsm8k/gsm8k_eval.py`);长上下文 100-shot GSM8K(约 17k tokens)在仅 4 GB GPU KV cache(`--kv-cache-memory-bytes 4G`)下从 host 服务;容量取启动报告值。
- 单测:`pytest tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_kv_cache_planning.py tests/v1/worker/test_attn_utils.py tests/v1/worker/test_utils.py tests/v1/kv_connector/unit/test_hisparse_connector.py tests/benchmarks/test_forward_cli.py`;`pytest tests/test_attention_backend_registry.py tests/kernels/attention/test_attention_selector.py -k "hisparse or opt_in"`。新增测试覆盖:hot layers 打包至最大 device page(22→4 类)、indexer pages 划分 pool block(`_pool_bytes_per_block == 1_049_472`)、HiSparse backend 选择/退出(Hopper FlashInfer 跳过、SM120 拒绝)、strided hot rows(非整块 stride)、fp8 decode 64-token pages、layer mirror 仅在 connector step 内等。
- 风险(正文明示):Flash + fp8 KV 在 Hopper 上仍会在首次 prefill 失败(FlashMLA backend),作者留作 follow-up 修复;SM120 的 FlashInfer sparse MLA 未经验证故不参与 HiSparse。

## 关联

- Fixes #59176(HiSparse 在 GLM-5.3-Flash 上不工作)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
