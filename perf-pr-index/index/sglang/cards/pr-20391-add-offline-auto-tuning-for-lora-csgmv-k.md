# PR #20391: Add offline auto-tuning for LoRA CSGMV kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @satyamk7054 |
| 合并 | 2026-04-10 |
| 分类 | Attention / Kernel(major) |
| Label | lora, run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 显示新增 tuning 脚本、config loader 与 8 个 H200 JSON 配置,另有 2 个文件因超长未收入 pack |
| 链接 | https://github.com/sgl-project/sglang/pull/20391 |

## 一句话结论

为 LoRA CSGMV shrink/expand Triton kernel 建立离线 auto-tuning 机制,H200 上 shrink kernel 最高提速 3.24x,E2E 吞吐最高 +24.3%。

## 动机

CSGMV(chunked segmented matrix-vector)LoRA kernel 此前使用硬编码 block size:shrink 为 BLOCK_N=16/BLOCK_K=256,expand 为 BLOCK_K=16/BLOCK_N=64,代码中已有 "fine-tune per operation" 的 TODO 但一直未落地。最优 tile 形状随 K、rank、num_slices、chunk_size 组合变化很大,固定默认值在不同 layer 上造成明显性能损失(如 qkv shrink chunk=128 仅 0.230ms 基线,可优化到 0.071ms)。

## 优化点

沿用 MoE `fused_moe_triton_config` 的 auto-tuning 模式,三件套:离线脚本网格搜索 + JSON 配置落盘 + 运行时 loader。搜索空间:shrink 遍历 BLOCK_N∈{16,32,64}、BLOCK_K∈{64,128,256}、num_warps∈{4,8}、num_stages∈{2,3,4};expand 额外搜索 maxnreg∈{96,112,128,160},通过寄存器上限提升 occupancy。配置按 chunk_size 键存储,文件名含 kernel/K/R/S/device 与 Triton 版本目录。运行时 `get_lora_shrink/expand_config` 以 lru_cache 加载,找不到匹配配置时回退到原硬编码默认值,Triton 版本不匹配时按版本目录降序回退。kernel 计算逻辑零改动,只替换 launch 参数。

## 关键改动

```text
benchmark/kernels/lora_csgmv/tune_lora_csgmv.py: 新增离线调优脚本,CUDA event 计时取中位数
python/sglang/srt/lora/triton_ops/lora_tuning_config.py: 新增配置加载器,lru_cache + SGLANG_LORA_CONFIG_DIR 环境变量
python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py: 硬编码 BLOCK_N=16/BLOCK_K=256 替换为 tuned config
python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py: 同上,并透传 maxnreg 等额外 launch 参数
python/sglang/srt/lora/triton_ops/csgmv_configs/triton_3_5_1/*.json: 8 个 H200 调优结果(qkv/o_proj/gate_up/down_proj)
```

## 性能声明

PR 注明测试条件:H200、Triton 3.5.1、Qwen3-Embedding-0.6B(rank=64)、fp16。

- kernel 级:shrink 最高 3.24x(qkv,chunk=128,0.230ms→0.071ms),o_proj shrink 2.44x,down_proj shrink 2.64x;expand 1.06x–1.55x。
- per-layer 合计(chunk=128,shrink+expand):2.549ms→1.945ms,净提速 1.31x;down_proj 2.63x。
- E2E:`bench_serving --backend sglang-embedding`,random dataset,120 prompts。相对 main@chunk=16 基线:tuned@chunk=128 请求吞吐 38.13 req/s(+24.3%)、token 吞吐 234,227 tok/s(+24.3%);tuned@chunk=16 为 +12.6%/+12.7%。

## 验证与回归风险

新增单测 `test.manual.lora.test_lora_tuning_config` 验证配置加载逻辑;kernel 数值逻辑不变,正确性由既有 LoRA correctness 测试覆盖(不同 block size 下输出一致)。回归风险:配置按 device+Triton 版本匹配,其他 GPU 型号无配置时走默认值,不劣化但也不受益;`maxnreg` 参数依赖较新 Triton 支持;fallback 读取其他版本目录的配置可能在跨版本时引入次优参数。

## 关联

机制上对应 MoE 的 auto-tuning 前序工作(`fused_moe_triton_config` 模式);本卡未收录同区域(LoRA CSGMV kernel)其他 PR,关联写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
