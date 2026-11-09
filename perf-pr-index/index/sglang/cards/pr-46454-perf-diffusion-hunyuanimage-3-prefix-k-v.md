# PR #46454: [Perf] diffusion: HunyuanImage-3 prefix K/V cache, opt-in FA3 attention and think-prompt support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @OnePunchMonk |
| 合并 | 2026-10-30 |
| 分类 | MoE(classify 给出;主体为 diffusion 前向与 MoE 执行顺序修复) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 7 个文件(素材另注明 3 个文件因超长被裁剪) |
| 链接 | https://github.com/sgl-project/sglang/pull/46454 |

## 一句话结论

HunyuanImage-3 跨去噪步缓存 text prefix K/V(端到端 1.28–1.30x),opt-in FA3 分段注意力 1.55x,并修复 CUDA 上 MoE 顺序导致的噪声纹理 bug。

## 动机

两项工作。修复:CUDA 生成出噪声纹理(main 亦然)——MoE block 先跑 routed experts 再跑 shared MLP,而 CUDA 的 Triton `FusedMoE` 路径 in-place(`inplace=True` 默认)覆写 `hidden_states`,shared MLP 读到 expert 输出而非层输入,每个 MoE 层都错;NPU 路径非 in-place,故 #36311 的原始 NPU 验证未发现,cookbook 也把 CUDA recipe 标为 unverified。性能:HunyuanImage-3 每个去噪步都把整个 text prefix 过每层(attention 与 MoE),但该 prefix 逐步不变;实现提案 P1(prefix KV cache)可省去每步 L_text/(L_text+L_img) 的 MoE 计算,对长或 think prompt 收益更大,同时让 FA3 重新可用。

## 优化点

`ImageKVCacheManager` 按 `cache_slot` 在第一步缓存每层 prefix K/V(clone 拥有独立存储,避免 pin 住整段 K/V),后续步把缓存 prepend 到 suffix K/V;`split_suffix_inputs` 只保留 prefix 之后的行(hidden/mask/RoPE),`pad_prefix_rows` 以零行填回全长度(调用者只读位于 suffix 的 image 位置)。`forward_block` 接收 `prefix_len`/`cache_slot`:step 0 跑全序列,后续只跑 suffix 行。`ar_stage` 把 prefix 定为第一个 noisy-image 或 timestep token 之前的行(mask 是 causal,这些行只看 prompt 与 clean conditioning;clean conditioning 图像也被缓存);clean image block 跨越边界时返回 None 走全算。CLA 开启、cache-dit adapter 驱动或无安全 prefix 时回落原路径;CFG cond/uncond 用独立 cache slot。prefix cache 默认用 dense SDPA(与 main+fix bitwise 一致);FA3 为 opt-in:`build_attention_segments` 把(causal text + bidirectional image blocks)mask 拆成 (start, end, causal) 段,每段一次 FA3 varlen 调用(bottom-right causal 对齐),GQA 原生无需 repeat_kv;仅 Hopper。另修复:shared MLP 先于 in-place experts;pipeline scheduler 对齐官方 `FlowMatchDiscreteScheduler`(`use_uniform_sigmas=True`,旧设置把训练 sigma range 二次 shift,10 步时最后 timestep 为 8.93 而非 250);`cot_text`/`bot_task`/`system_prompt` 声明进 `image_request_extra_fields`(image API 只转发声明过的字段,think 文本此前被静默丢弃)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3_utils.py: ImageKVCacheManager、build_attention_segments/segmented_attention、split/pad 工具
python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3.py: forward_block 前缀缓存路径;shared MLP 先于 in-place experts
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/hunyuan_image3/ar_stage.py: 步不变 prefix 长度与分段构建
python/sglang/multimodal_gen/runtime/pipelines/hunyuan_image3.py: FA3 段开关按 transformer self-attn 后端解析
python/sglang/multimodal_gen/configs/sample/hunyuan_image3.py: cot_text 等字段进 image_request_extra_fields
```

```python
def _prefix_kv_step(self, key, value, attn_metadata):
    slot = attn_metadata.cache_slot
    prefix_len = attn_metadata.prefix_len
    if attn_metadata.first_step:
        self._prefix_kv[slot] = (
            key[:, :prefix_len].clone(),
            value[:, :prefix_len].clone(),
        )
        return key, value
    prefix_k, prefix_v = self._prefix_kv[slot]
    return (
        torch.cat([prefix_k, key], dim=1),
        torch.cat([prefix_v, value], dim=1),
    )
```

## 性能声明

2x H200 TP2(cookbook recipe),1024x1024,50 steps,guidance 2.5,seed 42,一次 warmup + 一次计时。prefix cache(+fix)vs main+fix,输出 byte-identical(PSNR inf):short prompt 34.97→27.42 s(-21.6%,1.28x);long prompt 35.84→27.49 s(-23.3%,1.30x);峰值内存不变(106,318 MB)。FA3 opt-in:22.6 s vs 26.8 s(1.55x vs main+fix),非 bitwise——PSNR 21–24 dB vs SDPA、构图相同;GPU 上分段在 384 次调用中精确复现 attention mask,FA3 与 SDPA 每次调用差至多 1 个 bf16 ulp。think prompts(cot_text 修复后):比 main+fix 快 1.32x–1.50x,输出一致。缓存内存:每层每 CFG slot `bs x P x kv_heads x head_dim x 4` 字节,1248-token prefix 双 slot 实测 156 MB/GPU。

## 验证与回归风险

CPU 单测 11 项全过:fix 回归(in-place experts 时 shared MLP 见到层输入、timesteps 遵循官方 schedule;只回退两个源文件时恰好这两项失败);prefix cache(suffix-only 步与全量重算一致、CFG slot 不互覆、无安全 prefix 时为 None);FA3 segments(与 masked SDPA 相等、重叠块回落);image API 字段透传。默认路径 bitwise 等于 main+fix;FA3 仅当 effective transformer self-attn 后端解析为 fa/fa3/fa4 时启用,非 Hopper 保留 SDPA。缓存以 `clear_prefix_kv_cache` 在 finally 中释放。pack 中 CI:Base 通过,Extra 与 AMD ROCm 10 失败标记。

## 关联

依赖 #46600(CUDA MoE noise fix,其修复 #46461;PR 将在其落地后 rebase);前序 #36311(原始 NPU 移植验证)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
