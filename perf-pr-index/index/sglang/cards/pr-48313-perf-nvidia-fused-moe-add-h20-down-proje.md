# PR #48313: [Perf] [NVIDIA] fused_moe: Add H20 down-projection (TMA) configs for GLM-5.3-Flash TP8/TP4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @QNan0513 |
| 合并 | 2026-11-18 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/48313 |

## 一句话结论

为 GLM-5.3-Flash 在 H20 的 TP8/TP4 plain-TP Triton fused MoE 补齐下投影 `_down` 配置(带 TMA)并重调四个上投影配置:内核级 TP8 每 M 点 1.02x–1.28x 快于 main,TP8 服务 c64 无 MTP 输出吞吐 +5.2%、c1 +5.5%。

## 动机

plain TP(`--tp 8`/`--tp 4`,无 `--ep`)下 GLM-5.3-Flash 走 Triton fused MoE,目标层查 `E=289`(共享专家默认融合进路由专家)、NextN/MTP 草稿层及关闭共享专家融合时查 `E=288`。#45089 曾加 H20 配置,但用 stock `tuning_fused_moe_triton.py` 只按 batch size 出一份两 GEMM 共用的配置,且没有 `_down` 文件——服务日志打 `Down MoE config file not found ...; reusing the tuned up-projection config without TMA`,下投影 GEMM 用上投影配置且无 TMA。`tuning_fused_moe_triton_sep.py` 可分别调两个 GEMM 并写 `_down` 配置(#45591 已为 H200 出过同类文件),本 PR 对四个 H20 形状做同样的事,上配置一并重调,替换 #45089 的四个文件。

## 优化点

- 八个 Triton 3.8.0 配置(main 钉 torch 2.14.1,自带 Triton 3.8.0):`E=289/288 × N=256/512` 各一份 up + 一份 `_down`,均 `device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]`。
- 调参用 `--enable-tune-up-tma`(先调 down 再调 up,up 输出按所选 down 配置的读取布局写);路由从 `--topk-ids-dir` 采集(H20 TP4 真实服务 3 条约 8k token ShareGPT 提示,42 个 MoE 层);全默认搜索空间(过滤后 1,280 配置)× 默认 18 个 batch size,每 batch size 一张 GPU。
- 结果:down GEMM 全部 72 个条目启用 TMA;up GEMM 70/72 启用(M=4096 的 N=256 除外);71/72 个 up 条目与 #45089 不同;`BLOCK_SIZE_M` 按条目对齐 down 读取要求。
- 未覆盖:H20-3e 配置(#48314)与 EP 布局。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/
  E=289,N=256,...NVIDIA_H20...json           重调: num_stages 2-3→4-5, GROUP_SIZE_M 重排, +USE_TMA true (M=4096 false)
  E=288,N=256,...NVIDIA_H20...json           同上
  E=289,N=512,...NVIDIA_H20...json           同上
  E=288,N=512,...NVIDIA_H20...json           同上
  E=288,N=256,..._down.json / E=289,N=256,..._down.json / E=288,N=512,..._down.json / E=289,N=512,..._down.json
                                             新增: 全 18 桶 USE_TMA true
```

## 性能声明

(全部照抄 PR 正文,未独立核实)

- 内核基准(运行时 `fused_moe` 入口,CUDA graph 持 10 次调用 replay 100 次,ABBA 交替 6 轮取中位):每个 M 点都快于 main,TP8 1.02x–1.28x、TP4 1.03x–1.18x(M=1 TP8 E=289 1.28x:29.2→22.9 us;M=4096 1.10x:1500.5→1362.9 us)。M=48–256 双臂轮间波动可达约 10%,个别条目如 TP8 E=289 M=128 曾测得 1.08x。
- 服务端(TP8:c64 输出 tok/s 2900.4→3052.6, +5.2%;c16 1494.7→1544.9, +3.4%;c1 195.9→206.7, +5.5%。TP4:c16 +0.9%、c4 +0.3%、c1 164.9→172.6, +4.7%)。无 MTP 时 TP8 3.4%–5.5% 更快;prefill-only 负载输入吞吐约 +2%(15929→16259)。
- MTP(c16/c4/c64 等):TP8 c16 +13.1%(第二对 +5.1%)、TP4 c4 +17.3%(第二对 +14.9%);TP4 c1 一对 −7.4%、另一对 +8.2%(随机 token 提示上噪声大)。GSM8K 上 accept length:TP4 0.2%、TP8 0.1% 更高。
- 精度:GSM8K 20-shot 500 题 TP4 0.924→0.926、TP8 0.938→0.932(同臂重复差可达 1.6 点,单次标准误约 1.1 点);数值上两臂 fused_moe 输出相对 L2 差 ≤1.1e-3。

## 验证与回归风险

纯 JSON 配置数据(格式由 `save_configs_sep()` 固定,检查过仅含 JSON、尾换行、无尾随空白);GSM8K 与数值检查如上。风险面:仅覆盖 H20 标准版(`NVIDIA_H20`,H20-3e 在 #48314)与 plain TP 布局;调参基于 GLM-5.3-Flash 采集的路由分布,其他模型/负载不适用。Base CI 通过,Extra 与 AMD CI 显示失败图标(pack 未提供细节)。

## 关联

#45089(H20 初版 up 配置,被替换)、#45591(H200 `_down` 先例)、#43647(tuner `E=289` 一行修改,关闭未合并,本 PR 本地应用后未改 tuner)、#48314(H20-3e)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
