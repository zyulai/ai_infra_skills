# PR #25510: [diffusion] tighten selected perf baselines

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-05-17 |
| 分类 | Other (tier: major) |
| Label | npu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用(1 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/25510 |

## 一句话结论

用 #25457 最新成功 Base CI 的连贯快照收紧 H100 diffusion 性能基线,让回归阈值反映当前真实性能(非运行时优化)。

## 动机

CI 的 H100 diffusion `perf_baselines.json` 基线已落后于当前代码的实际性能(部分场景快 5% 以上),阈值过宽会让性能回归检测不敏感。需要用一次真实成功 CI run 的连贯快照系统性收紧,并保证每个场景内部的数字来自同一次 CI 请求、内部一致。

## 优化点

无运行时代码优化,属测试基线维护,更新规则为:(1) 仅当最新 CI run 显示某场景至少一项 top-level 指标 >= 5% 提升且无 top-level 回归时才更新;(2) 每个被更新场景整组替换(含 stages_ms、denoise_step_ms、expected_e2e/avg/median),值全部来自同一 CI request,保持内部一致;(3) 刷新 H100 基线 metadata 日期为 2026-05-17。数据源:CI run 25963872007,head SHA `4d2aa5b32694a713af808c6cf7ee5e41a3817b37`。被更新的场景包括 flux_image_t2i_2_gpus(expected_e2e 5613.23 → 5242.81 ms)、ltx_2_two_stage_t2v(8500.0 → 7209.55 ms)、ltx_2.3_one_stage_ti2v(21000.0 → 17881.82 ms)、wan2_1_t2v_1_3b_lora_1gpu(9624.48 → 8579.79 ms)、qwen_image_t2i_cache_dit_enabled、qwen_image_layered_i2i、fastwan2_2_ti2v_5b、wan2_1_t2v_1.3b_frame_interp_2x、ltx_2_3_two_stage_ti2v_2gpus、ltx_2_3_hq_pipeline 等。快 e2e 但 top-level denoise 更慢的场景被跳过以免扩大聚合阈值。diff 中另含 server_args.py 的 CLI 参数注释/顺序整理与 NPU 基线文件的小改动(denoise_step "0" 101.91 → 330.0、补 `estimated_full_test_time_s`),与正文"NPU 基线未改"的表述不完全一致。

## 关键改动

```text
python/sglang/multimodal_gen/test/server/perf_baselines.json: H100 各场景基线值整组收紧 + metadata 日期刷新
python/sglang/multimodal_gen/test/server/ascend/perf_baselines_npu.json: denoise_step "0" 值调整并补 estimated_full_test_time_s
python/sglang/multimodal_gen/runtime/server_args.py: CLI 参数注释与声明顺序整理(无行为变化)
```

## 性能声明

本 PR 不含运行时性能声明;"5% top-level improvement" 是基线更新的入选门槛而非加速数字。基线值本身取自 CI run 25963872007(H100 80GB CI pool,metadata 标注),例如 flux_image_t2i_2_gpus 的 DenoisingStage 5299.62 → 5000.66 ms、ltx_2.3_one_stage_ti2v 的 LTX2AVDenoisingStage 18500.0 → 15447.81 ms。

## 验证与回归风险

作者注明"Not run locally; values were extracted from the linked CI logs",数据完全来自链接的 CI run;PR Test (Base) 已通过。风险面:基线收紧后,CI perf 检查对噪声更敏感,可能出现偶发超阈值;NPU 基线在无稳定成功信号时只做了局部调整;`estimated_full_test_time_s` 等新增字段需被测试框架识别。

## 关联

#25457:提供基线快照的 CI run 所属 PR;同区域为 multimodal_gen diffusion server perf 基线体系(H100/NPU/MUSA 三套 json);pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
