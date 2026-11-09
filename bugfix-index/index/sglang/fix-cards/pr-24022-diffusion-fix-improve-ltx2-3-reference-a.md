# PR #24022: [diffusion] fix: improve LTX2.3 reference accuracy controls

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24022](https://github.com/sgl-project/sglang/pull/24022) |
| 作者 | mickqian |
| 合并日期 | 2026-04-29 |
| labels | documentation, quant, run-ci, diffusion |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
LTX-2.3 一阶段 TI2V 与官方实现存在偏差;text connector 正负 CFG 分支独立处理、修音频 cross-attention RoPE scale,并按重生成的 V2A GT 更新 CI 一致性阈值。

## 问题现象
LTX-2.3 one-stage TI2V 输出与参考(GT)存在差异,需改进 reference alignment controls。正文给出当前 code-only 分支(不含 transformer fp8-cast、不含 legacy V2A skip)对重新生成 GT 的测量:min_clip `0.8053`、min_ssim `0.7934`、min_psnr `20.8080`、max_mean_abs_diff `10.3797`;按 min SSIM 的对齐率为 `79.34%`。正文未描述终端用户可见的具体症状(如画面异常截图)。

## 根因
正文未设独立 Root Cause 段,由 Changes 归纳:1) LTX2 text connector 此前把正负 CFG 分支拼接后一起过 connector,与官方行为不一致;2) LTX2.3 音频 cross-attention 的 RoPE scale factors 有错;3) `--disable-autocast` 在应用 config dict overrides 时会丢失。第 1 点的机制 diff 注释有披露:"Official LTX-2.3 processes positive and negative prompts through the connector independently; batching shifts output numerics."(批处理改变数值);第 2 点具体错法正文未细说,diff 显示音频 RoPE 模块此前未传入 `scale_factors=self.audio_scale_factors`(由 diff 推断)。

## 修复方案
- `stages/text_connector.py`:CFG 开启时正负 prompt 各自独立调用 `self.connectors(...)`(各自构造 additive mask),不再拼接后 chunk;同时补上对 embeddings/mask 缺失的 `ValueError` 校验。
- `models/dits/ltx_2.py`:音频位置编码模块构造时传入 `scale_factors=self.audio_scale_factors`,修复音频 cross-attention RoPE scale。
- `configs/sample/ltx_2.py`:新增 `skip_v2a_cross_attn_for_video_gt: bool = False` 采样旗标(经 `build_request_extra` 透传为 `ltx2_skip_v2a_cross_attn_for_video_gt`),`ltx_2_denoising.py` 各模型调用点读取并传 `disable_v2a_cross_attn`,仅用于复现 legacy GT 行为;新 GT 默认保持 V2A cross-attention 开启。
- `configs/utils.py`:把 `disable_autocast` 加入 config 覆盖时保留的参数列表;新增单测 `test_disable_autocast_is_preserved_after_pipeline_config_resolution`。
- `consistency_threshold.json` / `perf_baselines.json`:更新 `ltx_2.3_one_stage_ti2v`(clip 0.57→0.64、psnr 9.0→8.8、mean_abs_diff 57.0→59.0)、`ltx_2.3_two_stage_t2v_2gpus` 等阈值,新增 `ltx_2_3_hq_pipeline`、`ltx_2_3_two_stage_ti2v_2gpus` 条目,更新 `LTX2UpsampleStage` 基线(216.72→752.07)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有走 LTX2 text connector 的 CFG 路径由一次批处理调用改为两次独立调用(计算量近似翻倍,正文未给端到端耗时);音频 RoPE scale 变化影响 LTX2.3 音频生成;阈值调整影响对应 CI 一致性判定的松紧;`skip_v2a_cross_attn_for_video_gt` 默认 False 不改变默认输出。正文如实记录:本地 pytest 因 shell 环境导入了其他 repo 路径、缺 `triton.compiler` 而在 collection 阶段失败,未跑到断言。

## 关联
- [PR #24024](https://github.com/sgl-project/sglang/pull/24024)(transformer fp8-cast 兼容分支,对本 PR 的 min_ssim=0.8291,未并入本 PR)
- sglang-ci-data/main commit `03cba05fc35219f5e7407d6d1ac33ecee3819191`(重生成的 faithful V2A-enabled GT)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
