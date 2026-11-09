# PR #54142: [Mypy] Fix mypy typing for N/O models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54142](https://github.com/vllm-project/vllm/pull/54142) |
| 作者 | taneem-ibrahim |
| 合并日期 | 2026-09-18 |
| labels | ready |
| 热度 | 👍 2 · 💬 30 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
把 mypy 检查范围扩展到 `models/[nNoO]*` 文件(主分支上 96 个类型错误),通过收紧类型契约、补 assert/cast/注解逐一修复,是 #26533 增量推行的一部分。

## 问题现象
N/O 开头的模型文件未被 mypy 覆盖。正文给出复现:
```bash
pre-commit run mypy-3.12 --files vllm/model_executor/models/[nNoO]*.py --hook-stage manual
```
Main 分支结果:`Found 96 errors in 14 files (checked 20 source files)`;分支修复后(写作时)剩 `nemotron_h.py:712: error: Cannot override class variable (previously declared on base class "SupportsLoRA") with instance variable [misc]` 1 个错误。无运行时故障现象,属静态类型层面的覆盖缺口。

## 根因
正文披露:mypy 增量推行(#26533)尚未覆盖 N/O 模型文件——`tools/pre_commit/mypy.py` 的 EXCLUDE 列表把 `vllm/model_executor/models/[nN]`、`[oO]` 整体排除,这些文件长期积累了 96 个类型错误(缺 None 检查、注解过宽/错误、Optional 属性未声明等)。

## 修复方案
1. `tools/pre_commit/mypy.py`:从 EXCLUDE 删除 `[nN]`、`[oO]` 两行,启用对 N/O 模型文件的检查。
2. 逐文件收紧类型契约(节选):`nemotron_h.py` 中 `is_non_gated_moe: bool = True` 改为 `is_non_gated_moe = True`(消除与 `SupportsLoRA` 基类类变量冲突,即复现器里剩下的那个错误),`get_mamba_state_shape_from_config` 返回类型改为 `MambaStateShapes`,`fc1/fc2_latent_proj` 声明 `| None`;`interfaces.py` 的 `MambaStateShapes` 类型并集新增 ReplaySSM 五元组形状;`mamba_utils.py` `append_replayssm_ring` 收紧 base_shapes 入参与返回的元组结构。
3. 大量模式化修复:`nano_nemotron_vl.py` 补 `assert isinstance(...)`/`cast(NemotronHForCausalLM, ...)`、`audio_num_clips: list[int] | torch.Tensor`、新增 `NanoNemotronVLMultiModalInputs(TypedDict)`、`torch.is_tensor` 改 `isinstance`、`_process_image_input` 接受 image/video 两种输入等;`aimv2.py` 的 `quant_config` 参数放宽为 `QuantizationConfig | None`;`olmo_hybrid.py` 声明 `self.rotary_emb: nn.Module | None`;`nemotron_h_mtp.py`/`openpangu_mtp.py` 的 forward 签名把 `inputs_embeds` 改为 keyword-only 可选参数并补 None 断言;`openpangu.py` 修 `q_b_proj` 第一参从局部 `q_lora_rank` 改为 `self.q_lora_rank`,`opencua.py` 改用 `model_config.get_multimodal_config()` 并以 `TYPE_CHECKING` 分支给 `OpenCUADummyInputsBuilder` 换泛型基类。
4. 注意:pack 的 diff 在 40000 字符处截断(文件粒度),最后可见文件为 `openpangu_mtp.py`,其后可能还有其他 O 开头文件的改动未展示。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:N/O 模型文件若干处运行时可见的小改动(新增 assert、`get_multimodal_config()`、`self.q_lora_rank`、forward 签名调整、`load_weights` 里 `maybe_remap_kv_scale_name` 返回值改名),以及 mypy pre-commit hook 覆盖面扩大——今后这些文件的新类型错误会阻塞 CI。作者验证:373 registry 测试、96 Nemotron processing 测试、14 ReplaySSM 测试通过,`pre-commit run --all-files` 通过。

## 关联
- #26533(继续其增量 mypy 推行)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
