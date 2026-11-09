# PR #56372: [Bugfix][Multimodal] Fix flat/scoped mm_processor_kwargs merge and resolution

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56372](https://github.com/vllm-project/vllm/pull/56372) |
| 作者 | danigarciaoca |
| 合并日期 | 2026-09-29 |
| labels | bug, mistral, multi-modality, qwen, ready, verified |
| 热度 | 👍 0 · 💬 60 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
定义 mm_processor_kwargs 的四级优先级(engine flat/scoped < request flat/scoped)、flat→HF modality scope 路由与 Qwen size 别名补全,修复合并/解析混乱。

## 问题现象
vLLM 同时接受来自 engine 配置与 per-request 覆盖的 `mm_processor_kwargs`,且同时支持 legacy flat 形式与 HuggingFace modality-scoped 形式(`images_kwargs`、`videos_kwargs` 等),但两维度的语义不一致:同一有效设置既可平铺又可在 modality scope 内表达时,仅按 source 优先级合并不够;flat kwargs 无法正确路由到支持的 HF scope;Qwen 的 `min_pixels`/`max_pixels` 与 `size.shortest_edge`/`size.longest_edge` 别名在合并后丢失 request 优先级。修复 #56363 与 #58406。

## 根因
正文披露:旧实现 `merged = dict(config) | dict(inference)` 只做一层 source 优先级替换,既不区分 flat 与 scoped 表示,也不做递归合并——高优先级的部分覆盖会整体丢弃低优先级的不相关条目;同时缺少「哪些 flat kwargs 属于哪个 HF scope」的 schema 发现,导致 flat/scoped 优先级与 modality 解析没有统一契约。

## 修复方案
三层契约:
1. **通用合并**(`vllm/config/multimodal.py` 新增 `_merge_mm_processor_kwargs`):有效设置优先级为 `engine flat < engine scoped < request flat < request scoped`;两侧均为 mapping 时递归合并,否则高优先级替换;缺失 scope、`None` 值 scope(`{"images_kwargs": None}`)与空 mapping 视为 absent 不参与替换(scope 本身如此;非空 scope 内显式 `None` 仍是值)。
2. **schema 发现与路由**:`BaseProcessingInfo` 从 processor 契约与模型支持的 modalities 推导每个 HF scope 支持的 processor kwargs(`supported_mm_processor_kwargs` 改为 `cached_property`);`InputProcessingContext` 按 `0 个支持 scope→保持 flat / 1 个→路由 / N 个→fan out` 解析,如 `{"size": X}` 在 image/video 均支持时解析为 `images_kwargs` 与 `videos_kwargs` 各一份;无 schema 时保守保留未知 kwargs、只去重已进 scope 的冗余 flat 副本。
3. **模型特定**:Qwen 在通用合并前做 `size.shortest_edge ↔ min_pixels`、`size.longest_edge ↔ max_pixels` 别名补全(非规范化);Qwen3-VL 把 `min_pixels`/`max_pixels` 加进 video 路由 schema(vLLM 历史兼容输入),并在调用 HF 前把别名折入 `videos_kwargs.size` 后移除;部分 scoped size 覆盖先按 processor defaults 补全成完整 `size`;多视频 list 值 `fps`/`num_frames` 逐视频标量化。sizing 消费者(Qwen2-VL/Qwen3-VL/Mistral3)改为直接读解析后的 `images_kwargs`/`videos_kwargs` scope;Mistral3 走同一 `_merge_and_resolve_mm_processor_kwargs` 路径。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响所有使用 `mm_processor_kwargs`(flat 或 scoped)的多模态模型。
- 回归面:`vllm/config/multimodal.py`(merge 逻辑重写)、`vllm/multimodal/processing/context.py`(resolve)、`models/mistral3.py`/`qwen2_vl.py`/`qwen3_vl.py`(sizing 与 HF 调用边界)。兼容性:legacy flat 与 HF scoped 均保留;未知 kwargs 保留。Qwen 有效 vision size 仍为 `processor defaults < size < 非 None min/max_pixels`。回归测试覆盖 merge 优先级、空 scope 归一化、schema 路由、Qwen 别名/部分 scoped size/多视频、Mistral3 size 覆盖;Qwen 对 HF 的多模态正确性测试(qwen3_vl / qwen2_5_vl)通过。diff 在 pack 中 40000 字符处截断,后半部分改动(如 context.py 全文)未纳入本卡。

## 关联
- #56363(所修 issue)
- #58406(所修 issue)
- #58408(相关工作:仅递归合并同路径嵌套 mapping,flat/scoped 优先级留待本 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
