# PR #24663: Feat: Support newer EAGLE-3 drafters

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Dogacel |
| 合并 | 2026-05-12 |
| 分类 | Speculative Decoding |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 2 个文件(llama_eagle3.py、eagle_info.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/24663 |

## 一句话结论

扩展 EAGLE-3 草稿模型支持(多层、任意数量 aux hidden states、可选 norm 布局),使新 gpt-oss-20b drafter 在 MT-Bench 上吞吐 224.53→365.15 tok/s、accept length 1.547→2.646。

## 动机

团队即将发布论文与 gpt-oss-20b / gpt-oss-120b 的 EAGLE-3 drafter checkpoints,需要 day-0 支持。SGLang 现有 EAGLE-3 实现有三处硬编码限制:只支持单层草稿(`num_hidden_layers != 1` 直接 raise)、固定消费 3 个 aux hidden states(`hidden_size * 3`、aux_norm_low/mid/high 三个固定 norm)、以及固定 pre-norm 输出约定;新 drafter 采用多层结构、更多 aux hidden states 与不同的 norm 摆放,均无法加载。

## 优化点

三点改动:(1) 多层 EAGLE-3——移除单层断言,`midlayer` 改为 `layers` ModuleList 按需建多层,旧权重名经 legacy map(`midlayer→layers.0`、`aux_norm_low/mid/high→fc_norm.0/1/2`)向后兼容;`LlamaDecoderLayer` 区分 input layer(layer_id==0,消费 target hidden states,`hidden_norm` 后与 embeds concat,qkv 输入维度 2x)与非 input layer(把上一层 MLP residual 融合进 `hidden_norm` 的 fused add-norm)。(2) per-aux norm 泛化——`fc_norm` ModuleList 按 `num_aux_hidden_states` 建 norm,兼容旧 `use_aux_norm` 标志。(3) aux 数量泛化——`num_aux_hidden_states` 按显式 attr > eagle_config 的 `eagle_aux_hidden_state_layer_ids` 长度 > 默认 3 解析,`fc` 输入维度改为 `hidden_size_in * num_aux`;`eagle_info.hidden_size_for` 相应返回 `target_hidden * num_aux`(支持 `target_hidden_size` attr),aux 输出按 `norm_output` 配置可选 normed/pre-norm。

## 关键改动

```text
python/sglang/srt/models/llama_eagle3.py: 多层 layers + legacy 权重名映射 + fc_norm + num_aux 泛化 + input layer 特化 forward
python/sglang/srt/speculative/eagle_info.py: hidden_size_for 按 num_aux_hidden_states 计算 spec 通道宽度
```

关键机制(input layer 的 concat 与后续层 fused residual):

```python
if self.is_input_layer:
    residual = hidden_states
    hidden_states = self.hidden_norm(hidden_states)
    embeds = self.input_layernorm(embeds)
    hidden_states = torch.cat([embeds, hidden_states], dim=-1)
else:
    hidden_states, residual = self.hidden_norm(hidden_states, residual)
```

## 性能声明

GH200(96 GB VRAM),MT-Bench(temperature=0.7,bs=1,depth=5,topk=1,max_tokens=6),gpt-oss-20b:

- 新 drafter vs 现 `zhuyksir/EAGLE3-gpt-oss-20b-bf16`:Latency 624.09 → 384.67 s(-38.4%),Throughput 224.53 → 365.15 tok/s(+62.6%),Accept Length 1.547 → 2.646(+71.1%);分项中 Math accept length 1.912→3.714(+94.2%)、Extraction 吞吐 +95.7%。
- vs D-Flash:吞吐 +26.9%(287.67 → 365.15 tok/s),accept length 2.622 → 2.646 基本持平,自述 "~27% higher throughput at the same draft quality"。
- vs 无投机解码基线:Throughput 315.86 → 365.15 tok/s(+15.6%),Latency -12.9%。

注意:这些提升主要来自新 drafter checkpoint 本身的质量,本 PR 的作用是让其在 SGLang 中可运行,而非纯代码路径加速。

## 验证与回归风险

PR checklist 全部勾选(含单元测试项),但正文未列出具体测试名;CI 未在 pack 中展示。风险点:input layer 与后续层的 norm 语义不同,旧单层权重经 legacy map 加载后行为需与旧实现等价(input layer 分支正是为兼容旧 checkpoint 而设);`hidden_size_for` 改变了 spec hidden states 通道宽度的推导,依赖 `spec_hidden_size` / `target_hidden_size` 配置正确;多层草稿下 CUDA graph 捕获与 KV 分配按层数线性增长。

## 关联

前序:#3582(EAGLE/MTP 支持)、#6606(draft extend CUDA graph)、#25748(多层 EAGLE 加载 OOM 修复)、#6081(MTP DP attention)。同区域后续:#24663 即多层 EAGLE-3 的能力铺垫,与 #6059(XiaomiMiMo MTP)同属 drafter 生态。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
