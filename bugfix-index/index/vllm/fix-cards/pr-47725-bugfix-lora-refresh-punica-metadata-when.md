# PR #47725: [BugFix][LoRA] Refresh punica metadata when LoRA slots are reassigned under an unchanged mapping

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47725](https://github.com/vllm-project/vllm/pull/47725) |
| 作者 | AmeenP |
| 合并日期 | 2026-07-07 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论
LoRA 槽位被 LRU 换入换出而 mapping 不变时 punica 元数据不刷新,在飞请求用错适配器;memo 增加槽位布局键修复。

## 问题现象
带外(out-of-band)`add_lora()` 触发 GPU 槽位 LRU 驱逐并轮转 `lora_index_to_id` 后,正在解码的请求会用错适配器权重:输出内容悄然错误。复现脚本(#47724 的三阶段:reference、重复 reference 做确定性对照、add_lora() churn 下两路 greedy 解码)在 stock 0.22.1 / 0.24.0 / main @ `ba2215209` 上,A1 流分别在 churn 第 509/510/515 个 token 处发散;修复后 bitwise 一致、跳过的刷新次数为 0(修复前每次运行约跳过 58-59 次)。FIX #47724。

## 根因
正文披露:`LoRAModelManager.set_adapter_mapping` 以 `self._last_mapping`(当前批次的 per-token LoRA ID 映射)做 memo,mapping 不变就跳过 punica 元数据刷新;但槽位被重新分配时 memo 从不失效——`add_lora()` 立即激活新适配器(`LRUCacheWorkerLoRAManager.add_adapter` → `activate_adapter`),LRU 驱逐在解码适配器占用的槽位并轮转 `lora_index_to_id`;由于在飞请求集合未变,`LoRAMapping` 完全相同,memo 命中,`punica_wrapper.update_metadata` 被跳过,`token_lora_indices` 仍编码旧槽位布局。

## 修复方案
`vllm/lora/model_manager.py`:memo 增加 slot layout 键。新增 `self._last_slot_layout`;`set_adapter_mapping` 每次计算 `tuple(self.lora_index_to_id)`,仅当 mapping 与槽位布局都未变时才跳过刷新:

```diff
-        if self._last_mapping != mapping:
+        slot_layout = tuple(self.lora_index_to_id)
+        if self._last_mapping != mapping or self._last_slot_layout != slot_layout:
             self._set_adapter_mapping(mapping)
             self._last_mapping = mapping
+            self._last_slot_layout = slot_layout
```
每步无变化的常规调用仍命中 memo,优化保留。新增回归测试 `test_set_adapter_mapping_refreshes_after_slot_reassignment`(在 main 上无修复失败 `assert [0, 1] == [1, 0]`,修复后通过)。

## 影响与回归风险
- 影响窗口:正文未说明(0.22.1、0.24.0、main @ ba2215209 均复现)。
- 回归面:仅 `set_adapter_mapping` 的 memo 判断;正常路径(mapping 与槽位均不变)行为不变,槽位变动时会多做一次元数据刷新(轻微开销);不影响 punica wrapper 本身。

## 关联
- Fixes [#47724](https://github.com/vllm-project/vllm/issues/47724)
- 正文提及(无关)的 #36187、#39935

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
