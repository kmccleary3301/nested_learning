# 模型评估指南 (正确版本)

## ⚠️ 重要：配置文件使用说明

### 训练 vs 评估的配置区别

```
训练时使用:  configs/pilot_paper_faithful_optimized.yaml
  • 包含训练优化参数 (batch_size=8, online_updates=false 等)
  • 通过 defaults: [/pilot, _self_] 继承基础配置

评估时使用:  configs/pilot.yaml
  • 包含完整模型架构配置 (titan_level, cms_levels 等)
  • 这是 README 官方示例使用的方式
  • 评估脚本只需要模型架构，不需要训练优化参数
```

**关键点**: 虽然训练用的是 `pilot_paper_faithful_optimized.yaml`，但它继承自 `pilot.yaml`，所以 checkpoint 的模型架构定义在 `pilot.yaml` 中。

---

## 🚀 快速开始 (基于 README 官方示例)

### 方式 1: 使用交互式脚本 (推荐)

```bash
./eval_correct.sh
# 选择评估任务 (1-4)
```

### 方式 2: 直接运行命令

#### 1. 快速测试 - PIQA (5 分钟)

```bash
uv run python scripts/eval/zeroshot.py \
  --config configs/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_000200.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa --max-samples 100 --device cuda:0 \
  --output eval/zeroshot_piqa.json
```

#### 2. 完整 Zero-shot 评估 (30 分钟)

```bash
uv run python scripts/eval/zeroshot.py \
  --config configs/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_000200.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all --max-samples 256 --device cuda:0 \
  --output eval/zeroshot_all.json
```

**支持的任务**: piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq, siqa, commonsenseqa, openbookqa

查看所有任务：
```bash
uv run python scripts/eval/zeroshot.py --list-tasks
```

#### 3. Passkey 记忆检索测试 (10 分钟)

这是测试 HOPE 记忆机制效果的关键任务：

```bash
uv run python scripts/eval/passkey.py \
  --config configs/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_000200.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --samples 64 --device cuda:0 \
  --output eval/passkey.json \
  --memorize \
  --memorize-steps 2 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02
```

**参数说明**:
- `--memorize`: 启用 HOPE 记忆机制
- `--memorize-steps 2`: 记忆迭代次数
- `--memorize-paths titan,cms_fast`: 更新的记忆层级
- `--memorize-surprise-threshold 0.02`: Surprise 阈值（低于此值不更新）

#### 4. NIAH (Needle-in-a-Haystack) 测试 (20 分钟)

测试不同上下文长度的检索能力：

```bash
uv run python scripts/eval/niah.py \
  --config configs/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_000200.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --context-lengths 2048 --context-lengths 4096 \
  --samples-per-length 20 \
  --device cuda:0 \
  --output eval/niah.json \
  --memorize \
  --memorize-steps 2 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02
```

#### 5. PG-19 困惑度测试 (30 分钟)

评估长文本语言建模质量：

```bash
uv run python scripts/eval/pg19_perplexity.py \
  --config configs/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_000200.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --max-samples 64 \
  --device cuda:0 \
  --output eval/pg19.json
```

---

## 📊 查看评估结果

### 快速查看 JSON 结果

```bash
# Zero-shot 结果
python3 -c "
import json
with open('eval/zeroshot_piqa.json', 'r') as f:
    data = json.load(f)
    for task, metrics in data.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            print(f'{task}: {metrics[\"accuracy\"]*100:.2f}% (samples: {metrics[\"samples\"]})')
"

# Passkey 结果
python3 -c "
import json
with open('eval/passkey.json', 'r') as f:
    data = json.load(f)
    baseline = data.get('baseline_accuracy', 0)
    memorize = data.get('memorize_accuracy', baseline)
    print(f'Baseline: {baseline*100:.2f}%')
    print(f'Memorize: {memorize*100:.2f}%')
    print(f'Improvement: {(memorize-baseline)*100:+.2f}%')
"
```

---

## 📈 预期性能 (200 步训练的小模型)

| 任务 | 预期范围 | 说明 |
|------|---------|------|
| **PIQA** | 50-65% | 物理常识推理 (随机 50%) |
| **HellaSwag** | 30-45% | 场景理解 (随机 25%) |
| **WinoGrande** | 50-60% | 常识推理 (随机 50%) |
| **Passkey (baseline)** | 30-50% | 不使用记忆机制 |
| **Passkey (memorize)** | 60-85% | 使用 HOPE 记忆 |
| **提升幅度** | +20-40% | ⬅️ 关键指标 |

**注意**:
- 这是只训练了 200 步的小模型
- 性能会远低于完全训练的模型
- 重点是验证：
  1. ✅ 优化配置不破坏模型功能
  2. ✅ HOPE 记忆机制正常工作（Passkey 提升）
  3. ✅ 训练质量正常（损失收敛）

---

## 🔧 完整评估套件 (官方脚本)

README 提供了完整的评估套件脚本 (`scripts/eval/run_pilot_suite.sh`)，但需要设置环境变量：

```bash
export HOPE_CONFIG=configs/pilot.yaml
export HOPE_CHECKPOINT=artifacts/checkpoints/pilot/step_000200.pt
export TOKENIZER_PATH=artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model
export DEVICE=cuda:0
export MAX_SAMPLES=256
export PASSKEY_SAMPLES=64

# 运行完整套件 (2-3 小时)
bash scripts/eval/run_pilot_suite.sh
```

这会运行所有评估任务：zero-shot, NIAH, continual, passkey, PG-19。

---

## 🆘 常见问题

### Q: 为什么评估时不用 `pilot_paper_faithful_optimized.yaml`?

**A**: 因为它只覆盖了训练相关参数，缺少完整的模型架构配置（`titan_level`, `cms_levels` 等）。评估脚本需要完整配置来重建模型结构。

### Q: Checkpoint 是用优化配置训练的，为什么评估要用基础配置？

**A**: `pilot_paper_faithful_optimized.yaml` 通过 `defaults: [/pilot, _self_]` 继承了 `pilot.yaml` 的所有模型配置，只覆盖了训练优化参数。Checkpoint 保存的模型架构实际上来自 `pilot.yaml`。

### Q: 如果 tokenizer 文件缺失怎么办？

**A**: 检查是否存在：
```bash
ls -lh artifacts/tokenizer/refinedweb_mix/
```
如果缺失，使用测试 tokenizer：
```bash
--tokenizer-path tests/data/tiny_tokenizer.model
```

### Q: 评估太慢怎么办？

**A**: 
1. 减少样本数: `--max-samples 50`
2. 只测试单个任务: `--tasks piqa`
3. 禁用记忆机制（去掉 `--memorize`）

---

## 📚 参考文档

- **README.md**: 第 140-198 行 (Evaluation 章节)
- **docs/zeroshot_eval.md**: Zero-shot 评估详细说明
- **docs/continual_eval.md**: 持续学习评估
- **scripts/eval/run_pilot_suite.sh**: 完整评估套件脚本示例

---

## ✅ 推荐评估流程

### 第 1 步: 快速验证 (5 分钟)

```bash
./eval_correct.sh
# 选择 1 - PIQA 快速测试
```

→ 验证模型能正常加载和运行

### 第 2 步: 记忆机制测试 (10 分钟)

```bash
./eval_correct.sh
# 选择 3 - Passkey 测试
```

→ 验证 HOPE 记忆机制的效果（baseline vs memorize 对比）

### 第 3 步: 完整评估 (可选, 30 分钟)

```bash
./eval_correct.sh
# 选择 2 - 完整 zero-shot 评估
```

→ 全面评估模型在多个任务上的性能

---

**重要提醒**: 所有评估命令都应该使用 `configs/pilot.yaml`，而不是训练时用的 `configs/pilot_paper_faithful_optimized.yaml`。这是 README 官方示例的标准做法。
