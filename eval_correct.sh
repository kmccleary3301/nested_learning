#!/bin/bash
# Correct evaluation commands based on README examples
# Use base config (pilot.yaml) for evaluation, not the training optimization config

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         正确的评估命令 (基于 README)                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 配置说明："
echo "  • 训练使用: configs/pilot_paper_faithful_optimized.yaml"
echo "  • 评估使用: configs/pilot.yaml (包含完整模型配置)"
echo ""

# Configuration (from README examples)
CONFIG="configs/pilot.yaml"
CHECKPOINT="artifacts/checkpoints/pilot/step_000200.pt"
TOKENIZER="artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model"
DEVICE="cuda:0"

# Verify files
echo "🔍 验证文件..."
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "❌ Tokenizer 不存在: $TOKENIZER"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    exit 1
fi

echo "✅ 所有文件就绪"
echo ""

# Create output directory
mkdir -p eval

# Ask user what to run
echo "请选择评估任务："
echo "  1) 快速测试 - PIQA (100 样本, ~5 分钟)"
echo "  2) 标准评估 - 所有 zero-shot 任务 (256 样本, ~30 分钟)"
echo "  3) Passkey 检索测试 (64 样本, ~10 分钟)"
echo "  4) NIAH 测试 (多种上下文长度, ~20 分钟)"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 运行快速测试 (PIQA)..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "命令:"
        echo "uv run python scripts/eval/zeroshot.py \\"
        echo "  --config $CONFIG \\"
        echo "  --checkpoint $CHECKPOINT \\"
        echo "  --tokenizer-path $TOKENIZER \\"
        echo "  --tasks piqa --max-samples 100 --device $DEVICE \\"
        echo "  --output eval/zeroshot_piqa_quick.json"
        echo ""
        
        uv run python scripts/eval/zeroshot.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --tokenizer-path "$TOKENIZER" \
            --tasks piqa \
            --max-samples 100 \
            --device "$DEVICE" \
            --output eval/zeroshot_piqa_quick.json
        
        echo ""
        echo "✅ 快速测试完成！"
        echo "📊 查看结果:"
        python3 -c "
import json
with open('eval/zeroshot_piqa_quick.json', 'r') as f:
    data = json.load(f)
    for task, metrics in data.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            print(f'  {task}: {metrics[\"accuracy\"]*100:.2f}% (samples: {metrics[\"samples\"]})')
"
        ;;
    
    2)
        echo ""
        echo "🚀 运行标准评估 (所有 zero-shot 任务)..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⏱️  预计时间: 30-40 分钟"
        echo ""
        
        uv run python scripts/eval/zeroshot.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --tokenizer-path "$TOKENIZER" \
            --tasks all \
            --max-samples 256 \
            --device "$DEVICE" \
            --output eval/zeroshot_step200_all.json
        
        echo ""
        echo "✅ 标准评估完成！"
        echo "📊 查看结果:"
        python3 -c "
import json
with open('eval/zeroshot_step200_all.json', 'r') as f:
    data = json.load(f)
    print('\\n任务                  准确率    样本数')
    print('─' * 50)
    for task, metrics in data.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            print(f'{task:20s} {metrics[\"accuracy\"]*100:6.2f}%  {metrics[\"samples\"]:>6d}')
"
        ;;
    
    3)
        echo ""
        echo "🚀 运行 Passkey 检索测试..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "测试 HOPE 记忆机制效果"
        echo ""
        
        # According to README, passkey uses --memorize flag
        uv run python scripts/eval/passkey.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --tokenizer-path "$TOKENIZER" \
            --samples 64 \
            --device "$DEVICE" \
            --output eval/passkey_step200.json \
            --memorize \
            --memorize-steps 2 \
            --memorize-paths titan,cms_fast \
            --memorize-surprise-threshold 0.02
        
        echo ""
        echo "✅ Passkey 测试完成！"
        echo "📊 查看结果:"
        python3 -c "
import json
with open('eval/passkey_step200.json', 'r') as f:
    data = json.load(f)
    baseline_acc = data.get('baseline_accuracy', data.get('accuracy', 0))
    mem_acc = data.get('memorize_accuracy', baseline_acc)
    print(f'\\nPasskey 检索准确率:')
    print('=' * 50)
    print(f'基线 (无记忆):    {baseline_acc*100:.2f}%')
    print(f'记忆机制:         {mem_acc*100:.2f}%')
    print(f'提升:             {(mem_acc-baseline_acc)*100:+.2f}%')
    if 'memorize_stats' in data:
        print('\\n记忆模块统计:')
        for key, val in data['memorize_stats'].items():
            print(f'  {key}: {val}')
"
        ;;
    
    4)
        echo ""
        echo "🚀 运行 NIAH 测试..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "测试不同上下文长度的检索能力"
        echo ""
        
        uv run python scripts/eval/niah.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --tokenizer-path "$TOKENIZER" \
            --context-lengths 2048 \
            --context-lengths 4096 \
            --samples-per-length 20 \
            --device "$DEVICE" \
            --output eval/niah_step200.json \
            --memorize \
            --memorize-steps 2 \
            --memorize-paths titan,cms_fast \
            --memorize-surprise-threshold 0.02
        
        echo ""
        echo "✅ NIAH 测试完成！"
        echo "📊 结果保存在: eval/niah_step200.json"
        ;;
    
    *)
        echo "❌ 无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 评估完成！"
echo ""
echo "📁 结果文件: eval/"
echo "📖 更多评估选项请查看: README.md (Evaluation 章节)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
