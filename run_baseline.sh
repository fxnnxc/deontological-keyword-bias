#!/bin/bash

# =============================================================================
# Deontological Keyword Bias - Baseline Experiments
# =============================================================================
# 
# This script runs baseline experiments to evaluate Deontological Keyword Bias (DKB)
# in large language models. It tests how models interpret sentences with/without
# modal expressions (must, should, etc.) and compares their judgments to human
# normative assessments.
#
# Experiments:
# 1. Deontology baseline: Test model judgments on deontic sentences
# 2. Commonsense baseline: Control experiments with non-deontic sentences
# 3. Negation experiments: Test robustness with negated sentences (commented out)
# 4. Debiasing experiments: Few-shot learning and in-context reasoning methods
#
# Author: Bumjin Park (KAIST AI)
# Paper: "Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models"
# =============================================================================

# Model configuration
# Uncomment models you want to evaluate
models=(
    "llama3_1--8b"              # Meta's Llama 3.1 8B model
    # "gemma2--9b"              # Google's Gemma 2 9B model
    # "qwen2--7b"               # Alibaba's Qwen 2 7B model
    # "exaone--8b"              # LG AI's EXAONE 8B model
    # "llama3_1_instruct--70b"  # Meta's Llama 3.1 70B model
    # "openai--gpt-4o-mini"     # OpenAI's GPT-4o-mini model
)

echo "🚀 Starting Baseline Experiments for Deontological Keyword Bias"
echo "📊 Models to evaluate: ${models[@]}"
echo "=" * 60

# =============================================================================
# Phase 1: Baseline Experiments
# =============================================================================
echo "📋 Phase 1: Running Baseline Experiments"
echo "----------------------------------------"

for model in ${models[@]}; do
    echo "🔬 Testing model: $model"
    
    # Deontology baseline - test modal expression effects
    echo "  📝 Running deontology baseline..."
    python run_deontology.py --model $model 
    
    # Commonsense baseline - control experiments
    echo "  🧠 Running commonsense baseline..."
    python run_commonsense.py --model $model
    
    # Negation experiments (commented out by default)
    # echo "  🔄 Running negation experiments..."
    # python run_deontology_negation.py --model $model 
    # python run_commonsense_negation.py --model $model 
    
    echo "✅ Completed baseline experiments for $model"
    echo ""
done

# =============================================================================
# Phase 2: Debiasing Experiments
# =============================================================================
echo "🎯 Phase 2: Running Debiasing Experiments"
echo "----------------------------------------"

for model in ${models[@]}; do
    echo "🔧 Testing debiasing methods for: $model"
    
    # Few-shot learning experiments
    # Test different combinations of positive/negative examples
    for fewshot_pos in {0..2}; do
        for fewshot_neg in {0..2}; do
            echo "  📚 Few-shot learning: ${fewshot_pos} positive, ${fewshot_neg} negative examples"
            python run_reduce_effects_deontology.py --model $model --prompt-version "fewshot_${fewshot_pos}_${fewshot_neg}"
            python run_reduce_effects_commonsense.py --model $model --prompt-version "fewshot_${fewshot_neg}_${fewshot_pos}"
        done
    done
    
    # In-context reasoning experiments
    echo "  🧮 In-context reasoning with logical prompts"
    python run_reduce_effects_deontology.py --model $model --prompt-version "logicshot_2_2"
    python run_reduce_effects_commonsense.py --model $model --prompt-version "logicshot_2_2"
    
    echo "✅ Completed debiasing experiments for $model"
    echo ""
done

echo "🎉 All experiments completed successfully!"
echo "📈 Check the output files for detailed results and analysis."