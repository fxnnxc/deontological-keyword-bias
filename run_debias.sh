#!/bin/bash

# =============================================================================
# Deontological Keyword Bias - Debiasing Experiments
# =============================================================================
# 
# This script focuses specifically on debiasing experiments to reduce
# Deontological Keyword Bias (DKB) in large language models. It implements
# and evaluates various debiasing methods including few-shot learning and
# in-context reasoning approaches.
#
# Debiasing Methods:
# 1. Few-shot Learning: Provide positive/negative examples to guide model behavior
# 2. In-context Reasoning: Use logical reasoning prompts to improve judgment
# 3. Prompt Engineering: Systematic variations of prompt structures
#
# The script runs both baseline (for comparison) and debiasing experiments
# to measure the effectiveness of bias reduction techniques.
#
# Author: Bumjin Park (KAIST AI)
# Paper: "Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models"
# =============================================================================

# Model configuration
# Uncomment models you want to evaluate for debiasing
models=(
    "llama3_1--8b"              # Meta's Llama 3.1 8B model
    # "gemma2--9b"              # Google's Gemma 2 9B model
    # "qwen2--7b"               # Alibaba's Qwen 2 7B model
    # "exaone--8b"              # LG AI's EXAONE 8B model
    # "llama3_1_instruct--70b"  # Meta's Llama 3.1 70B model
    # "openai--gpt-4o-mini"     # OpenAI's GPT-4o-mini model
)

echo "🎯 Starting Debiasing Experiments for Deontological Keyword Bias"
echo "📊 Models to evaluate: ${models[@]}"
echo "=" * 60

# =============================================================================
# Phase 1: Baseline Experiments (for comparison)
# =============================================================================
echo "📋 Phase 1: Running Baseline Experiments (for comparison)"
echo "--------------------------------------------------------"

for model in ${models[@]}; do
    echo "🔬 Testing baseline performance for: $model"
    
    # Deontology baseline - establish baseline bias levels
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
    # Test different combinations of positive/negative examples (0-2 each)
    # This creates a 3x3 grid of experiments to find optimal few-shot configuration
    for fewshot_pos in {0..2}; do
        for fewshot_neg in {0..2}; do
            echo "  📚 Few-shot learning: ${fewshot_pos} positive, ${fewshot_neg} negative examples"
            echo "     Testing deontology dataset..."
            python run_reduce_effects_deontology.py --model $model --prompt-version "fewshot_${fewshot_pos}_${fewshot_neg}"
            echo "     Testing commonsense dataset..."
            python run_reduce_effects_commonsense.py --model $model --prompt-version "fewshot_${fewshot_neg}_${fewshot_pos}"
        done
    done
    
    # In-context reasoning experiments
    # Use logical reasoning prompts with 2 positive and 2 negative examples
    echo "  🧮 In-context reasoning with logical prompts (2+2 examples)"
    echo "     Testing deontology dataset..."
    python run_reduce_effects_deontology.py --model $model --prompt-version "logicshot_2_2"
    echo "     Testing commonsense dataset..."
    python run_reduce_effects_commonsense.py --model $model --prompt-version "logicshot_2_2"
    
    echo "✅ Completed debiasing experiments for $model"
    echo ""
done

echo "🎉 All debiasing experiments completed successfully!"
echo "📈 Check the output files for detailed results and bias reduction analysis."
echo "📊 Compare baseline vs. debiased performance to measure effectiveness."