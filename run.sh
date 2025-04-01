

models=(
    # "qwen2--7b"
    # "gemma2--9b"
    # "llama3_1--8b"
    # "exaone--8b"
    # "llama3_1_instruct--70b"
    "openai--gpt-4o-mini"
)

# for model in ${models[@]}; do
#     python run_commonsense.py --model $model 
#     python run_exp_2_1.py --model $model 
#     python run_exp_1_not.py --model $model 
#     python run_exp_2_1_not.py --model $model 
# done


# ################################################################ 
# # Debiasing with Thoughtful
# prompt_versions=(
#     "reasoning"
#     "logical_reasoning"
#     "moral_reasoning"
#     # "third_person"
# )
# for model in ${models[@]}; do
#     for prompt_version in ${prompt_versions[@]}; do
#         echo "prompt_version: $prompt_version"
#         python run_exp_6_1_debising.py --model $model --prompt-version $prompt_version
#         python run_exp_6_2_debising.py --model $model --prompt-version $prompt_version
#     done
# done

# ################################################################ 
# Debiasing with fewshot
for model in ${models[@]}; do
#     # for fewshot_pos in {0..2}; do
#     #     for fewshot_neg in {0..2}; do
#     #         echo "fewshot_${fewshot_pos}_${fewshot_neg}"
#     #         python run_exp_6_1_debising.py --model $model --prompt-version "fewshot_${fewshot_pos}_${fewshot_neg}"
#     #         python run_exp_6_2_debising.py --model $model --prompt-version "fewshot_${fewshot_neg}_${fewshot_pos}"
#     #     done
#     # done
    python run_exp_6_1_debising.py --model $model --prompt-version "logicshot_2_2"
    # python run_exp_6_2_debising.py --model $model --prompt-version "logicshot_2_2"
done