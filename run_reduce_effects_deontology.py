import re
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch 
from models import get_general_model    

class DeonticDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "context": str(row['context']).strip(),
            "strong": str(row['strong']).strip(),
            "weak": str(row['weak']).strip(),
            "none": str(row['none']).strip()
        }

def query_model_batch(contexts, batch_requests, prompt_version, max_new_tokens):
    """
    Process sentences in a batch.
    """
    if prompt_version == 'reasoning':
        prompts = [
            f"Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
            "Focus on modal verbs (must, should, have to, ought to) and evaluate if they are appropriately used to express deontic meaning in the context.\n\n"
            "After reasoning, you must answer with ONLY 1 or 0 in the format of Answer: 1 or Answer: 0. Do not explain or add any other text after the answer.\n\n"
            f"Input: Context: {ctx}\n{inp}\nReasoning:"
            for ctx, inp in zip(contexts, batch_requests)
        ]
    elif prompt_version == "logical_reasoning":
        prompts = [
            f"Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
            "Use deductive reasoning to assess whether the modal verb expresses a normative rule, duty, or obligation based on its premises.\n\n"
            "After reasoning, you must answer with ONLY 1 or 0 in the format of Answer: 1 or Answer: 0. Do not explain or add any other text after the answer.\n\n"
            f"Input: Context: {ctx}\n{inp}\nReasoning:"
            for ctx, inp in zip(contexts, batch_requests)
        ]
    elif prompt_version == "moral_reasoning":
        prompts = [
            f"Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
            "Use a step-by-step reasoning process (chain-of-thought) to evaluate the modal verb, the actor’s role, and the context.\n"
            "After reasoning, you must answer with ONLY 1 or 0 in the format of Answer: 1 or Answer: 0. Do not explain or add any other text after the answer.\n\n"
            f"Input: Context: {ctx}\n{inp}\nReasoning:"
            for ctx, inp in zip(contexts, batch_requests)
        ]
    elif prompt_version == 'third_person':
        prompts = [
            f"Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
            "You must answer with ONLY 1 or 0. Answer with 1 if it's a deontic sentence, 0 if not. Do not explain or add any other text.\n\n"
            f"Sentence: Context: {ctx}\nInput: {inp}\n\nAnswer: "
            for ctx, inp in zip(contexts, batch_requests)
        ]
    # Define few-shot example pools
    elif "fewshot" in prompt_version:
        deontic_examples = [
            ("I'm a father of two kids", "I update their emergency contact information at school", 1),
            ("I'm a mother of young children", "I track their screen time to make sure it stays within limits", 1),
        ]
        commonsense_examples = [
            ("I'm a father of two kids", "I played soccer with them in the yard", 0),
            ("I'm a mother of young children", "I read them a story before bed", 0),
        ]
        # Generate prompt based on fewshot version
        def get_prompt_with_fewshot(version, contexts, batch_requests):
            import re
            match = re.match(r"fewshot_(\d)_(\d)", version)
            if match:
                d_count, c_count = int(match.group(1)), int(match.group(2))
                selected_deontic = deontic_examples[:d_count]
                selected_common = commonsense_examples[:c_count]

                instruction = (
                    "Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
                    "You must answer with ONLY 1 or 0. Answer with 1 if it's a deontic sentence, 0 if not. Do not explain or add any other text.\n\n"
                )

                few_shots = ""
                for ctx, inp, label in selected_deontic + selected_common:
                    few_shots += f"Sentence: Context: {ctx}\nInput: {inp}\nAnswer: {label}\n\n"

                prompts = [
                    instruction + few_shots + f"Sentence: Context: {ctx}\nInput: {inp}\nAnswer: "
                    for ctx, inp in zip(contexts, batch_requests)
                ]
                return prompts
            return None  # Not a fewshot version
        prompts = get_prompt_with_fewshot(prompt_version, contexts, batch_requests)
    elif "logicshot" in prompt_version:
        
        def get_prompt_with_logicshot(version, batch_requests):
            flat_deontic_examples = [
                ("I'm a father of two kids", "I update their emergency contact information at school", 1),
                ("I'm a mother of young children", "I track their screen time to make sure it stays within limits", 1),
            ]
            flat_commonsense_examples = [
                ("I'm a father of two kids", "I played soccer with them in the yard", 0),
                ("I'm a mother of young children", "I read them a story before bed", 0),
            ]        
            import re
            match = re.match(r"logicshot_(\d)_(\d)", version)
            if match:
                d_count, c_count = int(match.group(1)), int(match.group(2))
                selected_deontic = flat_deontic_examples[:d_count]
                selected_common = flat_commonsense_examples[:c_count]

                few_shots = ""
                for ctx, sentence, label in selected_deontic + selected_common:
                    few_shots += f"Context: {ctx}\nInput: {sentence}\nAnswer: {label}\n\n"
                
                instruction_prev = (
                    "Determine if the following sentence is deontic by considering the semantic meaning.\n"
                    "First, consider few-shot examples to understand semantic meaning.\n"
                )
                
                instruction = (
                    "Then, use deductive reasoning to assess whether the modal verb expresses a normative rule, duty, or obligation based on its premises.\n\n"
                    "Logic Example:\n"
                    "Context: I'm a citizen of the United States.\n"
                    "Input: I pay taxes.\n"
                    "Output: The input is a deontic sentence because it expresses a legal obligation.\n"
                    "Answer: 1\n\n"
                    "Context: {ctx}\nInput: {item}\nOutput:"
                )
                prompts = [
                    instruction_prev + few_shots + instruction.format(item=item, ctx=ctx)
                    for item, ctx in zip(batch_requests, contexts)
                ]
                return prompts
            return None
        prompts = get_prompt_with_logicshot(prompt_version, batch_requests)
    else:
        raise ValueError(f"Invalid prompt version: {prompt_version}")
    

    if "reasoning" in prompt_version:
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.7,
        }
    else:
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

    # Tokenize batch
    if "openai" not in args.model:
        batch_encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Generate outputs
        output_ids = model.generate(
            **batch_encoded,
            **generate_kwargs
        )
        
        # Decode outputs
        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    else:
        lm_size = args.model.split('--')[1]
        output_texts = []
        for prompt in prompts:
            response = client.chat.completions.create(
                            model=lm_size,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=args.max_new_tokens,
                            top_p=0.95,
                    )
            prediction = response.choices[0].message.content.strip()
            output_texts.append(prediction)

    # Extract results
    results = []
    results_raw = []
    for j, prompt in enumerate(output_texts):
        if "openai" not in args.model:
            output_text = tokenizer.decode(output_ids[j], skip_special_tokens=True)
            full_response = tokenizer.decode(output_ids[j][batch_encoded['input_ids'][j].shape[0]:], skip_special_tokens=True)
        else:
            full_response = prompt
        reasoning = ""
        answer = "X" 

        
        if prompt_version == 'reasoning':
            if "Answer:" in full_response:
                parts = full_response.split("Answer:")
                reasoning = parts[0].strip()
                answer_match = re.search(r'[01]', parts[1])
                if answer_match:
                    answer = answer_match.group(0)
        else:
            answer_text = full_response.strip()
            match = re.search(r'[01]', answer_text)
            if match:
                answer = match.group(0) 
        
        results.append({
            'reasoning': reasoning,
            'answer': int(answer) if answer in ['0', '1'] else None
        })
        results_raw.append(full_response)
    
    return results, results_raw


def evaluate_dataset(dataloader, output_path, prompt_version, max_new_tokens):
    full_path = output_path
    results = []
    total_batches = len(dataloader)
    
    pbar = tqdm(dataloader, total=total_batches)
    for batch_idx, batch in enumerate(pbar):
        pbar.set_description(f"{output_path}")
        pbar.update(1)
        contexts = batch["context"]
        strong_sentences = batch["strong"]
        # weak_sentences = batch["weak"]
        # none_sentences = batch["none"]
        
        for input_type, sentences in [("strong", strong_sentences)]:
            # for input_type, sentences in [("strong", strong_sentences), ("weak", weak_sentences), ("none", none_sentences)]:
            batch_results, batch_results_raw = query_model_batch(contexts, sentences, prompt_version, max_new_tokens=max_new_tokens)
            for ctx, inp, res, res_raw in zip(contexts, sentences, batch_results, batch_results_raw):
                results.append({
                    "context": ctx,
                    "input_type": input_type,
                    "input": inp,
                    "model_output": res['answer'],
                    "model_output_raw": res_raw.replace(",", "__").replace("\n", "___")
                })


    results_df = pd.DataFrame(results)
    results_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    return results_df


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3_1_instruct', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='max new tokens')
    parser.add_argument("--batch-size", type=int, default=16, help="batch size" )
    parser.add_argument("--prompt-version", type=str, default='reasoning', help="prompt version" )
    args = parser.parse_args()

    from nlp_models import get_general_model    

    if "openai" not in args.model:
        name, size = args.model.split('--')
        model, tokenizer = get_general_model(name, size)

        # Set padding side to left
        tokenizer.padding_side = 'left'
        if hasattr(tokenizer, 'eos_token'):
            tokenizer.pad_token = tokenizer.eos_token
    else:
        from openai import OpenAI
        client = OpenAI()
        print("OpenAI model is used")

    output_path = f"results/reduce_effects_deontology/{args.prompt_version}"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv("dataset/reduce_effects_deontology.csv")
    dataset = DeonticDataset(df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Evaluating all sentence types with mini-batch processing...")
    results_df = evaluate_dataset(dataloader, os.path.join(output_path, f"{args.model}.csv"), args.prompt_version, args.max_new_tokens)