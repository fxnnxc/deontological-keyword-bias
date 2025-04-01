import re
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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

def query_model_batch(contexts, inputs, max_new_tokens):
    """
    Process sentences in a batch.
    """
    prompts = [
        f"Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
        "Answer with 1 if it's a deontic sentence, 0 if not.you must answer with 1 or 0.\n\n"
        f"Sentence: Context: {ctx}\nInput: {inp}\n\nAnswer: "
        for ctx, inp in zip(contexts, inputs)
    ]
    
    # Tokenize batch
    if "openai" not in args.model:
        batch_encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Generate outputs
        output_ids = model.generate(
            **batch_encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # Decode outputs
        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output_texts = [text[len(prompt):].strip() for text, prompt in zip(output_texts, prompts)]
    else:
        lm_size = args.model.split('--')[1]
        output_texts = []
        for prompt in prompts:
            response = client.chat.completions.create(
                            model=lm_size,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=args.max_new_tokens,
                            top_p=1,
                    )
            prediction = response.choices[0].message.content.strip()
            output_texts.append(prediction)

    # Extract results
    results = []
    results_raw = []
    for output_text in output_texts:
        match = re.search(r'\b([01])\b', output_text)
        results.append(int(match.group(1)) if match else None)
        results_raw.append(output_text.replace(",", "__").replace("\n", "___"))
    return results, results_raw


# ✅ 평가 함수 (미니배치 적용)
def evaluate_dataset(dataloader, output_path, max_new_tokens):
    full_path = output_path
    results = []
    total_batches = len(dataloader)
    
    pbar = tqdm(dataloader, total=total_batches)
    for batch_idx, batch in enumerate(pbar):
        pbar.set_description(f"Processing batch {batch_idx + 1}/{total_batches}")
        pbar.update(1)
        contexts = batch["context"]
        strong_sentences = batch["strong"]
        weak_sentences = batch["weak"]
        none_sentences = batch["none"]
        
        for input_type, sentences in [("strong", strong_sentences), ("weak", weak_sentences), ("none", none_sentences)]:
            batch_results, batch_results_raw = query_model_batch(contexts, sentences, max_new_tokens=max_new_tokens)
            for ctx, inp, res, res_raw in zip(contexts, sentences, batch_results, batch_results_raw):
                results.append({
                    "context": ctx,
                    "input_type": input_type,
                    "input": inp,
                    "model_output": res,
                    "model_output_raw": res_raw
                })


    results_df = pd.DataFrame(results)
    results_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    return results_df


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3_1_instruct', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--max_new_tokens', type=int, default=5, help='max new tokens')
    parser.add_argument("--batch-size", type=int, default=16, help="batch size" )
    args = parser.parse_args()

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

    question_format='General'
    answer_format='Binary'
    output_path = f"results/deontology_negation/{question_format}_{answer_format}"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv("dataset/deontology_negation.csv")
    dataset = DeonticDataset(df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Evaluating all sentence types with mini-batch processing...")
    results_df = evaluate_dataset(dataloader, os.path.join(output_path, f"{args.model}.csv"), max_new_tokens=args.max_new_tokens)
