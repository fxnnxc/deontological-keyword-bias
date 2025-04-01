import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_general_model(lm_name, lm_size, lm_cache_dir=None, precision=None):
    if lm_name == "openai":
        return None 
    else:
        if lm_name[:len("llama3_1")] == "llama3_1":
            full_name = f"meta-llama/Llama-3.1-{lm_size}-Instruct"
            
        elif lm_name[:len("exaone")] == "exaone":
            full_name = f"LGAI-EXAONE/EXAONE-3.0-7.{lm_size}-Instruct"
            
        elif lm_name[:len("llama2")] == "llama2":
            full_name = f"meta-llama/Llama-2-{lm_size}-chat-hf"
            
        elif lm_name[:len("gemma2")] == "gemma2":
            full_name = f"google/gemma-2-{lm_size}"
            
        elif lm_name[:len("qwen2")] == "qwen2":
            full_name = f"Qwen/Qwen2-{lm_size}-Instruct"
        else:
            raise ValueError(f"Invalid model name: {lm_name}")
        
        precision = torch.bfloat16 if precision is None else precision
        model = AutoModelForCausalLM.from_pretrained(
            full_name,
            torch_dtype=precision,
            trust_remote_code=True,
            cache_dir=lm_cache_dir,
            device_map='auto',
        )
        tokenizer = AutoTokenizer.from_pretrained(lm_name,
                                                  cache_dir=lm_cache_dir 
                                                  )
        tokenizer.sep_token_id =  tokenizer.eos_token_id 
        tokenizer.pad_token_id =  tokenizer.eos_token_id 
        
        return model, tokenizer
