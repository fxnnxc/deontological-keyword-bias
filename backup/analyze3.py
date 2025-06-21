import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import os 
import json 
import torch 

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)

def save_averaged_results(exp_name):

    dir_path = f"results/{exp_name}/General_Binary"
    models = [
            'openai--gpt-4o-mini', 
            'llama3_1_instruct--70b', 
            'llama3_1--8b', 
            'gemma2--9b', 
            #   'exaone--8b', 
            'qwen2--7b'
            ]
    result_df = {
        'model': [],
        'diff' : [],  # strong - none 
        'mv' : []
    }
    values = {
    }
    strong_index = 0 
    none_index = 0 

    def process_model_output(x):
        x =str(x)
        if "1" in x and "0" in x:
            return 2
        elif "1" in x:
            return 1
        elif "0" in x:
            return 0
        else:
            return -1 

    def mv_func(x):
        values = ['must', 'should', 'ought']
        for v in values:
            if v in x:
                return v 
        if "has to" in x or "have to" in x:
            return "have to"
        return "none"

    for model in models:
        result_df = pd.read_csv(os.path.join(dir_path, f"{model}.csv"))
        for line in result_df.iterrows():
            line = line[1]
            mv = mv_func(line['input'])
            if line['input_type'] == 'strong':
                if strong_index not in values:
                    values[strong_index] = {}
                values[strong_index]['strong'] = {'value':process_model_output(line['model_output']), 
                                            'mv': mv,
                                            'input': line['input'],
                                            'model': model}
                strong_index += 1
            elif line['input_type'] == 'none':
                if none_index not in values:
                    values[none_index] = {}
                values[none_index]['none'] = {'value':process_model_output(line['model_output']), 
                                            'mv': mv,
                                            'input': line['input'],
                                            'model': model}
                none_index += 1 
                
    result_df = {
        'model': [],
        'diff' : [],  # strong - none 
        'strong': [],
        'none': [],
        'mv' : [],
        # 'sample_index':[],
    }
    for i in range(len(values)):
        result_df['model'].append(values[i]['strong']['model'])
        result_df['diff'].append(values[i]['strong']['value'] - values[i]['none']['value'])
        result_df['strong'].append(values[i]['strong']['value'])
        result_df['none'].append(values[i]['none']['value'])
        result_df['mv'].append(values[i]['strong']['mv'])
        # result_df['sample_index'].append(i)
        
    result_df = pd.DataFrame(result_df)
    new_df = result_df.groupby(['model', 'mv']).mean()
    new_df.to_csv(os.path.join(f"mv_wise_{exp_name}_mean.csv"))


if __name__ == "__main__":
    save_averaged_results("exp_1_not")
    save_averaged_results("exp_1")
    save_averaged_results("exp_2_1_not")
    save_averaged_results("exp_2_1")