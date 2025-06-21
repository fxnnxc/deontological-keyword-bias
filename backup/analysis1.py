import pandas as pd 
import numpy as np 
import os 

def save_averaged_results(exp_name):
    dir_path = f"results/{exp_name}/General_Binary"
    models = [
            'openai--gpt-4o',
            'openai--gpt-4o-mini', 
            'llama3_1_instruct--70b', 
            'llama3_1--8b', 
            'gemma2--9b', 
            'exaone--8b', 
            'qwen2--7b'
            ]
    result_dfs = []
    for model in models:
        result_df = pd.read_csv(os.path.join(dir_path, f"{model}.csv"))
        result_df['model'] = model
        result_dfs.append(result_df)

    def mv_func(x):
        values = ['must', 'should', 'ought']
        for v in values:
            if v in x:
                return v 
        if "has to" in x or "have to" in x:
            return "have to"
        return "none"

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
    result_df = pd.concat(result_dfs)
    result_df['mv'] = result_df['input'].apply(mv_func)
    result_df['model_output'] =  result_df['model_output_raw'].apply(process_model_output)
    result_df.head()

    targets = ['input_type', 'model', 'model_output']
    groupby = targets[:-1]
    result_averaged = result_df[targets].groupby(groupby).mean().reset_index()
    result_averaged
    new_df = pd.DataFrame(columns=['type'] + models)
    for input_type in ['none', 'strong', 'weak']:
        new_row = [input_type]
        for model in models:
            temp = result_averaged[result_averaged['input_type'] == input_type]
            temp = temp[temp['model'] == model]
            new_row.append(np.round(temp['model_output'].item(), 2))
        new_df.loc[len(new_df)] = new_row
    new_df
    new_df.to_csv(f'averaged_{exp_name}.csv', index=False)
    
    
if __name__ == "__main__":
    save_averaged_results("exp_1_not")
    save_averaged_results("exp_1")
    save_averaged_results("exp_2_1_not")
    save_averaged_results("exp_2_1")