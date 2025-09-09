import os
import pandas as pd
from datasets import load_dataset

human_eval = load_dataset("openai/openai_humaneval")['test']
print(human_eval)
mbpp = load_dataset("google-research-datasets/mbpp")['test']
print(mbpp)

human_eval_df = pd.DataFrame(human_eval)[['prompt', 'canonical_solution']]
human_eval_df['text'] = human_eval_df['prompt'] + '\n' + human_eval_df['canonical_solution']
human_eval_df = human_eval_df.drop(columns=['prompt', 'canonical_solution'])
human_eval_df['corpus'] = 'human_eval'
print(human_eval_df)
mbpp_df = pd.DataFrame(mbpp)[['code']]
mbpp_df = mbpp_df.rename(columns={'code': 'text'})
mbpp_df['corpus'] = 'mbpp'
print(mbpp_df)

combined_df = pd.concat([human_eval_df, mbpp_df], ignore_index=True)
combined_df['doc_id'] = combined_df.index
print(combined_df)
combined_df.to_csv('../unlearning/data/human_eval_and_mbpp/unseen_data.csv', index=False)
