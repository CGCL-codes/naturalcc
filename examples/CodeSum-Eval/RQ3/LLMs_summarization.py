import openai
import pandas as pd
import time
import re
import numpy as np
import sys
sys.path.append("../..")
import os
import prettytable as pt
from scipy import stats
import tqdm

# openai.api_key = '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'
openai.api_key = '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'



def generate(num, model):
    # 构建角色和评估标准

    # 结合数据
    df = pd.read_excel('../../dataset/RQ3/final/TLC/code0.xlsx').iloc[:num]
    # Define the columns for the results DataFrame
    columns = ['Code']

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=columns)
    pbar = tqdm.tqdm(df['Code'])

    for idx, code_to_display in enumerate(pbar):
        scores_dict = {
            'Code': code_to_display,
        }

        prompt =f"""
        Please generate a short comment for the following function:
        {code_to_display}
        """

        answer = model_api(model, prompt)
        print(f'{prompt}\n{answer}')
        scores_dict["Summarization"] = answer
        results_df = results_df.append(scores_dict, ignore_index=True)

    return results_df

def model_api(model, prompt):
    # print(f"new prompt:\n {prompt}")
    if model == 'gpt-4' or model == 'gpt-3.5-turbo-0613':
        message = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
            )
            generated_answer = ' '.join(response.choices[0]['message']['content'].strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    elif model == 'gpt-4.1-nano' or model == 'gpt-4o-mini' or model == 'gpt-4o':
        try:
            message = [
                      {"role": "user", "content": prompt}
               ]
            response = openai.chat.completions.create(
              model=model,
              messages=message,
              response_format={
                "type": "text"
              },
              temperature=1,
              max_completion_tokens=1000,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0,
              store=False
            )
            generated_answer = ' '.join(response.choices[0].message.content.strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    else:
        # try:
        #     print('codex')
            # response = openai.Completion.create(
            #     engine=model,  # gpt-4, gpt-3.5-turbo, text-davinci-003, text-davinci-002
            #     prompt=prompt,
            #     max_tokens=100,
            # )
            # generated_answer = ' '.join(response.choices[0].text.strip().split())
        response = openai.Completion.create(
            engine="code-davinci-002",  # Codex 专用模型
            prompt=prompt,
            # max_tokens=100,
            # temperature=0.2,
        )

        generated_answer = response["choices"][0]["text"]
        # except Exception as e:
        #     time.sleep(25)
        #     return model_api(model, prompt)
    return generated_answer


if __name__ == '__main__':
    num = 100

    # model = "text-davinci-003"
    # model = 'gpt-3.5-turbo-0613'
    # model = 'gpt-4'
    # model = 'codex'
    model = 'gpt-4o-mini'
    df = generate(num, model)

    df.to_excel(f"./results/generate_{num}_by_{model}.xlsx", index=False)