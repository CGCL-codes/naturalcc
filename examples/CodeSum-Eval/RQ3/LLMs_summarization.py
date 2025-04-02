import openai
import pandas as pd
import time
import sys
sys.path.append("../..")
import tqdm

openai.api_key = 'your openai key'


def generate(num, model):

    df = pd.read_excel('../../dataset/RQ3/final/code.xlsx').iloc[:num]
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
        Please generate a short comment in one sentence for the following function:
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
    else:
        try:
            response = openai.Completion.create(
                engine=model,  # gpt-4, gpt-3.5-turbo, text-davinci-003, text-davinci-002
                prompt=prompt,
                max_tokens=100,
            )
            generated_answer = ' '.join(response.choices[0].text.strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    return generated_answer


if __name__ == '__main__':
    num = 100

    # model = "text-davinci-003"
    model = 'gpt-3.5-turbo-0613'
    # model = 'gpt-4'

    df = generate(num, model)

    df.to_excel(f"./generate_{num}_by_{model}.xlsx", index=False)
