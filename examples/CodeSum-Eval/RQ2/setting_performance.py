import openai
import pandas as pd
import time
import re
from tqdm import tqdm
import sys
sys.path.append("../..")

openai.api_key = 'your openai key'


def evaluate(model, reference, nshot):

    df_score = pd.read_excel('../../dataset/human_evaluation/RQ1-2/human_evaluation.xlsx')
    
    coh_example = df_score[df_score['Coherence'].apply(lambda x: x.is_integer())].groupby('Coherence').head(nshot)
    con_example = df_score[df_score['Consistency'].apply(lambda x: x.is_integer())].groupby('Consistency').head(nshot)
    flu_example = df_score[df_score['Fluency'].apply(lambda x: x.is_integer())].groupby('Fluency').head(nshot)
    ref_example = df_score[df_score['Relevance'].apply(lambda x: x.is_integer())].groupby('Relevance').head(nshot)

    criteria = {
        "Coherence": "The summary should exhibit clear structural organization, progressing logically from sentence "
                     "to sentence to form a coherent body of information about the topic.",
        "Consistency": "Evaluating the alignment of facts between the summary and the code snippet. A consistent "
                       "summary should contain only statements supported by the source code, while penalizing any "
                       "inclusion of hallucinated facts.",
        "Fluency": "Assessing the quality of each sentence. Sentences should be free from repetition, formatting "
                   "issues, capitalization errors, or clear grammatical problems (e.g., fragments) that affect "
                   "readability.",
        "Relevance": "Evaluating the selection of vital content from the source code. The summary should include only "
                     "essential information from the source document, with penalties for redundancies and excessive "
                     "details.",
        # "Coherence": "the summary should be well-structured and well-organized. The summary should not just be a heap "
        #              "of related information, but should build from sentence to sentence to a coherent body of "
        #              "information about a topic.",
        #
        # "Consistency": "the factual alignment between the summary and the summarized code. A factually consistent "
        #                "summary contains only statements that are entailed by the source code. Annotators were "
        #                "also asked to penalize summaries that contained hallucinated facts. ",
        #
        # "Fluency": "the quality of individual sentences. The sentence should have no repetitive word, formatting "
        #            "problems, capitalization errors or obviously ungrammatical sentences ( "
        #            "e.g., fragments, missing components) that make the text difficult to understand.",
        #
        # "Relevance": "selection of important content from the source. The summary should include only important "
        #              "information from the source document. Annotators were instructed to penalize summaries that "
        #              "contained redundancies and excess information.",
    }

    # example

    if reference:
        roles = {
            # coherence
            "Original Code Author": "As the Original Code Author, having written the code, you ensure the coherence of the "
                                    "code summary, ensuring that it clearly conveys the main logic of the code.",
            # Consistency
            "Code Reviewer1": "As a Code Reviewer, serving as an experienced developer, you guarantee that the summary "
                              "remains consistent with the original code. You ensure that the summary captures the "
                              "primary functionality and logic of the code without introducing any additional or "
                              "unrelated content.",
            # Fluency
            "Code Reviewer2": "As a Code Reviewer, serving as an experienced developer, you focus on ensuring that the summary is written smoothly, with clear "
                              "sentences and appropriate wording. You challenge other judgments and provide alternative "
                              "solutions when necessary.",
            # Relevance
            "Code Editor": "As a Code Editor, concentrating on the business or functional relevance of the code, "
                           "you ensure that the summary captures the key significance of the code in the larger "
                           "system or project.",
        }
        evaluation_step = {
            'Coh': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its main functionality and key operations.'
            # '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
            # 'the main functionality and key operations of the code, and if they present them in a clear and '
            # 'logical order. '
            # '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Con': '',
            # 'Evaluation Steps:'
            # '1. Read the Source Code carefully and understand its main functionality and any key operations.'
            # '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
            # 'Ensure that the summary contains only statements that are present or implied in the source code.'
            # 'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
            # 'source code. If any are found, they should be penalized in your evaluation.'
            # '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Flu': '',
                   # 'Evaluation Steps:'
                   # '1. Read the code comments carefully and examine each sentence to ensure it is grammatically correct.'
                   # '2. Identify any glaring grammatical errors, such as sentence fragments, missing components like verbs or subjects, or any other issue that makes the text difficult to understand '
                   # '3. Check for any instances of repetitive words that can hamper clarity and ensure proper capitalization throughout the comments.'
                   # '4. Assign a score for fluency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   # 'based on the Evaluation Criteria. ',
            'Ref': ''
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its key information and primary actions of the code.'
            # '2. Read the code comments and compare them to the source code. '
            # 'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
            # '3. Check if the code comments include repetitive or unnecessary information. '
            # 'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
            # '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
        }
        rating = {
            'Coh': 'Evaluation Form (scores ONLY):',
            # 'Con': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            # 'Flu': 'Evaluation Form (scores ONLY):',
            # 'Ref': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            'Con': 'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')',
            'Flu': 'Evaluation Form (scores ONLY):',
            'Ref': 'Evaluation Form (scores ONLY):',
        }
        example = {
            'Coh': [f"""
            Source Code: {coh_example['Code'].iloc[i]}
            Reference Summary: {coh_example['Target'].iloc[i]}
            Summary: {coh_example['Generated'].iloc[i]}
            {rating['Coh']}
            Rating: {coh_example['Coherence'].iloc[i]}""" for i in range(nshot*4)],
            'Con': [f"""
            Source Code: {con_example['Code'].iloc[i]}
            Reference Summary: {con_example['Target'].iloc[i]}
            Summary: {con_example['Generated'].iloc[i]}
            {rating['Con']}
            Rating: {con_example['Consistency'].iloc[i]}""" for i in range(nshot*4)],
            'Flu': [f"""
            Source Code: {flu_example['Code'].iloc[i]}
            Reference Summary: {flu_example['Target'].iloc[i]}
            Summary: {flu_example['Generated'].iloc[i]}
            {rating['Flu']}
            Rating: {flu_example['Fluency'].iloc[i]}""" for i in range(nshot*4)],
            'Ref': [f"""
            Source Code: {ref_example['Code'].iloc[i]}
            Reference Summary: {ref_example['Target'].iloc[i]}
            Summary: {ref_example['Generated'].iloc[i]}
            {rating['Ref']}
            Rating: {ref_example['Relevance'].iloc[i]}""" for i in range(nshot*4)],
        }

    else:
        roles = {
            # coherence
            "Systems Analyst1": "As a Systems Analyst, you ensure the coherence of the "
                                "code summary, ensuring that it clearly conveys the main logic of the code.",
            # Consistency
            "Code Reviewer1": "As a Code Reviewer, serving as an experienced developer, you guarantee that the summary "
                              "remains consistent with the original code. You ensure that the summary captures the "
                              "primary functionality and logic of the code without introducing any additional or "
                              "unrelated content.",
            # Fluency
            "Systems Analyst2": "As a Systems Analyst, you focus on ensuring that the summary is written smoothly, with clear "
                                "sentences and appropriate wording. You challenge other judgments and provide alternative "
                                "solutions when necessary.",
            # Relevance
            "Code Reviewer2": "As a Code Reviewer, serving as an experienced developer, concentrating on the business or functional relevance of the code, "
                              "you ensure that the summary captures the key significance of the code in the larger "
                              "system or project.",
        }
        evaluation_step = {
            'Coh': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its main functionality and key operations.'
            # '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
            # 'the main functionality and key operations of the code, and if they present them in a clear and '
            # 'logical order. '
            # '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Con': '',
            # 'Evaluation Steps:'
            # '1. Read the Source Code carefully and understand its main functionality and any key operations.'
            # '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
            # 'Ensure that the summary contains only statements that are present or implied in the source code.'
            # 'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
            # 'source code. If any are found, they should be penalized in your evaluation.'
            # '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Flu': ''
                   'Evaluation Steps:'
                   '1. Read the code comments carefully and examine each sentence to ensure it is grammatically correct.'
                   '2. Identify any glaring grammatical errors, such as sentence fragments, missing components like verbs or subjects, or any other issue that makes the text difficult to understand '
                   '3. Check for any instances of repetitive words that can hamper clarity and ensure proper capitalization throughout the comments.'
                   '4. Assign a score for fluency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   'based on the Evaluation Criteria. ',
            'Ref': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its key information and primary actions of the code.'
            # '2. Read the code comments and compare them to the source code. '
            # 'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
            # '3. Check if the code comments include repetitive or unnecessary information. '
            # 'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
            # '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
        }
        rating = {
            'Coh': 'Evaluation Form (scores ONLY):',
            # 'Con': 'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')',
            # 'Flu': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            # 'Ref': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            'Con': 'Evaluation Form (scores ONLY):',
            'Flu':  'Evaluation Form (scores ONLY):',
            'Ref': 'Evaluation Form (scores ONLY):',

        }
        example = {
            'Coh': [f"""
            Source Code: {coh_example['Code'].iloc[i]}
            Summary: {coh_example['Generated'].iloc[i]}
            {rating['Coh']}
            Rating: {coh_example['Coherence'].iloc[i]}""" for i in range(nshot*5)],
            'Con': [f"""
            Source Code: {con_example['Code'].iloc[i]}
            Summary: {con_example['Generated'].iloc[i]}
            {rating['Con']}
            Rating: {con_example['Consistency'].iloc[i]}""" for i in range(nshot*5)],
            'Flu': [f"""
            Source Code: {flu_example['Code'].iloc[i]}
            Summary: {flu_example['Generated'].iloc[i]}
            {rating['Flu']}
            Rating: {flu_example['Fluency'].iloc[i]}""" for i in range(nshot*5)],
            'Ref': [f"""
            Source Code: {ref_example['Code'].iloc[i]}
            Summary: {ref_example['Generated'].iloc[i]}
            {rating['Ref']}
            Rating: {ref_example['Relevance'].iloc[i]}""" for i in range(nshot*5)],
        }

    df = pd.read_excel('../../dataset/RQ1-2/final/recode.xlsx')
    # Define the columns for the results DataFrame
    columns = ['Code', 'Target', 'Generated']

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=columns)

    # for idx, row in df.iterrows():
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        code_to_display = row['Code']
        target = row['Target']
        generated = row['Generated']
        print(idx)
        print(f"Code: {code_to_display}")
        print(f"Reference: {target}")
        print(f"Summary (To Be Evaluated): {generated}")
        scores_dict = {
            'Code': code_to_display,
            'Target': target,
            'Generated': generated
        }

        for (role_name, role_description), (criterion_name, criterion_task), (eval_name, eval_step), \
            (example_name, example_data),(rating_name, rating_data) in zip(roles.items(), criteria.items(), evaluation_step.items(),
                                                example.items(), rating.items()):
            demonstration = "\n".join(example_data)
            prompt = f"""
            {role_description}
            You will be given one summary written for a source code. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.
            Evaluation Criteria:
            {criterion_name}(0-4) - {criterion_task}
            {eval_step}
            Example:
            {demonstration}
            Evaluate item:
            Source Code: {code_to_display}
            Reference Summary: {target}
            Summary: {generated}
            {rating_data}
            """
            score = model_api(model, prompt)
            # print(prompt)
            column_name = f"{role_name} ({criterion_name} Score)"
            if reference:
                if rating_name in ['Con']:
                    match = re.search(r'Rating:\s*(\d+\.?\d*)', score)
                    if match:
                         match = float(match.group(1))
                    else:
                         match = 0
                else:
                    match = re.search(r'\d+', score)
                    if match:
                         match = match.group()
                    else:
                         match = 0
            else:
                match = re.search(r'\d+', score)
                if match:
                     match = match.group()
                else:
                     match = 0
            scores_dict[column_name] = match
            # Printing out the desired information:
            # print(f"Role: {role_name}")
            # print(f"Criterion: {criterion_name}")
            # print(f"Score: {score}")
        # print("------" * 10)
        # Append the result to the DataFrame
        results_df = results_df.append(scores_dict, ignore_index=True)
    # results_df.to_excel(f"evaluated_by_{model}_reference{reference}.xlsx", index=False)
    return results_df

def model_api(model, prompt):
    # print(f"new prompt:\n {prompt}")
    if model == 'gpt-4' or model == 'gpt-3.5-turbo':
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
    reference = 0  # 0-false, 1-ture

    model = "text-davinci-003"
    # model = 'gpt-3.5-turbo'
    # model = 'gpt-4'

    turn_num = 1
    print("reference:", reference, "turns:", turn_num)
    nshot = 3

    df_turn = evaluate(model, reference, nshot)
    df_turn.to_excel(f"evaluated_by_{model}_reference{reference}_turn{turn_num}_nshot{nshot}.xlsx", index=False)
