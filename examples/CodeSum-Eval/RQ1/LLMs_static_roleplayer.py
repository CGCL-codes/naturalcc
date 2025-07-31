import openai
from openai import OpenAI
import pandas as pd
import time
import re
import sys

# openai.api_key = ''

def evaluate(model, reference,role):
    
    roles_specific = {
        # coherence
        "Code Reviewer": "As a Code Reviewer, serving as an experienced developer, you ensure the coherence of the "
                         "code summary, ensuring that it clearly conveys the main logic of the code.",
        # Consistency
        "Original Code Author": "As the Original Code Author, having written the code, you guarantee that the summary "
                                "remains consistent with the original code. You ensure that the summary captures the "
                                "primary functionality and logic of the code without introducing any additional or "
                                "unrelated content.",
        # Fluency
        "Code Editor": "As a Code Editor, you focus on ensuring that the summary is written smoothly, with clear "
                       "sentences and appropriate wording. You challenge other judgments and provide alternative "
                       "solutions when necessary.",
        # Relevance
        "Systems Analyst": "As a Systems Analyst, concentrating on the business or functional relevance of the code, "
                           "you ensure that the summary captures the key significance of the code in the larger "
                           "system or project.",
    }

    roles = {
        "Code Reviewer": "As a Code Reviewer, serving as an experienced developer, ",
        "Original Code Author": "As the Original Code Author, having written the code, ",
        "Code Editor": "As a Code Editor, ",
        "Systems Analyst": "As a Systems Analyst, ",
    }
    dimension_roles = {
        "Coherence": "you ensure the coherence of the code summary, ensuring that it clearly conveys the main logic "
                     "of the code and is easy to follow.",
        "Consistency": "you guarantee that the summary remains consistent with the original code, without hallucinated or unsupported content, similar to fact-checking to prevent any fabricated functionality.",
        "Fluency": "you focus on ensuring that the summary is written smoothly, with clear sentences and appropriate "
                   "wording, ensuring it reads naturally, like it was written by a fluent native speaker.",
        "Relevance": "You identify and preserve the most important parts of the code, avoiding unnecessary or off-topic content—like aiming at the core message without distraction.",
    }

    criteria = {
        "Coherence": "the summary should be well-structured and well-organized. The summary should not just be a heap "
                     "of related information, but should build from sentence to sentence to a coherent body of "
                     "information about a topic.",

        "Consistency": "the factual alignment between the summary and the summarized code. A factually consistent "
                       "summary contains only statements that are entailed by the source code. Annotators were "
                       "also asked to penalize summaries that contained hallucinated facts. ",

        "Fluency": "the quality of individual sentences. The sentence should have no repetitive word, formatting "
                   "problems, capitalization errors or obviously ungrammatical sentences ( "
                   "e.g., fragments, missing components) that make the text difficult to understand.",

        "Relevance": "selection of important content from the source. The summary should include only important "
                     "information from the source document. Annotators were instructed to penalize summaries that "
                     "contained redundancies and excess information.",
    }

    evaluation_step = {
            'Coh': '1. Read the source code carefully and understand its main functionality and key operations.'
                   '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
                   'the main functionality and key operations of the code, and if they present them in a clear and '
                   'logical order. '
                   '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   'based on the Evaluation Criteria. ',
            'Con': '1. Read the Source Code carefully and understand its main functionality and any key operations.'
                   '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
                   'Ensure that the summary contains only statements that are present or implied in the source code.'
                   'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
                   'source code. If any are found, they should be penalized in your evaluation.'
                   '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   'based on the Evaluation Criteria. ',
            'Flu': '1. Read the code comments carefully and examine each sentence to ensure it is grammatically correct.'
                   '2. Identify any glaring grammatical errors, such as sentence fragments, missing components like verbs or subjects, or any other issue that makes the text difficult to understand '
                   '3. Check for any instances of repetitive words that can hamper clarity and ensure proper capitalization throughout the comments.'
                   '4. Assign a score for fluency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   'based on the Evaluation Criteria. ',
            'Ref': '1. Read the source code carefully and understand its key information and primary actions of the code.'
                   '2. Read the code comments and compare them to the source code. '
                   'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
                   '3. Check if the code comments include repetitive or unnecessary information. '
                   'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
                   '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   'based on the Evaluation Criteria. ',
        }
    if reference:
        example = {
            'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
                   'Reference Summary: creates a new zip entry with the specified name.'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 4'
                   'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
                   'Reference Summary:  a runnable that implements delayed execution of a transitsectionaction'
                   'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                   'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 0'
                   'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                   'Reference Summary: methods for starting asynchronous execution .'
                   'Summary: methods for starting asynchronous execution . process process process process parent parent'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 2'
                   'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                   'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                   'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 1',
            'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
                   'Reference Summary: creates a new zip entry with the specified name.'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 4'
                   'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                   'Reference Summary: remove a scanning callback .'
                   'Summary: if a scanning and .'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 1'
                   'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                   'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                   'Summary: creates a new dexportprivatekeypvk dialog .'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 0'
                   'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }\n'
                   'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                   'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 3',
            'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
                   'Reference Summary: creates a new zip entry with the specified name.'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 4'
                   'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                   'Reference Summary:  remove a scanning callback .'
                   'Summary: if a scanning and .'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0'
                   'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                   'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                   'Summary:  creates a new dexportprivatekeypvk dialog .'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 4'
                   'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                   'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                   'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0',
            'Ref': 'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                   'Reference Summary:  remove a scanning callback .'
                   'Summary: if a scanning and .'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 1'
                   'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                   'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                   'Summary:  creates a new dexportprivatekeypvk dialog .'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 0'
                   'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                   'Reference Summary:  invoked if the computation completes successfully'
                   'Summary: invoked completes a computation successfully successfully'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 4'
                   'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                   'Reference Summary: modifies the given file in place .'
                   'Summary: modifies the the file given . for prefixes . the file'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 2',
        }
    else:
        example = {
        'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
               'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
               'name ; }'
               # 'Reference Summary: creates a new zip entry with the specified name.\n'
               'Summary: creates a new zip entry with the name'
               'Evaluation Form (scores ONLY): '
               '- Coherence: 4'
               'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
               # 'Reference Summary: creates a new zip entry with the specified name.\n'
               'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
               'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
               'Evaluation Form (scores ONLY): '
               '- Coherence: 0'
               'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
               # 'Reference Summary: methods for starting asynchronous execution .'
               'Summary: methods for starting asynchronous execution . process process process process parent parent'
               'Evaluation Form (scores ONLY): '
               '- Coherence: 2'
               'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
               # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
               'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
               'Evaluation Form (scores ONLY): '
               '- Coherence: 1',
        'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
               'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
               'name ; }'
               # 'Reference Summary: creates a new zip entry with the specified name.\n'
               'Summary: creates a new zip entry with the name'
               'Evaluation Form (scores ONLY): '
               '- Consistency: 4'
               'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
               # 'Reference Summary: remove a scanning callback .'
               'Summary: if a scanning and .'
               'Evaluation Form (scores ONLY): '
               '- Consistency: 1'
               'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
               # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
               'Summary: creates a new dexportprivatekeypvk dialog .'
               'Evaluation Form (scores ONLY): '
               '- Consistency: 0'
               'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
               # 'Reference Summary: modifies the given file in place .'
               'Summary: modifies the the file given . for prefixes . the file'
               'Evaluation Form (scores ONLY): '
               '- Consistency: 2',
        'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
               'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
               'name ; }'
               # 'Reference Summary: creates a new zip entry with the specified name.\n'
               'Summary: creates a new zip entry with the name'
               'Evaluation Form (scores ONLY): '
               '- Fluency: 4'
               'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
               # 'Reference Summary:  remove a scanning callback .'
               'Summary: if a scanning and .'
               'Evaluation Form (scores ONLY): '
               '- Fluency: 0'
               'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
               # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
               'Summary:  creates a new dexportprivatekeypvk dialog .'
               'Evaluation Form (scores ONLY): '
               '- Fluency: 4'
               'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
               # 'Reference Summary:  invoked if the computation completes successfully'
               'Summary: invoked completes a computation successfully successfully'
               'Evaluation Form (scores ONLY): '
               '- Fluency: 0',
        'Ref': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
               'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
               'name ; }'
               # 'Reference Summary: creates a new zip entry with the specified name.\n'
               'Summary: creates a new zip entry with the name'
               'Evaluation Form (scores ONLY): '
               '- Relevance: 4'
               'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
               # 'Reference Summary:  remove a scanning callback .'
               'Summary: if a scanning and .'
               'Evaluation Form (scores ONLY): '
               '- Relevance: 1'
               'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
               # 'Reference Summary: modifies the given file in place .'
               'Summary: modifies the the file given . for prefixes . the file'
               'Evaluation Form (scores ONLY): '
               '- Relevance: 2'
               'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
               # 'Reference Summary: methods for starting asynchronous execution .'
               'Summary: methods for starting asynchronous execution . process process process process parent parent'
               'Evaluation Form (scores ONLY): '
               '- Relevance: 3',
    }

    df = pd.read_excel('../dataset/RQ1-2/final/recode.xlsx')

    # Define the columns for the results DataFrame
    columns = ['Code', 'Target', 'Generated']

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=columns)

    count = 0
    for idx, row in df.iterrows():
        code_to_display = row['Code']
        target = row['Target']
        generated = row['Generated']
        print(idx)
        # print(f"Code: {code_to_display}")
        # print(f"Reference: {target}")
        # print(f"Summary (To Be Evaluated): {generated}")
        scores_dict = {
            'Code': code_to_display,
            'Target': target,
            'Generated': generated
        }
        if role:
            for role_name,role_description in roles.items():
                for (dimension_name,role_dimension), (criterion_name, criterion_task), (eval_name, eval_step), \
                    (example_name, example_data) in zip(dimension_roles.items(), criteria.items(), evaluation_step.items(), example.items()):

                    if reference:
                        prompt = f"""
                        {role_description}{role_dimension}
                        You will be given one summary written for a source code. 
                        Your task is to rate the summary on one metric.
                        Please make sure you read and understand these instructions carefully. 
                        Please keep this document open while reviewing, and refer to it as needed.
                        Evaluation Criteria:
                        {criterion_name}(0-4) - {criterion_task}
                        Example:
                        {example_data}
                        Evaluate item:
                        Source Code: {code_to_display}
                        Reference Summary: {target}
                        Summary: {generated}
                        Evaluation Form (scores ONLY): 
                        - {criterion_name}: 
                        """
                    else:
                        prompt = f"""
                        {role_description}{role_dimension}
                        You will be given one summary written for a source code. 
                        Your task is to rate the summary on one metric.
                        Please make sure you read and understand these instructions carefully. 
                        Please keep this document open while reviewing, and refer to it as needed.
                        Evaluation Criteria:
                        {criterion_name}(0-4) - {criterion_task}
                        Example:
                        {example_data}
                        Evaluate item:
                        Source Code: {code_to_display}
                        Summary: {generated}
                        Evaluation Form (scores ONLY): 
                        - {criterion_name}: 
                        """

                    score = model_api(model, prompt)
                    # print(score)
                    column_name = f"{role_name} ({criterion_name} Score)"
                    # match = re.search(r'\d+', score)
                    matches = re.findall(r'\d+', score)
                    match = matches[-1]  
                    # if match:
                    #     match = match.group()
                    # else:
                    #     match = 0
                    scores_dict[column_name] = match
                    # Printing out the desired information:
                    # print(prompt)
                    print(f"Role: {role_name}")
                    print(f"Criterion: {criterion_name}")
                    print(f"Score: {match}")
        else:
            for (dimension_name, role_dimension), (criterion_name, criterion_task), (eval_name, eval_step), \
                (example_name, example_data) in zip(dimension_roles.items(), criteria.items(),
                                                    evaluation_step.items(), example.items()):
                if reference:
                    prompt = f"""
                    You will be given one summary written for a source code. 
                    Your task is to rate the summary on one metric.
                    Please make sure you read and understand these instructions carefully. 
                    Please keep this document open while reviewing, and refer to it as needed.
                    Evaluation Criteria:
                    {criterion_name}(0-4) - {criterion_task}
                    Example:
                    {example_data}
                    Evaluate item:
                    Source Code: {code_to_display}
                    Reference Summary: {target}
                    Summary: {generated}
                    Evaluation Form (scores ONLY): 
                    - {criterion_name}: 
                    """
                else:
                    prompt = f"""
                    You will be given one summary written for a source code. 
                    Your task is to rate the summary on one metric.
                    Please make sure you read and understand these instructions carefully. 
                    Please keep this document open while reviewing, and refer to it as needed.
                    Evaluation Criteria:
                    {criterion_name}(0-4) - {criterion_task}
                    Example:
                    {example_data}
                    Evaluate item:
                    Source Code: {code_to_display}
                    Summary: {generated}
                    Evaluation Form (scores ONLY): 
                    - {criterion_name}: 
                    """

                score = model_api(model, prompt)
                # print(score)
                column_name = f"{criterion_name} Score"
                # match = re.search(r'\d+', score)
                matches = re.findall(r'\d+', score)
                match = matches[-1]  
                # if match:
                #     match = match.group()
                # else:
                #     match = 0
                scores_dict[column_name] = match
                # Printing out the desired information:
                # print(prompt)
                print(f"Criterion: {criterion_name}")
                print(f"Score: {score}")
        print("------" * 10)
        # Append the result to the DataFrame
        # results_df = results_df.append(scores_dict, ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame([scores_dict])], ignore_index=True)
        count += 1  # increment counter
    # Save the results to an Excel file
    results_df.to_excel(f"evaluated_by_{model}_reference{reference}_final.xlsx", index=False)

def model_api(model, prompt):
    # print(f"new prompt:\n {prompt}")
    message = [
        {"role": "user", "content": prompt}
    ]
    if model == 'gpt-4' or model == 'gpt-3.5-turbo':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
            )
            generated_answer = ' '.join(response.choices[0].message.content.strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    elif model == 'deepseek':
        try:
            client = OpenAI(
                api_key='',
                base_url="https://api.deepseek.com/beta",
            )
            response = client.completions.create(
                model="deepseek-chat",
                prompt=prompt,
                suffix="",
                max_tokens=1000)
            generated_answer = response.choices[0].text
            matches = re.findall(r'\d+', generated_answer)
            match = matches[-1]  
            print(match)
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    elif model == 'Qwen3-32B':
        print('Qwen')
        message = [
            {"role": "user", "content": prompt}
        ]
        #
        client = OpenAI(
            base_url="https://router.huggingface.co/hf-inference/models/Qwen/Qwen3-32B/v1",
            api_key="")
        try:
            completion = client.chat.completions.create(
                model="Qwen/Qwen3-32B",
                messages=message,
                max_tokens=1000,
            )
            generated_answer = completion.choices[0].message.content
            print(generated_answer)
            matches = re.findall(r'\d+', generated_answer)
            match = matches[-1]  
            print(match)
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    elif model == 'Qwen3-235B':
        print('Qwen-new')
        message = [
            {"role": "user", "content": prompt}
        ]
        #
        client = OpenAI(
            base_url="https://router.huggingface.co/fireworks-ai/inference/v1",
            api_key="",)
        # try:
        completion = client.chat.completions.create(
            model="accounts/fireworks/models/qwen3-235b-a22b",
            messages=message,
            max_tokens=1000,
        )
        generated_answer = completion.choices[0].message.content
        print(generated_answer)
        matches = re.findall(r'\d+', generated_answer)
        match = matches[-1] 
        print(match)
        # except Exception as e:
        #     time.sleep(25)
        #     return model_api(model, prompt)
    elif model == 'gpt-4.1-nano' or model == 'gpt-4o-mini' or model == 'gpt-4o':
        try:
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
            matches = re.findall(r'\d+', generated_answer)
            match = matches[-1]  
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
    # model = "gpt-4.1-nano"
    model = 'gpt-4o-mini'
    # model = 'gpt-3.5-turbo'
    # model = 'gpt-4o'
    # model = 'gpt-4.1'
    # model = 'deepseek'
    # model = 'Qwen3-32B'
    role = 1
    reference = 0 # 0-free, 1-reference
    print(model, reference, role)
    evaluate(model, reference, role)

