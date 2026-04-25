import os
import json
import random
import time
import logging
import re
import ast 
from typing import List, Dict, Any, Optional

MODEL_NAME = "gemini-2.5-flash"
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_GENAI_AVAILABLE = False
    logging.error("google-generativeai library not found. Please install it: pip install google-generativeai")

API_KEY = "API"

gemini_model_instance = None
if GOOGLE_GENAI_AVAILABLE:
    if not API_KEY or API_KEY == "xxxx":
        logging.error("GEMINI_API_KEY environment variable not set or is placeholder 'xxxx'.")
    else:
        try:
            genai.configure(api_key=API_KEY)
            gemini_model_instance = genai.GenerativeModel(MODEL_NAME)
            logging.info("Gemini API configured successfully with gemini-2.5-flash.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")
else:
    logging.error("Gemini library not available.")

def call_gemini_api(prompt: str, task_description: str, retries=100, delay=200) -> Optional[str]:
    if not gemini_model_instance:
        logging.error(f"Gemini model not initialized. Cannot perform {task_description}.")
        return None

    for attempt in range(retries):
        try:
            logging.info(f"Calling Gemini for {task_description} (Attempt {attempt+1})...")
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            response = gemini_model_instance.generate_content(prompt, safety_settings=safety_settings)

            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 logging.warning(f"Gemini call for {task_description} blocked. Reason: {response.prompt_feedback.block_reason}")
                 return None
            else:
                 logging.warning(f"Gemini response for {task_description} did not contain text or block reason. Response parts: {response.parts}")
                 return None

        except Exception as e:
            logging.error(f"Error calling Gemini API for {task_description} (Attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    logging.error(f"Max retries reached for {task_description}. Failed to get response.")
    return None
def call_gemini_synonym(summary: str) -> List[str]:
    example_summary = "The function takes two integer parameter num1 and num2. It calculates the sum of these two integers using the add operator and returns the result as an integer."

    prompt = f"""
You will be given a summary.
Generate 6 synonyms of the summary and return a list to me. 
Make sure the generated synonyms are meaningful sentences. 
Do not add any information that's not provided in the answer. Just return the list.

For example:
If the summary is: {example_summary}
Your output should be something like
['The function is given integers num1 and num2, employs + to add them, and yields the sum as an integer.', 'The function takes two integers, num1 and num2, adds them using the + operator, and returns the sum as an integer.','The function accepts two integer parameters, num1 and num2, computes their sum with addition, and returns the result as an integer.','Given two integers, num1 and num2, the function adds them together and outputs the total as an integer.','This function receives two integer inputs, num1 and num2, combines them using the + operator, and returns the sum in integer form.','The function takes num1 and num2 as integer arguments, performs addition on them, and gives back the result as an integer.']

Just return the list. Do not add anything before or after the list.

The summary: {summary}
"""
    response_text = call_gemini_api(prompt, f"generating synonyms...\n")
    return response_text

def call_gemini_verification(code: str, summary: str) -> str:
    prompt = f"""
    Assume you are an expert in understanding JAVA code.
    Your task is to verify whether the summary is hallucinated or not with respect to the code.
    Only output one of the following labels: ["HALLUCINATED", "GROUNDED", "NOT SURE"].
    Please examine very carefully, as the summary is very very likely to be hallucinated. Pay extreme attention to every detail. Ungrounded assumption is regarded as hallucination.

    
    Summary:
    {summary}

    [CODE]
    ```java
    {code}
    ```
    [/CODE]

    Label:
    """
    response_text = call_gemini_api(prompt, f"validating....\n").upper()
    if "HALLUCINATED" in response_text:
        return "HALLUCINATED"
    elif "GROUNDED" in response_text:
        return "GROUNDED"
    else:
        return "NOT SURE"

def check_consistency(code: str, summary: str) -> int:
    flag = True
    while flag == True:
        try:
            _synonyms = call_gemini_synonym(summary)
            synonyms = ast.literal_eval('[' + _synonyms.strip().split('[',1)[1].rsplit(']',1)[0] + ']')
            flag = False
        except Exception as e:
            print(f"An exception has occurred, but we are redoing the opertaion")
            print(f"the exception is {e}")
            flag = True

    synonym_hallucination_score = 0

    scores = []
    for idx,synonym in enumerate(synonyms):
        response = call_gemini_verification(code, synonym)
        logging.info(f"Validating synonym{idx} with code......\nThe response is {response}")
        if response == "HALLUCINATED":
            synonym_hallucination_score += 1
            scores.append(1)
        elif response == "GROUNDED":
            synonym_hallucination_score += 0
            scores.append(0)
        else:
            scores.append(0.5)
    
    print(f"scores are {scores}")
    synonym_hallucination_score /= len(synonyms)
    hallucination_score = synonym_hallucination_score
    return hallucination_score

VARIATION_NUM = 6

INPUT_FILE = "INPUT-FILE"
OUTPUT_FILE = "OUTPUT-FILE"


print(f"The INPUT_FILE is {INPUT_FILE}\nThe OUTPUT_FILE is {OUTPUT_FILE}")

import itertools
BEGIN_LINE = 0
END_LINE = None

with open(INPUT_FILE,"r") as input_file, open(OUTPUT_FILE,"a") as output_file:
    for idx,line in enumerate(itertools.islice(input_file,BEGIN_LINE,END_LINE)):
        p = json.loads(line)
        
        print(f"We are at line {idx + BEGIN_LINE}")

        if p["decision"] == True:
            print(f"The decision is true, nothing happen")
            json.dump(p,output_file)
            output_file.write('\n')
            continue
        score = check_consistency(p["code"], p["summary"])
        print(f"The score is {score}")
        if score == 0:
            p["decision"] = True
        else:
            p["decision"] = False
        json.dump(p,output_file)
        output_file.write('\n')