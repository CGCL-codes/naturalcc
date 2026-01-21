eval_prompt = """
Below is a markdown table regarding "[TITLE]":

[TABLE]

You're tasked with answering the following question:

[QUESTION]

You have 2 answers derived by two different methods. Answer A was derived by prompting the AI to think step-by-step. Answer B was derived by interacting with a Python Shell.

Answer A is [COT_ANSWER].
Answer B is [AGENT_ANSWER].

Your task is to determine which is the correct answer. It is crucial that you strictly adhere to the following evaluation process:

1. **Preliminary Evaluation**: Begin by evaluating which of the two answers directly addresses the question in a straightforward and unambiguous manner. A direct answer provides a clear response that aligns closely with the query without introducing additional or extraneous details. If one of the answers is not a direct response to the question, simply disregard it.
2. **Nature of the Question**: If both answers appear to be direct answers, then evaluate the nature of the question. For tasks involving computation, counting, and column-locating, especially when for extensive table, the Python Shell (Answer B) might be more precise. However, always remain cautious if the Python Shell's output appears off (e.g., error messages, success notifications, etc.). Such outputs may not be trustworthy for a correct answer.
3. **Final Verdict**: Finally, after thorough evaluation and explanation, provide your verdict strictly following the given format:
  - Use "[[A]]" if Answer A is correct.
  - Use "[[B]]" if Answer B is correct.

Note: 
1. Each method has its own strengths and weaknesses. Evaluate them with with an unbiased perspective. When in doubt, consider the nature of the question and lean towards the method that is most suited for such queries.
2. Ensure that your verdict is provided after evaluation, at the end.
"""