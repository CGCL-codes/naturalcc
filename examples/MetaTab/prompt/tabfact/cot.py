cot_prompt = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]".

[TABLE]

Based on the given table, check the following statement is true or false:

[QUESTION]

Let's think step by step to verify the statement, and then give the final answer. Ensure the final answer format is only "Final Answer: Yes/No" form, no other form. And ensure the final answer is only a Yes or No, without any explanation.
"""