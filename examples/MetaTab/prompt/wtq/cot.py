cot_prompt = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]".

[TABLE]

Based on the given table, answer the following question:

[QUESTION]

Let's think step by step, and then give the final answer. Ensure the final answer format is only "Final Answer: AnswerName1, AnswerName2..." form, no other form. And ensure the final answer is a number or entity names, as short as possible, without any explanation.
"""