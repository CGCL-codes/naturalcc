header_check_prompt = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]".

[TABLE]

Headings of a table are labels or titles given to rows or columns to provide a brief description of the data they contain.

Based on the given table, the headings of the table are more likely to be:

(A) [FIRST_ROW]
(B) [FIRST_COLUMN]
(C) None of the above

Directly give your choice. Ensure the format is only "Choice: (A)/(B)/(C)" form, no other form, without any explanation.
"""