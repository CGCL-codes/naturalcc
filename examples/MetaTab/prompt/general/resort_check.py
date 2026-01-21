sort_prompt = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]":

[TABLE]

Note: Only selected rows from the beginning and end of the table are displayed for brevity. Intermediate rows are omitted and represented by "..." for clarity.

The table column headings are provided below, separated by semicolons:

[HEADINGS]

In order to optimize the interpretability and readability of the data, follow these guidelines to determine the most suitable sorting method:

Sorting Guidelines:

1. Evaluate columns based on data types such as numerical, alphabetical, chronological, categorical, or other relevant sorting methods.
2. Identify any patterns or relationships in the data that would be highlighted by certain sorting methods.
3. Consider column position, as those on the left may sometimes have sorting priority.
4. If applicable, consider sorting by multiple columns in a prioritized sequence.

Provide your decision using one of the following statements:

- For sorting using a single column: "Sort by: [Name of Column]".
- For sorting using multiple columns: "Sort by: [Primary Column Name], [Secondary Column Name], ...".
- If no specific sorting seems advantageous: "Sort by: N/A".

Your response should strictly follow the formats provided.
"""