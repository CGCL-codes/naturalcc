import pandas as pd
import ast
import csv
from contextlib import redirect_stdout
from io import StringIO
import unicodedata
import re

###################################
### Dataframe related functions ###
###################################

def remove_merged_suffixes(df):
    # define a pattern to match the merged suffixes
    pattern = re.compile(r'^(.*) \.\d+$')
    
    # iterate over the columns
    for col in df.columns:
        # iterate over the values in the column
        for idx, value in df[col].items():
            match = pattern.match(str(value))
            if match:
                # if the value matches the pattern, replace it with the matched group
                new_value = match.group(1).strip()
                # check if the new value is in the column, including column name
                if new_value in df[col].drop(idx).values or new_value == col:
                    df.at[idx, col] = new_value
    return df

def markdown_to_df(markdown_string):
    """
    Parse a markdown table to a pandas dataframe.
    
    Parameters:
    markdown_string (str): The markdown table string.
    
    Returns:
    pd.DataFrame: The parsed markdown table as a pandas dataframe.
    """
    
    # Split the markdown string into lines
    lines = markdown_string.strip().split("\n")

    # strip leading/trailing '|'
    lines = [line.strip('|') for line in lines]

    # Check if the markdown string is empty or only contains the header and delimiter
    if len(lines) < 2:
        raise ValueError("Markdown string should contain at least a header, delimiter and one data row.")
        
    # Check if the markdown string contains the correct delimiter for a table
    if not set(lines[1].strip()) <= set(['-', '|', ' ', ':']):
        # means the second line is not a delimiter line
        # we do nothing
        pass
    # Remove the delimiter line
    else:
        del lines[1]

    # Join the lines back into a single string, and use StringIO to make it file-like
    markdown_file_like = StringIO("\n".join(lines))

    # Use pandas to read the "file", assuming the first row is the header and the separator is '|'
    df = pd.read_csv(markdown_file_like, sep='|', skipinitialspace=True, quoting=csv.QUOTE_NONE)

    # Strip whitespace from column names and values
    df.columns = df.columns.str.strip()

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # normalize unicode characters
    df = df.map(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

    return df

def print_partial_markdown(df, keep: int=3):
    # Concatenate the first `keep` and last `keep` rows of the dataframe
    combined_df = pd.concat([df.head(keep), df.tail(keep)])
    
    # Convert the combined dataframe to markdown
    markdown_output = combined_df.to_markdown(index=True)
    
    # Insert the "..." separator in the appropriate line
    markdown_lines = markdown_output.split('\n')
    separator_index = len(df.head(keep).to_markdown(index=True).split('\n'))
    markdown_lines.insert(separator_index, '...')
    
    # Join the lines to form the final markdown string and print
    final_output = '\n'.join(markdown_lines)
    
    return final_output

def convert_cells_to_numbers(df):
    # Helper function to remove commas and try to convert to numeric
    def to_numeric(cell):
        if isinstance(cell, str):  # Check if the cell is of string type
            no_comma = cell.replace(',', '')  # Remove commas
            # Check if the string without commas can be a float
            try:
                float(no_comma)
                return pd.to_numeric(no_comma, errors='coerce')
            except ValueError:
                return cell  # If it can't be a number, return the original cell
        return pd.to_numeric(cell, errors='coerce')

    
    # Apply the function to each cell in the dataframe
    return df.map(to_numeric)

def infer_dtype(df):
    """
    Attempt to convert columns in a DataFrame to a more appropriate data type.
    
    :param df: Input DataFrame
    :return: DataFrame with updated dtypes
    """
    
    for col in df.columns:
        # Try converting to numeric
        df[col] = pd.to_numeric(df[col], errors='ignore')

        # If the column type is still object (string) after trying numeric conversion, try datetime conversion
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            except:
                pass

    return df


def parse_code_from_string(input_string):
    """
    Parse executable code from a string, handling various markdown-like code block formats.

    Parameters:
    input_string (str): The input string.

    Returns:
    str: The parsed code.
    """

    # Pattern to match code blocks wrapped in triple backticks, with optional language specification
    triple_backtick_pattern = r"```(\w*\s*)?(.*?)```"
    match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()

    # Pattern to match code blocks wrapped in single backticks
    single_backtick_pattern = r"`(.*?)`"
    match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Default return if no code block patterns are matched
    return input_string.strip()


def python_repl_ast(code, custom_globals=None, custom_locals=None, memory=None):
    """
    Run command with own globals/locals and returns anything printed.

    Parameters:
    code (str): The code to execute.
    custom_globals (dict): The globals to use.
    custom_locals (dict): The locals to use.
    memory (dict): The state/memory to retain between invocations.

    Returns:
    tuple: (str: The output of the code, dict: updated memory).
    """

    if memory is None:
        memory = {}

    if custom_globals is None:
        custom_globals = globals().copy()
    else:
        custom_globals = {**globals(), **custom_globals}

    if custom_locals is None:
        custom_locals = memory.copy()
    else:
        custom_locals = {**custom_locals, **memory}

    try:
        tree = ast.parse(code)
        module = ast.Module(tree.body[:-1], type_ignores=[])

        # Execute all lines except the last
        exec(ast.unparse(module), custom_globals, custom_locals)

        # Prepare the last line
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)

        io_buffer = StringIO()

        # Redirect stdout to our buffer and attempt to evaluate the last line
        with redirect_stdout(io_buffer):
            try:
                ret = eval(module_end_str, custom_globals, custom_locals)
                if ret is not None:
                    output = str(ret)
                else:
                    output = io_buffer.getvalue()
            except Exception:
                # If evaluating fails, try executing it instead
                #try:
                    # 如果评估失败，尝试执行
                #    exec(module_end_str, custom_globals, custom_locals)
                #    output = io_buffer.getvalue()
                #except:
                #    # 忽略所有错误
                #    output = ""  # 设置一个默认输出
                exec(module_end_str, custom_globals, custom_locals)
                output = io_buffer.getvalue()

        # Update memory with new variable states
        memory.update(custom_locals)

        # Return any output captured during execution along with the updated memory
        return output, memory

    except Exception as e:
        return "{}: {}".format(type(e).__name__, str(e)), memory
