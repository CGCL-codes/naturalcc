import pandas as pd
from pandas import DataFrame
from dateutil.parser import parse
from fuzzywuzzy import process
import re
import unicodedata



def best_match(target, choices):
    # Find the best match for a target string from a list of choices
    best_match, score = process.extractOne(target, choices)
    return best_match if score > 80 else None # Only return a match if the score is above 80

def best_match_based_on_content(df, unmatched_name):
    for col in df.columns:
        # Check if the unmatched name is in every cell of the column
        matched_cells = df[col].head().apply(lambda x: unmatched_name in str(x)).sum()
        if matched_cells == len(df[col].head()):
            return col
    return None

def transpose(df: DataFrame, heading_deduplication=True) -> DataFrame:
    # Convert column names to a new row
    df = pd.concat([pd.DataFrame([df.columns], columns=df.columns), df]).reset_index(drop=True)
    
    # Transpose the dataframe
    df_transposed = df.T
    
    # Rename columns based on the first row
    df_transposed.columns = df_transposed.iloc[0]
    
    # Drop the first row
    df_transposed = df_transposed.drop(df_transposed.index[0])
    
    # Handle duplicate column names
    if heading_deduplication:
        cols = pd.Series(df_transposed.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + f" .{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        df_transposed.columns = cols

    # Reset the index without adding a new column
    df_transposed.reset_index(drop=True, inplace=True)

    df_transposed.columns.name = None
    
    return df_transposed

def determine_column_type(values):
    """Determine the type of the column based on all non-null values."""
    
    # Flags for determining the type
    is_float = True
    is_datetime = True
    
    for value in values:
        if value is not None and value != '':
            # Check for float
            if is_float:
                try:
                    # Handle numbers with commas as thousands separators
                    value_without_commas = str(value).replace(',', '')
                    float(value_without_commas)
                except ValueError:
                    is_float = False
            
            # Check for datetime
            if is_datetime:
                try:
                    parse(value)
                except:
                    is_datetime = False

            # If neither float nor datetime, it's a string
            if not is_float and not is_datetime:
                return 'string'

    if is_float:
        return 'float'
    elif is_datetime:
        return 'datetime'
    else:
        return 'string'

def try_convert(value, col_type):
    """Attempt to convert a value to a number or datetime based on the column type."""
    if col_type == 'float':
        try:
            # Handle numbers with commas as thousands separators
            value = str(value).replace(',', '')
            return float(value)
        except ValueError:
            return None
    elif col_type == 'datetime':
        try:
            return parse(value)
        except:
            return None
    else:
        return value

def natural_key(s):
    # convert to string
    s = str(s)

    # extract numbers
    numbers = [int(x) for x in re.findall(r'\d+', s)]

    # If no numbers are found, return a large number to ensure the string is placed at the end
    if not numbers:
        numbers = [float('inf')]
    
    # extract non-numbers
    non_numbers = re.sub(r'\d+', '', s)
    
    return (numbers, non_numbers)

def natural_sort(df, col):
    return df.loc[sorted(df.index, key=lambda idx: natural_key(df[col][idx]))]

def sort_dataframe(df, column_names):
    # if the column_names is ['N/A']
    if len(column_names) == 1 and column_names[0] == 'N/A':
        # no need to sort, just return the dataframe
        return df

    # Find the closest matching columns
    df.columns = df.columns.str.replace("\\n", "")

    actual_columns = []

    # first try to match columns with commas
    for col in df.columns:
        if ',' in str(col):
            # split the column name by commas
            parts = col.split(", ")
            matched = all(any(part in name for name in column_names) for part in parts)
            if matched:
                actual_columns.append(col)
                # remove the matched parts from column_names
                for part in parts:
                    column_names = [name for name in column_names if part not in name]

    # then try to match columns 
    for col in column_names:
        if col in df.columns:
            actual_columns.append(col)
        else:
            match = best_match(col, df.columns)
            if match:
                actual_columns.append(match)
            else:
                match_based_on_content = best_match_based_on_content(df, col)
                if match_based_on_content:
                    actual_columns.append(match_based_on_content)
                else:
                    print(f"Warning: No close match found for column: {col} vs {df.columns}")

    sort_columns = []
    # Convert columns to sortable format
    for col in actual_columns:
        # get the position of the column (in case there are multiple columns with the same name)
        col_positions = [i for i, name in enumerate(df.columns) if name == col]
        # if there are multiple columns with the same name, use the leftmost one
        leftmost_col_position = col_positions[0]

        # Get column data and type
        col_data = df.iloc[:, leftmost_col_position]
        col_type = determine_column_type(col_data)

        # If column type is string, apply natural sorting
        df[col + "_sort"] = col_data.apply(lambda x: try_convert(x, col_type))

        # add the column to the sort_columns
        sort_columns.append((col + "_sort", col_type))

    # Sort dataframe sequentially for each column in reverse order
    for col, col_type in reversed(sort_columns):
        
        # If the column is a string type, use natsorted
        if col_type == 'string':
            df = natural_sort(df, col)
        
        # Otherwise, use sort_values
        else:
            df = df.sort_values(by=col)


    # Drop temporary columns
    for col in actual_columns:
        df = df.drop(columns=[col + "_sort"])

    # Reset index
    df = df.reset_index(drop=True)

    return df

def normalize_string(s):
    """Normalize the string using NFC normalization."""
    return unicodedata.normalize('NFC', s)

def parse_markdown_table(table):
    """Parse a markdown table and return a list of lists."""
    lines = normalize_string(table).strip().split('\n')
    # Remove the separator line
    lines = [line for i, line in enumerate(lines) if not (i == 1 and set(line.replace(':', '-')) in [{'-', ' ', '|'}, {'-', '|'}])]
    # Split each line into cells
    return [list(map(str.strip, line.split('|')[1:-1])) for line in lines if line]

def markdown_tables_equal(table1, table2):
    """Check if two markdown tables are equal. If not, return the differences."""
    parsed_table1 = parse_markdown_table(table1)
    parsed_table2 = parse_markdown_table(table2)

    if parsed_table1 == parsed_table2:
        return True, []

    differences = []
    for row_idx, (row1, row2) in enumerate(zip(parsed_table1, parsed_table2)):
        for col_idx, (cell1, cell2) in enumerate(zip(row1, row2)):
            if cell1 != cell2:
                differences.append((row_idx, col_idx, cell1, cell2))

    return False, differences
