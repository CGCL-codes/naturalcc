import re
import string
import unicodedata

# parse the reuslt from header checking
def parse_header_checking_result(output):
    """
    1. "Choice: (A)" -> "(A)", "Choice: (B)" -> "(B)", "Choice: (C)" -> "(C)"
    2. "(A)" -> False, "(B)" -> True, "(C)" -> False
    """
    # parse the choice
    match = re.search(r'\(A\)|\(B\)|\(C\)', output)
    
    result = match.group(0) if match else None
    
    # if (A) or (C), return False, if (B) return True
    if "B" in result:
        return True
    else:
        return False


def parse_header_sorting_result(output):
    """
    Format: Sort by: Name1, Name2 -> [Name1, Name2]
    """
    
    # regex pattern to match the sorting criteria
    pattern = r'Sort by: (.*?)(\n|$)'
    
    match = re.search(pattern, output, re.I)  # Using re.I for case-insensitive matching
    if not match:
        return None
    
    # split the captured string by commas to extract individual names
    sorting_criteria = match.group(1).split(',')
    
    # process names to optionally drop "ascending" or "descending"
    names = []
    for name in sorting_criteria:
        name = name.strip()
        if re.search(r' ascending$', name, re.I):
            name = re.sub(r' ascending$', '', name, flags=re.I)
        elif re.search(r' \(ascending\)$', name, re.I):
            name = re.sub(r' \(ascending\)$', '', name, flags=re.I)
        elif re.search(r' descending$', name, re.I):
            name = re.sub(r' descending$', '', name, flags=re.I)
        elif re.search(r' \(descending\)$', name, re.I):
            name = re.sub(r' \(descending\)$', '', name, flags=re.I)
        names.append(name)
    
    return names


def extract_markdown_tables(text):
    """
    Extracts markdown tables from a text.

    Parameters:
    text (str): The response text.

    Returns:
    list: A list of markdown tables, usually only one.
    """
    # Regular expression for markdown tables
    pattern = r"((?:\|.*\|\s*\n?)+)"
    tables = re.findall(pattern, text)

    # Strip any leading/trailing white spaces from the tables
    tables = [table.strip() for table in tables]

    return tables

def normalize_md_table(table):
    """
    Normalizes a markdown table by removing markdown syntax and extra white space.

    Parameters:
    table (str): The markdown table.

    Returns:
    str: The normalized markdown table.
    """
    # Split the table into lines
    lines = table.strip().split("\n")

    # Filter out the lines that only contain '|', '-', and spaces
    lines = [line for line in lines if not set(line.strip()).issubset({"|", "-", " "})]

    # Remove markdown symbols and strip extra white space from each line
    lines = [line.replace("|", "").strip() for line in lines]

    # Split cells by spaces, remove empty cells and join them again
    lines = [' '.join(filter(None, line.split(" "))) for line in lines]

    return lines

def check_md_tables_equal(pred, target):
    """
    Checks if two markdown tables are equal.

    Parameters:
    pred (str): The predicted markdown table.
    target (str): The target markdown table.

    Returns:
    bool: Whether the two tables are equal.
    """
    
    return normalize_md_table(pred) == normalize_md_table(target)


def count_rows_columns_markdown_table(markdown_table):
    lines = markdown_table.strip().split('\n')

    # check whether delimiter line exists
    delimiter_line_exists = False
    for line in lines:
        if set(line.strip()) == set(['-', '|', ' ']):
            delimiter_line_exists = True
            break

    # get number of rows
    if delimiter_line_exists:
        row_count = len(lines) - 1
    else:
        row_count = len(lines) 

    # get number of columns
    column_count = lines[0].count("|") - 1  # excluding extra '|' at start and end

    return row_count, column_count

def extract_answer(text:str, patterns:list = [r"Final Answer: (.*)", r": (.*)", r"is (.*)"], return_match_flag=False):
    """
    Extracts the answer from a response text.

    Parameters:
    text (str): The response text.

    Returns:
    str: The extracted answer.
    """
    # Regular expression patterns
    patterns = patterns
    answer = None
    match_flag = False

    # convert text to lower case to ignore case
    text = text.lower()

    for pattern in patterns:
        # find matches
        matches = re.findall(pattern, text, re.IGNORECASE)
        # if matches found, update answer with the last match
        if matches:
            answer = matches[-1]
            if "final answer" in pattern.lower():
                match_flag = True
            
            if return_match_flag:
                return answer, match_flag
            return answer

    if return_match_flag:
        return answer, match_flag
    return answer


def normalize_tabfact_answer(answer) -> bool:
    if not answer:
        return answer
    # normalize the answer Yes/True/yes, etc. to True
    if "yes" in answer.lower() or "true" in answer.lower():
        return True
    # normalize the answer No/False/no, etc. to False
    elif "no" in answer.lower() or "false" in answer.lower():
        return False
    else:
        return answer

def maybe_normalize_float(span: str):
    if span and (re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
                 or (re.match(r"^[0-9]*[.]?[0-9]*$", span))) and span != '.':
        # FIXME: We did this(instead of try except) to convert a string into a float
        #  since the try catch will lead to an error when using 8 V100 gpus with cuda 11.0,
        #  and we still don't know why that could happen....
        return str(float(span))
    else:
        return span


def maybe_normalize_number(text: str) -> str:
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    for index, unit in enumerate(units):
        if text == unit:
            return str(float(index))
    return text

def normalize_false_true(text: str) -> str:
    if text.lower() == "false":
        return "no"
    elif text.lower() == "true":
        return "yes"
    else:
        return text
    
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def remove_punc(text: str) -> str:
    # Step 1: Remove inner content of parentheses using regular expressions
    text = re.sub(r'\([^)]*\)', '', text).strip()

    # Step 2: Remove all punctuation
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def remove_articles(text: str) -> str:
    return re.sub(r'\b(a|an|the)\b', ' ', text)



def eval_ex_match(pred, gold_result, log=False):
    pred = pred.lower()
    gold_result = gold_result.lower()

    # Replace and with comma
    if ' and ' in pred and '|' in gold_result:
        pred = pred.replace(' and ', ', ')

    pred = [span.strip() for span in pred.split(', ')]

    if '|' in gold_result:
        gold_result = [span.strip() for span in gold_result.split('|')]
    else:
        gold_result = [span.strip() for span in gold_result.split(', ')]

    pred = [normalize_unicode(normalize_false_true(maybe_normalize_number(remove_punc(remove_articles(span.strip()))))) for span in pred]
    gold_result = [normalize_unicode(normalize_false_true(maybe_normalize_number(remove_punc(remove_articles(span.strip()))))) for span in gold_result]

    # print(pred, ' # ', gold_result)
    clean_float = True  # TODO
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]
    
    res = sorted(pred) == sorted(gold_result)
    
    if not res:
        # it is possible that the answer is a number, but they add a unit to it, like 1.5 m vs 1.5 or 1.5 mile vs 1.5
        if len(pred) == len(gold_result) == 1:
            

            pred_no_unit = [re.sub(r'(\d+\.?\d*) \w+', r'\1', span)  for span in pred]
            gold_result_no_unit = [re.sub(r'(\d+\.?\d*) \w+', r'\1', span) for span in gold_result]


            pred_no_unit = [maybe_normalize_float(span) for span in pred_no_unit]
            gold_result_no_unit = [maybe_normalize_float(span) for span in gold_result_no_unit]

            # res = sorted(pred_no_unit) == sorted(gold_result_no_unit)

            res = pred_no_unit == gold_result_no_unit
    
    if res == 0 and log:
        print(f"{pred} # {gold_result}")

    return res