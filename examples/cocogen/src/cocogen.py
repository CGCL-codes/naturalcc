import argparse
import os
import tempfile
from tqdm import tqdm
from utils import *
import dataset_utils
from dataset_utils import read_example_data, read_codereval_data
from comp_utils import safe_completion, length_of_prompt
from parsing_utils import substitute_function_in_code
from diagnostic_handlers import E0602_handler, E1101_handler, E0102_handler


def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argument(parser)
    parser.add_argument('--style', type=str, default="standard")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_iterations', type=int, default=3, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=2)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=3)  # debug
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result', default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--with_context', default=True, action='store_true')
    parser.add_argument('--show_prompt', default=False, action='store_true')
    parser.add_argument('--split', default='python')
    parser.add_argument('--db_name', default='codereval')
    parser.add_argument('--db_pass', default='')
    args = parser.parse_args()
    specify_engine(args)
    return args


def result_cache_name(args):
    return "misc/Dense+Verification+Hint_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.json".format(args.split,
                                                                                                     args.engine_name,
                                                                                                     args.train_slice,
                                                                                                     args.train_slice + args.num_shot,
                                                                                                     args.dev_slice,
                                                                                                     args.dev_slice + args.num_dev,
                                                                                                     args.num_distractor,
                                                                                                     args.style)

def result_codereval_cache_name(args):
    return "misc/Dense+Verification+Hint_{}_predictions_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.jsonl".format(
        args.split,
        args.engine_name,
        args.train_slice,
        args.train_slice + args.num_shot,
        args.dev_slice,
        args.dev_slice + args.num_dev,
        args.num_distractor,
        args.style)

def convert_paragraphs_to_context(s, connction='\n', n=1):
    context_list = [x[0]['context'] for x in reversed(s['top_k_context_ada002'])]
    context_list = context_list[:n]
    context_list = reversed(context_list)  # place context of higher similarity near
    return '```\n' + connction.join(['{}'.format(p) for i, p in enumerate(context_list)]) + '\n```'

def run_pylint(code_content, file_to_check, temp_module_name, current_module_name):
    code_lines = code_content.splitlines()
    format_string = "{line}:{C}:{msg_id}:{obj}:{module}:{msg}:{symbol}"
    # Command to run pylint
    command = ['pylint', "--disable=C", file_to_check, f"--msg-template='{format_string}'"]

    results = []
    # Running the command and capturing output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    for line in stdout.decode().splitlines():
        parts = line.split(':')
        if (len(parts) < 6):
            continue
        try:
            resp = {
                "line": int(parts[0]),
                "line_content": code_lines[int(parts[0]) - 1],
                "category": parts[1],
                "diagnostic_type": parts[2],
                "related_object": parts[3],
                "module": parts[4].replace(temp_module_name, current_module_name),
                "message": parts[5],
                "message_symbolic_name": parts[6]
            }
            results.append(resp)
        except:
            pass
    return results




def llm_sql(error_line, error_message):
    prompt = sql_prompt.format(error_line, error_message)
    print(f"PROMPT={prompt}")
    comp = safe_completion("gpt-3.5-turbo", prompt, 1024, n=1, stop=dataset_utils._END_PREDICTION_SEP, temp=0.0,
                           logprobs=5)
    sql_query = comp['choices'][0]['text'].lstrip("```sql\n").rstrip('\n```')
    return sql_query


def sql_exec(sql_query, task_body):
    # we cache the extracted semantic database for efficiency reasons
    codeql_exec = "/home/bi/Code/ProjectEval/third_party/codeql/codeql-home/codeql"
    codeql_db = f"./data/CoderEval_codeql_db/{task_body['repo_dirname']}"
    sql_query = "import python\n" + sql_query
    # create a temporary '.ql' file to save the query
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.ql', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(sql_query)
        temp_file.flush()
    # write qlpack file
    with open("/tmp/qlpack.yml", "w") as f:
        f.write(qlpack_str)
    # execute the query and fetch the outputs from standard output
    cmd = f"{codeql_exec} query run {temp_filename} --database={codeql_db}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    os.remove(temp_filename)
    return stdout
    pass


def diagnostic_handler(diagnostic_body, task_body) -> (bool, str):
    diagnostic_type = diagnostic_body['diagnostic_type']
    default_diagnostic_message = "In line: " + diagnostic_body["line_content"] + " . " + diagnostic_body["message"]

    # handle the frequently appeared cases by hand
    if (diagnostic_type == 'E0213'):  # Skip 'no-self-argument' error, due to the inconsistency of CoderEval Dataset
        return False, ""
    if (diagnostic_type == 'E0001'):
        return True, default_diagnostic_message
    if (diagnostic_type == 'E0602'):  # undefined-variable
        diag_msg = E0602_handler(diagnostic_body, task_body)
        return True, diag_msg
    if (diagnostic_type == 'E1101'):  # no-member
        diag_msg = E1101_handler(diagnostic_body, task_body)
        return True, diag_msg
    if (diagnostic_type == 'E0102'):  # function redefined
        diag_msg = E0102_handler(diagnostic_body, task_body)
        return True, diag_msg
    if (diagnostic_body["category"] == "E"):
        llm_query = llm_sql(diagnostic_body['line_content'], diagnostic_body['message'])
        result_body = sql_exec(llm_query, task_body)
        return True, default_diagnostic_message + "\n" + "Possible related functions: \n" + result_body.decode("utf-8")
    return False, ""
    pass


def analyze_pylint_message(diagnostic_items, interested_start_line_num, interested_end_line_num, task_body):
    collected_messages = []
    has_error = False
    seen_messages = set()  # Set to store messages we've already seen

    for diagnostic_item in diagnostic_items:
        line_number = diagnostic_item["line"]
        message = diagnostic_item["message"]
        if interested_start_line_num <= line_number and line_number < interested_end_line_num:
            print(f"[VERIFICATION] MSG = \n{diagnostic_item}")

            # Skip if we've already seen this message
            if message in seen_messages:
                continue

            current_has_error, current_message = diagnostic_handler(diagnostic_body=diagnostic_item,
                                                                    task_body=task_body)
            if current_has_error:
                has_error = True
                collected_messages.append(current_message)
                seen_messages.add(message)  # Add the message to the set

    return has_error, "\n".join(collected_messages)


def run_semantic_checker(new_code, original_file_path, interested_start_line_num, interested_end_line_num, language,
                         task_body):
    # Get the directory of the original file
    task_directory = os.path.dirname(original_file_path)
    module_name = os.path.basename(original_file_path).split('.py')[0]
    # Determine the file extension based on the language
    if language == 'python':
        extension = '.py'
    elif language == 'java':
        extension = '.java'
    else:
        raise ValueError("Unsupported language")

    with tempfile.NamedTemporaryFile(mode='w+', dir=task_directory, suffix=extension, delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(new_code)
        temp_file.flush()

    diagnostic_itmes = run_pylint(code_content=new_code,
                                  file_to_check=temp_filename,
                                  temp_module_name=os.path.basename(temp_filename).split('.py')[0],
                                  current_module_name=module_name)
    os.remove(temp_filename)
    has_error, diagnostic_message = analyze_pylint_message(diagnostic_items=diagnostic_itmes,
                                                           interested_start_line_num=interested_start_line_num,
                                                           interested_end_line_num=interested_end_line_num,
                                                           task_body=task_body)
    return has_error, diagnostic_message


def verify_generated_code(generated_code, task, args):
    gold_text = task['gold']
    gold_text = remove_leading_indent(gold_text)

    test_case_code = task['file_content']
    substituted_code, start_line, end_line = substitute_function_in_code(code=test_case_code,
                                                                         function_name=task[
                                                                             'function_name'],
                                                                         new_body=gold_text,
                                                                         language=args.split)

    has_error, diagnostic_message = run_semantic_checker(new_code=substituted_code,
                                                         original_file_path=task['test_case_file'],
                                                         interested_start_line_num=start_line,
                                                         interested_end_line_num=end_line,
                                                         language=args.split,
                                                         task_body=task)
    if (has_error):
        return True, ""
    test_case_code = task['file_content']
    substituted_code, start_line, end_line = substitute_function_in_code(code=test_case_code,
                                                                         function_name=task[
                                                                             'function_name'],
                                                                         new_body=generated_code,
                                                                         language=args.split)

    has_error, diagnostic_message = run_semantic_checker(new_code=substituted_code,
                                                         original_file_path=task['test_case_file'],
                                                         interested_start_line_num=start_line,
                                                         interested_end_line_num=end_line,
                                                         language=args.split,
                                                         task_body=task)
    if (has_error):
        return False, diagnostic_message
    return True, ""
    pass


def in_context_prediction(context, ex, shots, engine, style="standard", length_test_only=False, previous_code=None,
                          fail_reason=None, n=10):
    print(f"engine = {engine}]")
    if style == "standard":
        if context:
            showcase_examples = [
                "Q: \"{}\"\nA:\n{}\n".format(s["problem"], s["solution"]) for s in shots
            ]
            if (fail_reason is not None):
                input_example = "{}\nQ: \"{}\"\n{}\nA:\n".format(
                    convert_paragraphs_to_context(ex, n=3 if engine == 'gpt-3.5-turbo' else 1),
                    ex["problem"],
                    f"\nPrevious Answer: \n```\n{previous_code}\n```\nFail Reason:{fail_reason}"
                )
            else:
                input_example = "{}\nQ: \"{}\"\nA:\n".format(
                    convert_paragraphs_to_context(ex, n=3 if engine == 'gpt-3.5-turbo' else 1), ex["problem"])
        else:
            showcase_examples = [
                "Q: \"{}\"\nA:\n{}\n".format(s["problem"], s["solution"]) for s in shots
            ]
            if (fail_reason is not None):
                input_example = "Q: \"{}\"\n{}\nA:\n".format(
                    ex["problem"],
                    f"\nPrevious Answer: \n```\n{previous_code}\n```\nFail Reason:{fail_reason}"
                )
            else:
                input_example = "{}\nQ: \"{}\"\nA:\n".format(
                    convert_paragraphs_to_context(ex, n=3 if engine == 'gpt-3.5-turbo' else 1), ex["problem"])

        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")

    try:

        if length_test_only:
            pred = length_of_prompt(prompt, 32)
            print("-----------------------------------------")
            print(pred)
            print(prompt)
            return pred
        else:
            comp = safe_completion(engine, prompt, 1024, n=n, stop=dataset_utils._END_PREDICTION_SEP, temp=0.0,
                                   logprobs=5)
        pred = {}

        pred["id"] = ex["id"]
        pred["prompt"] = prompt
        pred['texts'] = []

        choices = comp['choices']

        def extract_code_from_string(s):
            # Regular expression to match strings surrounded by ```python xxx ```
            match = re.search(r'```(python|java)\s+(.*?)\s+```', s, re.DOTALL)
            if match:
                # Extract and return the code part
                return match.group(2)
            else:
                # Return the string itself if no match is found
                return s

        for item in choices:
            print(f"---------------Predicted Text--------------\n{item['text']}")
            extracted_text = extract_code_from_string(item['text'])
            pred['texts'].append(extracted_text)

        return pred
    except:
        raise
        pred = {}

        pred["id"] = ex["id"]
        pred["prompt"] = prompt
        pred['texts'] = []
        return pred


def test_few_shot_performance(args):
    print("Running prediction")
    train_set = read_example_data(f"data/{args.split}_train.json",
                                  split=args.split,
                                  manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_codereval_data(f"data/CoderEval4{args.split}_search.json",
                                  workdir=os.path.abspath(f'./data/CoderEval_ds/{args.split}/'),
                                  args=args)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    if args.show_prompt:
        showcase_examples = [
            "Q: \"{}\"\nA:\n\"{}\"\n".format(s["question"], s["answer"]) for s in train_set
        ]
        prompt = "\n".join(showcase_examples)
        print(prompt)
        raise Exception('prompt shown')

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        predictions = []
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            # 1. first-round completion
            pred = in_context_prediction(args.with_context, x, train_set, engine=args.engine, \
                                         style=args.style, length_test_only=args.run_length_test, n=10)

            # 2. Verify the generated results
            verified_texts = []
            for text in pred['texts']:
                iterations = 0
                current_code = text
                while True:
                    success, reason = verify_generated_code(generated_code=current_code, task=x,
                                                            args=args)
                    if (success == True or iterations >= args.num_iterations):
                        break
                    pred = in_context_prediction(False, x, train_set, engine=args.engine, \
                                                 style=args.style, length_test_only=args.run_length_test,
                                                 previous_code=current_code, fail_reason=reason, n=1)
                    # if overflow, exit
                    if (pred['texts'][0] == 'Input Overflow'):
                        print("Overflow triggered, immediately return the previous generated code")
                        break
                    current_code = pred['texts'][0]
                    iterations += 1
                verified_texts.append(current_code)
            pred['texts'] = verified_texts
            if pred == None:
                raise AssertionError("Assertion Failed")
            else:
                predictions.append(pred)

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', 32)
            return
        # save
        dump_json(predictions, result_cache_name(args))
    export_to_codereval_evaluation_pipeline(args)


def export_to_codereval_evaluation_pipeline(args):
    predictions = read_json(result_cache_name(args))
    completions = []
    for p in predictions:
        completion = {
            "_id": p["id"],
            "generate_results": [x.lstrip() for x in p['texts']]
        }
        completions.append(completion)
    print(len(predictions))
    print(result_codereval_cache_name(args))
    dump_jsonl(completions, result_codereval_cache_name(args))


if __name__ == '__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        raise AssertionError("Performance Evaluation is finished using CoderEval Docker Environment")
