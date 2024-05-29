
from utils import *
from vector_database.utils import Tools
from vector_database.search_code import DependencyTableSearchWrapper

def search_dependency_table(query_text, task_body, table_name, symbol_names = None, symbol_field = None, search_scope = None, scope_field = None):
    # 1. Exact Match

    # 2. Fuzz Match
    Emb = Tools.ada002_embedding(query_text)
    DS = DependencyTableSearchWrapper()
    result = DS.search_dependency_table(repo_name=task_body['repo_dirname'], table_name=table_name,
                                        embedding_vector=Emb,
                                        symbol_names=symbol_names,
                                        symbol_field=symbol_field,
                                        search_scope=search_scope,
                                        scope_field=scope_field)
    return result


def filter_relevant_records(list_of_record_lists, max_items=10):
    if not list_of_record_lists:
        return []

    # Flatten the list of lists while keeping track of the category (sublist index)
    all_records = [(record, category) for category, records in enumerate(list_of_record_lists) for record in records]
    # Sort by similarity score in descending order
    sorted_records = sorted(all_records, key=lambda x: x[0][1], reverse=True)

    # Filtering
    filtered_records = []
    categories_represented = set()

    for record, category in sorted_records:
        if category not in categories_represented or len(filtered_records) < max_items:
            filtered_records.append(record)
            categories_represented.add(category)
            if len(filtered_records) == max_items:
                break

    for record, category in sorted_records:
        if category not in categories_represented:
            filtered_records.append(record)
            categories_represented.add(category)
    return filtered_records


def _E0602_extract_chain(lead_identifier, line):
    results = []
    split_line = split_identifiers_non_identifiers(line)
    for idx, item in enumerate(split_line):
        if(item == lead_identifier):
            for item2 in split_line[idx:]:
                if(is_identifier(item2)):
                    results.append(item2)
                else:
                    if(item2.strip() == '.'):
                        continue
                    else:
                        break
    return results


def retrieved_item_to_string(item):
    result_string = ""
    result_string += '```\n'
    if('gfname' in item[0]['metadata']): # global function
        result_string += item[0]['metadata']['m'].replace('Module', 'from') + " " + item[0]['metadata']["gf"].replace('Function', 'import') + '\n'
        result_string += item[0]['metadata']['content'] + " ..."
    elif('cname' in item[0]['metadata']): # class
        result_string += item[0]['metadata']['m'].replace('Module', 'from') + " " + item[0]['metadata']["c"].replace('Class', 'import') + '\n'
    elif('cfname' in item[0]['metadata']): # class function
        result_string += item[0]['metadata']['m'].replace('Module', 'from') + " " + \
                         item[0]['metadata']["c"].replace('Class', 'import') + '\n' + \
                         item[0]['metadata']['c'].replace('Class', 'class') + ':\n    ' + \
                         item[0]['metadata']['content'] + " ..."
    else:
        raise AssertionError("Assertion Failed")
    result_string += '\n```'
    return result_string
    pass

def E0602_handler(diagnostic_body, task_body, max_items = 5):
    # search M-C, and
    # M-C
    # extract symbol
    symbol_list = extract_single_quoted_strings(diagnostic_body['message'])
    if(len(symbol_list) != 1):
        raise AssertionError("E0602 pattern not match")
    symbol_names = _E0602_extract_chain(lead_identifier=symbol_list[0],
                                        line=diagnostic_body['line_content'])
    query_text = f"In line: {diagnostic_body['line_content']}\nError: {diagnostic_body['message']}\nCurrent Module: {diagnostic_body['module']}\n"
    # if current module == undefined symbol name, report it

    M_C_context = search_dependency_table(
        query_text=query_text,
        task_body=task_body,
        table_name="M_C",
        symbol_names=symbol_names,
        symbol_field="cname")
    # M_GF
    M_GF_context = search_dependency_table(
        query_text=query_text,
        task_body=task_body,
        table_name="M_GF_GFF_GFSL_GFEL",
        symbol_names=symbol_names,
        symbol_field="gfname"
    )

    # M_C_CF
    M_C_CF_context = search_dependency_table(
        query_text=query_text,
        task_body=task_body,
        table_name="M_C_CF_CFF_CFSL_CFEL",
        symbol_names=symbol_names,
        symbol_field="cfname"
    )

    #

    retrieved_contexts = filter_relevant_records([M_C_context, M_GF_context, M_C_CF_context], max_items=max_items)
    pass

    # generate report message
    message = f"In line: {diagnostic_body['line_content']}\nError: There is no symbol named '{symbol_names[0]}' in current context.\nCurrent Module: {diagnostic_body['module']}\n"
    if (diagnostic_body['module'].split('.')[-1] == symbol_names[0]):
        message += f"May not need to add module qualifier '{symbol_names[0]}'.\n"
    message += "I found other symbols defined in the repository, you may use them:\n"
    for item in retrieved_contexts:
        message += retrieved_item_to_string(item) + '\n'
    # Filtering, keep the most relevant 10 items (and ensure each category has at least one item
    # each return value is a list, the item in the list is a tuple, item[0] is retrieved context body, item[1] (you should consider) is similarity score
    return message

def E1101_handler(diagnostic_body, task_body, max_items = 5): # no name in module
    # M_C_CF
    symbol_list = extract_single_quoted_strings(diagnostic_body['message'])
    if(len(symbol_list) != 2 and len(symbol_list) != 3):
        raise AssertionError("E0602 pattern not match")
    class_name = symbol_list[0]
    symbol_name = symbol_list[1]
    symbol_names = _E0602_extract_chain(lead_identifier=symbol_name,
                                        line=diagnostic_body['line_content'])
    query_text = f"In line: {diagnostic_body['line_content']}\nError: {diagnostic_body['message']}\nCurrent Module: {diagnostic_body['module']}\n"

    # search with qualified scope
    M_C_CF_context = search_dependency_table(
        query_text=query_text,
        task_body=task_body,
        table_name="M_C_CF_CFF_CFSL_CFEL",
        symbol_names=symbol_names,
        symbol_field="cfname",
        search_scope=class_name,
        scope_field="c",
    )

    retrieved_contexts = filter_relevant_records([M_C_CF_context], max_items=max_items)

    # generate report message
    message = f"In line: {diagnostic_body['line_content']}\nError: The class '{class_name}' has no member named '{symbol_name}'.\nCurrent Module: {diagnostic_body['module']}\n"
    message += "I found other member functions defined in the class, you may use them:\n"
    for item in retrieved_contexts:
        message += retrieved_item_to_string(item) + '\n'
    # Filtering, keep the most relevant 10 items (and ensure each category has at least one item
    # each return value is a list, the item in the list is a tuple, item[0] is retrieved context body, item[1] (you should consider) is similarity score
    return message

    pass

def E0102_handler(diagnostic_body, task_body, max_items=5): # function redefined
    # todo: add function signature.
    message = f"In line: {diagnostic_body['line_content']}\nError: This function is already defined in previous context, you may directly use it."
    return message