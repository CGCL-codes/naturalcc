from modules.apk import infer_simple_type_from_assignment,infer_simple_type_from_return
from modules.llms import  call_LLM,judge,condition_LLM
from modules.dataflow import construct_graph_of_target_code
from modules.compile import condition_judge
def If_Analysis(p, condition):
    """处理 If_Analysis(condition)"""
    condition = condition_judge(condition)
    if (True or False) not in condition:
        condition = condition_LLM(condition)
    print(f"Processing If_Analysis: {condition}")
    return {"type": "If", "condition": condition}

def Assignment_Analysis(p, value,scope):
    """处理 Assignment_Analysis(value)"""
    scope = "local"
    simple_type = infer_simple_type_from_assignment(value)
    CGDG = construct_graph_of_target_code(p)
    answer_def = call_LLM(CGDG['def'],scope)
    answer_user = call_LLM(CGDG['user'],scope)
    answer_usee = call_LLM(CGDG['usee'],scope)
    final_type = judge(simple_type,answer_def,answer_user,answer_usee)

    value = final_type
    print(f"Processing Assignment_Analysis: {value}")
    return {"type": "Assignment", "value": value}

def Return_Analysis(value,scope):
    """处理 Return_Analysis(value)"""
    scope = "return"
    simple_type = infer_simple_type_from_return(value)
    CGDG = construct_graph_of_target_code(p)
    answer_def = call_LLM(CGDG['def'],scope)
    answer_user = call_LLM(CGDG['user'],scope)
    answer_usee = call_LLM(CGDG['usee'],scope)
    final_type = judge(simple_type, answer_def, answer_user, answer_usee)
    value = final_type
    print(f"Processing Return_Analysis: {value}")
    return {"type": "Return", "value": value}

def Function_Analysis(func_def):
    """处理 Function_Analysis(func_def)"""
    print(f"Processing Function_Analysis: {func_def}")
    return {"type": "Function", "definition": func_def}

def Argument_Analysis(args,scope,p):
    """处理 Argument_Analysis(args)"""
    scope = "args"
    simple_type = infer_simple_type_from_return(args)
    CGDG = construct_graph_of_target_code(p)
    answer_def = call_LLM(CGDG['def'],scope)
    answer_user = call_LLM(CGDG['user'],scope)
    answer_usee = call_LLM(CGDG['usee'],scope)
    final_type = judge(simple_type, answer_def, answer_user, answer_usee)
    value = final_type
    print(f"Processing Argument_Analysis: {args}")
    return {"type": "Argument", "args": args}

