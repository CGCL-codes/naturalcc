"""
The functions in this file are originated from the code for
Compound Probabilistic Context-Free Grammars for Grammar Induction,
"""

import re


def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w


def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn


def get_nonbinary_spans(actions, SHIFT=0, REDUCE=1):
    spans = []
    tags = []
    stack = []
    pointer = 0
    binary_actions = []
    nonbinary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            nonbinary_actions.append(SHIFT)
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == 'NT(':
            # stack.append('(')
            # stack.append(action[3:-1].split('-')[0])
            stack.append(action[3:-1].split('_')[-1])
            # stack.append(action[3:-1])
        elif action == "REDUCE":
            nonbinary_actions.append(REDUCE)
            right = stack.pop()
            left = right
            n = 1
            # while stack[-1] is not '(':
            while type(stack[-1]) is tuple:
                left = stack.pop()
                n += 1
            span = (left[0], right[1])
            tag = stack.pop()
            if left[0] != right[1]:
                spans.append(span)
                tags.append(tag)
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)
                num_reduce += 1
        else:
            assert False
    assert (len(stack) == 1)
    assert (num_shift == num_reduce + 1)
    return spans, tags, binary_actions, nonbinary_actions


def get_actions(single_ast):
    output_actions = []
    stack=[]
    stack_num=[]
    for item in single_ast:
        single_ast_item =single_ast[item]
        if 'value' in single_ast_item:
            output_actions.append('SHIFT')
            stack_num[-1] -= 1
            while (stack_num[-1] == 0):
                output_actions.append('REDUCE')
                stack.pop()
                stack_num.pop()
                if (len(stack_num) > 0):
                    stack_num[-1] -= 1
                else:
                    break
        else:
            output_actions.append('NT(' + single_ast_item['type'] + ')')
            stack.append(single_ast_item['type'])
            stack_num.append(len(single_ast_item['children']))
    return output_actions

def get_predict_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '《' or line_strip[i] == '》'  # 每一次都是从(或者)
        if line_strip[i] == '《':
            if is_next_open_bracket(line_strip, i):  # open non-terminal  #遇到"("返回
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1
                # get the next open bracket,
                # which may be a terminal or another non-terminal
                while line_strip[i] != '《':
                    i += 1
            else:  # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != '》':
                    i += 1
                i += 1
                while line_strip[i] != '》' and line_strip[i] != '《':
                    i += 1
        else:
            output_actions.append('REDUCE')
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != '》' and line_strip[i] != '《':
                i += 1
    assert i == max_idx
    return output_actions


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '《':
            return True
        elif char == '》':
            return False
    raise IndexError('Bracket possibly not balanced, '
                     'open bracket not followed by closed bracket')


def get_nonterminal(line, start_idx):
    assert line[start_idx] == '《'  # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not (char == '《') and not (char == '》')
        output.append(char)
    return ''.join(output)

#输出‘code tokens’
def get_tags_tokens(single_ast):
    output_tokens=[]
    for i in single_ast:
        single_ast_item=single_ast[i]
        if 'value' in single_ast_item:
            output_tokens.append(single_ast_item['value'])
    return  output_tokens


# def get_between_brackets(line, start_idx):
#     output = []
#     for char in line[(start_idx + 1):]:
#         if char == ')':
#             break
#         assert not (char == '(')
#         output.append(char)
#     return ''.join(output)
