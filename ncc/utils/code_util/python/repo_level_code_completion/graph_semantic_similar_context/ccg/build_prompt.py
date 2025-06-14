def make_an_extended_block(retrieved_context, tokenizer):
    """
    创建一个扩展的代码块，包括文件路径和代码内容的注释。

    参数:
    retrieved_context (list): 检索到的上下文，包括文件路径和代码内容。
    tokenizer (object): 用于分词的 tokenizer 对象。

    返回:
    tuple: 包含扩展的代码块字符串和其标记长度的元组。
    """
    content = retrieved_context[0]
    # 将文件路径放在注释中
    f_path_comment = f'# The below code fragment can be found in:\n'
    f_paths_str = '# ' + '/'.join(retrieved_context[-2]) + '\n'
    # 将代码行放在注释中
    code_lines = content.splitlines(keepends=True)
    content_lines_comment = [f'# {line.rstrip()}\n' for line in code_lines]
    # 聚合注释和代码行
    seperator = '# ' + '-' * 50 + '\n'
    block_str = "".join([f_path_comment, f_paths_str, seperator] + content_lines_comment + [seperator])
    # 对扩展的代码块进行分词
    tokenized_block = tokenizer.tokenize(block_str)
    # 计算标记长度
    token_len = len(tokenized_block)
    return block_str, token_len


def make_str_block_with_max_token_length(tokenizer, max_token_num: int, context_str: str, with_comment=False):
    """
    创建一个字符串块，其标记长度不超过最大标记数。

    参数:
    tokenizer (object): 用于分词的 tokenizer 对象。
    max_token_num (int): 最大标记数。
    context_str (str): 上下文字符串。
    with_comment (bool): 是否将字符串作为注释。

    返回:
    str: 生成的字符串块。
    """
    str_block = ""
    new_line = context_str.splitlines(keepends=True)
    if with_comment:
        # 将每一行上下文字符串作为注释
        context_str_lines_comment = [f'# {line}' for line in new_line]
        new_line = context_str_lines_comment
    curr_len = 0
    for i in range(1, len(new_line) + 1):
        # 对每一行进行分词并计算长度
        line_len = len(tokenizer.tokenize(new_line[-i]))
        if line_len + curr_len < max_token_num:
            # 如果当前长度加上新行长度小于最大标记数，则添加到字符串块中
            str_block = new_line[-i] + str_block
            curr_len += line_len
        else:
            break
    return str_block


def build_infile_prompt(case, tokenizer, max_num_tokens):
    """
    构建 infile 模式的提示。

    参数:
    case (dict): 包含上下文的案例。
    tokenizer (object): 用于分词的 tokenizer 对象。
    max_num_tokens (int): 最大标记数。

    返回:
    str: 生成的提示字符串。
    """
    comment = "# Complete the next statement of the following codes:\n"
    comment_length = len(tokenizer.tokenize(comment))
    max_num_tokens = max_num_tokens // 2 - comment_length
    context = "".join(case['context'])
    # 创建一个字符串块，其标记长度不超过最大标记数
    prompt = make_str_block_with_max_token_length(tokenizer, max_num_tokens, context)
    return comment + prompt


def build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k):
    """
    构建 retrieval 模式的提示。

    参数:
    case (dict): 包含上下文和检索到的代码片段的案例。
    tokenizer (object): 用于分词的 tokenizer 对象。
    max_num_tokens (int): 最大标记数。
    max_top_k (int): 最大检索片段数。

    返回:
    str: 生成的提示字符串。
    """
    # 原始上下文
    context_max_tokens = max_num_tokens // 2
    comment = "# Based on above, complete the next statement of the following codes:\n"
    comment_length = len(tokenizer.tokenize(comment))
    # 创建一个字符串块，其标记长度不超过最大标记数
    context = make_str_block_with_max_token_length(tokenizer, context_max_tokens - comment_length, "".join(case['context']))
    context_prompt = comment + context

    # 检索到的示例
    seperator = '# ' + '-' * 50
    retrieval_prompt = "# Here are some relevant code fragments from other files of the repo:\n"
    retrieval_prompt += seperator + '\n'

    num_chosen_context = 0
    max_retrieval_length = max_num_tokens // 2
    current_token_length = len(tokenizer.tokenize(retrieval_prompt))
    retrival_blocks = []
    top_k_context = case['top_k_context']
    for i in range(1, len(top_k_context) + 1):
        retrieval_context = top_k_context[-i]
        if num_chosen_context >= max_top_k:
            break
        # 创建一个扩展的代码块
        block_str, token_len = make_an_extended_block(retrieval_context, tokenizer)
        if current_token_length + token_len < max_retrieval_length:
            retrival_blocks.insert(0, block_str)
            current_token_length += token_len
            num_chosen_context += 1
        else:
            continue
    retrieval_prompt += ''.join(retrival_blocks)
    return retrieval_prompt + context_prompt


def build_prompt(case, tokenizer, max_num_tokens, max_top_k=10, mode='retrieval'):
    """
    构建提示字符串。

    参数:
    case (dict): 包含上下文和检索到的代码片段的案例。
    tokenizer (object): 用于分词的 tokenizer 对象。
    max_num_tokens (int): 最大标记数。
    max_top_k (int): 最大检索片段数。
    mode (str): 提示模式，可以是 'infile' 或 'retrieval'。

    返回:
    str: 生成的提示字符串。
    """
    prompt = ""
    if mode == 'infile':
        # 构建 infile 模式的提示
        prompt = build_infile_prompt(case, tokenizer, max_num_tokens)
    elif mode == 'retrieval':
        # 构建 retrieval 模式的提示
        prompt = build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k)
    return prompt