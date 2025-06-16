import re
import jsonlines

def remove_comments(code):
    """
    函数用于移除给定代码字符串（包含多种语言代码，此处考虑C++、Java、Python）中的注释
    """
    # 处理Python的块注释（三引号形式）
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    # 处理C++、Java、Python通用的块注释（/* */ 形式）
    code = re.sub(r"/\*([^*]|(\*+[^*/]))*\*+/", "", code)


    # 处理C++、Java、Python通用的行注释（// 形式）
    lines = code.splitlines()
    new_lines = []
    for line in lines:
        if line == "":
            new_lines.append(line)
            continue
        new_line = re.sub(r"(//.*)|(#.*)", "", line)
        if new_line == "":
            continue
        new_lines.append(new_line)
    return "\n".join(new_lines)

# 示例用法
# code = """
# // 这是一段混合多种语言代码示例，先写点Python代码
# def add(a, b):
#     \"\"\"
#     这个函数用于计算两个数的和
#     \"\"\"
#     result = a + b  # 执行加法运算
#     return result

# /* 下面来点C++代码，定义个简单函数 */
# int add_cpp(int a, int b) {
#     return a + b; // 返回两数之和
# }

# // 再写点Java代码
# class Test {
#     public static int add_java(int a, int b) {
#         return a + b; // 做加法运算
#     }
# }
# """
# print(remove_comments(code))


# 打开JSON文件，这里使用相对路径，你可以根据实际情况调整路径
json_objects = []
with jsonlines.open('/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl') as reader:
    for data in reader:
        data['output'] = remove_comments(data['output'])
        del data['CoT']
        json_objects.append(data)
with jsonlines.open('/home/wanyao/wucai/work/data/code_instructions_filtered_CoT_comment_remove.jsonl', 'a') as writer:
    for item in json_objects:
        writer.write(item)