import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from ncc.utils.code_util.apex.apex_code_utility import ApexCodeUtil

apex_code_util = ApexCodeUtil()

sample_code = """
    public class SampleClass {    
        public Integer myNumber;
        public Integer getMyNumber() {
            // Return the current value of myNumber
            return this.myNumber;
        }
    }
"""

ast = apex_code_util.parse(sample_code)

def print_ast(node, indent=0):
    # 打印当前节点的类型和内容
    print("  " * indent + f"Type: {node.type}, Text: {node.text.decode('utf-8')}")
    
    # 遍历子节点
    for child in node.children:
        print_ast(child, indent + 1)

# 打印 AST 的根节点及其子节点
print("AST Information:")
print_ast(ast.root_node)