import networkx as nx
from tree_sitter import Language, Parser
from .utils import CONSTANTS
import os

def python_control_dependence_graph(root_node, CCG, src_lines, parent):
    """
    根据给定的语法树节点、控制依赖图和源代码行，构建控制依赖图（CCG）。

    参数:
    root_node (Node): 语法树的根节点。
    CCG (nx.MultiDiGraph): 控制依赖图对象。
    src_lines (list): 源代码行列表。
    parent (int): 父节点的 ID。

    返回:
    None
    """
    # 给当前节点分配一个 ID
    node_id = len(CCG.nodes)

    # 处理 import 语句
    if root_node.type in ['import_from_statement', 'import_statement']:
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]

        # 如果没有父节点，将当前节点添加到图中，并将其设置为父节点
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                        startRow=start_row, endRow=end_row,
                        sourceLines=src_lines[start_row:end_row + 1],
                        defSet=set(),  # 当前节点中定义的变量
                        useSet=set())  # 当前节点中使用的变量
            parent = node_id
        else:
            # 如果当前节点的行号范围在父节点的行号范围内，跳过处理
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                # 否则，将当前节点添加到图中，并添加一条从父节点到当前节点的控制依赖边（CDG）
                CCG.add_node(node_id, nodeType=root_node.type,
                            startRow=start_row, endRow=end_row,
                            sourceLines=src_lines[start_row:end_row + 1],
                            defSet=set(),
                            useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理类和函数定义
    elif root_node.type in ['class_definition', 'decorated_definition', 'function_definition']:
        if root_node.type == 'function_definition':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('parameters').end_point[0]
        elif root_node.type == 'decorated_definition':
            def_node = root_node.child_by_field_name('definition')
            start_row = root_node.start_point[0]
            parameter_node = def_node.child_by_field_name('parameters')
            if parameter_node is not None:
                end_row = parameter_node.end_point[0]
            else:
                end_row = def_node.start_point[0]
        elif root_node.type == 'class_definition':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('name').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理循环语句
    elif root_node.type in ['while_statement', 'for_statement']:
        if root_node.type == 'for_statement':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('right').end_point[0]
        if root_node.type == 'while_statement':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理 if 语句
    elif root_node.type == 'if_statement':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理 elif 语句
    elif root_node.type == 'elif_clause':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理 else 和 except 语句
    elif root_node.type in ['else_clause', 'except_clause']:
        start_row = root_node.start_point[0]
        end_row = root_node.start_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理 with 语句
    elif root_node.type == 'with_statement':
        start_row = root_node.start_point[0]
        end_row = root_node.children[1].end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理一般语句和错误节点
    elif 'statement' in root_node.type or 'ERROR' in root_node.type:
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    # 处理子节点
    for child in root_node.children:
        # 检查子节点是否是标识符
        if child.type == 'identifier':
            # 获取标识符的起始行和列以及结束列
            row = child.start_point[0]
            col_start = child.start_point[1]
            col_end = child.end_point[1]
            # 从源代码行中提取标识符名称并去除两端空白字符
            identifier_name = src_lines[row][col_start:col_end].strip()
            
            # 如果没有父节点，跳过处理
            if parent is None:
                continue
            
            # 根据父节点的类型，决定如何处理标识符
            if 'definition' in CCG.nodes[parent]['nodeType']:
                # 如果父节点是定义节点，将标识符添加到定义集合中
                CCG.nodes[parent]['defSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] == 'for_statement':
                # 如果父节点是 for 语句，进一步检查标识符的位置
                p = child
                while p.parent.type != 'for_statement':
                    p = p.parent
                if p.parent.type == 'for_statement' and p.prev_sibling.type == 'for':
                    # 如果标识符在 for 语句的头部，将其添加到定义集合中
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    # 否则，将其添加到使用集合中
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] == 'with_statement':
                # 如果父节点是 with 语句，进一步检查标识符的位置
                if child.parent.type == 'as_pattern_target':
                    # 如果标识符在 as 语句中，将其添加到定义集合中
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    # 否则，将其添加到使用集合中
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] == 'expression_statement':
                # 如果父节点是表达式语句，进一步检查标识符的位置
                p = child
                while p.parent.type != 'assignment' and p.parent.type != 'expression_statement':
                    p = p.parent
                if p.parent.type == 'assignment' and p.next_sibling is not None:
                    # 如果标识符在赋值语句的左侧，将其添加到定义集合中
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    # 否则，将其添加到使用集合中
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif 'import' in CCG.nodes[parent]['nodeType']:
                # 如果父节点是 import 语句，将标识符添加到定义集合中
                CCG.nodes[parent]['defSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] in ['global_statement', 'nonlocal_statement']:
                # 如果父节点是 global 或 nonlocal 语句，将标识符添加到定义集合中
                CCG.nodes[parent]['defSet'].add(identifier_name)
            else:
                # 其他情况下，将标识符添加到使用集合中
                CCG.nodes[parent]['useSet'].add(identifier_name)
        
        # 递归处理子节点
        python_control_dependence_graph(child, CCG, src_lines, parent)

    return


def python_control_flow_graph(CCG):
    # 创建一个有向多重图，用于表示控制流图（CFG）
    CFG = nx.MultiDiGraph()

    # 用于存储每个节点的下一个兄弟节点
    next_sibling = dict()
    # 用于存储每个节点的第一个子节点
    first_children = dict()

    # 用于存储起始节点
    start_nodes = []
    for v in CCG.nodes:
        # 如果节点没有前驱节点，则为起始节点
        if len(list(CCG.predecessors(v))) == 0:
            start_nodes.append(v)
    # 对起始节点进行排序
    start_nodes.sort()
    for i in range(0, len(start_nodes) - 1):
        v = start_nodes[i]
        u = start_nodes[i + 1]
        # 设置每个起始节点的下一个兄弟节点
        next_sibling[v] = u
    next_sibling[start_nodes[-1]] = None

    for v in CCG.nodes:
        # 获取当前节点的所有子节点
        children = list(CCG.neighbors(v))
        if len(children) != 0:
            # 对子节点进行排序
            children.sort()
            for i in range(0, len(children) - 1):
                u = children[i]
                w = children[i + 1]
                # 如果当前节点是 if 语句且子节点是子句，则不设置下一个兄弟节点
                if CCG.nodes[v]['nodeType'] == 'if_statement' and 'clause' in CCG.nodes[w]['nodeType']:
                    next_sibling[u] = None
                else:
                    next_sibling[u] = w
            next_sibling[children[-1]] = None
            # 设置当前节点的第一个子节点
            first_children[v] = children[0]
        else:
            first_children[v] = None

    # 用于存储边列表
    edge_list = []

    for v in CCG.nodes:
        # 处理块的开始控制流
        if v in first_children.keys():
            u = first_children[v]
            if u is not None:
                # 添加从当前节点到第一个子节点的控制流边
                edge_list.append((v, u, 'CFG'))
        
        # 处理块的结束控制流
        if CCG.nodes[v]['nodeType'] in ['return_statement', 'raise_statement']:
            # 对于 return 和 raise 语句，不需要添加控制流边，因为它们会终止当前块的执行
            pass
        elif CCG.nodes[v]['nodeType'] in ['break_statement', 'continue_statement']:
            # 处理 break 和 continue 语句
            u = None
            p = list(CCG.predecessors(v))[0]
            # 找到最近的循环语句节点
            while CCG.nodes[p]['nodeType'] not in ['for_statement', 'while_statement']:
                # 获取当前节点 p 的前驱节点
                p = list(CCG.predecessors(p))[0]
            if CCG.nodes[v]['nodeType'] == 'break_statement':
                # break 语句跳转到循环的下一个兄弟节点
                u = next_sibling[p]
            if CCG.nodes[v]['nodeType'] == 'continue_statement':
                # continue 语句跳转到循环的开始
                u = p
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] == 'for_statement':
            # 处理 for 语句，添加从当前节点到第一个子节点的控制流边
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] == 'while_statement':
            # 处理 while 语句，添加从当前节点到第一个子节点的控制流边
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            # 添加从当前节点到下一个兄弟节点的控制流边
            u = next_sibling[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] in ['if_statement', 'try_statement']:
            # 处理 if 和 try 语句，添加从当前节点到第一个子节点的控制流边
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            # 添加从当前节点到所有子句的控制流边
            for u in CCG.neighbors(v):
                if 'clause' in CCG.nodes[u]['nodeType']:
                    edge_list.append((v, u, 'CFG'))
        elif 'clause' in CCG.nodes[v]['nodeType']:
            # 处理子句，添加从当前节点到第一个子节点的控制流边
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
    
        # 处理下一个兄弟节点的控制流
        u = next_sibling[v]  # 获取当前节点 v 的下一个兄弟节点 u
        if u is None:  # 如果没有下一个兄弟节点
            p = v
            # 找到最近的父节点，并处理其控制流
            while len(list(CCG.predecessors(p))) != 0:  # 循环查找最近的父节点
                p = list(CCG.predecessors(p))[0]  # 获取父节点
                if CCG.nodes[p]['nodeType'] == 'while_statement':  # 如果父节点是 while 语句
                    edge_list.append((v, p, 'CFG'))  # 添加从 v 到 p 的控制流边
                    break
                if CCG.nodes[p]['nodeType'] == 'for_statement':  # 如果父节点是 for 语句
                    edge_list.append((v, p, 'CFG'))  # 添加从 v 到 p 的控制流边
                    break
                if CCG.nodes[p]['nodeType'] in ['try_statement', 'if_statement']:  # 如果父节点是 try 或 if 语句
                    if next_sibling[p] is not None:  # 如果父节点有下一个兄弟节点
                        edge_list.append((v, next_sibling[p], 'CFG'))  # 添加从 v 到父节点的下一个兄弟节点的控制流边
                        break
        if u is not None:  # 如果有下一个兄弟节点
            edge_list.append((v, u, 'CFG'))  # 添加从 v 到 u 的控制流边
    
    # 将边添加到控制流图（CFG）中
    CFG.add_edges_from(edge_list)
    for v in CCG.nodes:
        if v not in CFG.nodes:
            CFG.add_node(v)
    reachable = {v: set(nx.descendants(CFG, v)) for v in CFG.nodes()}
    return CFG, edge_list, reachable


def python_data_dependence_graph(CFG, CCG, reachable):
    try:
        total_pairs = 0
        processed_pairs = 0
        for v in CCG.nodes:
            v_defSet = CCG.nodes[v]['defSet']
            if not v_defSet:
                continue
            for u in CCG.nodes:
                total_pairs += 1
                if v == u or 'import' in CCG.nodes[v]['nodeType']:
                    continue
                u_useSet = CCG.nodes[u]['useSet']
                common_vars = v_defSet & u_useSet
                if not common_vars:
                    continue
                if u not in reachable[v]:
                    continue
                CCG.add_edge(v, u, 'DDG')
                processed_pairs += 1
                # if processed_pairs % 1000 == 0:
                    # print(f"已处理 {processed_pairs}/{total_pairs} 节点对")
    except Exception as e:
        print(f"数据依赖构建错误：{str(e)}")
        raise e

def resolve_python_module_path(module_name, current_file_dir, repo_dir):
    """
    根据 module_name (如 'utils', 'utils.submod', '.localmod')，
    结合当前文件夹和仓库路径，试图找到目标文件(夹)的绝对路径。
    如果在仓库内找不到，返回 None。
    """
    # 1) 判断是否是相对导入(如 `from . import x` 或 `from .submod import y`)
    #    这里仅做简单演示，可根据 '.' 的个数来定位上级目录
    if module_name.startswith('.'):
        # 计算向上几层:
        dots = 0
        for ch in module_name:
            if ch == '.':
                dots += 1
            else:
                break
        # 去掉前置点
        real_module = module_name[dots:].lstrip('.')
        # 假设简单场景：只考虑 from . import ...
        # 如果要支持 from .. import ... 就多向上几级
        up_dir = current_file_dir
        for _ in range(dots - 1):
            up_dir = os.path.dirname(up_dir)

        # 拼出可能的路径
        # e.g. up_dir + real_module(用/分割)
        if real_module:
            candidate = os.path.join(up_dir, real_module.replace('.', '/'))
        else:
            # 仅一个 '.'，表示当前目录
            candidate = up_dir
    else:
        # 2) 绝对导入(如 'package.submod')
        #    从 repo_dir 往下匹配
        candidate = os.path.join(repo_dir, module_name.replace('.', '/'))

    # 检查 candidate 是否存在 .py 文件 or 目录
    py_candidate = candidate + ".py"
    if os.path.isfile(py_candidate):
        return py_candidate
    # 也可能是包目录 (含 __init__.py)
    if os.path.isdir(candidate):
        init_file = os.path.join(candidate, "__init__.py")
        if os.path.isfile(init_file):
            return candidate  # 是个包目录
    return None  # 找不到，则视为第三方或不存在

def create_graph(code_lines, repo_name):
    """
    根据给定的代码行和仓库名称创建控制依赖图（CCG）。

    参数:
    code_lines (list): 代码行列表。
    repo_name (str): 仓库名称。

    返回:
    nx.MultiDiGraph: 控制依赖图对象，如果代码行为空或全是注释则返回 None。
    """
    src_lines = "".join(code_lines)
    src_lines = src_lines.splitlines(keepends=True)
    # print(f"开始生成ccg文件: {repo_name}")
    # print(f"代码行数: {len(code_lines)}")
    # 移除最后一行的多余字符
    if len(src_lines) != 0:
        src_lines[-1] = src_lines[-1].rstrip().strip('(').strip('[').strip(',')

    # 定义 tree-sitter 解析器
    language = Language('/home/wanyao/talentan/naturalcc/ncc/tools/tree-sitter-prebuilts/Linux/python.so', CONSTANTS.repos_language[repo_name])
    parser = Parser()
    parser.set_language(language)

    if len(src_lines) == 0:
        return None

    # 移除注释
    comment_prefix = ""
    if language.name == "python":
        comment_prefix = "#"

    comment_lines = []
    for i in range(0, len(src_lines)):
        line = src_lines[i]
        if line.lstrip().startswith(comment_prefix):
            src_lines[i] = '\n'
            comment_lines.append(i)

    # 解析文件以获取语法树
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(src_lines) or column >= len(src_lines[row]):
            return None
        return src_lines[row][column:].encode('utf8')
    try:
        # print("开始解析语法树...")
        tree = parser.parse(read_callable)
        # print("语法树解析完成！")
        # 检查是否全是注释

        # print("检查是否全是注释...")
        all_comment = True
        for child in tree.root_node.children:
            if child.type not in 'comment':
                all_comment = False
        if all_comment:
            print("文件全是注释，返回 None")
            return None
        # print("文件包含有效代码，继续处理")

        # 初始化程序依赖图
        ccg = nx.MultiDiGraph()
        # print("初始化控制依赖图（CCG）")

        if language.name == 'python':
            # print("开始构建控制依赖图（CCG）")
            for child in tree.root_node.children:
                # print(f"正在处理节点类型: {child.type}（父节点: None）")
                python_control_dependence_graph(child, ccg, code_lines, parent=None)
                # print("控制依赖图（CCG）构建完成")

            # 验证 CCG 是否为空
            if not ccg.nodes:
                print(f"图 {repo_name} 构建失败，无有效节点！")
                return None
            
            # print("开始构建控制流图（CFG）")
            cfg, cfg_edge_list, reachable  = python_control_flow_graph(ccg)
            # print("控制流图（CFG）构建完成")

            # print("开始构建数据依赖图（DDG）")
            python_data_dependence_graph(cfg, ccg, reachable)
            # print("数据依赖图（DDG）构建完成")

            ccg.add_edges_from(cfg_edge_list)
            # print("所有依赖图合并完成")

        # 获取图中的所有节点并排序
        node_list = list(ccg.nodes)
        node_list.sort()

        # 反转注释行列表，以便从后往前处理注释行
        comment_lines.reverse()
        max_comment_line = 0

        # 遍历每一行注释
        for comment_line_num in comment_lines:
            insert_id = -1
            # 找到第一个开始行号大于注释行号的节点
            for v in ccg.nodes:
                if ccg.nodes[v]['startRow'] > comment_line_num:
                    insert_id = v
                    break
            # 如果没有找到合适的节点，更新最大注释行号
            if insert_id == -1:
                max_comment_line = max(max_comment_line, comment_line_num)
            else:
                # 将注释行号设置为节点的开始行号
                ccg.nodes[insert_id]['startRow'] = comment_line_num
                end_row = ccg.nodes[insert_id]['endRow']
                # 更新节点的源代码行
                ccg.nodes[insert_id]['sourceLines'] = code_lines[comment_line_num: end_row + 1]

        # 如果有未处理的最大注释行号，更新最后一个节点的结束行号和源代码行
        if max_comment_line != 0:
            last_node_id = node_list[-1]
            ccg.nodes[last_node_id]['endRow'] = max_comment_line
            start_row = ccg.nodes[last_node_id]['startRow']
            ccg.nodes[last_node_id]['sourceLines'] = code_lines[start_row: max_comment_line + 1]

        # print(f"文件 {repo_name} 处理完成！")
        return ccg
    except Exception as e:
        print(f"创建图时发生致命错误：{str(e)}")
        raise e