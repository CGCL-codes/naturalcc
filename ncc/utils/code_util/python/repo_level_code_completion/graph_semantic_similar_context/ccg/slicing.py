import queue
import networkx as nx
from .utils import CONSTANTS


class Slicing:
    """
    Slicing 类用于根据给定的图（包含控制依赖、数据依赖与控制流）和起始节点，
    进行前向依赖切片操作，收集相关语句并组合得到代码上下文。
    """

    def __init__(self, max_hop=CONSTANTS.max_hop, max_statement=CONSTANTS.max_statement):
        """
        初始化 Slicing 类实例。

        参数:
        max_hop - 在控制流图(CFG)中能够向前追踪的最大跳数
        max_statement - 限定在切片中最多能收集的语句数量
        """
        self.max_hop = max_hop
        self.max_statement = max_statement

    def forward_dependency_slicing(self, node, graph: nx.MultiDiGraph, contain_node=False):
        """
        forward_dependency_slicing 方法基于控制依赖(CDG)、数据依赖(DDG)和控制流图(CFG)，
        以指定节点 node 为起点进行前向依赖分析，收集可能影响或关联到该节点的语句。

        参数:
        node - 前向依赖分析的起始节点
        graph - 多重有向图，包含节点和边的依赖信息
        contain_node - 是否在最终收集到的语句中保留初始节点

        返回:
        (ctx_str, line_list, subgraph):
        1. ctx_str: 拼接的相关语句上下文字符串
        2. line_list: 语句对应的行号列表（升序）
        3. subgraph: 与切片内节点对应的子图
        """
        # 用于存储收集到的行号与对应的源代码
        line_ctx = dict()
        # 记录已被访问（入队）到或处理过的节点，防止重复处理
        visited = set()
        # 获取图中节点总数
        n_nodes = len(graph.nodes)

        # 定义一个队列，用于广度遍历。队列元素 (节点, 当前跳数)
        q = queue.Queue()
        # 将起始节点和初始跳数(0)加入队列
        q.put((node, 0))

        # 定义视图过滤函数，只保留 CDG 边
        def cdg_view(v, u, t):
            return t == 'CDG'

        # 定义视图过滤函数，只保留 CFG 边
        def cfg_view(v, u, t):
            return t == 'CFG'

        # 定义视图过滤函数，只保留 DDG 边
        def ddg_view(v, u, t):
            return t == 'DDG'

        # 使用 subgraph_view 获取三个子图：控制依赖图、控制流图、数据依赖图
        cdg = nx.subgraph_view(graph, filter_edge=cdg_view)
        cfg = nx.subgraph_view(graph, filter_edge=cfg_view)
        ddg = nx.subgraph_view(graph, filter_edge=ddg_view)

        # 用于记录已收集在切片中的节点
        n_statement = set()

        # 在没有达到最大语句数前持续遍历
        while len(n_statement) < self.max_statement:
            # 如果已收集节点数等于图中节点数 或 队列为空，则停止
            if len(n_statement) == n_nodes or q.empty():
                break

            # 从队列中获取下一个节点和其对应跳数
            curr_v, hop = q.get()

            # 获取当前节点对应的源码行范围，并将这部分行内容加入 line_ctx
            start_line = graph.nodes[curr_v]['startRow']
            end_line = graph.nodes[curr_v]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx[i] = graph.nodes[curr_v]['sourceLines'][i - start_line]

            # 将此节点计入已收集语句集合
            n_statement.add(curr_v)
            # 检查是否已达到收集的最大语句数限制
            if len(n_statement) >= self.max_statement:
                break

            # 将当前节点放入变量 p 中，后续用于查找控制依赖的前驱
            p = curr_v
            # 在控制依赖图(CDG)中，持续寻找前驱节点并收集
            if p in cdg.nodes:
                while len(list(cdg.predecessors(p))) != 0:
                    p = list(cdg.predecessors(p))[0]
                    start_line = graph.nodes[p]['startRow']
                    end_line = graph.nodes[p]['endRow']
                    for i in range(start_line, end_line + 1):
                        line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                    n_statement.add(p)
                    if len(n_statement) >= self.max_statement:
                        break

            # 在数据依赖图(DDG)中查找前驱节点并收集
            for u in ddg.predecessors(curr_v):
                p = u
                start_line = graph.nodes[p]['startRow']
                end_line = graph.nodes[p]['endRow']
                for i in range(start_line, end_line + 1):
                    line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                n_statement.add(p)
                if len(n_statement) >= self.max_statement:
                    break

                # 如果有控制依赖存在，也同时收集
                if p in cdg.nodes:
                    if len(list(cdg.predecessors(p))) != 0:
                        p = list(cdg.predecessors(p))[0]
                        start_line = graph.nodes[p]['startRow']
                        end_line = graph.nodes[p]['endRow']
                        for i in range(start_line, end_line + 1):
                            line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                        n_statement.add(p)

            # 如果下一跳 hop+1 大于允许的最大 hop，则不再继续遍历 CFG
            if hop + 1 > self.max_hop:
                continue
            else:
                # 否则在控制流图(CFG)中获取前驱节点，尝试加入队列
                for u in cfg.predecessors(curr_v):
                    # 排除已访问的节点以及 nodeType 为 'definition' 的节点
                    if u not in visited and 'definition' not in graph.nodes[u]['nodeType']:
                        q.put((u, hop + 1))

            # 将当前节点标记为已访问，避免重复处理
            visited.add(curr_v)

        # 如果不包含初始节点，则将其对应的行从结果中移除
        if not contain_node:
            n_statement.remove(node)
            start_line = graph.nodes[node]['startRow']
            end_line = graph.nodes[node]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx.pop(i)

        # 获取全部收集到的行号并排序
        line_list = list(line_ctx.keys())
        line_list.sort()
        # 用于拼接源码上下文的列表
        ctx = []
        for i in range(0, len(line_list)):
            ctx.append(line_ctx[line_list[i]])

        # 获取由收集节点组成的子图
        subgraph = nx.subgraph(graph, n_statement)

        # 返回拼接的上下文字符串、行号列表以及对应子图
        return "".join(ctx), line_list, subgraph