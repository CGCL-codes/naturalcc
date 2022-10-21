from .properties import Properties
from .edge import Edge
from ... import log as logger

node_labels = ["Block", "Call", "Comment", "ControlStructure", "File", "Identifier", "FieldIdentifier", "Literal",
               "Local", "Member", "MetaData", "Method", "MethodInst", "MethodParameterIn", "MethodParameterOut",
               "MethodReturn", "Namespace", "NamespaceBlock", "Return", "Type", "TypeDecl", "Unknown"]

operators = ['addition', 'addressOf', 'and', 'arithmeticShiftRight', 'assignment',
             'assignmentAnd', 'assignmentArithmeticShiftRight', 'assignmentDivision',
             'assignmentMinus', 'assignmentMultiplication', 'assignmentOr', 'assignmentPlus',
             'assignmentShiftLeft', 'assignmentXor', 'cast', 'conditionalExpression',
             'division', 'equals', 'fieldAccess', 'greaterEqualsThan', 'greaterThan',
             'indirectFieldAccess', 'indirectIndexAccess', 'indirection', 'lessEqualsThan',
             'lessThan', 'logicalAnd', 'logicalNot', 'logicalOr', 'minus', 'modulo', 'multiplication',
             'not', 'notEquals', 'or', 'postDecrement', 'plus', 'postIncrement', 'preDecrement',
             'preIncrement', 'shiftLeft', 'sizeOf', 'subtraction']

node_labels += operators

node_labels = {label: i for i, label in enumerate(node_labels)}

PRINT_PROPS = True


class Node:
    def __init__(self, node, indentation):
        self.id = node["id"].split(".")[-1]
        self.label = self.id.split("@")[0]
        self.indentation = indentation + 1
        self.properties = Properties(node["properties"], self.indentation)
        self.edges = {edge["id"].split(".")[-1]: Edge(edge, self.indentation) for edge in node["edges"]}
        self.order = None
        operator = self.properties.get_operator()
        self.label = operator if operator is not None else self.label
        self._set_type()

    def __str__(self):
        indentation = self.indentation * "\t"
        properties = f"{indentation}Properties: {self.properties}\n"
        edges_str = ""

        for edge in self.edges:
            edges_str += f"{self.edges[edge]}"

        return f"\n{indentation}Node id: {self.id}\n{properties if PRINT_PROPS else ''}{indentation}Edges: {edges_str}"

    def connections(self, connections, e_type):
        for e_id, edge in self.edges.items():
            if edge.type != e_type: continue

            if edge.node_in in connections["in"] and edge.node_in != self.id:
                connections["in"][self.id] = edge.node_in

            if edge.node_out in connections["out"] and edge.node_out != self.id:
                connections["out"][self.id] = edge.node_out

        return connections

    def has_code(self):
        return self.properties.has_code()

    def has_line_number(self):
        return self.properties.has_line_number()

    def get_code(self):
        return self.properties.code()

    def get_line_number(self):
        return self.properties.line_number()

    def get_column_number(self):
        return self.properties.column_number()

    def _set_type(self):
        # label = self.label if self.operator is None else self.operator
        self.type = node_labels.get(self.label)  # Label embedding

        if self.type is None:
            logger.log_warning("node", f"LABEL {self.label} not in labels!")
            self.type = len(node_labels) + 1
