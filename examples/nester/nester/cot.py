import json
import argparse
from hityper.tdg import TypeGenNode, SymbolNode, TypeNode, GraphBaseNode, MergeNode, BranchNode
from hityper.tdg_generator import TDGGenerator
from hityper.typeobject import TypeObject
from ast_operation import FunctionLocator, AnnotationCleaner, CommentRemover
import ast
import traceback
from tqdm import tqdm




class COTGenerator(object):
    def __init__(self):
        self.wordmap = {
            1: "First,",
            2: "Second,",
            3: "Third,",
            4: "Fourth,",
            5: "Fifth,",
            6: "Sixth,",
            7: "Seventh,",
            8: "Eighth,",
            9: "Ninth,",
            10: "Tenth,"
        }
        self.mathopmap = {
            "+": "add",
            "-": "subtract",
            "*": "multiple",
            "/": "div",
            "//": "floordiv",
            "%": "mod",
            "**": "pow",
            "<<": "left shift",
            ">>": "right shift",
            "|": "bit or",
            "^": "bit xor",
            "&": "bit and",
            "@": "matrix multiple",
            "~": "invert"
        }
        self.unopmap = {
            "+": "add",
            "-": "subtract"
        }
        self.boolopmap = {
            "not": "not",
            "and": "and",
            "or": "or"
        }
        self.cmpopmap = {
            "==": "==",
            "!=": "!=",
            "<": "<",
            "<=": "<=",
            ">": ">",
            ">=": ">=",
            "is": "is",
            "is not": "is not",
            "in": "in",
            "not in": "not in"
        }


    def get_TDG(self, instance, full = False):
        try:
            key = '{}--{}--{}--{}'.format(instance["file"], instance["loc"], instance["name"], instance["scope"])
            root = ast.parse(open(instance["file"], "r").read())
            usertypes = {"direct": [], "indirect": [], "init": [], "unrecognized": []}
            if not full:
                locations = [instance["loc"]]
            else:
                locations = None
            generator = TDGGenerator(instance["file"], True, locations, usertypes, alias = 1, repo = None)
            global_tg = generator.run(root)
            return global_tg
        except Exception as e:
            print('Cannot generate the TDG for {}, reason: {}'.format(key, e))
            return None


    def remove_duplicate_nodes(self, nodes, direction = "forward"):
        if len(nodes) == 1:
            return nodes
        else:
            newnodes = []
            for n in nodes:
                if direction == "forward" and len(n.ins) == 1 and ((isinstance(n.ins[0], SymbolNode) and n.ins[0].symbol == n.symbol) or isinstance(n.ins[0], MergeNode) or isinstance(n.ins[0], BranchNode)):
                    continue
                elif direction == "backward" and len(n.outs) == 1 and ((isinstance(n.outs[0], SymbolNode) and n.outs[0].symbol == n.symbol) or isinstance(n.outs[0], MergeNode) or isinstance(n.outs[0], BranchNode)):
                    continue
                else:
                    newnodes.append(n)
            return newnodes


    def handle_dict(self, node):
        nodes = []
        if isinstance(node, TypeGenNode) and node.op == "Dict_Read":
            nodes.append([node, "key"])
            if len(node.ins) > node.splitindex:
                nodes.append([node, "values"])
            return nodes
        else:
            return [node]

    def handle_ins(self, nodes):
        new_nodes = []
        for n in nodes:
            new_nodes += self.handle_dict(n)
        return new_nodes

    def process_duplicate_nodes(self, nodes, remove_type_node = False, direction = "forward", next_level = False):
        changed = True
        while changed:
            changed = False
            newnodes = []
            for n in nodes:
                if direction == "forward":
                    if isinstance(n, SymbolNode) and len(n.ins) == 1 and isinstance(n.ins[0], SymbolNode) and n.symbol == n.ins[0].symbol:
                        newnodes += self.handle_dict(n.ins[0])
                        changed = True
                    elif isinstance(n, MergeNode):
                        for nn in n.ins:
                            if next_level:
                                newnodes += self.handle_ins(nn.ins)
                            else:
                                newnodes += nn.ins
                        changed = True
                    elif isinstance(n, BranchNode):
                        if next_level:
                            newnodes += self.handle_ins(n.ins)
                        else:
                            newnodes += n.ins
                        changed = True
                    elif isinstance(n, TypeGenNode) and n.op == "=":
                        if next_level:
                            newnodes += self.handle_ins(n.ins)
                        else:
                            newnodes += n.ins
                        changed = True
                    elif remove_type_node and isinstance(n, TypeNode):
                        changed = True
                        continue
                    else:
                        newnodes.append(n)
                elif direction == "backward":
                    if isinstance(n, BranchNode):
                        newnodes += n.outs
                        changed = True
                    elif isinstance(n, MergeNode):
                        newnodes += n.outs
                        changed = True
                    else:
                        newnodes.append(n)
            nodes = newnodes
        return newnodes


    def _gen_object_for_op(self, node):
        if node.op == "=":
            return "an assignment"
        elif node.op in self.mathopmap:
            return "a math operation {}".format(self.mathopmap[node.op])
        elif node.op in self.unopmap:
            return "a unary operation {}".format(self.unopmap[node.op])
        elif node.op in self.boolopmap:
            return "a bool expression {}".format(self.boolopmap[node.op])
        elif node.op in self.cmpopmap:
            return "a comparsion operation {}".format(self.cmpopmap[node.op])
        elif node.op == "call":
            if node.func == None:
                return "a function call"
            else:
                return "a function call {}".format(node.func.replace("_@_", "."))
        elif node.op == "Subscript_Read":
            return "a subscription"
        elif node.op in "Subscript_Write":
            return "an assignment"
        elif node.op == "List_Read":
            return "a list"
        elif node.op == "List_Write":
            return "by unpacking a list"
        elif node.op == "Tuple_Read":
            return "a tuple"
        elif node.op == "Tuple_Write":
            return "by unpacking a tuple"
        elif node.op == "Set_Read":
            return "a set"
        elif node.op == "Dict_Read":
            return "a dict"
        elif node.op == ".":
            return "an attribute access"
        elif node.op == "IfExp":
            return "an IfExp expression"
        elif node.op == "JoinedStr":
            return "a JoinedStr"
        elif node.op == "forin":
            return "an iterator"
        elif node.op == "ListComp":
            return "a list comprehension"
        elif node.op == "DictComp":
            return "a dict comprehension"
        elif node.op == "SetComp":
            return "a set comprehension"
        elif node.op == "GeneratorExp":
            return "a generator expression"
        elif node.op == "yield":
            return "a yield expression"
        else:
            print("Unknown operation: {}".format(node.op))
        
        

    def _gen_subject_for_op(self, node, capitalize = False, part = None, plural = False):
        if plural:
            operand = "operands"
            argument = "arguments"
            element = "elements"
            key = "keys"
            value = "values"
        else:
            operand = "operand"
            argument = "argument"
            element = "element"
            key = "key"
            value = "value"
        if node.op == "=":
            subject = "the value of the assignment"
        elif node.op in self.mathopmap:
            subject = "the {} of the math operation {}".format(operand, self.mathopmap[node.op])
        elif node.op in self.unopmap:
            subject = "the operand of the unary operation {}".format(self.unopmap[node.op])
        elif node.op in self.boolopmap:
            subject = "the {} of the bool expression {}".format(operand, self.boolopmap[node.op])
        elif node.op in self.cmpopmap:
            subject = "the {} of the comparsion operation {}".format(operand, self.cmpopmap[node.op])
        elif node.op == "call":
            if node.func == None:
                subject = "the {} of the function call".format(argument)
            else:
                subject = "the {} of the function call {}".format(argument, node.func.replace('_@_', '.'))
        elif node.op == "Subscript_Read":
            subject = "the target of the subscription"
        elif node.op in "Subscript_Write":
            subject = "the value of the assignment"
        elif node.op in ["List_Read", "List_Write"]:
            subject = "the {} of the list".format(element)
        elif node.op in ["Tuple_Read", "Tuple_Write"]:
            subject = "the {} of the tuple".format(element)
        elif node.op == "Set_Read":
            subject = "the {} of the set".format(element)
        elif node.op == "Dict_Read":
            if part == "key":
                subject = "the {} of the dict".format(key)
            else:
                subject = "the {} of the dict".format(value)
        elif node.op == ".":
            subject = "the target of the attribute access"
        elif node.op == "IfExp":
            subject = "the {} of the IfExp expression".format(operand)
        elif node.op == "JoinedStr":
            subject = "the {} of the JoinedStr".format(operand)
        elif node.op == "forin":
            subject = "the iterator"
        elif node.op == "ListComp":
            subject = "the {} of the list comprehension".format(operand)
        elif node.op == "DictComp":
            subject = "the {} of the dict comprehension".format(operand)
        elif node.op == "SetComp":
            subject = "the {} of the set comprehension".format(operand)
        elif node.op == "GeneratorExp":
            subject = "the {} of the generator expression".format(operand)
        elif node.op == "yield":
            subject = "the {} of the yield expression".format(operand)

        if capitalize:
            subject = subject[0].upper() + subject[1:]
        
        return subject
        

    def _gen_verb_for_op(self, node, plural = False):
        if not plural:
            return "is"
        else:
            return "are"

    def _gen_object_for_symbol(self, node, capitalize = False, partial = False, add_noun = False):
        if "(arg)" in node.symbol:
            name = node.symbol.replace("(arg)", "")
            cot = ""
            if partial:
                cot = "a part of "
            cot += "the argument {}".format(name)
            if capitalize:
                cot = cot[0].upper() + cot[1:]
        elif "_@_" in node.symbol:
            name = node.symbol.replace("_@_", ".")
            cot = ""
            if partial:
                cot = "a part of "
            cot += "the attribute {}".format(name)
            if capitalize:
                cot = cot[0].upper() + cot[1:]
        elif "Return_Value@" in node.symbol:
            name = node.symbol.replace("Return_Value@", "")
            if "@" in name:
                name = name.split("@")[0]
            cot = ""
            if partial:
                cot = "a part of "
            cot += "the return value of {}".format(name)
            if capitalize:
                cot = cot[0].upper() + cot[1:]
        else:
            name = node.symbol
            if add_noun:
                cot = "a variable " + name
            else:
                cot = name
            if partial:
                cot = "a part of " + cot
        return cot
    
    def _gen_subject_for_symbol(self, node, scope = None, name = None, capitalize = False, partial = False):
        if scope == None:
            return self._gen_object_for_symbol(node, capitalize = capitalize, partial = partial)
        elif scope == "local":
            subject = "the variable {}".format(name)
        elif scope == "return":
            subject = "the return value of {}".format(name)
        elif scope == "arg":
            subject = "the argument {}".format(name)
        if partial:
            subject = "a part of " + subject
        if capitalize:
            subject = subject[0].upper() + subject[1:]
        return subject
        

    def _gen_object_for_type(self, node):
        typemap = {
            "typing.Text": "str"
        }
        cot = TypeObject.resolveTypeName(node.type)
        if cot in typemap:
            cot = typemap[cot]
        cot = "a " + cot 
        return cot

    def _gen_object_for_node(self, node, add_noun = False):
        if isinstance(node, TypeGenNode):
            return self._gen_object_for_op(node)
        elif isinstance(node, SymbolNode):
            return self._gen_object_for_symbol(node, add_noun = add_noun)
        elif isinstance(node, TypeNode):
            return self._gen_object_for_type(node)
        else:
            print(node)

    def _gen_subject_for_node(self, node, capitalize = False, partial = False, part = None, plural = False, scope = None, name = None):
        if isinstance(node, TypeGenNode):
            subject = self._gen_subject_for_op(node, capitalize = capitalize, part = part, plural = plural)
        elif isinstance(node, SymbolNode):
            subject = self._gen_subject_for_symbol(node, scope = scope, name = name, capitalize = capitalize, partial = partial)
        else:
            subject = ""
        
        return subject

    def _gen_verb_for_node(self, node, plural = False, by = False):
        if isinstance(node, TypeGenNode):
            return self._gen_verb_for_op(node, plural = plural)
        elif isinstance(node, SymbolNode):
            if not by:
                return "is assigned from"
            else:
                return "is assigned"
        else:
            return ""
    
    def _gen_cot_for_def(self, node, scope, name, hop = 1, base_hop = 0):
        cur_level_nodes = [node]
        next_level_nodes = []
        cur_hop = 0
        cot = []
        
        while cur_hop < hop and len(cur_level_nodes) > 0:
            cot.append(self.wordmap[cur_hop + 1 + base_hop])
            for i, n in enumerate(cur_level_nodes):
                dict_node = []
                if (isinstance(n, GraphBaseNode) and len(n.ins) == 0) or isinstance(n, TypeNode):
                    continue
                part = None
                if isinstance(n, list):
                    if n[1] == "key":
                        inputnodes = n[0].ins[:n[0].splitindex]
                        part = "key"
                    else:
                        inputnodes = n[0].ins[n[0].splitindex:]
                        part = "value"
                    dict_node = n
                    n = n[0]
                else:
                    inputnodes = []
                    for nn in n.ins:
                        inputnodes.append(nn)
                inputnodes = self.process_duplicate_nodes(inputnodes)
                if isinstance(n, TypeGenNode) and n.op in ["Subscript_Read", "Subscript_Write"]:
                    next_level_nodes.append(n.ins[0])
                else:
                    for nn in inputnodes:
                        if isinstance(nn, TypeGenNode) and nn.op == "Dict_Read":
                            next_level_nodes.append([nn, "key"])
                            if len(nn.ins) > nn.splitindex:
                                next_level_nodes.append([nn, "values"])
                        else:
                            if (isinstance(nn, GraphBaseNode) and len(nn.ins) == 0) or isinstance(nn, TypeNode):
                                continue
                            next_level_nodes.append(nn)
                next_level_nodes = self.process_duplicate_nodes(next_level_nodes, remove_type_node = True, next_level = True)
                if len(inputnodes) == 1:
                    if isinstance(inputnodes[0], TypeGenNode) and inputnodes[0].op == "Subscript_Write":
                        partial = True
                    else:
                        partial = False
                    if isinstance(inputnodes[0], TypeGenNode) and inputnodes[0].op in ["List_Write", "Tuple_Write"]:
                        by = True
                    else:
                        by = False
                    cot += [self._gen_subject_for_node(n, capitalize = True if i >= 1 else False, partial = partial, part = part, scope = scope if cur_hop == 0 else None, name = name if cur_hop == 0 else None), self._gen_verb_for_node(n, by = by), self._gen_object_for_node(inputnodes[0]) + "."]
                elif len(inputnodes) == 2:
                    cot += [self._gen_subject_for_node(n, capitalize = True if i >= 1 else False, part = part, plural = True, scope = scope if cur_hop == 0 else None, name = name if cur_hop == 0 else None), self._gen_verb_for_node(n, plural = True), self._gen_object_for_node(inputnodes[0]) + " and " + self._gen_object_for_node(inputnodes[1]) + "."]
                elif len(inputnodes) >= 3:
                    object_str = ""
                    for nn in inputnodes[:-1]:
                        object_str = object_str + self._gen_object_for_node(nn) + ", "
                    object_str = object_str[:-2] + " and " + self._gen_object_for_node(inputnodes[-1]) + "."
                    cot += [self._gen_subject_for_node(n, capitalize = True if i >= 1 else False, part = part, plural = True, scope = scope if cur_hop == 0 else None, name = name if cur_hop == 0 else None), self._gen_verb_for_node(n, plural = True), object_str]
            cur_level_nodes = next_level_nodes
            next_level_nodes = []
            cur_hop += 1
        return " ".join(cot), cur_hop

    
    def _gen_end(self, name, gttype, scope):
        if scope == "local":
            sub = "the variable {}".format(name)
        elif scope == "return":
            sub = "the return value of {}".format(name)
        elif scope == "arg":
            sub = "the argument {}".format(name)
        type_sub = "the type of " + sub
        end = "Therefore, " + type_sub + " is `{}`.".format(gttype)
        return end


    def gen_cot_for_def(self, name, nodes, scope, gttype, hop = 3, occurrance = 1):
        end = self._gen_end(name, gttype, scope)
        cots = []
        base_hop = 0
        for n in nodes[:occurrance]:
            cot, cur_hop = self._gen_cot_for_def(n, scope, name, hop = hop, base_hop = base_hop)
            cots.append(cot)
            base_hop += cur_hop
        final_cot = " ".join(cots) + " " + end
        #print(final_cot)
        return final_cot


    def gen_cot_for_usage(self, name, nodes, scope, gttype):
        end = self._gen_end(name, gttype, scope)
        usages = []
        for n in nodes:
            outputnodes = n.outs
            outputnodes = self.process_duplicate_nodes(outputnodes, direction = "backward")
            for o in outputnodes:
                usages.append(self._gen_object_for_node(o, add_noun = True))
        usages = list(set(usages))
        new_usages = []
        for u in usages:
            if u == "a variable {}".format(name):
                continue
            new_usages.append(u)
        usages = new_usages
        cot = ""
        if len(usages) > 0:
            usage_str = ""
            if len(usages) == 1:
                usage_str += usages[0]
                usage_str += "."
            elif len(usages) == 2:
                usage_str += " and ".join(usages)
                usage_str += "."
            elif len(usages) >= 3:
                for u in usages[:-1]:
                    usage_str += u
                    usage_str += ", "
                usage_str = usage_str[:-2] + " and " + usages[-1]
                usage_str += "."
            cot += "The argument {} is used in ".format(name) + usage_str
            cot += " Based on the usage and naming convention, it is reasonable to assume that the type of the argument {} is {}.".format(name, gttype)
        else:
            cot += "Based on the naming convention, it is reasonable to assume that the type of the argument {} is {}.".format(name, gttype)
        cot += " "
        cot += end
        #print(cot)
        return cot

    def gen_one(self, instance, hop = 3):
        key = '{}--{}--{}--{}'.format(instance["file"], instance["loc"], instance["name"], instance["scope"])
        try:
            global_tg = self.get_TDG(instance)
            if global_tg == None:
                return None
            if instance["loc"] == "global@global":
                if instance["name"] not in global_tg.globalsymbols:
                    print("Error occurs: Cannot find variable {} in the TDG".format(instance["name"]))
                    return None
                nodes = global_tg.globalsymbols[instance["name"]]
                nodes = self.remove_duplicate_nodes(nodes)
                return self.gen_cot_for_def(instance["name"], nodes, "local", instance["processed_gttype"], hop = hop)
            else:
                curtg = None
                for tg in global_tg.tgs:
                    if tg.name == instance["loc"]:
                        curtg = tg
                        break
                if curtg == None:
                    global_tg = self.get_TDG(instance, full = True)
                    if global_tg == None:
                        print("Error occurs: Cannot find the TDG for {}".format(key))
                        return None
                    else:
                        curtg = None
                        for tg in global_tg.tgs:
                            if tg.name == instance["loc"]:
                                curtg = tg
                                break
                        if curtg == None:
                            print("Error occurs: Cannot find the TDG for {}".format(key))
                            return None
                if instance["scope"] == "local":
                    if instance["name"] not in curtg.symbolnodes:
                        print("Error occurs: Cannot find variable {} in the TDG".format(instance["name"]))
                        return None
                    nodes = curtg.symbolnodes[instance["name"]]
                    nodes = self.remove_duplicate_nodes(nodes)
                    return self.gen_cot_for_def(instance["name"], nodes, "local", instance["processed_gttype"], hop = hop)
                elif instance["scope"] == "return":
                    nodes = curtg.returnvaluenodes
                    return self.gen_cot_for_def(instance["name"], nodes, "return", instance["processed_gttype"], hop = hop)
                elif instance["scope"] == "arg":
                    nodes = []
                    if instance["name"] + "(arg)" in curtg.symbolnodes:
                        nodes += curtg.symbolnodes[instance["name"] + "(arg)"]
                    if instance["name"] in curtg.symbolnodes:
                        nodes += curtg.symbolnodes[instance["name"]]   
                    if len(nodes) == 0:
                        print("Error occurs: Cannot find argument {} in the TDG".format(instance["name"]))
                        return None
                    nodes = self.remove_duplicate_nodes(nodes, direction = "backward")
                    return self.gen_cot_for_usage(instance["name"], nodes, "arg", instance["processed_gttype"])  
        except Exception as e:
            print("Error occurs: {}".format(e))
            traceback.print_exc()
            return  None               

    def gen_all(self, dataset, hop = 3):
        data = json.loads(open(dataset, "r").read())
        cots = {}
        i = 0
        for d in tqdm(data):
            i += 1
            key = '{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"])
            cot = self.gen_one(d, hop = hop)
            cots[key] = cot
        with open(dataset.replace(".json", f"_cots_hop{hop}.json"), "w", encoding = "utf-8") as df:
            df.write(json.dumps(cots, sort_keys=True, indent=4, separators=(',', ': ')))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required = True, type=str, help = "Path to the JSON file of dataset")
    parser.add_argument('-p', '--hop', required = False, default = 3, type=int, help = "Number of hops")
    args = parser.parse_args()
    generator = COTGenerator()
    generator.gen_all(args.source, hop = args.hop)

if __name__ == "__main__":
    main()
    






