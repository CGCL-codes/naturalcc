import ast
from hityper.tdg_generator import TDGGenerator
import traceback


class FunctionLocator(ast.NodeVisitor):
    def __init__(self):
        self.inclass = False
        self.inclass = False
        self.found = False
        self.node = None


    def visit_ClassDef(self, node):
        if not self.inclass and node.name == self.classname:
            self.inclass = True
            self.found = False
            self.generic_visit(node)
            if self.found and self.funcname == "global":
                self.node = node
        elif not self.inclass:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        if not self.infunc and node.name == self.funcname and self.inclass:
            if self.scope == 'return' and node.name == self.name:
                self.node = node
            else:
                self.infunc = True
                self.found = False
                self.generic_visit(node)
                if self.found:
                    self.node = node
        elif not self.infunc and self.inclass:
            self.generic_visit(node)
    
    def visit_Name(self, node):
        if node.id == self.name and self.scope == 'local' and self.infunc and self.inclass:
            self.found = True
    
    def visit_Attribute(self, node):
        if node.attr == self.name and hasattr(node.value, "id") and node.value.id == "self" and self.scope == "local" and self.infunc and self.inclass:
            self.found = True

    def visit_arg(self, node):
        if node.arg == self.name and self.scope == 'arg' and self.infunc and self.inclass:
            self.found = True


    def run(self, root, loc, name, scope):
        self.inclass = False
        self.infunc = False
        self.node = None
        self.found = False
        self.funcname, self.classname = loc.split('@')
        self.name = name
        self.scope = scope
        if self.classname == 'global':
            self.inclass = True
        if self.funcname == 'global':
            self.infunc = True
        if self.inclass and self.infunc:
            remover = GlobalNodeRemover()
            node = remover.run(root)
            return node
        else:
            self.visit(root)
        return self.node


class GlobalNodeRemover(ast.NodeTransformer):
    def __init__(self):
        pass
    
    def visit_Import(self, node):
        return None
    
    def visit_ImportFrom(self, node):
        return None
    
    def visit_FunctionDef(self, node):
        if not self.only_import:
            return None
        else:
            return node
    
    def visit_AsyncFunctionDef(self, node):
        if not self.only_import:
            return None
        else:
            return node

    def visit_ClassDef(self, node):
        if not self.only_import:
            return None
        else:
            return node
    
    def run(self, root, only_import = False):
        self.only_import = only_import
        self.visit(root)
        ast.fix_missing_locations(root)
        return root


class AnnotationCleaner(ast.NodeTransformer):
    def __init__(self):
        pass

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        node.returns = None
        if node.args.args:
            for arg in node.args.args:
                arg.annotation = None
        if node.args.kwonlyargs:
            for elem in node.args.kwonlyargs:
                elem.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        ast.fix_missing_locations(node)
        return node

    def visit_AnnAssign(self, node):
        if node.value != None:
            newnode = ast.Assign(targets = [node.target], value = node.value)
            ast.fix_missing_locations(newnode)
            return newnode
        else:
            return None
    
    def run(self, node):
        self.visit(node)
        placeholder = PlaceHolder()
        node = placeholder.run(node)
        return node


class AnnotationMask(ast.NodeTransformer):
    def __init__(self, mask = "<mask>"):
        self.mask = mask

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        if self.scope == "return":
            node.returns = ast.Name(id = self.mask)
            self.added = True
        elif self.scope == "arg":
            if node.args.args:
                for arg in node.args.args:
                    if arg.arg == self.name:
                        arg.annotation = ast.Name(id = self.mask)
                        self.added = True
            if node.args.kwonlyargs:
                for elem in node.args.kwonlyargs:
                    if elem.arg == self.name:
                        elem.annotation = ast.Name(id = self.mask)
                        self.added = True
        self.generic_visit(node)
        ast.fix_missing_locations(node)
        return node
    
    def visit_Assign(self, node):
        if len(node.targets) == 1 and hasattr(node.targets[0], "id") and node.targets[0].id == self.name and not self.added:
            newnode = ast.AnnAssign(target = node.targets[0], value = node.value, annotation = ast.Name(id = self.mask), simple = 1)
            ast.fix_missing_locations(newnode)
            self.added = True
            return newnode
        else:
            self.generic_visit(node)
            return node
    
    def run(self, root, name, scope, loc):
        self.name = name
        self.scope = scope
        self.loc = loc
        locator = FunctionLocator()
        node = locator.run(root, loc, name, scope)
        remover = CommentRemover()
        node = remover.run(node)
        cleaner = AnnotationCleaner()
        node = cleaner.run(node)
        self.added = False
        self.visit(node)
        if not self.added:
            newnode = ast.AnnAssign(target = ast.Name(id = self.name), value = None, annotation = ast.Name(id = self.mask), simple = 1)
            ast.fix_missing_locations(newnode)
            node.body.append(newnode)
            self.added = True
        return node
        

class CommentRemover(ast.NodeTransformer):
    def __init__(self):
        pass
    
    def visit_Expr(self, node):
        self.generic_visit(node)
        if type(node.value) == ast.Constant and isinstance(node.value.value, str):
            return None
        else:
            return node
    
    def run(self, root):
        self.visit(root)
        placeholder = PlaceHolder()
        root = placeholder.run(root)
        ast.fix_missing_locations(root)
        return root


class PlaceHolder(ast.NodeTransformer):
    def __init__(self):
        pass
    
    def visit_FunctionDef(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_AsyncFunctionDef(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node
    
    def visit_ClassDef(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_If(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_For(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node
    
    def visit_While(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_AsyncFor(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_With(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_AsyncWith(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_Try(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_TryStar(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node
    
    def run(self, root):
        self.visit(root)
        return root



class SimpleSlicer(ast.NodeTransformer):
    def __init__(self, mode = "name"):
        self.mode = mode

    def contain_lines(self, node):
        for line in self.lines:
            if line in range(node.lineno, node.end_lineno + 1):
                return True
        return False

    def visit_Return(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None
    
    def visit_Assign(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None
    
    def visit_AugAssign(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None
    
    def visit_AnnAssign(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None

    def visit_For(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node
    
    def visit_AsyncFor(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node

    def visit_While(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node
    
    def visit_If(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node

    def visit_With(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node
    
    def visit_AsyncWith(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node

    def visit_Match(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            return node
    
    def visit_Raise(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None

    def visit_Try(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node
    
    def visit_TryStar(self, node):
        if self.mode == "name" and self.name not in ast.unparse(node):
            return None
        elif self.mode == "line" and not self.contain_lines(node):
            return None
        else:
            self.generic_visit(node)
            if len(node.body) == 0:
                node.body.append(ast.Pass())
            return node
    
    def visit_Assert(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None

    def visit_Global(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None
    
    def visit_Nonlocal(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None
    
    def visit_Expr(self, node):
        if self.mode == "name" and self.name in ast.unparse(node):
            return node
        elif self.mode == "line" and self.contain_lines(node):
            return node
        else:
            return None

    def run(self, source, name, lines = None, output_node = False):
        try:
            root = ast.parse(source)
            self.name = name
            self.lines = lines
            self.visit(root)
            if output_node:
                return root
            else:
                return ast.unparse(root)
        except:
            if output_node:
                return root
            else:
                return source



class StaticSlicer(object):
    def __init__(self):
        pass

    def get_TDG(self, root, loc, full = False):
        try:
            usertypes = {"direct": [], "indirect": [], "init": [], "unrecognized": []}
            if not full:
                locations = [loc]
            else:
                locations = None
            generator = TDGGenerator("TEST", True, locations, usertypes, alias = 1, repo = None)
            global_tg = generator.run(root)
            return global_tg
        except Exception as e:
            print('Error Occurs: Cannot generate the TDG for {}, reason: {}'.format(self.name, e))
            return None

    def collect_lines_for_node(self, node, direction = "forward", hop = 3):
        lines = [node.lineno]
        if hop <= 1:
            return lines
        if direction == "forward":
            for n in node.ins:
                lines += self.collect_lines_for_node(n, direction = direction, hop = hop - 1)
        elif direction == "backward":
            for n in node.outs:
                lines += self.collect_lines_for_node(n, direction = direction, hop = hop - 1)
        lines = list(set(lines))
        return lines

    def collect_lines_for_nodes(self, nodes, direction = "forward", hop = 3):
        lines = []
        for n in nodes:
            lines += self.collect_lines_for_node(n, direction = direction, hop = hop)
        lines = list(set(lines))
        return lines
            
        

    def collect_lines(self, root, forward_hop = 3, backward_hop = 2):
        try:
            global_tg = self.get_TDG(root, self.loc)
            if self.func == "global":
                if self.scope != "local":
                    print("Error Occurs: Unexpected arguments or return values in non-function code.")
                    return None
                nodes = global_tg.globalsymbols[self.name]
                lines = self.collect_lines_for_nodes(nodes, hop = forward_hop)
                return lines
            else:
                curtg = None
                for tg in global_tg.tgs:
                    if tg.name == self.loc:
                        curtg = tg
                        break
                if curtg == None:
                    global_tg = self.get_TDG(root, self.loc, full = True)
                    for tg in global_tg.tgs:
                        if tg.name == self.loc:
                            curtg = tg
                            break
                    if curtg == None:
                        print("Error Occurs: Cannot find the TDG for function {}, skipped...".format(self.func))
                        return None
                if self.scope == "arg":
                    nodes = []
                    if self.name + "(arg)" in curtg.symbolnodes:
                        nodes += curtg.symbolnodes[self.name + "(arg)"]
                    if self.name in curtg.symbolnodes:
                        nodes += curtg.symbolnodes[self.name]
                    lines = self.collect_lines_for_nodes(nodes, direction = "backward", hop = backward_hop)
                    return lines
                elif self.scope == "return":
                    nodes = curtg.returnvaluenodes
                    lines = self.collect_lines_for_nodes(nodes, hop = forward_hop)
                    return lines
                else:
                    nodes = curtg.symbolnodes[self.name]
                    lines = self.collect_lines_for_nodes(nodes, hop = forward_hop)
                    return lines
        except Exception as e:
            print("Error Occurs: Cannot collect the lines for {}, reason: {}".format(self.name, e))
            traceback.print_exc()
            return []
    

    def run(self, filename, name, scope, loc, forward_hop = 3):
        try:
            source = open(filename, "r").read()
            root = ast.parse(ast.unparse(ast.parse(source)))
            remover = CommentRemover()
            root = remover.run(root)
            cleaner = AnnotationCleaner()
            root = cleaner.run(root)
            root = ast.parse(ast.unparse(root))
            self.name = name
            self.scope = scope
            self.loc = loc
            self.func = loc.split("@")[0]
            lines = self.collect_lines(root, forward_hop = forward_hop, backward_hop = 2)
            slicer = SimpleSlicer(mode = "line")
            root = slicer.run(root, self.name, lines = lines, output_node = True)
            locator = FunctionLocator()
            node = locator.run(root, self.loc, self.name, self.scope)
            if type(node) in [ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef] and len(node.body) == 0:
                node.body.append(ast.Pass())
            newsource = ast.unparse(node)
        except Exception as e:
            print("Error Occurs: Slicing failed, reason: {}".format(e))
            traceback.print_exc()
            newsource = source
        return newsource
    