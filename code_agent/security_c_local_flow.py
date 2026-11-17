"""Bounded local C value flow for native pointer and capacity candidates.

The domain retains integers, positive array extents plus constants, and array
aliases. Unknown calls/results stay unknown. Branches are joined, local names
are scoped, escaped pointer variables are invalidated, and only simple counted
loops receive a finite summary. No dataset names, labels or expected rates enter
the analysis. Extent expressions on unresolved member types remain candidates,
not a proof of the full C type or whole-program reachability.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from code_agent.security_c_rules import _int_value, call_parts, declared_name, text, walk
from code_agent.security_exam import switch_off


@dataclass(frozen=True)
class Extent:
    base: str
    offset: int = 0
    bytes: bool = False


@dataclass(frozen=True)
class Reference:
    base: str
    capacity: object
    offset: object = 0
    character: bool = False
    item_size: int | None = None


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class Range:
    lower: int
    upper: int


NONZERO = object()
WRAPPED = object()

# Parameter positions actually dereferenced by the upstream list APIs (RT-
# Thread include/rtservice.h and FreeRTOS include/list.h, list.c). In particular
# listIS_CONTAINED_WITHIN only compares its first pointer; do not dereference it.
LIST_ACCESS = {
    'rt_list_init': (0,), 'rt_list_remove': (0,), 'rt_list_isempty': (0,),
    'rt_list_insert_after': (0, 1), 'rt_list_insert_before': (0, 1),
    'vListInitialise': (0,), 'vListInitialiseItem': (0,),
    'listLIST_IS_EMPTY': (0,), 'listCURRENT_LIST_LENGTH': (0,),
    'listGET_HEAD_ENTRY': (0,), 'listGET_OWNER_OF_HEAD_ENTRY': (0,),
    'listIS_CONTAINED_WITHIN': (1,),
}


def unwrap(node):
    while node is not None and node.type == 'parenthesized_expression' and node.named_children:
        node = node.named_children[0]
    return node


def key(node):
    return re.sub(r'\s+', '', text(unwrap(node)))


def object_declarator(node):
    """The declarator nearest the name determines array vs pointer object.

    int *slots[8] is an array; int (*pointer)[8] is a pointer. The outer
    tree-sitter node alone reverses these two common engineering declarations.
    """
    kind, array, pointers = None, None, 0
    while node is not None and node.type != 'identifier':
        if node.type == 'pointer_declarator': kind, array, pointers = 'pointer', None, pointers + 1
        elif node.type == 'array_declarator': kind, array = 'array', node
        child = node.child_by_field_name('declarator')
        node = child if child is not None else (node.named_children[0] if node.type == 'parenthesized_declarator' and node.named_children else None)
    return kind, array, pointers


def arithmetic(left, op, right):
    if isinstance(left, Range) or isinstance(right, Range):
        a = (left.lower, left.upper) if isinstance(left, Range) else (left,)
        b = (right.lower, right.upper) if isinstance(right, Range) else (right,)
        values = [arithmetic(x, op, y) for x in a for y in b]
        if all(isinstance(v, int) for v in values): return Range(min(values), max(values))
        return None
    if isinstance(left, int) and isinstance(right, int):
        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/' and right and left % right == 0: return left // right
    if isinstance(left, Extent) and isinstance(right, int) and op in ('+', '-'):
        return replace(left, offset=left.offset + (right if op == '+' else -right))
    if isinstance(right, Extent) and isinstance(left, int) and op == '+':
        return replace(right, offset=right.offset + left)
    if isinstance(left, Extent) and isinstance(right, Extent) and op == '-' and (left.base, left.bytes) == (right.base, right.bytes):
        return left.offset - right.offset
    if isinstance(left, Reference) and op in ('+', '-'):
        return replace(left, offset=arithmetic(left.offset, op, right))
    return None


def truth(value):
    if isinstance(value, int): return bool(value)
    if value is NONZERO or isinstance(value, Reference): return True
    if isinstance(value, Extent) and value.offset >= 0: return True
    if isinstance(value, Range):
        if value.lower > 0 or value.upper < 0: return True
        if value.lower == value.upper == 0: return False
    return None


def comparison(left, op, right):
    difference = arithmetic(left, '-', right)
    if isinstance(difference, Range):
        if difference.lower == difference.upper: return comparison(difference.lower, op, 0)
        if op in ('==', '!='):
            if difference.lower > 0 or difference.upper < 0: return op == '!='
            return None
        results = [comparison(v, op, 0) for v in (difference.lower, difference.upper)]
        return results[0] if results[0] == results[1] else None
    if isinstance(difference, int):
        return {'==': difference == 0, '!=': difference != 0, '<': difference < 0,
                '<=': difference <= 0, '>': difference > 0, '>=': difference >= 0}.get(op)
    if isinstance(difference, Extent) and not difference.bytes and difference.offset >= 0:
        return {'==': False, '!=': True, '<': False, '<=': False, '>': True, '>=': True}.get(op)
    if right == 0 and left is NONZERO:
        return {'==': False, '!=': True}.get(op)
    if left == 0 and right is NONZERO:
        return {'==': False, '!=': True}.get(op)
    return None


def resolve(value, state):
    return state.get(('$symbol', value.name), value) if isinstance(value, Symbol) else value


class LocalFlow:
    def __init__(self, root, emit):
        self.emit = emit
        self.arrays, self.symbols, self.unsigned = {}, {}, set()
        self.pointer_names = set()
        self.symbol_index = 0
        # Only declarations outside functions seed the global facts. Local
        # declarations are interpreted at their actual position and scope.
        for node in walk(root):
            if node.type != 'array_declarator': continue
            parent = node.parent
            while parent is not None and parent.type not in ('function_definition', 'declaration', 'field_declaration'):
                parent = parent.parent
            if parent is None: continue
            ancestor = parent
            while ancestor is not None and ancestor.type != 'function_definition': ancestor = ancestor.parent
            if ancestor is not None: continue
            object_kind, array, _ = object_declarator(node)
            if object_kind != 'array': continue
            size = array.child_by_field_name('size') if array is not None else None
            name = declared_name(node)
            capacity = _int_value(size)
            if capacity is None:
                capacity = Extent(key(size) or name)
                if size is not None and unwrap(size).type == 'identifier': self.symbols[key(size)] = capacity
            kind = text(parent.child_by_field_name('type'))
            prefix = '.' if parent.type == 'field_declaration' else ''
            reference = Reference(prefix + name, capacity, character=kind == 'char',
                                  item_size={'char': 1, 'int': 4, 'long': 8}.get(kind))
            old = self.arrays.get(prefix + name, reference)
            self.arrays[prefix + name] = reference if old == reference else None

    def reference(self, node, state, *, infer=False, character=False):
        node = unwrap(node)
        name = key(node)
        value = state.get(name, self.arrays.get(name))
        if isinstance(value, Reference): return value
        if node is not None and node.type == 'field_expression':
            value = self.arrays.get('.' + text(node.child_by_field_name('field')))
            if isinstance(value, Reference): return replace(value, base=name)
        if infer and name not in self.pointer_names:
            return Reference(name, Extent(name), character=character)
        return None

    def bounds(self, node, reference, index, count=1):
        if reference is None or index is None or count is None or count == 0: return
        if isinstance(index, Range) or isinstance(count, Range):
            indices = (index.lower, index.upper) if isinstance(index, Range) else (index,)
            counts = (count.lower, count.upper) if isinstance(count, Range) else (count,)
            for i in indices:
                for n in counts: self.bounds(node, reference, i, n)
            return
        position = arithmetic(reference.offset, '+', index)
        end = arithmetic(position, '+', count)
        out = (position is WRAPPED or index is WRAPPED or count is WRAPPED
               or comparison(position, '<', 0) is True
               or comparison(end, '>', reference.capacity) is True)
        if isinstance(position, Extent) and position.bytes and position.base == reference.base:
            out = True  # object bytes are at least its element count
        if out:
            self.emit(node, 'string_overflow' if reference.character else 'array_oob',
                      f'Local value flow reaches {reference.base} with offset {position!r}, '
                      f'write/access size {count!r}, capacity {reference.capacity!r}.')
        # Extend the existing I teaching boundary to typed char arrays. This
        # intentionally conservative rule concerns a terminal NUL slot;
        # the reference configuration accepts that last valid byte.
        parent = node.parent
        subscript = node.child_by_field_name('index') if node.type == 'subscript_expression' else None
        left = unwrap(subscript.child_by_field_name('left')) if subscript is not None and subscript.type == 'binary_expression' else None
        if (not out and reference.character and switch_off('I') and parent is not None
                and parent.type == 'assignment_expression' and parent.child_by_field_name('left') == node
                and text(parent.child_by_field_name('right')) in ("'\\0'", '0')
                and left is not None
                and text(subscript.child_by_field_name('operator')) == '-'
                and _int_value(subscript.child_by_field_name('right')) == 1
                and comparison(end, '==', reference.capacity) is True):
            self.emit(node, 'string_overflow', 'Teaching boundary gap I conservatively rejects the final valid NUL slot.')

    def refine(self, condition, state, branch):
        node = unwrap(condition)
        if node is None: return
        if node.type == 'identifier':
            value = NONZERO if branch else 0
            previous = state.get(key(node))
            if isinstance(previous, Symbol): state[('$symbol', previous.name)] = value
            state[key(node)] = value
        elif node.type == 'unary_expression' and text(node.child_by_field_name('operator')) == '!':
            self.refine(node.child_by_field_name('argument'), state, not branch)
        elif node.type == 'binary_expression':
            left, right = node.child_by_field_name('left'), node.child_by_field_name('right')
            op = text(node.child_by_field_name('operator'))
            if op in ('==', '!=') and (_int_value(right) == 0 or key(right) in ('NULL', 'nullptr')) and unwrap(left).type == 'identifier':
                self.refine(left, state, branch != (op == '=='))

    def unknown(self, name):
        self.symbol_index += 1
        return Symbol(f'{name}@{self.symbol_index}')

    @staticmethod
    def join(states):
        if not states: return {}
        result = {}
        for name in set().union(*(s.keys() for s in states)):
            values = [resolve(s.get(name), s) for s in states]
            if all(v == values[0] for v in values): result[name] = values[0]
            elif all(isinstance(v, (int, Range)) for v in values):
                ends = [n for v in values for n in ((v.lower, v.upper) if isinstance(v, Range) else (v,))]
                result[name] = Range(min(ends), max(ends))
            else: result[name] = None
        return result

    def evaluate(self, node, state):
        node = unwrap(node)
        if node is None: return None
        kind = node.type
        if kind in ('number_literal', 'char_literal'):
            value = _int_value(node)
            if value is None and kind == 'number_literal':
                try: value = int(re.sub(r'[uUlL]+$', '', text(node)), 0)
                except ValueError: pass
            return value
        if kind == 'identifier':
            if key(node) in ('NULL', 'nullptr', '__null'): return 0
            name = key(node)
            if name not in state:
                state[name] = self.symbols.get(name, self.arrays.get(name))
                if state[name] is None: state[name] = self.unknown(name)
            return resolve(state[name], state)
        if kind == 'cast_expression':
            value = node.child_by_field_name('value')
            result = self.evaluate(value, state)
            if (result is None or isinstance(result, Symbol)) and re.search(r'\bchar\s*\*', text(node.child_by_field_name('type'))):
                return self.reference(value, state, infer=True, character=True)
            return result
        if kind == 'sizeof_expression':
            operand = node.child_by_field_name('value') or (node.named_children[-1] if node.named_children else None)
            if key(operand) in self.pointer_names: return 8  # explicit native64 audit profile
            reference = self.reference(operand, state)
            if reference:
                if reference.item_size is not None:
                    if reference.item_size == 1: return reference.capacity
                    if isinstance(reference.capacity, int): return reference.capacity * reference.item_size
                return Extent(reference.base, bytes=True)
            return None
        if kind == 'binary_expression':
            left, right = unwrap(node.child_by_field_name('left')), unwrap(node.child_by_field_name('right'))
            op = text(node.child_by_field_name('operator'))
            # Preserve native sizeof-array / sizeof-element relations even
            # when the array's declared bound is a configuration macro.
            if op == '/' and left.type == right.type == 'sizeof_expression':
                a = unwrap(left.child_by_field_name('value'))
                b = unwrap(right.child_by_field_name('value'))
                if b is not None and b.type == 'subscript_expression' and key(a) == key(b.child_by_field_name('argument')):
                    reference = self.reference(a, state, infer=True)
                    if reference:
                        state.setdefault(key(a), reference)
                        return reference.capacity
            a, b = self.evaluate(left, state), self.evaluate(right, state)
            if op in ('==', '!=', '<', '<=', '>', '>='): return comparison(a, op, b)
            return arithmetic(a, op, b)
        if kind in ('assignment_expression', 'update_expression'):
            target = node.child_by_field_name('left') if kind == 'assignment_expression' else node.child_by_field_name('argument')
            op = text(node.child_by_field_name('operator'))
            old = self.evaluate(target, state)
            right = self.evaluate(node.child_by_field_name('right'), state) if kind == 'assignment_expression' else 1
            value = right if op == '=' else arithmetic(old, '+' if op in ('+=', '++') else '-', right)
            if unwrap(target).type == 'identifier':
                if key(target) in self.unsigned and isinstance(value, int) and value < 0: value = WRAPPED
                state[key(target)] = value if value is not None else self.unknown(key(target))
            return value
        if kind == 'unary_expression':
            value = self.evaluate(node.child_by_field_name('argument'), state)
            op = text(node.child_by_field_name('operator'))
            if op == '-' and isinstance(value, int): return -value
            if op == '!' and truth(value) is not None: return not truth(value)
            return value if op == '+' else None
        if kind == 'comma_expression':
            value = None
            for child in node.named_children: value = self.evaluate(child, state)
            return value
        if kind == 'conditional_expression':
            condition = node.child_by_field_name('condition')
            known = truth(self.evaluate(condition, state))
            values, states = [], []
            for branch, field in ((True, 'consequence'), (False, 'alternative')):
                if known is not None and branch != known: continue
                copy = dict(state); self.refine(condition, copy, branch)
                values.append(self.evaluate(node.child_by_field_name(field), copy)); states.append(copy)
            state.update(self.join(states))
            return values[0] if values and all(v == values[0] for v in values) else None
        if kind == 'field_expression':
            value = self.evaluate(node.child_by_field_name('argument'), state)
            if text(node.child_by_field_name('operator')) == '->' and value == 0:
                self.emit(node, 'null_pointer', 'Local assignments and branch constraints force the member pointer to NULL.')
            return self.reference(node, state)
        if kind == 'subscript_expression':
            argument = node.child_by_field_name('argument')
            reference = self.reference(argument, state)
            if reference is None:
                value = self.evaluate(argument, state)
                reference = value if isinstance(value, Reference) else None
            self.bounds(node, reference, self.evaluate(node.child_by_field_name('index'), state))
            return None
        if kind == 'pointer_expression':
            argument = unwrap(node.child_by_field_name('argument'))
            op = text(node.child_by_field_name('operator'))
            if op == '&' and argument.type == 'subscript_expression':
                reference = self.reference(argument.child_by_field_name('argument'), state, infer=True)
                index = self.evaluate(argument.child_by_field_name('index'), state)
                if reference and (comparison(index, '<', 0) is True or comparison(index, '>', reference.capacity) is True or index is WRAPPED):
                    self.bounds(argument, reference, index)
                return replace(reference, offset=index) if reference else None
            value = self.evaluate(argument, state)
            if op == '*':
                if value == 0: self.emit(node, 'null_pointer', 'The locally propagated pointer is NULL at this dereference.')
                if isinstance(value, Reference): self.bounds(node, value, 0)
            return NONZERO if op == '&' else None
        if kind == 'call_expression':
            name, args = call_parts(node)
            values = [self.evaluate(arg, state) for arg in args]
            if '__typeof__(' in name and len(values) == 1: return values[0]
            if name in ('strncpy', 'rt_strncpy') and len(values) >= 3:
                ref = values[0]
                if isinstance(ref, Reference): self.bounds(node, ref, 0, values[2])
            for position in LIST_ACCESS.get(name, ()):
                if position >= len(values): continue
                value = values[position]
                if isinstance(value, Reference): self.bounds(node, value, 0)
                elif value == 0: self.emit(node, 'null_pointer', 'The list API dereferences this NULL argument.')
            # Address arguments escape: a callee can assign through &pointer.
            for arg in args:
                arg = unwrap(arg)
                if arg.type == 'pointer_expression' and text(arg.child_by_field_name('operator')) == '&':
                    operand = unwrap(arg.child_by_field_name('argument'))
                    if operand.type == 'identifier' and (key(operand) in self.pointer_names or not isinstance(state.get(key(operand)), Reference)):
                        state[key(operand)] = None
            return None
        # Unknown expression kinds may contain a dereference, but their value
        # cannot be inferred merely by taking their final syntactic child.
        for child in node.named_children: self.evaluate(child, state)
        return None

    def statement(self, node, state):
        if node is None: return False
        kind = node.type
        if kind == 'compound_statement':
            original = set(state); unsigned = set(self.unsigned); pointers = set(self.pointer_names)
            declared = set()
            saved = {}
            terminated = False
            for child in node.named_children:
                if child.type == 'declaration':
                    for target in child.children_by_field_name('declarator'):
                        name = declared_name(target); declared.add(name); saved.setdefault(name, state.get(name))
                terminated = self.statement(child, state)
                if terminated: break
            for name in declared:
                if name in original: state[name] = saved[name]
                else: state.pop(name, None)
            self.unsigned, self.pointer_names = unsigned, pointers
            return terminated
        elif kind == 'declaration':
            typ = text(node.child_by_field_name('type'))
            for declarator in node.children_by_field_name('declarator'):
                target = declarator.child_by_field_name('declarator') if declarator.type == 'init_declarator' else declarator
                name = declared_name(target)
                state[name] = None
                object_kind, array, pointers = object_declarator(target)
                if re.search(r'\bunsigned\b', typ) and object_kind != 'pointer': self.unsigned.add(name)
                if object_kind == 'pointer': self.pointer_names.add(name)
                if object_kind == 'array':
                    size = array.child_by_field_name('size')
                    capacity = self.evaluate(size, state)
                    if capacity is None or isinstance(capacity, Symbol):
                        capacity = Extent(key(size) or name)
                        if size is not None and unwrap(size).type == 'identifier': state[key(size)] = capacity
                    state[name] = Reference(name, capacity, character=typ == 'char' and not pointers,
                                           item_size=8 if pointers else {'char': 1, 'int': 4, 'long': 8}.get(typ))
                elif declarator.type == 'init_declarator':
                    value = self.evaluate(declarator.child_by_field_name('value'), state)
                    state[name] = value if value is not None else self.unknown(name)
        elif kind == 'if_statement':
            condition = node.child_by_field_name('condition')
            known = truth(self.evaluate(condition, state))
            states = []
            for branch, field in ((True, 'consequence'), (False, 'alternative')):
                if known is not None and branch != known: continue
                copy = dict(state); self.refine(condition, copy, branch)
                terminated = self.statement(node.child_by_field_name(field), copy)
                if not terminated: states.append(copy)
            state.update(self.join(states))
            return not states
        elif kind == 'for_statement':
            self.statement(node.child_by_field_name('initializer'), state)
            condition = unwrap(node.child_by_field_name('condition'))
            update = unwrap(node.child_by_field_name('update'))
            body = node.child_by_field_name('body')
            if condition is not None and condition.type == 'binary_expression' and update is not None and update.type == 'update_expression':
                variable = key(condition.child_by_field_name('left'))
                limit = self.evaluate(condition.child_by_field_name('right'), state)
                op = text(condition.child_by_field_name('operator'))
                simple = (state.get(variable) == 0 and op in ('<', '<=')
                          and key(update.child_by_field_name('argument')) == variable
                          and text(update.child_by_field_name('operator')) == '++'
                          and not any(n.type in ('break_statement', 'return_statement', 'goto_statement')
                              or (n.type == 'assignment_expression' and key(n.child_by_field_name('left')) == variable)
                              or (n.type == 'update_expression' and key(n.child_by_field_name('argument')) == variable)
                              for n in walk(body)))
                if simple and (comparison(limit, '>', 0) is True or isinstance(limit, Extent)):
                    end = arithmetic(limit, '-', 1) if op == '<' else limit
                    first, last = dict(state), dict(state)
                    first[variable], last[variable] = 0, end
                    self.statement(body, first); self.statement(body, last)
                    state.update(self.join([first, last]))
                    state[variable] = limit if op == '<' else arithmetic(limit, '+', 1)
                    return False
            # Without a finite summary, don't propagate facts from an arbitrary
            # loop iteration into the following code.
            if truth(self.evaluate(condition, state)) is False: return False
            copy = self.loop_input(body, state)
            self.statement(body, copy)
            for name in state:
                if copy.get(name) != state[name]: state[name] = None
        elif kind in ('while_statement', 'do_statement'):
            condition = node.child_by_field_name('condition')
            if kind == 'while_statement' and truth(self.evaluate(condition, state)) is False: return False
            body = node.child_by_field_name('body')
            copy = self.loop_input(body, state)
            self.statement(body, copy)
            for name in state:
                if copy.get(name) != state[name]: state[name] = None
        elif kind in ('assignment_expression', 'update_expression'):
            self.evaluate(node, state)
        else:
            for child in node.named_children:
                if child.type.endswith('_statement'): self.statement(child, state)
                else: self.evaluate(child, state)
        return kind in ('return_statement', 'break_statement', 'goto_statement')

    def loop_input(self, body, state):
        """A later iteration need not retain the entry values of loop writes.

        Without a finite summary, forget those scalar values before inspecting
        the body. Local declarations and assignments can establish fresh facts.
        Array storage itself remains the same object; only elements may change.
        """
        result = dict(state)
        for node in walk(body):
            target = node.child_by_field_name('left') if node.type == 'assignment_expression' else (
                node.child_by_field_name('argument') if node.type == 'update_expression' else None)
            target = unwrap(target)
            if target is not None and target.type == 'identifier' and key(target) in result:
                result[key(target)] = self.unknown(key(target))
        return result


def scan_local_value_flow(root, emit):
    flow = LocalFlow(root, emit)
    for function in walk(root):
        if function.type != 'function_definition' or function.has_error: continue
        flow.unsigned = set()
        flow.pointer_names = {declared_name(n) for n in walk(function.child_by_field_name('declarator'))
                              if n.type in ('pointer_declarator', 'array_declarator')}
        flow.statement(function.child_by_field_name('body'), {})
