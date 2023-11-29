import re
import codecs

# Clean Gadget
# Author https://github.com/johnb110/VDPython:
# For each gadget, replaces all user variables with "VAR#" and user functions with "FUN#"
# Removes content from string and character literals keywords up to C11 and C++17; immutable set
from typing import List

keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'})
# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}


def to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])


regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)


# input is a list of string lines
def clean_gadget(gadget):
    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()')

    # final cleaned gadget output to return to interface
    cleaned_gadget = []

    for line in gadget:
        # replace any non-ASCII characters with empty string
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line)
        # remove all hexadecimal literals
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line)
        # return, in order, all regex matches at string list; preserves order for semantics
        user_fun = rx_fun.findall(hex_line)
        user_var = rx_var.findall(hex_line)

        # Could easily make a "clean gadget" type class to prevent duplicate functionality
        # of creating/comparing symbol names for functions and variables in much the same way.
        # The comparison frozenset, symbol dictionaries, and counters would be class scope.
        # So would only need to pass a string list and a string literal for symbol names to
        # another function.
        for fun_name in user_fun:
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                # check to see if function name already in dictionary
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = 'FUN' + str(fun_count)
                    fun_count += 1
                # ensure that only function name gets replaced (no variable name with same
                # identifier); uses positive lookforward
                hex_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line)

        for var_name in user_var:
            # next line is the nuanced difference between fun_name and var_name
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                # check to see if variable name already in dictionary
                if var_name[0] not in var_symbols.keys():
                    var_symbols[var_name[0]] = 'VAR' + str(var_count)
                    var_count += 1
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                # print(var_name, gadget, user_var)
                hex_line = re.sub(r'\b(' + var_name[0] + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()',
                                  var_symbols[var_name[0]], hex_line)

        cleaned_gadget.append(hex_line)
    # return the list of cleaned lines
    return cleaned_gadget


# Cleaner & Tokenizer
# Author https://github.com/hazimhanif/svd-transformer/blob/master/transformer_svd.ipynb

def tokenizer(code, flag=False):
    gadget: List[str] = []
    tokenized: List[str] = []
    # remove all string literals
    no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '', code)
    # remove all character literals
    no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
    code = no_char_lit_line

    if flag:
        code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]

    for line in code.splitlines():
        if line == '':
            continue
        stripped = line.strip()
        # if "\\n\\n" in stripped: print(stripped)
        gadget.append(stripped)

    clean = clean_gadget(gadget)

    for cg in clean:
        if cg == '':
            continue

        # Remove code comments
        pat = re.compile(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)')
        cg = re.sub(pat, '', cg)

        # Remove newlines & tabs
        cg = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', '', cg)
        # Mix split (characters and words)
        splitter = r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)'
        cg = re.split(splitter, cg)

        # Remove None type
        cg = list(filter(None, cg))
        cg = list(filter(str.strip, cg))
        # code = " ".join(code)
        # Return list of tokens
        tokenized.extend(cg)

    return tokenized

#test = "((uint32_t *)(&s->boncop))[addr/sizeof(uint32_t)]"
#test2 = "(type_code[hw_breakpoint[n].type] << (16 + n*4)) |\n\n                ((uint32_t)len_code[hw_breakpoint[n].len] << (18 + n*4))"
#asd = re.split(regex_split_operators + r'|(\/)', test)
#print(list(filter(None,asd)))
#print(tokenizer(test2))
