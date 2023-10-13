import tokenize
from io import BytesIO


def python_code_tokenize(full_code_text):
    '''
    :param full_code_text:
    :return:
    '''
    g = tokenize.tokenize(BytesIO(full_code_text.encode('utf-8')).readline)
    tokens = []
    prev_token = None
    try:
        for x in g:
            if x.type == tokenize.ENDMARKER:  # End Marker
                continue
            # if x.type == tokenize.COMMENT:
            #     continue
            elif x.type == tokenize.NEWLINE:
                # tokens.append('NEW_LINE')
                tokens.append('[EOL]')
            elif x.type == tokenize.INDENT:
                # tokens.append('INDENT')
                continue
            elif x.type == tokenize.DEDENT:
                # tokens.append('DEDENT')
                continue
            elif x.type == tokenize.STRING:  # String
                s = x.string.strip()
                if s.startswith('"""') or s.startswith("'''"):
                    if prev_token is not None and (prev_token == '=' or prev_token == '(' or prev_token == ','):
                        tokens.append(x.string)
                    continue
                tokens.append(x.string)
                pass
            elif x.string == '\n':
                continue
            elif x.type < 57:
                tokens.append(x.string)
            prev_token = x.string.strip()
    except:
        return []
        pass
    return tokens


if __name__ == '__main__':
    expected = ["def", "ensure_dir", "(", "d", ")", ":", "if", "not", "os", ".", "path", ".", "exists", "(", "d", ")",
                ":", "try", ":", "os", ".", "makedirs", "(", "d", ")", "except", "OSError", "as", "oe", ":",
                "# should not happen with os.makedirs", "# ENOENT: No such file or directory", "if", "os", ".", "errno",
                "==", "errno", ".", "ENOENT", ":", "msg", "=", "twdd", "(",
                "\"\"\"One or more directories in the path ({}) do not exist. If\n                           you are specifying a new directory for output, please ensure\n                           all other directories in the path currently exist.\"\"\"",
                ")", "return", "msg", ".", "format", "(", "d", ")", "else", ":", "msg", "=", "twdd", "(",
                "\"\"\"An error occurred trying to create the output directory\n                           ({}) with message: {}\"\"\"",
                ")", "return", "msg", ".", "format", "(", "d", ",", "oe", ".", "strerror", ")"]
    code = "def ensure_dir(d):\n    \"\"\"\n    Check to make sure the supplied directory path does not exist, if so, create it. The\n    method catches OSError exceptions and returns a descriptive message instead of\n    re-raising the error.\n\n    :type d: str\n    :param d: It is the full path to a directory.\n\n    :return: Does not return anything, but creates a directory path if it doesn't exist\n             already.\n    \"\"\"\n    if not os.path.exists(d):\n        try:\n            os.makedirs(d)\n        except OSError as oe:\n            # should not happen with os.makedirs\n            # ENOENT: No such file or directory\n            if os.errno == errno.ENOENT:\n                msg = twdd(\"\"\"One or more directories in the path ({}) do not exist. If\n                           you are specifying a new directory for output, please ensure\n                           all other directories in the path currently exist.\"\"\")\n                return msg.format(d)\n            else:\n                msg = twdd(\"\"\"An error occurred trying to create the output directory\n                           ({}) with message: {}\"\"\")\n                return msg.format(d, oe.strerror)"
    generated = python_code_tokenize(code)
    eidx = 0
    for token in generated:
        if token in ['NEW_LINE', 'INDENT', 'DEDENT']:
            continue
        if expected[eidx] != token:
            print(eidx, token, expected[eidx], sep='\n')
            break
        eidx += 1
