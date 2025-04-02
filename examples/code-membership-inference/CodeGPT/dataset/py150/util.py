import os
import json
import re
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from io import BytesIO


lits = json.load(open("/pathto/membership_inference/CodeGPT/dataset/py150/literals.json"))

def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits['str']
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )

def py_tokenize(base_dir, file_name, output_dir, file_type):
    # file_name should be the full path name
    file_paths = open(file_name).readlines()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wf = open(os.path.join(output_dir, f"{file_type}.txt"), 'w')
    for ct,path in enumerate(file_paths):
        try:
            code = open(os.path.join(base_dir, path.strip())).read()
            token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
            out_tokens = []
            prev_eol = False
            for toknum, tokval, _, _, _ in token_gen:
                tokval = " ".join(tokval.split())
                if toknum == STRING:
                    add_token = process_string(tokval)
                    out_tokens.append(add_token)
                    prev_eol = False
                elif toknum == NUMBER:
                    if tokval in lits['num']:
                        out_tokens.append(f"<NUM_LIT:{tokval}>")
                    else:
                        out_tokens.append(f"<NUM_LIT>")
                    prev_eol = False
                elif toknum in [NEWLINE, NL]:
                    if not prev_eol:
                        out_tokens.append("<EOL>")
                        prev_eol = True
                elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                    continue
                else:
                    out_tokens.append(tokval)
                    prev_eol = False
            if out_tokens[0] == "<EOL>":
                out_tokens = out_tokens[1:]
            if out_tokens[-1] == "<EOL>":
                out_tokens = out_tokens[:-1]
        except Exception:
            out_tokens = []
        out_tokens = ["<s>"] + out_tokens + ["</s>"]
        out = " ".join(out_tokens)
        wf.write(out+"\n")

        if ct % 10000 == 0:
            print(f"{file_type}: {ct} are done")
    wf.close()