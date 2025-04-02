import pickle
import json
import subprocess
import regex as re

sql_prompt = """
Please generate SQL according to given error line content and error message. 
You can only use the 'getScope()' function to get the parent scope (which is a module or class) of a symbol, and use 'getId()' to get the identifier name of a symbol.
Do not use other functions.
Do not use getName(), use getId().
The only possible data types are Module m, Function f, Variable v, and Class c.
Error Line Content:
```python
from keys import hashkey
```
Error Message:
```
Unable to import 'keys'
```
SQL:
```sql
from Module m where m.inSource() and v.getScope() = m select m
```

Error Line Content:
```python
        for name in logging.root.manager.loggerDict:
```
Error Message:
```
Instance of 'RootLogger' has no 'loggerDict' member
```
SQL:
```sql
from Module m, Class c, Function cf where m.inSource() and m.contains(c) and c.contains(cf) and cf.getScope() = c and c.getId() = "RootLogger" select m, c, cf
```

Error Line Content:
```python
            from ._bolt5 import AsyncBolt5x3
```
Error Message:
```
No name 'AsyncBolt5x3' in module 'neo4j._sync.io._bolt5'
```
SQL:
```sql
from Module m, Variable v where m.inSource() and v.getScope() = m and m.getId() = "neo4j._sync.io._bolt5" select m, v
```

Error Line Content:
```python
        error_indices = self._get_err_indices()
```
Error Message:
```
No value for argument 'coord_name' in method call
```
SQL:
```sql
from Module m, Function f where m.inSource() and m.contains(f) and f.getId() = "coord_name" select m, f
```

Error Line Content:
```python
{0}
```
Error Message:
```
{1}
```
SQL:
"""

qlpack_str = """name: my-python-project
version: 0.0.0
libraryPathDependencies: codeql-python"""

def run_cmd(cmds, is_pred_task=False):
    if is_pred_task:
        cmds.append('--run_pred')
    print("Running CMD")
    print(" ".join(cmds))
    try:
        output = subprocess.check_output(cmds, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)
        raise e
    output = output.decode()
    return output

def convert_paragraphs_to_context(s, connction='\n', pars = None):
    if pars == None: # use s 
        return connction.join(['{}'.format(p) for i, p in enumerate(s['pars'])])
    else:
        return connction.join(['{}'.format(p) for i, p in enumerate(pars)])

def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)

def read_jsonlines(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [json.loads(x) for x in lines]

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def dump_jsonl(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        for item in obj:
            json.dump(item, f, indent=indent)
            f.write('\n')


def add_engine_argument(parser):
    parser.add_argument('--engine',
                            default='text-davinci-003',
                            choices=['davinci', 'text-davinci-001', 'text-davinci-002', 'code-davinci-002', 'gpt-3.5-turbo'])

def specify_engine(args):
    if args.model=='gptj':
        args.engine_name = 'gptj'
        args.engine = 'gptj'
    elif args.model == 'code-llama':
        args.engine_name = 'code-llama'
        args.engine = 'code-llama'
    elif args.model == 'starcoder':
        args.engine_name = 'starcoder'
        args.engine = 'starcoder'
    elif args.model == 'codegen':
        args.engine_name = 'codegen'
        args.engine = 'codegen'
    elif args.model == 'codegen-1b':
        args.engine_name = 'codegen-1b'
        args.engine = 'codegen-1b'
    elif args.model == 'codegen-2b':
        args.engine_name = 'codegen-2b'
        args.engine = 'codegen-2b'
    elif args.model == 'gpt-3.5-turbo':
        args.engine_name = 'gpt-3.5-turbo'
        args.engine = 'gpt-3.5-turbo'
    elif args.model == 'code-llama':
        args.engine_name = 'code-llama'
        args.engine = 'code-llama'
    else:
        args.engine_name = args.engine



def codereval_project_to_directory_name(project_name):
    repo_name_to_directory = {
        "ansible-security/ansible_collections.ibm.qradar": "ansible-security---ansible_collections.ibm.qradar",
        "champax/pysolbase": "champax---pysolbase",
        "gopad/gopad-python": "gopad---gopad-python",
        "mozilla/relman-auto-nag": "mozilla---relman-auto-nag",
        "openstack/neutron-lib": "openstack---neutron-lib",
        "pre-commit/pre-commit": "pre-commit---pre-commit",
        "scieloorg/packtools": "scieloorg---packtools",
        "SoftwareHeritage/swh-lister": "SoftwareHeritage---swh-lister",
        "witten/atticmatic": "witten---atticmatic",
        "awsteiner/o2sclpy": "awsteiner---o2sclpy",
        "cloudmesh/cloudmesh-common": "cloudmesh---cloudmesh-common",
        "ikus060/rdiffweb": "ikus060---rdiffweb",
        "MozillaSecurity/lithium": "MozillaSecurity---lithium",
        "ossobv/planb": "ossobv---planb",
        "rak-n-rok/Krake": "rak-n-rok---Krake",
        "scrolltech/apphelpers": "scrolltech---apphelpers",
        "standalone": "standalone",
        "witten/borgmatic": "witten---borgmatic",
        "bastikr/boolean": "bastikr---boolean",
        "commandline/flashbake": "commandline---flashbake",
        "infobloxopen/infoblox-client": "infobloxopen---infoblox-client",
        "mwatts15/rdflib": "mwatts15---rdflib",
        "pexip/os-python-cachetools": "pexip---os-python-cachetools",
        "redhat-openstack/infrared": "redhat-openstack---infrared",
        "SEED-platform/py-seed": "SEED-platform---py-seed",
        "sunpy/radiospectra": "sunpy---radiospectra",
        "ynikitenko/lena": "ynikitenko---lena",
        "bazaar-projects/docopt-ng": "bazaar-projects---docopt-ng",
        "cpburnz/python-sql-parameters": "cpburnz---python-sql-parameters",
        "jaywink/federation": "jaywink---federation",
        "neo4j/neo4j-python-driver": "neo4j---neo4j-python-driver",
        "pexip/os-python-dateutil": "pexip---os-python-dateutil",
        "rougier/matplotlib": "rougier---matplotlib",
        "sipwise/repoapi": "sipwise---repoapi",
        "turicas/rows": "turicas---rows",
        "zimeon/ocfl-py": "zimeon---ocfl-py",
        "burgerbecky/makeprojects": "burgerbecky---makeprojects",
        "eykd/prestoplot": "eykd---prestoplot",
        "kirankotari/shconfparser": "kirankotari---shconfparser",
        "openstack/cinder": "openstack---cinder",
        "pexip/os-zope": "pexip---os-zope",
        "santoshphilip/eppy": "santoshphilip---eppy",
        "skorokithakis/shortuuid": "skorokithakis---shortuuid",
        "ufo-kit/concert": "ufo-kit---concert",
        "atmosphere-atmosphere-2.7.x": "atmosphere",
        "fastjson2-main": "fastjson2",
        "framework-master": "framework",
        "hasor-master": "hasor",
        "jgrapht-master": "jgrapht",
        "jjwt-master": "jjwt",
        "logging-log4j1-main": "logging-log4j1",
        "protostuff-master": "protostuff",
        "skywalking-master": "skywalking",
        "interviews-master": "interviews"
    }
    return repo_name_to_directory[project_name]


def extract_single_quoted_strings(input_string):
    # Regex pattern to match substrings within single quotes
    pattern = r"'([^']*)'"
    # Find all matches and return them as a list
    return re.findall(pattern, input_string)


def remove_leading_indent(gold_text):
    """
    Remove leading indent in a string containing Python code.
    The first line's indent is set to 0 spaces, and other lines have corresponding indents removed.
    """

    lines = gold_text.split('\n')
    if len(lines) <= 1:
        return gold_text.lstrip()
    leading_spaces = len(lines[0]) - len(lines[0].lstrip(' '))
    lines[0] = lines[0].lstrip(' ')
    for i in range(1, len(lines)):
        lines[i] = lines[i][leading_spaces:]

    # Join the lines back into a single string
    return '\n'.join(lines)

def split_identifiers_non_identifiers(statement):
    # Splitting the string based on Python identifier rules
    # A Python identifier must start with a letter (a-z, A-Z) or underscore (_)
    # and can be followed by any number of letters, digits (0-9), or underscores
    identifier_pattern = r'\b[_a-zA-Z][_a-zA-Z0-9]*\b'
    parts = re.findall(identifier_pattern, statement)

    # Finding non-identifier parts
    non_identifier_parts = re.split(identifier_pattern, statement)

    # Combining identifier and non-identifier parts in order
    result = []
    for i in range(len(non_identifier_parts)):
        if non_identifier_parts[i]:
            result.append(non_identifier_parts[i])
        if i < len(parts):
            result.append(parts[i])

    return result

def is_identifier(string):
    if not string:  # Check if the string is empty
        return False

    if not (string[0].isalpha() or string[0] == '_'):  # Check if first character is a letter or underscore
        return False

    for char in string[1:]:  # Check remaining characters
        if not (char.isalnum() or char == '_'):
            return False

    return True

