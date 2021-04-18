import gzip
import pathlib
import re
from typing import Optional, Iterable

import jsonlines
import torch
import tqdm
from loguru import logger

# Possible keys:
#   'identifier' for method name which may be blank for anonymous fns
#   'function' for the function as a string
#   'function_tokens'
#   'docstring' for the docstring (blank for most, filled in for about 100k)
#   'docstring_tokens'
FUNCTION_ONLY_FIELDS = {"function": "function"}

_num_invalid_id = 0
_num_valid_id = 0
_fix_function_crop_regexes = [
    re.compile(r + r"(\s+|\()") for r in
    [r"\A^unction", r"\A^nction", r"\A^ction", r"\A^tion", r"\A^ion", r"\A^on", r"\A^n"]
]
_valid_identifier_regex = re.compile(r"^[a-zA-Z_$][0-9a-zA-Z_$]*$")
_url_regex = re.compile(r"https?://\S+\b")
_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")

def normalize_docstring(docstring: str):
    # Substitute urls with [URL]
    # docstring = _newline_regex.sub(r" [EOL]", docstring)
    # docstring = _whitespace_regex.sub(" ", docstring)
    docstring = _url_regex.sub("[URL]", docstring)
    return docstring


def _fix_json_dict(json_dict, require_fields, src_function_key, src_method_name_key):
    if require_fields:
        for field in require_fields:
            if field not in json_dict or not json_dict[field]:
                return None

    # Fix cropped "function" token at the begging of the function string
    for regex in _fix_function_crop_regexes:
        json_dict[src_function_key] = regex.sub(r"function\1", json_dict[src_function_key], count=1)

    if src_method_name_key in json_dict and json_dict[src_method_name_key]:
        if require_fields is not None and src_method_name_key in require_fields:
            # We need the identifier (method name) as a label. Filter invalid identifiers
            global _num_invalid_id, _num_valid_id
            if _valid_identifier_regex.match(json_dict[src_method_name_key]):
                _num_valid_id += 1
            else:
                # Skip this data point, it's not valid
                _num_invalid_id += 1
                return None

        # Remove function name from declaration, but leave it in the function body
        _function_name_regex = r"(function\s*)" + re.escape(json_dict[src_method_name_key])
        replaced_fn = re.sub(_function_name_regex, r"\1x", json_dict[src_function_key], count=1)
        json_dict[src_function_key] = replaced_fn
    else:
        json_dict[src_function_key] = "const x = " + json_dict[src_function_key]

    return json_dict


def _make_example(json_dict, fields, require_fields, src_function_key, src_method_name_key):
    json_dict = _fix_json_dict(json_dict, require_fields, src_function_key, src_method_name_key)

    if json_dict is None:
        return None

    # Normalize docstring (replace URLs)
    if require_fields and "docstring" in require_fields:
        json_dict["docstring"] = normalize_docstring(json_dict["docstring"])

    return {out_key: json_dict[json_key] for json_key, out_key in fields.items()}


class JSONLinesDataset(torch.utils.data.Dataset):
    """Defines a Dataset of columns stored in jsonlines format."""

    def __init__(
        self,
        path,
        fields=FUNCTION_ONLY_FIELDS,
        require_fields: Optional[Iterable[str]] = None,
        limit_size=-1,
        debug_charset=False,
        src_function_key="function",
        src_method_name_key="identifier",
        **kwargs,
    ):
        """Create a JSONLinesDataset given a path and field mapping dictionary.
        Arguments:
            path (str): Path to the data file. Must be in .jsonl.gz or .jsonl format.
            fields (dict[str: str]):
                The keys should be a subset of the JSON keys,
                and the values should be desired names.
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON key names
                and also enables selecting a subset of columns to load.
            require_fields:
                Set of remapped data fields required to be present
        """
        label_char_set = set()
        nl = 0
        full_path = pathlib.Path(path).resolve()
        f = open(full_path, "rb") if path.endswith(".jsonl") else full_path.open("r")
        reader = jsonlines.Reader(f)

        self.examples = []
        logger.debug(f"Loading {full_path}")
        for line in tqdm.tqdm(reader, desc=full_path.name, total=limit_size if limit_size >= 0 else 1843099):
            example = _make_example(line, fields, require_fields, src_function_key, src_method_name_key)
            if example:
                self.examples.append(example)
                if "label" in example.keys():
                    label_char_set.update(example["label"])
                if limit_size >= 0 and len(self.examples) >= limit_size:
                    print()
                    logger.info(f"WARNING: Limiting dataset size to {limit_size}")
                    break
            if debug_charset and len(label_char_set) != nl:
                logger.debug(f"update label char set: {label_char_set}")
                nl = len(label_char_set)
        f.close()

        logger.debug(f"Loaded {len(self.examples)} examples")
        if require_fields is not None and "identifier" in require_fields:
            logger.debug(f"Num examples with valid identifier field: {_num_valid_id}")
            logger.debug(f"Num examples with invalid identifier field:{_num_invalid_id}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_csnjs_dataset(filepath, label_mode, limit_size):
    """
    Returns dataset for code_search_net JavaScript language,
    which contains datapoints as dicts with keys "function" and "label"
    """
    if label_mode == "identifier":
        src_function_key = "code"
        src_method_name_key = "func_name"
        dataset_fields = {"code": "function", "func_name": "label"}
        dataset_require_fields = ["func_name"]
    elif label_mode == "docstring":
        src_function_key = "code"
        src_method_name_key = "func_name"
        dataset_fields = {"code": "function", "docstring": "label"}
        dataset_require_fields = ["docstring"]
    else:
        # Unsupervised (full) dataset has different key names
        src_function_key = "function"
        src_method_name_key = "identifier"
        dataset_fields = {"function": "function"}
        dataset_require_fields = []

    dataset = JSONLinesDataset(
        filepath,
        fields=dataset_fields,
        require_fields=dataset_require_fields,
        limit_size=limit_size,
        src_function_key=src_function_key,
        src_method_name_key=src_method_name_key,
    )
    return dataset
