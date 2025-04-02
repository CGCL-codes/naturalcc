[utils](./dataset/utils) is a directory containing:
   - [_parser.py](./dataset/utils/parser/_parser.py): to parse code into (child-value) AST
   - [child_only.py](./dataset/utils/ast/child_only.py): 
        1. transform child-value into child-only


2 AST formats:
1) child-value format:

    non-leaf nodes:
    ```shell script
    {
      "type": "statement",
      "parent": 12,
      "children": [23, 24, 25, ...], 
    }
    ```
    leaf nodes:
    ```shell script
    {
      "type": "AddOp",
      "parent": 24,
      "value": "+",
    }
    ```
2) child-only format:

    non-leaf nodes:
    ```shell script
    {
      "type": "statement",
      "parent": 12,
      "children": [23, 24, 25, ...], 
    }
    ```
    leaf nodes:
    ```shell script
    {
      "type": "AddOp",
      "parent": 24,
      "children": ["+"],
    }
    ```